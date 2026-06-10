"""
生成式推理引擎 — Qwen2.5 LoRA 问答

采用方案A（即用即加载）：每次请求加载目标 LoRA，用完释放。
GPU 上同一时刻只保留一个 LoRA。

注意：需要安装 torch + transformers + peft 才能使用。
"""

import os
from pathlib import Path
from typing import Optional

# torch 和 peft 在需要时惰性导入，避免未安装时模块加载崩溃

# 全局变量：当前加载的模型信息
_current_lora = {"model_id": None, "model": None, "tokenizer": None}


def _check_deps():
    """检查依赖是否安装，返回缺失列表"""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import peft
    except ImportError:
        missing.append("peft")
    return missing


class GenerativeInferenceEngine:
    """生成式推理引擎"""

    def __init__(self):
        self.base_model_name = "Qwen/Qwen2.5-3B-Instruct"
        self._base_model = None
        self._tokenizer_base = None
        self._loaded = False

    def _ensure_base(self):
        """加载基座模型（全局共享）"""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("[生成式] 加载基座模型:", self.base_model_name)
        self._tokenizer_base = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer_base.pad_token is None:
            self._tokenizer_base.pad_token = self._tokenizer_base.eos_token

        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._loaded = True
        print("[生成式] 基座模型加载完成")

    def _load_lora(self, model: dict):
        """加载 LoRA 权重（如果当前已是目标 LoRA，跳过）"""
        global _current_lora
        import torch
        from peft import PeftModel

        model_id = model["id"]
        if _current_lora["model_id"] == model_id:
            return _current_lora["model"]

        if _current_lora["model"] is not None:
            print(f"[生成式] 卸载 LoRA: {_current_lora['model_id']}")
            _current_lora["model"] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        lora_path = model.get("file_path")
        if not lora_path:
            raise ValueError("模型没有关联的 LoRA 权重文件")

        lora_dir = Path(lora_path)
        if not lora_dir.exists() or not (lora_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"LoRA 权重不存在: {lora_path}")

        print(f"[生成式] 加载 LoRA: {model_id}")
        peft_model = PeftModel.from_pretrained(self._base_model, str(lora_dir))
        peft_model.eval()

        _current_lora["model_id"] = model_id
        _current_lora["model"] = peft_model
        return peft_model

    def predict(self, model: dict, question: str) -> dict:
        """
        执行生成式问答。

        如果依赖缺失、无 GPU、或无 LoRA 权重，返回友好提示而非崩溃。
        """
        # 1. 检查依赖
        missing = _check_deps()
        if missing:
            return {
                "answer": f"[生成式问答] 服务端缺少依赖: {', '.join(missing)}。请安装: pip install torch transformers peft",
                "confidence": 0.0,
                "source": "missing_deps",
            }

        import torch

        # 2. 检查 GPU
        if not torch.cuda.is_available():
            return {
                "answer": "[生成式问答] 当前没有可用 GPU，生成式模型需要 GPU 推理。请在有 GPU 的机器上运行。",
                "confidence": 0.0,
                "source": "no_gpu",
            }

        # 3. 检查 LoRA 权重
        lora_path = model.get("file_path")
        if not lora_path:
            return {
                "answer": "[生成式问答] 模型未上传 LoRA 权重。请先上传。",
                "confidence": 0.0,
                "source": "no_lora",
            }

        lora_dir = Path(lora_path)
        if not lora_dir.exists() or not (lora_dir / "adapter_config.json").exists():
            return {
                "answer": f"[生成式问答] LoRA 权重文件不存在: {lora_path}",
                "confidence": 0.0,
                "source": "lora_not_found",
            }

        # 4. 推理
        try:
            self._ensure_base()
            lora_model = self._load_lora(model)

            from transformers import AutoTokenizer

            messages = [
                {"role": "system", "content": "你是一个专业的中文问答助手，请准确简洁地回答问题。"},
                {"role": "user", "content": question},
            ]
            prompt = self._tokenizer_base.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self._tokenizer_base(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(lora_model.device)

            with torch.no_grad():
                outputs = lora_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self._tokenizer_base.pad_token_id,
                    eos_token_id=self._tokenizer_base.eos_token_id,
                )

            answer = self._tokenizer_base.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            return {
                "answer": answer or "[生成式] 模型未生成有效答案",
                "confidence": 0.8,
                "source": "qwen_lora",
            }

        except Exception as e:
            return {
                "answer": f"[生成式问答] 推理失败: {str(e)}",
                "confidence": 0.0,
                "source": "inference_error",
            }
