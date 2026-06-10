"""
千问 LoRA 微调训练脚本

基于 HuggingFace TRL + PEFT 实现。
使用 Qwen2.5 系列模型对 FAQ 数据进行指令微调。

用法:
    python generative_qa/train_qwen_lora.py
    python generative_qa/train_qwen_lora.py --config generative_qa/configs/train.yaml

环境:
    conda activate nlp
    pip install torch transformers datasets accelerate peft trl
"""

import sys
import os
import json
import argparse
from pathlib import Path

# 将项目根目录加入 sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def parse_args():
    parser = argparse.ArgumentParser(description="千问 LoRA 微调")
    parser.add_argument("--config", "-c", type=str,
                        default=str(Path(__file__).resolve().parent / "configs" / "train.yaml"))
    parser.add_argument("--model_name", type=str, help="覆盖模型的名称")
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数")
    parser.add_argument("--batch_size", type=int, help="覆盖 batch size")
    parser.add_argument("--lr", type=float, help="覆盖学习率")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_training_data(data_path: str, max_length: int = 512) -> list:
    """
    将 FAQ 数据格式化为指令微调格式。

    输出格式:
        {
            "instruction": "用户问题",
            "output": "标准答案",
        }
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted = []
    for item in raw_data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if not question or not answer:
            continue
        formatted.append({
            "instruction": question,
            "output": answer,
        })

    print(f"[数据] 共 {len(formatted)} 条训练样本")
    return formatted


def train():
    args = parse_args()
    cfg = load_config(args.config)

    # 命令行参数覆盖
    if args.model_name:
        cfg["model"]["name"] = args.model_name
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    # ── 加载数据 ──────────────────────────────────────────────────────
    data_path = Path(root) / cfg["data"]["train_path"]
    formatted_data = format_training_data(str(data_path), cfg["data"]["max_length"])

    # ── 训练配置 ──────────────────────────────────────────────────────
    print(f"[配置] 模型: {cfg['model']['name']}")
    print(f"[配置] 训练参数: epochs={cfg['training']['num_epochs']}, "
          f"batch={cfg['training']['batch_size']}, "
          f"lr={cfg['training']['learning_rate']}, "
          f"grad_accum={cfg['training']['gradient_accumulation_steps']}")
    print(f"[配置] LoRA: r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}")
    print(f"[GPU] 可用: {os.system('python -c \"import torch; print(torch.cuda.is_available())\"') == 0}")

    # TODO: 完整训练循环实现
    # 需要安装: pip install transformers datasets accelerate peft trl
    #
    # 核心步骤:
    # 1. 加载 tokenizer 和模型 (AutoModelForCausalLM, AutoTokenizer)
    # 2. 应用 LoRA (PEFT LoraConfig + get_peft_model)
    # 3. 格式化数据为 chat template
    # 4. 使用 SFTTrainer 或手写训练循环
    # 5. 保存 LoRA 权重
    #
    # 参考代码结构:
    #   from transformers import AutoModelForCausalLM, AutoTokenizer
    #   from peft import LoraConfig, get_peft_model
    #   from trl import SFTTrainer
    #
    #   model = AutoModelForCausalLM.from_pretrained(
    #       cfg['model']['name'],
    #       torch_dtype=torch.bfloat16,
    #       device_map="auto",
    #   )
    #   tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    #
    #   lora_config = LoraConfig(
    #       r=cfg['lora']['r'],
    #       lora_alpha=cfg['lora']['alpha'],
    #       target_modules=cfg['lora']['target_modules'],
    #       lora_dropout=cfg['lora']['dropout'],
    #       task_type="CAUSAL_LM",
    #   )
    #   model = get_peft_model(model, lora_config)
    #
    #   trainer = SFTTrainer(
    #       model=model,
    #       train_dataset=dataset,
    #       args=transformers.TrainingArguments(**train_args),
    #       tokenizer=tokenizer,
    #   )
    #   trainer.train()
    #   trainer.save_model(cfg['training']['output_dir'])

    print("\n[提示] 训练脚本为骨架代码。运行前请安装依赖:")
    print("  pip install torch transformers datasets accelerate peft trl")
    print(f"\n准备就绪后运行: python {__file__}")


if __name__ == "__main__":
    train()
