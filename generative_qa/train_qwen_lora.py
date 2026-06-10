"""
千问 LoRA 微调训练脚本

基于 HuggingFace TRL + PEFT 实现。
使用 Qwen2.5 系列模型对 FAQ 数据进行指令微调。

用法:
    python generative_qa/train_qwen_lora.py
    python generative_qa/train_qwen_lora.py --epochs 5 --batch_size 8
    python generative_qa/train_qwen_lora.py --model_name Qwen/Qwen2.5-7B-Instruct

环境:
    conda activate nlp
    pip install torch transformers datasets accelerate peft trl tensorboard

输出:
    checkpoints/qwen_lora/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
"""

import sys
import os
import json
import argparse
import yaml
from pathlib import Path

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="千问 LoRA 微调")
    parser.add_argument("--config", "-c", type=str,
                        default=str(Path(__file__).resolve().parent / "configs" / "train.yaml"))
    parser.add_argument("--model_name", type=str, default=None, help="覆盖模型名称")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖输出目录")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_instruction_data(data_path: str) -> list:
    """
    将 FAQ 数据格式化为指令微调格式。

    输出每条样本格式:
        {"instruction": "什么是机器学习", "output": "机器学习是..."}
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


def main():
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
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir

    model_name = cfg["model"]["name"]
    output_dir = cfg["training"]["output_dir"]
    lora_r = cfg["lora"]["r"]
    lora_alpha = cfg["lora"]["alpha"]
    lora_dropout = cfg["lora"]["dropout"]
    target_modules = cfg["lora"]["target_modules"]

    print("=" * 60)
    print(f"  千问 LoRA 微调")
    print(f"  模型: {model_name}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  epochs={cfg['training']['num_epochs']}, "
          f"batch={cfg['training']['batch_size']}, "
          f"lr={cfg['training']['learning_rate']}")
    print(f"  输出: {output_dir}")
    print("=" * 60)

    # ── 加载数据 ──────────────────────────────────────────────────────
    data_path = ROOT / cfg["data"]["train_path"]
    formatted_data = format_instruction_data(str(data_path))

    dataset = Dataset.from_list(formatted_data)
    train_test = dataset.train_test_split(
        test_size=cfg["data"].get("val_split", 0.1),
        seed=42,
    )
    print(f"[数据] 训练集: {len(train_test['train'])} 条, "
          f"验证集: {len(train_test['test'])} 条")

    # ── 加载 Tokenizer ────────────────────────────────────────────────
    print(f"[加载] Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Chat template 格式
    def format_func(example):
        messages = [
            {"role": "system", "content": "你是一个专业的中文问答助手。"},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # ── 加载模型 ──────────────────────────────────────────────────────
    print(f"[加载] 模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # 训练时禁用 KV 缓存
    model.config.pad_token_id = tokenizer.pad_token_id

    # 启用梯度检查点
    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("[优化] 梯度检查点已启用")

    # ── 应用 LoRA ─────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 预期输出: trainable params: ~8.4M / total: ~3.1B (0.27%)

    # ── 训练参数 ──────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        eval_steps=train_cfg.get("eval_steps", 200),
        evaluation_strategy="steps" if train_test["test"] else "no",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", not train_cfg.get("fp16", False)),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to=train_cfg.get("report_to", "tensorboard"),
        run_name=train_cfg.get("run_name", "qwen_lora_faq"),
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )

    # ── 创建 Trainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        tokenizer=tokenizer,
        formatting_func=format_func,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
        ),
        max_seq_length=cfg["data"].get("max_length", 512),
    )

    # ── 开始训练 ──────────────────────────────────────────────────────
    print("[训练] 开始...")
    trainer.train()

    # ── 保存 LoRA 权重 ────────────────────────────────────────────────
    print(f"[保存] LoRA 权重到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[完成] 训练结束！")

    # ── 打包为 zip（方便 Web 上传） ────────────────────────────────────
    import shutil
    zip_path = output_dir.rstrip("/") + ".zip"
    shutil.make_archive(output_dir.rstrip("/"), "zip", output_dir)
    print(f"[打包] {zip_path}")


if __name__ == "__main__":
    main()
