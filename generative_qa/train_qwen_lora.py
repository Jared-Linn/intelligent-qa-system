"""
千问 LoRA 微调训练脚本

基于 HuggingFace TRL SFTTrainer + PEFT 实现。
使用 Qwen2.5-3B-Instruct 对 FAQ 数据进行指令微调。

用法:
    python generative_qa/train_qwen_lora.py
    python generative_qa/train_qwen_lora.py --epochs 30 --batch_size 8
    HF_HOME=/autodl-pub/huggingface python generative_qa/train_qwen_lora.py
    python generative_qa/train_qwen_lora.py --resume /autodl-pub/checkpoints/qwen_lora/checkpoint-xxx
    python generative_qa/train_qwen_lora.py --dry_run

Web 对接:
    训练产出 zip 后通过 POST /api/models/{id}/upload 上传。
    web/services/infer_generative.py 自动加载 LoRA 推理。
"""

import sys
import os
import json
import math
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_qwen_lora")


def parse_args():
    parser = argparse.ArgumentParser(description="千问 LoRA 微调训练")
    parser.add_argument("--config", "-c", type=str,
                        default=str(Path(__file__).resolve().parent / "configs" / "train.yaml"))
    parser.add_argument("--model_name", type=str, default=None, help="覆盖模型名称")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖输出目录")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印配置, 不训练")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        logger.error("请安装 PyYAML: pip install pyyaml")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_deps():
    missing = []
    try:
        import torch
        logger.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            mem = getattr(p, 'total_memory', getattr(p, 'total_mem', 0))
            logger.info(f"GPU: {p.name}, VRAM: {mem / 1024**3:.1f}GB")
    except ImportError:
        missing.append("torch")

    for pkg in ["transformers", "peft", "trl", "accelerate", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"缺少依赖: {', '.join(missing)}")
        logger.error("安装: pip install torch transformers peft trl accelerate datasets")
        sys.exit(1)


def load_faq_data(data_path: str, val_split: float = 0.1):
    """加载 FAQ JSON, 分割训练/验证集"""
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs = []
    for item in raw:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            pairs.append({"instruction": q, "output": a})

    import random
    random.shuffle(pairs)

    split = max(1, int(len(pairs) * (1 - val_split)))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    logger.info(f"[数据] 总 {len(pairs)}, 训练 {len(train_pairs)}, 验证 {len(val_pairs)}")
    return train_pairs, val_pairs


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI 覆盖
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

    hf_home = os.environ.get("HF_HOME", "")

    # ── 打印配置 ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  千问 LoRA 微调训练")
    logger.info("=" * 60)
    logger.info(f"  模型:          {cfg['model']['name']}")
    logger.info(f"  缓存目录:      {cfg['model'].get('cache_dir', '默认')}")
    if hf_home:
        logger.info(f"  HF_HOME:       {hf_home}")
    logger.info(f"  LoRA r:        {cfg['lora']['r']}")
    logger.info(f"  LoRA alpha:    {cfg['lora']['alpha']}")
    logger.info(f"  Epochs:        {cfg['training']['num_epochs']}")
    logger.info(f"  Batch/GPU:     {cfg['training']['batch_size']}")
    logger.info(f"  Grad accum:    {cfg['training']['gradient_accumulation_steps']}")
    eff = cfg['training']['batch_size'] * cfg['training']['gradient_accumulation_steps']
    logger.info(f"  等效 batch:    {eff}")
    logger.info(f"  LR:            {cfg['training']['learning_rate']}")
    logger.info(f"  Warmup ratio:  {cfg['training']['warmup_ratio']}")
    logger.info(f"  FP16:          {cfg['training']['fp16']}")
    logger.info(f"  Output:        {cfg['training']['output_dir']}")
    logger.info(f"  Data:          {cfg['data']['train_path']}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[Dry run] 跳过训练")
        return

    check_deps()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # ── 1. Tokenizer ─────────────────────────────────────────────────
    logger.info("[1/5] 加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"].get("cache_dir", None),
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 2. Qwen Chat Template format ─────────────────────────────────
    def format_func(example):
        messages = [
            {"role": "system", "content": "你是一个专业的中文问答助手，请基于知识库准确简洁地回答问题。"},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # ── 3. 模型 ──────────────────────────────────────────────────────
    logger.info("[2/5] 加载基座模型 ...")
    use_bf16 = cfg["training"].get("bf16", False)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"].get("cache_dir", None),
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("[优化] 梯度检查点已启用")

    # ── 4. LoRA ──────────────────────────────────────────────────────
    logger.info("[3/5] 配置 LoRA ...")
    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 预期: ~8.4M / ~3.1B (0.27%)

    # ── 5. 数据 ──────────────────────────────────────────────────────
    logger.info("[4/5] 准备数据 ...")
    data_path = ROOT / cfg["data"]["train_path"]
    train_pairs, val_pairs = load_faq_data(
        str(data_path), cfg["data"].get("val_split", 0.1),
    )

    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)

    # ── 6. SFTTrainer ────────────────────────────────────────────────
    logger.info("[5/5] 初始化 SFTTrainer ...")

    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    save_strategy = cfg["training"].get("save_strategy", "steps")
    eval_strategy = cfg["training"].get("eval_strategy", "steps")

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=cfg["training"]["num_epochs"],
            per_device_train_batch_size=cfg["training"]["batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            learning_rate=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            warmup_ratio=cfg["training"]["warmup_ratio"],
            fp16=cfg["training"]["fp16"],
            bf16=use_bf16,
            max_grad_norm=cfg["training"]["max_grad_norm"],
            logging_steps=cfg["training"]["logging_steps"],
            save_strategy=save_strategy,
            save_steps=cfg["training"].get("save_steps", 200),
            save_total_limit=cfg["training"]["save_total_limit"],
            eval_strategy=eval_strategy if val_dataset else "no",
            eval_steps=cfg["training"].get("eval_steps", 200),
            report_to=cfg["training"].get("report_to", "tensorboard"),
            run_name=cfg["training"]["run_name"],
            logging_dir=cfg["training"].get("logging_dir", None),
            remove_unused_columns=False,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            load_best_model_at_end=True if val_dataset and eval_strategy != "no" else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=42,
            max_length=cfg["data"]["max_length"],
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        formatting_func=format_func,
    )

    # ── 训练 ─────────────────────────────────────────────────────────
    logger.info("[训练] 开始 ...")
    if args.resume:
        logger.info(f"恢复训练: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── 保存 ─────────────────────────────────────────────────────────
    logger.info(f"[保存] LoRA 权重到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存训练配置 (方便复现)
    with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    logger.info(f"[完成] 训练结束! 文件: {os.listdir(output_dir)}")

    # ── 打包提示 ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  上传至 Web 系统:")
    logger.info(f"    cd {output_dir}")
    logger.info("    zip -r lora.zip adapter_config.json adapter_model.safetensors \\")
    logger.info("      tokenizer_config.json tokenizer.json special_tokens_map.json")
    logger.info("  POST /api/models/{id}/upload + lora.zip")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
