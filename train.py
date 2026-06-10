"""
训练脚本

支持：
- BERT 微调（抽取式问答 / 分类）
- LoRA 微调
- 混合精度训练
- 断点续训
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from utils.config import Config
# from models.qa_model import QAModel
# from models.lora import LoRAModel
# from data.dataloader import QADataset, create_dataloader


def main():
    parser = argparse.ArgumentParser(description="智能问答系统 — 训练脚本")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", "-r", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = Config(args.config)
    train_cfg = cfg.get("training", default={})

    # 命令行覆盖
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.lr:
        train_cfg["learning_rate"] = args.lr

    print("=" * 50)
    print("  智能问答系统 — 训练")
    print("=" * 50)
    print(f"[配置] {args.config}")
    print(f"[训练参数] epochs={train_cfg.get('epochs')}, "
          f"batch_size={train_cfg.get('batch_size')}, "
          f"lr={train_cfg.get('learning_rate')}")

    # TODO: 实现完整训练循环
    # 当前为骨架代码，后续课程逐步填充
    print("\n[提示] train.py 为骨架代码，请按学习路线逐步实现训练逻辑。")
    print("  第4课: 实现 BERT 微调训练循环")
    print("  第5课: 实现评估与调参")


if __name__ == "__main__":
    main()
