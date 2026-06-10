"""
训练脚本 — 检索式问答模型

当前检索式问答（TF-IDF / BM25）为无监督方法，无需训练。
首次运行 predict.py 时自动构建索引。

本脚本为后续课程预留，用于：
- BERT 微调（抽取式阅读理解）
- 双编码器检索模型训练（Sentence-BERT）
- LoRA 微调（分类/匹配）

用法：
    # 当前：构建 TF-IDF 索引（等同于运行 predict.py）
    python retrieval_qa/predict.py -q "测试"

    # 后续（安装 torch + transformers 后）：
    python retrieval_qa/train.py --config configs/default.yaml

训练数据格式：
    data/raw/faq_expanded.json 中的问答对，
    或自定义 JSON 数据（见 README.md）
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="智能问答系统 — 训练脚本（预留）")
    parser.add_argument("--config", "-c", type=str,
                        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"))
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
    print("  检索式问答 — 训练脚本")
    print("=" * 50)
    print(f"[配置] {args.config}")
    print(f"[训练参数] epochs={train_cfg.get('epochs')}, "
          f"batch_size={train_cfg.get('batch_size')}, "
          f"lr={train_cfg.get('learning_rate')}")

    # ── 检查是否已安装深度学习依赖 ─────────────────────────────────────
    try:
        import torch
        import transformers
        print(f"[OK] PyTorch {torch.__version__}, Transformers {transformers.__version__}")
    except ImportError:
        print("\n[提示] 当前检索式问答使用 TF-IDF/BM25，无需训练。")
        print("  要使用 BERT 等深度模型，请安装:")
        print("  pip install torch transformers")
        print("\n  否则直接运行 predict.py 即可:")
        print("  python retrieval_qa/predict.py -m faq")
        return

    # ── 训练逻辑骨架（后续课程填充） ──────────────────────────────────
    print(f"\n[数据] 加载: {cfg.get('data', 'faq_path')}")
    print("[模型] 加载预训练编码器...")
    print("[训练] 开始训练循环...")
    print("[保存] 模型保存到 checkpoints/")

    # TODO: 完整训练循环
    # 1. 加载 tokenizer + 模型
    # 2. 准备数据集
    # 3. 训练循环（前向 → 损失 → 反向 → 更新）
    # 4. 保存 checkpoint


if __name__ == "__main__":
    main()
