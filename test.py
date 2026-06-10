"""
测试脚本

验证模型在测试集上的功能、准确率和推理速度。
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import Config
from modules.knowledge_retrieval import TfidfRetriever
from modules.answer_generation import AnswerGenerator
from utils.metrics import MetricsCalculator


def main():
    parser = argparse.ArgumentParser(description="智能问答系统 — 测试脚本")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, help="测试数据路径（可选，默认使用训练数据子集）")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    cfg = Config(args.config)
    retriever_path = cfg.get("paths", "vectorizer")

    # 加载检索器
    print("[测试] 加载检索器...")
    retriever = TfidfRetriever(config=cfg.config)
    retriever.load(retriever_path)

    # 准备测试数据（取前 N 条）
    raw_path = args.data or cfg.get("data", "faq_path")
    with open(raw_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 留出 20% 作为测试集
    test_size = max(1, len(test_data) // 5)
    test_data = test_data[:test_size]
    print(f"[测试] 测试集大小: {len(test_data)}")

    # 测试推理
    generator = AnswerGenerator(method="direct")
    metrics = MetricsCalculator()

    predictions = []
    ground_truths = []
    latencies = []

    print("[测试] 开始测试...")
    for item in test_data:
        query = item["question"]
        true_answer = item["answer"]

        start = time.time()
        results = retriever.retrieve(query, top_k=args.top_k)
        result = generator.generate(query, results)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

        predictions.append(result["answer"])
        ground_truths.append(true_answer)

    # 计算指标
    em_f1 = metrics.compute_em_f1(predictions, ground_truths)
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "=" * 50)
    print("  测试报告")
    print("=" * 50)
    print(f"  样本数:     {len(predictions)}")
    print(f"  Exact Match: {em_f1['exact_match']:.4f}")
    print(f"  F1 Score:    {em_f1['f1']:.4f}")
    print(f"  平均延迟:    {avg_latency:.1f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
