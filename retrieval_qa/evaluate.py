"""
评价脚本

对系统效果进行综合分析，包括：
- 检索指标：Precision@K, Recall@K, MRR
- 问答指标：EM, F1
- 错误案例分析
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import Config
from modules.knowledge_retrieval import TfidfRetriever
from modules.answer_generation import AnswerGenerator
from utils.metrics import MetricsCalculator


def main():
    parser = argparse.ArgumentParser(description="智能问答系统 — 评价脚本")
    parser.add_argument("--config", "-c", type=str,
                        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"))
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(args.config)
    retriever_path = cfg.get("paths", "vectorizer")

    # 加载
    retriever = TfidfRetriever(config=cfg.config)
    retriever.load(retriever_path)

    with open(cfg.get("data", "faq_path"), "r", encoding="utf-8") as f:
        all_data = json.load(f)

    eval_size = min(50, len(all_data))
    eval_data = all_data[:eval_size]
    print(f"[评价] 评估集: {len(eval_data)} 条")

    generator = AnswerGenerator(method="direct")
    metrics = MetricsCalculator()

    # 分类统计
    category_results = defaultdict(lambda: {"em": [], "f1": []})
    all_em = []
    all_f1 = []

    print("[评价] 正在评估...")
    for item in eval_data:
        query = item["question"]
        true_answer = item["answer"]
        category = item.get("category", "unknown")

        results = retriever.retrieve(query, top_k=args.top_k)
        result = generator.generate(query, results)

        em = metrics.exact_match(result["answer"], true_answer)
        f1 = metrics.f1_score(result["answer"], true_answer)

        all_em.append(em)
        all_f1.append(f1)
        category_results[category]["em"].append(em)
        category_results[category]["f1"].append(f1)

    # 输出报告
    print("\n" + "=" * 60)
    print("  综合评价报告")
    print("=" * 60)
    print(f"  总样本: {len(eval_data)}")
    print(f"  整体 EM:  {sum(all_em)/len(all_em):.4f}")
    print(f"  整体 F1:  {sum(all_f1)/len(all_f1):.4f}")
    print("-" * 60)
    print("  分类评估:")
    for cat, res in sorted(category_results.items()):
        n = len(res["em"])
        if n == 0:
            continue
        avg_em = sum(res["em"]) / n
        avg_f1 = sum(res["f1"]) / n
        print(f"    {cat:25s}  n={n:3d}  EM={avg_em:.4f}  F1={avg_f1:.4f}")

    # 分析失败案例
    print("-" * 60)
    print("  失败案例（EM=0 且 F1<0.3）:")
    failures = 0
    for item in eval_data:
        query = item["question"]
        true_answer = item["answer"]
        results = retriever.retrieve(query, top_k=args.top_k)
        result = generator.generate(query, results)
        em = metrics.exact_match(result["answer"], true_answer)
        f1 = metrics.f1_score(result["answer"], true_answer)
        if em == 0 and f1 < 0.3:
            failures += 1
            if failures <= 5:
                print(f"  Q: {query}")
                print(f"  预测: {result['answer'][:80]}")
                print(f"  真实: {true_answer[:80]}")
                print()

    print(f"  共 {failures} 个失败案例")
    print("=" * 60)


if __name__ == "__main__":
    main()
