"""
智能问答系统 — 推理/预测脚本

两种运行模式：
  1. 单次查询： python predict.py --question "什么是人工智能"
  2. 交互模式： python predict.py --interactive

流程：
  加载数据 → 构建 TF-IDF 索引 → 检索 → 生成答案
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List

# 将项目根目录加入系统路径，确保模块可以正确导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import Config
from modules.knowledge_retrieval import TfidfRetriever
from modules.answer_generation import AnswerGenerator


def load_faq_data(raw_path: str):
    """加载 FAQ 数据"""
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    return questions, answers


def build_or_load_retriever(
    questions: List[str], answers: List[str], cfg: Config
) -> TfidfRetriever:
    """
    构建或加载检索器。

    如果已存在保存的索引则加载，否则重新构建并保存。
    这样第二次运行就不需要重新训练 TF-IDF 了。
    """
    import os

    retriever_path = cfg.get("paths", "vectorizer", default="data/processed/vectorizer.pkl")

    if os.path.exists(retriever_path):
        print("检测到已保存的索引，直接加载...")
        retriever = TfidfRetriever(config=cfg.config)
        retriever.load(retriever_path)
    else:
        print("未检测到索引，重新构建...")
        retriever = TfidfRetriever(config=cfg.config)
        retriever.build_index(questions, answers)
        retriever.save(retriever_path)

    return retriever


def answer_question(
    query: str,
    retriever: TfidfRetriever,
    generator: AnswerGenerator,
    top_k: int = 3,
) -> dict:
    """
    回答问题

    完整流程：问题理解（本节简化） → 知识检索 → 答案生成
    """
    # 1. 检索
    results = retriever.retrieve(query, top_k=top_k)

    # 2. 生成答案
    result = generator.generate(query, results)

    return result


def print_result(query: str, result: dict):
    """美观地输出结果"""
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"[问题] {query}")
    print(sep)
    print(f"[答案] {result['answer']}")
    if result["confidence"] > 0:
        print(f"[置信度] {result['confidence']:.4f}")
    if result["source"] != "none":
        print(f"[来源] {result['source']}")
    if result.get("details"):
        print(f"[候选结果]:")
        for i, d in enumerate(result["details"], 1):
            print(f"  {i}. [{d['score']:.4f}] {d['question']}")
    print(sep)


def interactive_mode(retriever: TfidfRetriever, generator: AnswerGenerator, top_k: int):
    """交互式问答模式"""
    print("\n[智能问答系统已启动！（输入 exit 或 quit 退出）")
    print("   " + "-" * 40)

    while True:
        try:
            query = input("\n[请输入问题]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("再见！")
            break

        result = answer_question(query, retriever, generator, top_k)
        print_result(query, result)


def main():
    parser = argparse.ArgumentParser(description="智能问答系统 — 推理脚本")
    parser.add_argument("--question", "-q", type=str, help="单次查询的问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式问答模式")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="返回 Top-K 候选")
    args = parser.parse_args()

    # 加载配置
    cfg = Config(args.config)
    top_k = args.top_k or cfg.get("retrieval", "top_k", default=3)
    min_score = cfg.get("answer", "min_score", default=0.1)

    # 加载数据
    raw_path = cfg.get("data", "raw_path")
    questions, answers = load_faq_data(raw_path)
    print(f"[数据] 加载了 {len(questions)} 条 FAQ 数据")

    # 构建/加载检索器
    retriever = build_or_load_retriever(questions, answers, cfg)

    # 创建答案生成器
    generator = AnswerGenerator(
        method=cfg.get("answer", "method", default="direct"),
        min_score=min_score,
    )

    # 执行模式
    if args.question:
        result = answer_question(args.question, retriever, generator, top_k)
        print_result(args.question, result)
    elif args.interactive:
        interactive_mode(retriever, generator, top_k)
    else:
        # 默认：交互模式
        interactive_mode(retriever, generator, top_k)


if __name__ == "__main__":
    main()
