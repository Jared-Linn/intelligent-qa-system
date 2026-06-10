"""
智能问答系统 — 推理/预测主入口

支持三种模式：
  1. 单次查询：    python retrieval_qa/predict.py -q "什么是机器学习"
  2. FAQ 交互：    python retrieval_qa/predict.py -m faq
  3. 对话交互：    python retrieval_qa/predict.py -m dialogue

问答流程：
  用户输入 → 问题理解 → 知识检索 → 答案生成 → 输出

首次运行自动构建索引，后续秒级加载。

运行环境：
  conda activate nlp
  pip install -r retrieval_qa/requirements.txt
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List

# 项目根目录加入系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import Config
from modules.question_understanding import QuestionUnderstanding
from modules.knowledge_retrieval import TfidfRetriever
from modules.answer_generation import AnswerGenerator, DialogueManager


# ══════════════════════════════════════════════════════════════════════
# FAQ 模式
# ══════════════════════════════════════════════════════════════════════

def load_faq_data(raw_path: str) -> tuple:
    """加载 FAQ 数据"""
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    return questions, answers


def build_or_load_retriever(questions: List[str], answers: List[str], cfg: Config) -> TfidfRetriever:
    """构建或加载检索器（有缓存则加载，否则新建）"""
    import os
    retriever_path = cfg.get("paths", "vectorizer")
    method = cfg.get("retrieval", "method", default="tfidf")

    retriever = TfidfRetriever(config=cfg.config)

    if os.path.exists(retriever_path):
        print(f"[检索器] 检测到缓存，加载: {retriever_path}")
        retriever.load(retriever_path)
    else:
        print("[检索器] 未检测到缓存，重新构建索引...")
        retriever.build_index(questions, answers)
        retriever.save(retriever_path)

    return retriever


def answer_question(
    query: str,
    retriever: TfidfRetriever,
    generator: AnswerGenerator,
    top_k: int = 3,
    question_understanding=None,
) -> dict:
    """
    完整问答链路：
      问题理解 → 知识检索 → 答案生成
    """
    # 1. 问题理解
    if question_understanding:
        understanding = question_understanding.understand(query)
        print(f"[理解] 分类: {understanding['category']}  "
              f"关键词: {', '.join(understanding['keywords'])}")
        search_query = understanding["query"]
    else:
        search_query = query

    # 2. 知识检索
    results = retriever.retrieve(search_query, top_k=top_k)

    # 3. 答案生成
    result = generator.generate(query, results)
    return result


def print_result(query: str, result: dict):
    """格式化输出 FAQ 结果"""
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


def interactive_faq(retriever, generator, top_k, question_understanding=None):
    """FAQ 交互模式"""
    print("\n" + "=" * 50)
    print("  智能问答系统 — FAQ 模式")
    print("  输入问题获取答案，输入 exit 退出")
    print("=" * 50)

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

        result = answer_question(query, retriever, generator, top_k, question_understanding)
        print_result(query, result)


# ══════════════════════════════════════════════════════════════════════
# 对话模式
# ══════════════════════════════════════════════════════════════════════

def load_dialogue_data(raw_path: str) -> tuple:
    """加载对话数据集，转为 (questions, answers, dialogue_map)"""
    with open(raw_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    questions = []
    answers = []
    dialogue_map = {}

    for d in dialogues:
        dialogue_map[d["id"]] = d
        turns = d.get("turns", [])

        # 为每一轮 user → assistant 对建立索引
        for i, turn in enumerate(turns):
            if turn["role"] == "user" and i + 1 < len(turns):
                next_turn = turns[i + 1]
                if next_turn["role"] == "assistant":
                    questions.append(turn["text"])
                    answers.append({
                        "text": next_turn["text"],
                        "dialogue_id": d["id"],
                        "title": d.get("title", ""),
                        "scenario": d.get("scenario", ""),
                        "category": d.get("category", ""),
                    })

    return questions, answers, dialogue_map


class DialogueRetriever:
    """对话检索器：TF-IDF 匹配对话场景"""

    def __init__(self, config: dict):
        self.config = config
        self.retriever = TfidfRetriever(config)
        self.questions: List[str] = []
        self.answer_meta: List[dict] = []
        self.dialogue_map: dict = {}

    def load_dialogues(self, data_path: str):
        """加载对话数据"""
        qs, ans, dmap = load_dialogue_data(data_path)
        self.questions = qs
        self.answer_meta = ans
        self.dialogue_map = dmap

    def build_index(self):
        """构建检索索引"""
        answers = [a["text"] for a in self.answer_meta]
        self.retriever.build_index(self.questions, answers)

    def retrieve(self, query: str, top_k: int = 3):
        """检索匹配的对话回复"""
        results = self.retriever.retrieve(query, top_k)
        # 丰富元数据
        for r in results:
            for meta in self.answer_meta:
                if meta["text"] == r["answer"]:
                    r.update(meta)
                    break
        return results

    def save(self, path: str):
        data = {
            "questions": self.questions,
            "answer_meta": self.answer_meta,
        }
        import pickle
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.questions = data["questions"]
        self.answer_meta = data["answer_meta"]
        # 重建索引
        answers = [a["text"] for a in self.answer_meta]
        self.retriever.build_index(self.questions, answers)


def build_or_load_dialogue_retriever(cfg: Config):
    """构建或加载对话检索器"""
    import os
    index_path = cfg.get("paths", "dialogue_index")
    data_path = cfg.get("data", "dialogue_path", default="data/raw/dialogues.json")

    retriever = DialogueRetriever(config=cfg.config)
    retriever.load_dialogues(data_path)

    if os.path.exists(index_path):
        print(f"[对话] 检测到缓存，加载: {index_path}")
        retriever.load(index_path)
    else:
        print("[对话] 未检测到缓存，重新构建索引...")
        retriever.build_index()
        retriever.save(index_path)

    return retriever


def interactive_dialogue(retriever: DialogueRetriever, cfg: Config):
    """对话交互模式"""
    max_history = cfg.get("dialogue", "max_history", default=4)
    manager = DialogueManager(retriever, max_history=max_history)

    print("\n" + "=" * 50)
    print("  智能问答系统 — 对话模式")
    print("  支持多轮上下文感知问答")
    print("  输入 exit 退出 | reset 重置对话")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n[你]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if user_input.lower() == "reset":
            manager.reset()
            print("[系统] 对话已重置")
            continue

        result = manager.chat(user_input)
        print(f"\n[助手] {result['answer']}")
        if result["confidence"] > 0:
            print(f"[置信度] {result['confidence']:.4f}")
        if result["matched_scenario"] != "none":
            print(f"[场景] {result['matched_scenario']}")


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="智能问答系统 — 推理脚本")
    parser.add_argument("--question", "-q", type=str, help="单次查询的问题（FAQ 模式）")
    parser.add_argument("--mode", "-m", type=str, choices=["faq", "dialogue"],
                        default=None, help="运行模式: faq | dialogue")
    parser.add_argument("--config", "-c", type=str,
                        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"),
                        help="配置文件路径")
    parser.add_argument("--top_k", "-k", type=int, default=None, help="返回 Top-K 候选")
    parser.add_argument("--no-understand", action="store_true",
                        help="禁用问题理解模块（仅用原文检索）")
    args = parser.parse_args()

    # 加载配置
    cfg = Config(args.config)

    # 判断模式
    mode = args.mode
    if mode is None:
        mode = "faq" if args.question else "dialogue"

    if mode == "faq":
        run_faq_mode(cfg, args)
    else:
        run_dialogue_mode(cfg, args)


def run_faq_mode(cfg: Config, args):
    """运行 FAQ 问答模式"""
    top_k = args.top_k or cfg.get("retrieval", "top_k", default=3)
    min_score = cfg.get("answer", "min_score", default=0.1)

    # 加载数据
    raw_path = cfg.get("data", "faq_path")
    questions, answers = load_faq_data(raw_path)
    print(f"[数据] 加载了 {len(questions)} 条 FAQ 数据")

    # 构建/加载检索器
    retriever = build_or_load_retriever(questions, answers, cfg)

    # 创建答案生成器
    generator = AnswerGenerator(
        method=cfg.get("answer", "method", default="direct"),
        min_score=min_score,
    )

    # 创建问题理解模块
    question_understanding = None
    if not args.no_understand:
        qu_config = cfg.get("question_understanding", default={})
        question_understanding = QuestionUnderstanding(qu_config)

        # 如果关键词提取用 TF-IDF，需要训练
        if qu_config.get("keyword_method") == "tfidf":
            print("[理解] 训练关键词提取器...")
            corpus = questions + answers
            question_understanding.fit_keywords_tfidf(corpus)

    # 执行
    if args.question:
        result = answer_question(args.question, retriever, generator,
                                 top_k, question_understanding)
        print_result(args.question, result)
    else:
        interactive_faq(retriever, generator, top_k, question_understanding)


def run_dialogue_mode(cfg: Config, args):
    """运行对话模式"""
    retriever = build_or_load_dialogue_retriever(cfg)
    interactive_dialogue(retriever, cfg)


if __name__ == "__main__":
    main()
