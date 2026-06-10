"""
语料预处理模块

功能：
- 读取 FAQ / 对话等原始语料
- 文本清洗与规范化
- 分词与词表构建
- 问答对拆分与保存
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

from utils.tokenizer import ChineseTokenizer, clean_text


class Preprocessor:
    """语料预处理器"""

    def __init__(self, tokenizer_method: str = "jieba"):
        self.tokenizer = ChineseTokenizer(method=tokenizer_method)

    # ── 数据加载 ──────────────────────────────────────────────────────

    def load_json(self, path: str) -> List[Dict]:
        """加载 JSON 格式语料"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_csv(self, path: str, q_field="question", a_field="answer") -> List[Dict]:
        """加载 CSV 格式语料"""
        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append({"question": row[q_field], "answer": row[a_field]})
        return items

    # ── 文本清洗 ──────────────────────────────────────────────────────

    def clean_items(self, items: List[Dict]) -> List[Dict]:
        """批量清洗问答对"""
        cleaned = []
        seen_questions = set()
        for item in items:
            q = clean_text(item.get("question", ""))
            a = clean_text(item.get("answer", ""))
            if not q or not a:
                continue
            if q in seen_questions:
                continue  # 去重
            seen_questions.add(q)
            item["question"] = q
            item["answer"] = a
            cleaned.append(item)
        return cleaned

    # ── 词表构建 ──────────────────────────────────────────────────────

    def build_vocab(
        self,
        items: List[Dict],
        max_size: int = 10000,
        min_freq: int = 2,
        save_path: Optional[str] = None,
    ) -> Dict[str, int]:
        """构建词表（基于分词结果）"""
        counter = Counter()
        for item in items:
            for text in [item.get("question", ""), item.get("answer", "")]:
                tokens = self.tokenizer.tokenize(text)
                counter.update(tokens)

        # 过滤低频词
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        for word, freq in counter.most_common(max_size):
            if freq >= min_freq:
                vocab[word] = len(vocab)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            print(f"[词表] 已保存到 {save_path} ({len(vocab)} 词)")

        return vocab

    # ── 统计 ──────────────────────────────────────────────────────────

    def print_stats(self, items: List[Dict]):
        """打印数据集统计信息"""
        n = len(items)
        categories = {}
        for item in items:
            cat = item.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print(f"[数据] 总样本数: {n}")
        print(f"[数据] 分类分布:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"       {cat}: {count}")

        # 平均长度
        q_lens = [len(item.get("question", "")) for item in items]
        a_lens = [len(item.get("answer", "")) for item in items]
        print(f"[数据] 问题平均长度: {sum(q_lens)/n:.1f} 字")
        print(f"[数据] 答案平均长度: {sum(a_lens)/n:.1f} 字")

    # ── 主流程 ────────────────────────────────────────────────────────

    def run(
        self,
        input_path: str,
        output_dir: str = "data/processed/",
        build_vocab: bool = True,
    ) -> List[Dict]:
        """
        完整预处理流程：
        1. 加载数据
        2. 清洗
        3. 保存处理结果
        4. 可选构建词表
        """
        items = self.load_json(input_path)
        print(f"[预处理] 原始数据: {len(items)} 条")

        items = self.clean_items(items)
        print(f"[预处理] 清洗后: {len(items)} 条")

        # 保存
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "faq_processed.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"[预处理] 已保存到 {out_path}")

        self.print_stats(items)

        if build_vocab:
            self.build_vocab(items, save_path=str(out_dir / "vocab.json"))

        return items
