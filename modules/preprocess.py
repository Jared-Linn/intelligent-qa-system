"""
语料预处理模块

功能：读取原始语料 → 分词 → 保存处理结果
知识点：中文分词、文本清洗、JSON 操作
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import jieba


class TextPreprocessor:
    """文本预处理器：清洗、分词、构建词典"""

    def __init__(self):
        # 正则：保留中英文、数字和基本标点
        self.clean_pattern = re.compile(r"[^一-鿿\w\s，。！？、：；""''（）]")

    def clean_text(self, text: str) -> str:
        """清洗文本：去除特殊符号和多余空白"""
        text = self.clean_pattern.sub("", text)      # 去除非中英文数字的符号
        text = re.sub(r"\s+", "", text)               # 中文分词前去除空格
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """使用 jieba 进行中文分词"""
        return list(jieba.cut(text, cut_all=False))   # 精确模式

    def build_vocab(
        self, texts: List[str], min_freq: int = 1, max_size: int = 5000
    ) -> Dict[str, int]:
        """
        构建词表（vocabulary）

        参数:
            texts: 文本列表
            min_freq: 最小词频（低于此值的词被忽略）
            max_size: 词表最大容量

        返回:
            {词: 索引} 的映射字典
        """
        # 第一步：统计词频
        freq = {}
        for text in texts:
            for word in self.tokenize(text):
                freq[word] = freq.get(word, 0) + 1

        # 第二步：过滤低频词并按频率排序
        sorted_words = sorted(
            [(w, f) for w, f in freq.items() if f >= min_freq],
            key=lambda x: -x[1]
        )[:max_size]

        # 第三步：构建索引映射（预留特殊标记位置）
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        for i, (word, _) in enumerate(sorted_words):
            vocab[word] = i + 4  # 从 4 开始分配

        return vocab

    def process_faq_data(
        self, input_path: str, output_dir: str, config: dict = None
    ) -> Dict:
        """
        完整的 FAQ 数据预处理流程

        流程:
            1. 读取原始 JSON
            2. 清洗问题和答案文本
            3. 分词
            4. 构建词表
            5. 保存处理结果

        参数:
            input_path: 原始数据路径
            output_dir: 输出目录
            config: 配置参数
        """
        # 1. 读取数据
        print(f"[1/5] 读取数据: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = [item["question"] for item in data]
        answers = [item["answer"] for item in data]
        print(f"      共加载 {len(questions)} 条问答对")

        # 2. 清洗文本
        print("[2/5] 清洗文本...")
        clean_questions = [self.clean_text(q) for q in questions]
        clean_answers = [self.clean_text(a) for a in answers]

        # 3. 分词
        print("[3/5] 分词...")
        tokenized_questions = [self.tokenize(q) for q in clean_questions]

        # 4. 构建词表
        print("[4/5] 构建词表...")
        all_texts = clean_questions + clean_answers
        vocab = self.build_vocab(all_texts)
        print(f"      词表大小: {len(vocab)}")

        # 5. 保存结果
        print(f"[5/5] 保存到: {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存处理后的数据
        processed = {
            "questions": clean_questions,
            "answers": clean_answers,
            "tokenized_questions": [[" ".join(t)] for t in tokenized_questions],
            "vocab_size": len(vocab),
            "total_pairs": len(questions),
        }
        output_path = output_dir / "faq_processed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        # 保存词表
        vocab_path = output_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        # 打印前几条样例
        print("\n=== 预处理样例 ===")
        for i in range(min(3, len(questions))):
            print(f"  问题: {questions[i]}")
            print(f"  分词: {' | '.join(tokenized_questions[i])}")
            print(f"  答案: {answers[i][:50]}...")
            print()

        return processed


if __name__ == "__main__":
    # 独立运行：预处理 FAQ 数据
    import sys
    # 把项目根目录加入路径
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.config import Config

    cfg = Config("configs/default.yaml")
    preprocessor = TextPreprocessor()
    result = preprocessor.process_faq_data(
        input_path=cfg.get("data", "raw_path"),
        output_dir=cfg.get("data", "processed_dir"),
    )
    print(f"预处理完成！词表大小: {result['vocab_size']}, 问答对数: {result['total_pairs']}")
