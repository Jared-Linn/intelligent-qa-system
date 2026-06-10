"""
知识检索模块

本节实现：TF-IDF + 余弦相似度的基础检索
后续课程：BM25 → 稠密向量检索（Sentence-BERT） → 混合检索

知识点：TF-IDF 原理、余弦相似度、向量空间模型
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """
    TF-IDF 检索器

    核心思想：
    - TF (词频)：词语在文档中出现的频率
    - IDF (逆文档频率)：词语在多少个文档中出现，出现越少越重要
    - TF-IDF = TF × IDF，值越大表示该词对文档越重要

    检索流程：
        1. 对所有 FAQ 问题计算 TF-IDF 向量
        2. 对用户输入的问题同样计算 TF-IDF 向量
        3. 计算余弦相似度，返回 Top-K 最相似的问题及其答案
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.questions: List[str] = []
        self.answers: List[str] = []

    def _tokenize_chinese(self, text: str) -> str:
        """
        中文分词预处理

        TF-IDF 对英文是按空格分词的，所以中文需要先分词再用空格连接。
        这一步叫"分词 + 空格连接"，是中文 TF-IDF 的标准做法。
        """
        words = jieba.cut(text, cut_all=False)
        return " ".join(words)

    def build_index(self, questions: List[str], answers: List[str]):
        """
        构建 TF-IDF 索引

        流程：
            1. 保存问答数据
            2. 中文分词
            3. 训练 TF-IDF 向量化器
            4. 计算所有问题的 TF-IDF 向量矩阵
        """
        self.questions = questions
        self.answers = answers

        # 中文分词（TF-IDF 要求输入是空格分隔的 token 序列）
        print(f"[1/3] 中文分词 ({len(questions)} 条问题)...")
        tokenized = [self._tokenize_chinese(q) for q in questions]

        # 创建 TF-IDF 向量化器
        print("[2/3] 训练 TF-IDF 向量化器...")
        tfidf_config = self.config.get("tfidf", {})
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get("max_features", 5000),
            ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2])),
            use_idf=tfidf_config.get("use_idf", True),
            token_pattern=r"(?u)\S+",   # 匹配所有非空 token（中文已分词）
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(tokenized)

        # 打印信息
        print(f"[3/3] 索引构建完成!")
        print(f"      词典大小: {self.vectorizer.get_feature_names_out().shape[0]}")
        print(f"      矩阵形状: {self.tfidf_matrix.shape}")
        print(f"      问题总数: {len(questions)}")

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        检索与查询最相似的 FAQ

        参数:
            query: 用户问题
            top_k: 返回前 K 个结果

        返回:
            [(问题, 答案, 相似度), ...] 按相似度从高到低排列

        原理：
            1. 将用户问题转为 TF-IDF 向量
            2. 计算与所有 FAQ 问题的余弦相似度
            3. 余弦相似度 = cos(θ) = A·B / (|A|×|B|)
               - 值域 [-1, 1]，值越大表示两个向量方向越一致
               - 在 TF-IDF 空间中 = 语义越相似
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("请先调用 build_index() 构建索引")

        # 对查询分词并向量化
        query_vec = self.vectorizer.transform([self._tokenize_chinese(query)])

        # 计算与所有文档的余弦相似度
        # cosine_similarity 返回形状为 (1, n_docs) 的矩阵
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # 获取 Top-K 索引（按相似度降序）
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 组装结果
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有相关性的结果
                results.append((self.questions[idx], self.answers[idx], float(scores[idx])))

        return results

    def save(self, path: str):
        """保存检索器到磁盘"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "questions": self.questions,
                "answers": self.answers,
            }, f)
        print(f"检索器已保存到: {path}")

    def load(self, path: str):
        """从磁盘加载检索器"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.questions = data["questions"]
        self.answers = data["answers"]
        print(f"检索器已从 {path} 加载")
        print(f"      问题数: {len(self.questions)}, 词典大小: {self.vectorizer.get_feature_names_out().shape[0]}")


def test_retriever():
    """测试检索器的基本功能"""
    # 准备简单的测试数据
    questions = [
        "什么是人工智能",
        "什么是机器学习",
        "Python 是什么",
        "什么是深度学习",
    ]
    answers = [
        "人工智能是计算机科学的分支",
        "机器学习是 AI 的子领域",
        "Python 是编程语言",
        "深度学习是机器学习的子集",
    ]

    # 构建索引
    retriever = TfidfRetriever()
    retriever.build_index(questions, answers)

    # 测试检索
    test_queries = [
        "AI 是什么",
        "Python 编程语言",
        "什么是机器学习算法",
    ]

    print("\n=== 检索测试 ===")
    for query in test_queries:
        print(f"\n用户问题: {query}")
        results = retriever.retrieve(query, top_k=2)
        for q, a, score in results:
            print(f"  [{score:.4f}] {q} → {a}")


if __name__ == "__main__":
    test_retriever()
