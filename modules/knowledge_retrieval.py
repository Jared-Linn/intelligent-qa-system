"""
知识检索模块

支持多种检索方法：
- TF-IDF: 基于词频-逆文档频率的稀疏检索
- BM25: 概率检索模型，对 TF-IDF 的改进
- Hybrid: TF-IDF / BM25 混合（预留向量检索接口）

流程：问题表示 → 候选召回 → 相关性排序 → 输出候选段落
"""

import json
import math
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter

from utils.tokenizer import ChineseTokenizer


# ══════════════════════════════════════════════════════════════════════
# TF-IDF 检索器
# ══════════════════════════════════════════════════════════════════════

class TfidfRetriever:
    """基于 TF-IDF + 余弦相似度的检索器"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.tokenizer = ChineseTokenizer(method="jieba")

        # TF-IDF 参数
        self.max_features = config.get("tfidf", {}).get("max_features", 5000)
        self.ngram_range = tuple(config.get("tfidf", {}).get("ngram_range", [1, 2]))
        self.use_idf = config.get("tfidf", {}).get("use_idf", True)

        # 数据
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.corpus: List[str] = []  # 分词后的文档
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: np.ndarray = None  # [n_docs, n_terms]

    # ── 索引构建 ──────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """分词并生成 n-gram"""
        tokens = list(self.tokenizer.tokenize(text))

        # 添加 n-gram
        ngrams = []
        n_min, n_max = self.ngram_range
        for n in range(n_min, n_max + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append("".join(tokens[i:i + n]))

        return tokens + ngrams

    def _build_vocab(self):
        """从语料构建词表"""
        counter = Counter()
        for doc in self.corpus:
            counter.update(doc)

        # 按频率排序，取 top-k
        most_common = counter.most_common(self.max_features)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

    def _compute_idf(self):
        """计算 IDF"""
        n_docs = len(self.corpus)
        doc_freq = Counter()

        for doc in self.corpus:
            for word in set(doc):
                if word in self.vocab:
                    doc_freq[word] += 1

        self.idf = {}
        for word, df in doc_freq.items():
            self.idf[word] = math.log((n_docs + 1) / (df + 1)) + 1

    def _vectorize(self, tokens: List[str]) -> np.ndarray:
        """将分词结果转为 TF-IDF 向量"""
        vec = np.zeros(len(self.vocab))

        if not tokens:
            return vec

        # 计算 TF
        tf = Counter(tokens)
        max_tf = max(tf.values())

        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                # TF: 归一化后的词频
                tf_value = 0.5 + 0.5 * count / max_tf
                # IDF
                idf_value = self.idf.get(word, 1.0) if self.use_idf else 1.0
                vec[idx] = tf_value * idf_value

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def build_index(self, questions: List[str], answers: List[str]):
        """构建 TF-IDF 索引"""
        self.questions = questions
        self.answers = answers

        # 分词
        print("[索引] 正在分词...")
        self.corpus = [self._tokenize(q) for q in questions]
        print(f"[索引] 分词完成，{len(self.corpus)} 篇文档")

        # 构建词表
        self._build_vocab()
        print(f"[索引] 词表大小: {len(self.vocab)}")

        # 计算 IDF
        self._compute_idf()

        # 构建 TF-IDF 矩阵
        print("[索引] 正在计算 TF-IDF 矩阵...")
        matrix = []
        for doc in self.corpus:
            matrix.append(self._vectorize(doc))
        self.tfidf_matrix = np.array(matrix)
        print(f"[索引] TF-IDF 矩阵形状: {self.tfidf_matrix.shape}")

    # ── 检索 ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索最相关的问答对。

        Returns:
            [{"question": str, "answer": str, "score": float}, ...]
        """
        if self.tfidf_matrix is None or len(self.questions) == 0:
            return []

        # 向量化查询
        query_tokens = self._tokenize(query)
        query_vec = self._vectorize(query_tokens)

        # 计算余弦相似度
        similarities = self.tfidf_matrix @ query_vec

        # 获取 top-k 索引
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            results.append({
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "score": round(score, 4),
            })

        return results

    # ── 持久化 ────────────────────────────────────────────────────────

    def save(self, path: str):
        """保存检索器到磁盘"""
        data = {
            "questions": self.questions,
            "answers": self.answers,
            "vocab": self.vocab,
            "idf": self.idf,
            "tfidf_matrix": self.tfidf_matrix,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[检索器] 已保存到 {path}")

    def load(self, path: str):
        """从磁盘加载检索器"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.questions = data["questions"]
        self.answers = data["answers"]
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.config = data.get("config", {})
        print(f"[检索器] 已从 {path} 加载 ({len(self.questions)} 条)")


# ══════════════════════════════════════════════════════════════════════
# BM25 检索器
# ══════════════════════════════════════════════════════════════════════

class Bm25Retriever:
    """BM25 概率检索模型"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.tokenizer = ChineseTokenizer(method="jieba")
        self.k1 = config.get("bm25", {}).get("k1", 1.5)
        self.b = config.get("bm25", {}).get("b", 0.75)

        self.questions: List[str] = []
        self.answers: List[str] = []
        self.corpus: List[List[str]] = []
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.n_docs: int = 0
        self.df: Dict[str, int] = {}  # 文档频率

    def build_index(self, questions: List[str], answers: List[str]):
        self.questions = questions
        self.answers = answers

        print("[BM25] 正在分词...")
        self.corpus = [self.tokenizer.tokenize(q) for q in questions]
        self.n_docs = len(self.corpus)
        self.doc_lens = [len(doc) for doc in self.corpus]
        self.avg_doc_len = sum(self.doc_lens) / max(self.n_docs, 1)

        # 计算文档频率
        print("[BM25] 计算文档频率...")
        self.df = {}
        for doc in self.corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

        print(f"[BM25] 索引完成，{self.n_docs} 篇文档，平均长度 {self.avg_doc_len:.1f}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """BM25 检索"""
        if self.n_docs == 0:
            return []

        query_tokens = self.tokenizer.tokenize(query)
        scores = np.zeros(self.n_docs)

        for token in query_tokens:
            if token not in self.df:
                continue

            df = self.df[token]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for i, doc in enumerate(self.corpus):
                # 词频
                tf = doc.count(token)
                if tf == 0:
                    continue

                doc_len = self.doc_lens[i]
                score = idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                )
                scores[i] += score

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "question": self.questions[idx],
                    "answer": self.answers[idx],
                    "score": round(float(scores[idx]), 4),
                })
        return results

    def save(self, path: str):
        data = {
            "questions": self.questions,
            "answers": self.answers,
            "df": self.df,
            "doc_lens": self.doc_lens,
            "avg_doc_len": self.avg_doc_len,
            "n_docs": self.n_docs,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.questions = data["questions"]
        self.answers = data["answers"]
        self.df = data["df"]
        self.doc_lens = data["doc_lens"]
        self.avg_doc_len = data["avg_doc_len"]
        self.n_docs = data["n_docs"]
        self.config = data.get("config", {})


# ══════════════════════════════════════════════════════════════════════
# 混合检索器
# ══════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """混合检索：结合 TF-IDF 和 BM25 结果"""

    def __init__(self, tfidf_retriever: TfidfRetriever, bm25_retriever: Bm25Retriever,
                 weights: Tuple[float, float] = (0.5, 0.5)):
        self.tfidf = tfidf_retriever
        self.bm25 = bm25_retriever
        self.weights = weights

    def build_index(self, questions: List[str], answers: List[str]):
        self.tfidf.build_index(questions, answers)
        self.bm25.build_index(questions, answers)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        tfidf_results = self.tfidf.retrieve(query, top_k * 2)
        bm25_results = self.bm25.retrieve(query, top_k * 2)

        # 合并分数
        merged = {}
        w1, w2 = self.weights
        for r in tfidf_results:
            idx = self._get_index(r["question"])
            merged[idx] = {"score": w1 * r["score"], "count": 1}
        for r in bm25_results:
            idx = self._get_index(r["question"])
            if idx in merged:
                merged[idx]["score"] += w2 * r["score"]
                merged[idx]["count"] += 1
            else:
                merged[idx] = {"score": w2 * r["score"], "count": 1}

        # 按分数排序
        sorted_items = sorted(merged.items(), key=lambda x: -x[1]["score"])
        all_questions = self.tfidf.questions
        all_answers = self.tfidf.answers

        results = []
        for idx, info in sorted_items[:top_k]:
            results.append({
                "question": all_questions[idx],
                "answer": all_answers[idx],
                "score": round(info["score"], 4),
            })
        return results

    def _get_index(self, question: str) -> int:
        try:
            return self.tfidf.questions.index(question)
        except ValueError:
            return -1

    def save(self, path: str):
        self.tfidf.save(path + ".tfidf")
        self.bm25.save(path + ".bm25")

    def load(self, path: str):
        self.tfidf.load(path + ".tfidf")
        self.bm25.load(path + ".bm25")


# ── 工厂函数 ────────────────────────────────────────────────────────────

def create_retriever(method: str = "tfidf", config: dict = None):
    """创建检索器实例"""
    if method == "bm25":
        return Bm25Retriever(config)
    elif method == "hybrid":
        tfidf = TfidfRetriever(config)
        bm25 = Bm25Retriever(config)
        return HybridRetriever(tfidf, bm25)
    else:
        return TfidfRetriever(config)
