"""
答案生成模块

本节实现：直接返回检索到的答案（"抽取式"的最简形式）
后续课程：阅读理解抽取 → RAG 生成

目前策略：
- 如果检索到相似问题，直接返回对应的答案
- 如果相似度低于阈值，返回"未找到答案"的提示
"""

from typing import List, Tuple, Optional


class AnswerGenerator:
    """答案生成器"""

    def __init__(self, method: str = "direct", min_score: float = 0.1):
        """
        参数:
            method: 生成方式
                "direct"   - 直接返回检索到的答案（本节使用）
                "extract"  - 阅读理解抽取（后续课程）
                "generate" - 生成式（后续课程）
            min_score: 最低相似度阈值
        """
        self.method = method
        self.min_score = min_score

    def generate(
        self, query: str, retrieval_results: List[Tuple[str, str, float]]
    ) -> dict:
        """
        生成答案

        参数:
            query: 用户原始问题
            retrieval_results: 检索结果 [(问题, 答案, 相似度), ...]

        返回:
            {
                "answer": str,          # 最终答案
                "source": str,          # 来源
                "confidence": float,    # 置信度
                "details": [...]        # 候选详情
            }
        """
        if self.method == "direct":
            return self._direct_answer(query, retrieval_results)
        else:
            raise ValueError(f"不支持的方法: {self.method}")

    def _direct_answer(self, query: str, results: List[Tuple[str, str, float]]) -> dict:
        """直接返回最相似问题对应的答案"""
        if not results:
            return {
                "answer": "抱歉，我没有找到相关的答案。",
                "source": "none",
                "confidence": 0.0,
                "details": [],
            }

        best_q, best_a, best_score = results[0]

        # 低于阈值 → 未找到
        if best_score < self.min_score:
            return {
                "answer": f"抱歉，我没有找到与「{query}」相关的答案。（最相似问题相似度仅为 {best_score:.2f}）",
                "source": "none",
                "confidence": best_score,
                "details": [(best_q, best_a, best_score)],
            }

        # 构建详情（供调试和参考）
        details = [
            {"question": q, "answer": a, "score": round(s, 4)}
            for q, a, s in results
        ]

        return {
            "answer": best_a,
            "source": "faq_retrieval",
            "confidence": round(best_score, 4),
            "details": details,
        }


def test_generator():
    """测试答案生成"""
    from knowledge_retrieval import TfidfRetriever

    # 准备数据
    questions = [
        "什么是人工智能",
        "什么是机器学习",
        "什么是 Python",
    ]
    answers = [
        "人工智能是计算机科学的分支，研究模拟人类智能。",
        "机器学习是人工智能的子领域，让计算机从数据中学习。",
        "Python 是一种高级编程语言。",
    ]

    # 检索
    retriever = TfidfRetriever()
    retriever.build_index(questions, answers)

    # 生成答案
    generator = AnswerGenerator(min_score=0.1)

    test_queries = ["AI 是什么", "Python 语言", "今天天气怎么样"]

    print("=== 答案生成测试 ===\n")
    for q in test_queries:
        results = retriever.retrieve(q, top_k=2)
        result = generator.generate(q, results)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"  置信度: {result['confidence']}")
        if result["details"]:
            for d in result["details"]:
                print(f"  → 匹配: {d['question']} ({d['score']:.4f})")
        print()


if __name__ == "__main__":
    test_generator()
