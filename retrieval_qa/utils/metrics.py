"""
评价指标模块

提供问答系统常用的评价指标计算：
- Exact Match (EM)
- F1 Score
- Precision / Recall
- BLEU (预留)
- ROUGE (预留)
"""

import re
from typing import List, Union


class MetricsCalculator:
    """评价指标计算器"""

    @staticmethod
    def normalize(text: str) -> str:
        """标准化文本（去空格、标点、统一小写）"""
        text = re.sub(r"[^\w]", "", text)
        return text.lower().strip()

    # ── 检索指标 ──────────────────────────────────────────────────────

    @staticmethod
    def precision_at_k(relevant: List[str], retrieved: List[str], k: int = None) -> float:
        """Precision@K: 前 K 个结果中相关结果的比例"""
        if k is None:
            k = len(retrieved)
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            return 0.0
        hits = sum(1 for r in retrieved_k if r in relevant)
        return hits / len(retrieved_k)

    @staticmethod
    def recall_at_k(relevant: List[str], retrieved: List[str], k: int = None) -> float:
        """Recall@K: 前 K 个结果覆盖正确答案的比例"""
        if k is None:
            k = len(retrieved)
        if not relevant:
            return 0.0
        retrieved_k = retrieved[:k]
        hits = sum(1 for r in retrieved_k if r in relevant)
        return hits / len(relevant)

    @staticmethod
    def mrr(relevant: List[str], retrieved: List[str]) -> float:
        """MRR: 第一个正确结果的平均排名倒数"""
        for i, r in enumerate(retrieved, 1):
            if r in relevant:
                return 1.0 / i
        return 0.0

    # ── 抽取式问答指标 ───────────────────────────────────────────────

    @staticmethod
    def exact_match(prediction: str, ground_truth: Union[str, List[str]]) -> int:
        """Exact Match: 预测与标准答案完全一致"""
        pred_norm = MetricsCalculator.normalize(prediction)

        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        for gt in ground_truth:
            if pred_norm == MetricsCalculator.normalize(gt):
                return 1
        return 0

    @staticmethod
    def f1_score(prediction: str, ground_truth: Union[str, List[str]]) -> float:
        """F1 Score: 预测与标准答案的 Token 重合度"""
        pred_tokens = set(MetricsCalculator.normalize(prediction))

        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        best_f1 = 0.0
        for gt in ground_truth:
            gt_tokens = set(MetricsCalculator.normalize(gt))
            if not pred_tokens or not gt_tokens:
                continue
            common = pred_tokens & gt_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        return best_f1

    @staticmethod
    def compute_em_f1(predictions: List[str], ground_truths: List[Union[str, List[str]]]) -> dict:
        """批量计算 EM 和 F1"""
        em_scores = []
        f1_scores = []

        for pred, gt in zip(predictions, ground_truths):
            em_scores.append(MetricsCalculator.exact_match(pred, gt))
            f1_scores.append(MetricsCalculator.f1_score(pred, gt))

        return {
            "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "total": len(em_scores),
        }
