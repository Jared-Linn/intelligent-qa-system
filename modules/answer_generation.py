"""
答案生成模块

根据检索结果组织最终回答，支持三种模式：
  - direct:  直接返回检索到的答案（适用于 FAQ）
  - extractive: 从候选段落中抽取答案片段（预留，需阅读理解模型）
  - generative: 基于检索结果 + 语言模型生成答案（预留，第4课）

流程：候选段落 → 答案抽取/生成 → 答案验证 → 最终答案
"""

import math
import re
from typing import List, Dict, Optional


class AnswerGenerator:
    """答案生成器"""

    def __init__(self, method: str = "direct", min_score: float = 0.1):
        """
        Args:
            method: direct | extractive | generative
            min_score: 最低相似度阈值
        """
        self.method = method
        self.min_score = min_score

    def generate(self, query: str, candidates: List[Dict]) -> Dict:
        """
        从候选结果生成答案。

        Args:
            query: 用户原始问题
            candidates: [{"question": str, "answer": str, "score": float}, ...]

        Returns:
            {
                "answer": str,          # 最终答案
                "confidence": float,    # 置信度
                "source": str,          # 来源（匹配到的问题）
                "details": List[Dict],  # 候选详情
                "method": str,          # 使用的生成方法
            }
        """
        if self.method == "direct":
            return self._direct_answer(query, candidates)
        elif self.method == "extractive":
            return self._extractive_answer(query, candidates)
        elif self.method == "generative":
            return self._generative_answer(query, candidates)
        else:
            return self._direct_answer(query, candidates)

    # ── 直接返回 ──────────────────────────────────────────────────────

    def _direct_answer(self, query: str, candidates: List[Dict]) -> Dict:
        """直接返回最高分候选的答案"""
        if not candidates:
            return self._no_answer()

        best = candidates[0]
        if best["score"] < self.min_score:
            return self._no_answer(best["score"], [candidates[0]])

        return {
            "answer": best["answer"],
            "confidence": self._normalize_confidence(best["score"]),
            "source": best["question"],
            "details": candidates,
            "method": "direct",
        }

    # ── 抽取式（预留） ───────────────────────────────────────────────

    def _extractive_answer(self, query: str, candidates: List[Dict]) -> Dict:
        """从候选文本中抽取答案片段（占位，后续集成阅读理解模型）"""
        # TODO: 集成 BERT 阅读理解模型
        return self._direct_answer(query, candidates)

    # ── 生成式（预留，第4课） ────────────────────────────────────────

    def _generative_answer(self, query: str, candidates: List[Dict]) -> Dict:
        """基于 LLM 生成答案（占位，后续接入 Qwen/ChatGLM）"""
        # TODO: 接入 LLM
        return self._direct_answer(query, candidates)

    # ── 工具方法 ──────────────────────────────────────────────────────

    def _no_answer(self, score: float = 0.0, details: List = None) -> Dict:
        """返回无答案结果"""
        return {
            "answer": "抱歉，我暂时无法回答这个问题。请换个方式提问试试。",
            "confidence": 0.0,
            "source": "none",
            "details": details or [],
            "method": self.method,
        }

    @staticmethod
    def _normalize_confidence(score: float) -> float:
        """将检索分数映射到 [0, 1] 置信度"""
        # Sigmoid-like 映射
        return 1.0 / (1.0 + math.exp(-5 * (score - 0.5))) if score > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# 对话管理器（多轮对话）
# ══════════════════════════════════════════════════════════════════════

class DialogueManager:
    """多轮对话管理器，支持上下文感知检索"""

    def __init__(self, retriever, max_history: int = 4):
        self.retriever = retriever
        self.max_history = max_history
        self.history: List[Dict] = []
        self.current_scenario: Optional[Dict] = None
        self.current_dialogue_id: Optional[str] = None

    def reset(self):
        """重置对话"""
        self.history = []
        self.current_scenario = None
        self.current_dialogue_id = None

    def get_context(self) -> str:
        """构建上下文文本"""
        context_parts = []
        for turn in self.history[-self.max_history:]:
            role = turn["role"]
            text = turn["text"]
            context_parts.append(f"{role}: {text}")
        return "\n".join(context_parts)

    def chat(self, user_input: str) -> Dict:
        """
        处理一轮对话。

        Returns:
            {
                "answer": str,
                "confidence": float,
                "matched_scenario": str,
                "category": str,
                "candidates": List[Dict],
            }
        """
        # 添加到历史
        self.history.append({"role": "user", "text": user_input})

        # 检索最匹配的场景
        results = self.retriever.retrieve(user_input, top_k=3)

        if not results or results[0]["score"] < 0.1:
            self.history.append({
                "role": "assistant",
                "text": "抱歉，我还没学会这个话题。换个话题聊聊？",
            })
            return self._build_dialogue_result(
                answer="抱歉，我还没学会这个话题。换个话题聊聊？",
                confidence=0.0,
                scenario="none",
                candidates=results,
            )

        best = results[0]
        self.history.append({"role": "assistant", "text": best["answer"]})

        # 保持历史在限制长度内
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

        return self._build_dialogue_result(
            answer=best["answer"],
            confidence=self._normalize_confidence(best["score"]),
            scenario=best.get("title", best["question"]),
            candidates=results,
        )

    def _build_dialogue_result(
        self, answer: str, confidence: float,
        scenario: str, candidates: List[Dict],
    ) -> Dict:
        return {
            "answer": answer,
            "confidence": confidence,
            "matched_scenario": scenario,
            "category": candidates[0].get("category", "general") if candidates else "general",
            "candidates": candidates,
        }

    @staticmethod
    def _normalize_confidence(score: float) -> float:
        return 1.0 / (1.0 + math.exp(-5 * (score - 0.5))) if score > 0 else 0.0

