"""
检索式推理引擎 — 根据用户上传的 JSON 数据构建 TF-IDF 索引并问答

每个模型独立索引，首次加载时构建，后续从缓存加载。
"""

import sys
import json
import os
from pathlib import Path
from typing import Optional

# 确保 retrieval_qa/ 在 sys.path 中（内部模块使用 from utils.xxx 导入）
_root = Path(__file__).resolve().parent.parent.parent
_retrieval_path = _root / "retrieval_qa"
if str(_retrieval_path) not in sys.path:
    sys.path.insert(0, str(_retrieval_path))

from web.services.file_manager import UPLOAD_DIR


class RetrievalInferenceEngine:
    """检索式推理引擎（带缓存）"""

    def __init__(self):
        self._cache = {}  # model_id -> (retriever, generator)

    def _load_data(self, model: dict) -> tuple:
        """
        加载模型数据，返回 (questions, answers)
        支持从上传文件或默认 FAQ 数据
        """
        model_id = model["id"]

        # 优先使用用户上传的文件
        if model.get("file_path"):
            data_path = Path(model["file_path"])
            if not data_path.is_absolute():
                data_path = UPLOAD_DIR.parent.parent / data_path
            if data_path.exists():
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                questions = [item["question"] for item in data]
                answers = [item["answer"] for item in data]
                return questions, answers

        # fallback：默认 FAQ 数据
        default_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "faq_expanded.json"
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [item["question"] for item in data], [item["answer"] for item in data]

        raise FileNotFoundError("没有找到数据文件")

    def predict(self, model: dict, question: str) -> dict:
        """
        执行检索式问答

        Returns:
            { "answer": str, "confidence": float, "source": str }
        """
        model_id = model["id"]

        # 检查缓存
        if model_id not in self._cache:
            questions, answers = self._load_data(model)
            from retrieval_qa.modules.knowledge_retrieval import TfidfRetriever
            from retrieval_qa.modules.answer_generation import AnswerGenerator

            retriever = TfidfRetriever()
            retriever.build_index(questions, answers)
            generator = AnswerGenerator(method="direct", min_score=0.1)
            self._cache[model_id] = (retriever, generator)

        retriever, generator = self._cache[model_id]
        results = retriever.retrieve(question, top_k=3)
        result = generator.generate(question, results)

        return {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "source": result.get("source", ""),
        }
