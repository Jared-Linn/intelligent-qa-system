"""
问答路由 — 单模型问答 + 多模型对比

延迟加载推理引擎，首次请求时初始化。
"""

import time
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from web.database import get_model_by_id, log_usage, increment_downloads
from web.routers.auth import get_current_user

router = APIRouter(prefix="/api/qa", tags=["问答"])


class AskRequest(BaseModel):
    model_id: int
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    model_id: int
    model_name: str
    answer: str
    confidence: float
    latency_ms: int


class CompareRequest(BaseModel):
    model_ids: List[int] = Field(..., min_length=1, max_length=10)
    question: str = Field(..., min_length=1)


class CompareItem(BaseModel):
    model_id: int
    model_name: str
    model_type: str
    username: str
    answer: str
    confidence: float
    latency_ms: int


class CompareResponse(BaseModel):
    question: str
    results: List[CompareItem]


# ── 推理引擎缓存 ─────────────────────────────────────────────────

_retrieval_engine = None


def _get_retrieval_engine():
    """延迟加载检索式推理引擎"""
    global _retrieval_engine
    if _retrieval_engine is None:
        from web.services.infer_retrieval import RetrievalInferenceEngine
        _retrieval_engine = RetrievalInferenceEngine()
    return _retrieval_engine


def _do_inference(model_id: int, question: str) -> dict:
    """
    执行推理（根据模型类型分发到不同引擎）。
    返回 { answer, confidence, latency_ms }
    """
    model = get_model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")

    start = time.time()

    if model["type"] == "retrieval":
        engine = _get_retrieval_engine()
        result = engine.predict(model, question)
    else:
        from web.services.infer_generative import GenerativeInferenceEngine
        engine = GenerativeInferenceEngine()
        result = engine.predict(model, question)

    latency = int((time.time() - start) * 1000)
    result["latency_ms"] = latency

    # 记录使用量
    log_usage(model_id, api_path="/api/qa/ask", latency_ms=latency)
    increment_downloads(model_id)

    return result


# ── API 端点 ──────────────────────────────────────────────────────


@router.post("/ask", response_model=AskResponse)
async def ask(data: AskRequest):
    """单模型问答"""
    model = get_model_by_id(data.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")

    result = _do_inference(data.model_id, data.question)
    return AskResponse(
        model_id=data.model_id,
        model_name=model["name"],
        **result,
    )


@router.post("/compare", response_model=CompareResponse)
async def compare(data: CompareRequest):
    """多模型对比（同一问题，多模型同时出答案）"""
    results = []
    for mid in data.model_ids:
        model = get_model_by_id(mid)
        if model is None:
            continue
        try:
            result = _do_inference(mid, data.question)
            results.append(CompareItem(
                model_id=mid,
                model_name=model["name"],
                model_type=model["type"],
                username=model.get("username", ""),
                **result,
            ))
        except Exception as e:
            results.append(CompareItem(
                model_id=mid,
                model_name=model.get("name", "未知"),
                model_type=model.get("type", ""),
                username="",
                answer=f"推理失败: {str(e)}",
                confidence=0.0,
                latency_ms=0,
            ))

    return CompareResponse(question=data.question, results=results)
