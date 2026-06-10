"""Pydantic 数据模型 — 评分与排行榜"""
from pydantic import BaseModel, Field
from typing import Optional, List


class RatingCreate(BaseModel):
    model_id: int
    question: str
    answer: str
    score: int = Field(..., ge=1, le=5)
    session_id: Optional[str] = None


class RatingResponse(BaseModel):
    id: int
    user_id: int
    username: Optional[str] = None
    model_id: int
    question: str
    answer: str
    score: int
    session_id: Optional[str]
    created_at: str


class RankingItem(BaseModel):
    rank: int
    model_id: int
    user_id: int
    username: str
    model_name: str
    model_type: str
    avg_score: float
    downloads: int
    rating_count: int
    call_count: int


class RankingResponse(BaseModel):
    items: List[RankingItem]
    total: int
