"""Pydantic 数据模型 — 模型"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    type: str = Field(..., pattern="^(retrieval|generative)$")
    description: str = ""
    public: bool = False


class ModelResponse(BaseModel):
    id: int
    user_id: int
    username: Optional[str] = None
    name: str
    type: str
    file_path: Optional[str] = None
    description: str
    public: bool
    status: str
    created_at: str
    updated_at: str
    downloads: int
    avg_score: float
    rating_count: Optional[int] = None
    call_count: Optional[int] = None


class ModelListResponse(BaseModel):
    models: List[ModelResponse]
    total: int
