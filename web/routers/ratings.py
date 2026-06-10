"""
评分路由 — 模型评分
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from web.models.rating import RatingCreate, RatingResponse
from web.database import create_rating, get_model_ratings, get_model_by_id, increment_downloads
from web.routers.auth import get_current_user

router = APIRouter(prefix="/api/ratings", tags=["评分"])


@router.post("")
async def submit_rating(data: RatingCreate, user: dict = Depends(get_current_user)):
    """提交评分"""
    model = get_model_by_id(data.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")

    rating = create_rating(
        user_id=user["id"],
        model_id=data.model_id,
        question=data.question,
        answer=data.answer,
        score=data.score,
        session_id=data.session_id,
    )
    return rating


@router.get("/{model_id}")
async def list_ratings(model_id: int):
    """获取模型的评分列表"""
    ratings = get_model_ratings(model_id)
    return {"ratings": ratings, "total": len(ratings)}
