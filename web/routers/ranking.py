"""
排行榜路由
"""

from fastapi import APIRouter, Query
from web.models.rating import RankingItem, RankingResponse
from web.database import get_ranking

router = APIRouter(prefix="/api/ranking", tags=["排行榜"])


@router.get("", response_model=RankingResponse)
async def ranking(
    model_type: str = Query("all", pattern="^(all|retrieval|generative)$"),
    sort_by: str = Query("avg_score", pattern="^(avg_score|downloads|official)$"),
    period: str = Query("all", pattern="^(7d|30d|all)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """排行榜"""
    items = get_ranking(model_type, sort_by, period, limit)
    ranked = []
    for i, item in enumerate(items, 1):
        ranked.append(RankingItem(
            rank=i,
            model_id=item["id"],
            user_id=item["user_id"],
            username=item["username"],
            model_name=item["name"],
            model_type=item["type"],
            avg_score=item["avg_score"] or 0.0,
            downloads=item["downloads"] or 0,
            rating_count=item["rating_count"] or 0,
            call_count=item["call_count"] or 0,
        ))
    return RankingResponse(items=ranked, total=len(ranked))
