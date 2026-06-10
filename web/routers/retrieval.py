from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

# ── 延迟加载检索器（首次请求时初始化） ────────────────────────────────

_retriever = None
_generator = None
_question_understanding = None
_faq_questions = []
_faq_answers = []


def _ensure_loaded():
    """懒加载检索式问答模块"""
    global _retriever, _generator, _question_understanding, _faq_questions, _faq_answers

    if _retriever is not None:
        return

    import sys
    import json
    from pathlib import Path

    # 确保 retrieval_qa/ 在 sys.path 中（内部模块使用 from utils.xxx 导入）
    root = Path(__file__).resolve().parent.parent.parent
    retrieval_path = root / "retrieval_qa"
    if str(retrieval_path) not in sys.path:
        sys.path.insert(0, str(retrieval_path))

    config_path = retrieval_path / "configs" / "default.yaml"

    from retrieval_qa.utils.config import Config
    from retrieval_qa.modules.knowledge_retrieval import TfidfRetriever
    from retrieval_qa.modules.answer_generation import AnswerGenerator
    from retrieval_qa.modules.question_understanding import QuestionUnderstanding

    cfg = Config(str(config_path))

    # 加载 FAQ 数据
    faq_path = root / cfg.get("data", "faq_path")
    with open(faq_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _faq_questions = [item["question"] for item in data]
    _faq_answers = [item["answer"] for item in data]

    # 检索器
    vectorizer_path = root / cfg.get("paths", "vectorizer")
    _retriever = TfidfRetriever(config=cfg.config)
    if vectorizer_path.exists():
        _retriever.load(str(vectorizer_path))
    else:
        _retriever.build_index(_faq_questions, _faq_answers)
        _retriever.save(str(vectorizer_path))

    # 答案生成器
    _generator = AnswerGenerator(
        method=cfg.get("answer", "method", default="direct"),
        min_score=cfg.get("answer", "min_score", default=0.1),
    )

    # 问题理解
    qu_config = cfg.get("question_understanding", default={})
    _question_understanding = QuestionUnderstanding(qu_config)
    if qu_config.get("keyword_method") == "tfidf":
        _question_understanding.fit_keywords_tfidf(_faq_questions + _faq_answers)


# ── 请求/响应模型 ────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    use_understanding: bool = True


class CandidateItem(BaseModel):
    question: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    source: str
    category: Optional[str] = None
    keywords: List[str] = []
    candidates: List[CandidateItem] = []


# ── API 端点 ─────────────────────────────────────────────────────────


@router.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """检索式问答：问题理解 → 知识检索 → 答案生成"""
    _ensure_loaded()

    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 1. 问题理解
    category = "general"
    keywords = []
    search_query = query
    if request.use_understanding:
        understanding = _question_understanding.understand(query)
        category = understanding["category"]
        keywords = understanding["keywords"]
        search_query = understanding["query"]

    # 2. 知识检索
    results = _retriever.retrieve(search_query, top_k=request.top_k)

    # 3. 答案生成
    result = _generator.generate(query, results)

    return QueryResponse(
        question=query,
        answer=result["answer"],
        confidence=result["confidence"],
        source=result["source"],
        category=category,
        keywords=keywords,
        candidates=[
            CandidateItem(question=c["question"], score=c["score"])
            for c in result.get("details", [])
        ],
    )


@router.get("/ask")
async def ask_get(
    q: str = Query(..., description="问题"),
    top_k: int = Query(3, description="返回候选数"),
):
    """GET 方式的问答接口（便于浏览器调试）"""
    return await ask(QueryRequest(question=q, top_k=top_k))
