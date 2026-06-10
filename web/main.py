"""
智能问答系统 — Web API 入口

FastAPI 应用，整合用户系统、模型管理、问答评测、排行榜。

启动:
    uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# 加入项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.database import init_db
from web.routers import auth, models, qa, ratings, ranking


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库"""
    init_db()
    print("[数据库] SQLite 初始化完成")
    yield


app = FastAPI(
    title="智能问答系统",
    description="支持检索式 + 生成式问答、用户系统、模型管理、排行榜",
    version="2.0.0",
    lifespan=lifespan,
)

# 挂载静态文件
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)

# 注册路由
app.include_router(auth.router)
app.include_router(models.router)
app.include_router(qa.router)
app.include_router(ratings.router)
app.include_router(ranking.router)


@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "智能问答系统", "version": "2.0.0"}
