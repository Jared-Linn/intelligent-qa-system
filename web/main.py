"""
智能问答系统 — Web API 入口

FastAPI 应用，提供检索式和生成式两种问答接口。

启动:
    uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# 加入项目根目录，便于导入 retrieval_qa / generative_qa
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.routers import retrieval, generative

app = FastAPI(
    title="智能问答系统",
    description="检索式问答 + 生成式问答（千问微调）",
    version="1.0.0",
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# 注册路由
app.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["检索式问答"])
app.include_router(generative.router, prefix="/api/v1/generative", tags=["生成式问答"])

# 模板
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/")
async def root():
    """Web 主页"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "智能问答系统"}
