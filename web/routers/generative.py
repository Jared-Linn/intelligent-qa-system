from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# 预留：安装依赖后取消注释即可启用
# _generative_ready = False
# _gen_model = None
# _gen_tokenizer = None


class GenRequest(BaseModel):
    question: str
    max_length: int = 512
    temperature: float = 0.7


class GenResponse(BaseModel):
    question: str
    answer: str
    model: str = "Qwen2.5-3B-Instruct (LoRA)"


@router.post("/ask", response_model=GenResponse)
async def ask(request: GenRequest):
    """生成式问答（预留 — 千问 LoRA 微调后启用）"""
    raise HTTPException(
        status_code=501,
        detail="生成式问答尚未启用。请先训练模型: python generative_qa/train_qwen_lora.py"
    )


@router.get("/ask")
async def ask_get(q: str = "你好", max_length: int = 512, temperature: float = 0.7):
    return await ask(GenRequest(question=q, max_length=max_length, temperature=temperature))


@router.get("/status")
async def status():
    """生成式服务状态"""
    return {
        "enabled": False,
        "model": "Qwen2.5-3B-Instruct",
        "fine_tuned": False,
        "message": "请先运行训练脚本: python generative_qa/train_qwen_lora.py",
    }
