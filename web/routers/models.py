"""
模型路由 — 模型 CRUD + 文件上传
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from web.models.model import ModelCreate, ModelUpdate, ModelResponse, ModelListResponse
from web.database import (
    create_model, get_user_models, get_public_models,
    get_model_by_id, update_model_file, update_model, delete_model,
)
from web.routers.auth import get_current_user
from web.services.file_manager import (
    save_retrieval_data, save_lora_weights, delete_model_files,
)

router = APIRouter(prefix="/api/models", tags=["模型管理"])


@router.get("", response_model=ModelListResponse)
async def list_my_models(user: dict = Depends(get_current_user)):
    """我的模型列表"""
    models = get_user_models(user["id"])
    return ModelListResponse(models=models, total=len(models))


@router.post("", response_model=ModelResponse)
async def create_new_model(data: ModelCreate, user: dict = Depends(get_current_user)):
    """创建新模型"""
    model = create_model(
        user_id=user["id"],
        name=data.name,
        model_type=data.type,
        description=data.description,
        public=data.public,
    )
    if model is None:
        raise HTTPException(status_code=400, detail="创建模型失败")
    return ModelResponse(**model)


@router.post("/{model_id}/upload")
async def upload_model_file(
    model_id: int,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """上传模型文件（LoRA zip / JSON 数据）"""
    # 验证模型归属
    model = get_model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    if model["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="无权操作此模型")

    # 根据类型处理
    content = await file.read()

    try:
        if model["type"] == "retrieval":
            file_path = save_retrieval_data(user["id"], model_id, content)
        else:
            file_path = save_lora_weights(user["id"], model_id, content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    update_model_file(model_id, file_path)
    return {"message": "上传成功", "file_path": file_path}


@router.get("/public", response_model=ModelListResponse)
async def list_public_models(model_type: str = None):
    """公开模型广场"""
    models = get_public_models(model_type)
    return ModelListResponse(models=models, total=len(models))


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model_detail(model_id: int):
    """模型详情"""
    model = get_model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    return ModelResponse(**model)


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model_by_id(model_id: int, data: ModelUpdate, user: dict = Depends(get_current_user)):
    """更新模型属性（名称/描述/可见性）"""
    model = get_model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    if model["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="无权操作此模型")
    ok = update_model(model_id, name=data.name, description=data.description, public=data.public)
    if not ok:
        raise HTTPException(status_code=400, detail="更新失败，没有要修改的字段")
    updated = get_model_by_id(model_id)
    return ModelResponse(**updated)



@router.delete("/{model_id}")
async def delete_model_by_id(model_id: int, user: dict = Depends(get_current_user)):
    """删除模型"""
    model = get_model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    if model["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="无权操作此模型")

    delete_model_files(user["id"], model_id)
    ok = delete_model(model_id)
    if not ok:
        raise HTTPException(status_code=400, detail="删除失败")
    return {"message": "已删除"}
