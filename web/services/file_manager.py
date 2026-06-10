"""
文件管理服务 — 上传 / 解压 / 校验

- LoRA 权重上传为 zip，解压到 models/{user_id}/{model_id}/
- 检索式数据上传为 JSON，存到 models/{user_id}/{model_id}/
"""

import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import Optional

# 上传根目录
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "uploads"


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_model_dir(user_id: int, model_id: int) -> Path:
    """获取模型文件目录"""
    return UPLOAD_DIR / str(user_id) / str(model_id)


def save_retrieval_data(user_id: int, model_id: int,
                        content: bytes, filename: str = "data.json") -> Optional[str]:
    """
    保存检索式数据文件（JSON）。
    校验格式合法性后保存，返回相对路径。
    """
    model_dir = get_model_dir(user_id, model_id)
    _ensure_dir(model_dir)
    file_path = model_dir / filename

    # 校验 JSON 格式
    try:
        data = json.loads(content.decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 必须是数组格式")
        for item in data:
            if "question" not in item or "answer" not in item:
                raise ValueError("每条数据必须包含 question 和 answer 字段")
    except json.JSONDecodeError:
        raise ValueError("文件不是合法的 JSON 格式")
    except Exception as e:
        raise ValueError(f"数据校验失败: {e}")

    file_path.write_bytes(content)
    return str(file_path.relative_to(UPLOAD_DIR.parent.parent))


def save_lora_weights(user_id: int, model_id: int,
                      content: bytes, filename: str = "lora.zip") -> Optional[str]:
    """
    保存生成式 LoRA 权重。
    接收 zip 压缩包，解压到模型目录。
    """
    model_dir = get_model_dir(user_id, model_id)
    _ensure_dir(model_dir)

    zip_path = model_dir / filename
    zip_path.write_bytes(content)

    # 解压
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # 校验 zip 中是否包含 adapter_config.json
            file_list = zf.namelist()
            if not any("adapter_config.json" in f for f in file_list):
                raise ValueError("LoRA zip 中必须包含 adapter_config.json")

            zf.extractall(model_dir)
    except zipfile.BadZipFile:
        raise ValueError("文件不是合法的 zip 格式")

    # 删除原始 zip
    zip_path.unlink()
    return str(model_dir)


def delete_model_files(user_id: int, model_id: int):
    """删除模型关联的文件"""
    model_dir = get_model_dir(user_id, model_id)
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
