"""
配置管理模块

功能：
- 加载 YAML 配置文件
- 支持通过点号路径访问嵌套字段
- 支持默认值回退
"""

import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """配置管理器，封装 YAML 配置的加载与访问"""

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load()

    def _load(self) -> dict:
        """加载 YAML 配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        通过键路径获取配置值。

        用法:
            cfg.get("retrieval", "top_k")         # -> 3
            cfg.get("paths", "vectorizer")        # -> "data/processed/vectorizer.pkl"
            cfg.get("nonexistent", default=42)     # -> 42
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """设置顶层配置项（用于命令行覆盖）"""
        self.config[key] = value

    def __repr__(self) -> str:
        return f"Config({self.config_path})"
