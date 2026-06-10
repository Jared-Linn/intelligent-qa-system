"""
配置文件管理模块

统一管理 YAML 配置的加载、合并和访问。
支持命令行参数覆盖配置项。
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict


class Config:
    """配置类，封装了配置的加载和访问"""

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config: Dict[str, Any] = self._load(config_path)

    def _load(self, path: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, *keys: str, default=None) -> Any:
        """
        安全地获取嵌套配置项。
        用法: config.get("retrieval", "top_k") -> 3
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def __repr__(self) -> str:
        return json.dumps(self.config, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 测试配置加载
    cfg = Config("configs/default.yaml")
    print("=== 配置内容 ===")
    print(cfg)
    print("\n=== 单项访问 ===")
    print(f"检索方法: {cfg.get('retrieval', 'method')}")
    print(f"Top-K: {cfg.get('retrieval', 'top_k')}")
    print(f"不存在的键: {cfg.get('nonexist', default='默认值')}")
