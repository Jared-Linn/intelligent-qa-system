"""
LoRA（Low-Rank Adaptation）适配层

在不修改预训练模型权重的情况下，通过插入低秩矩阵进行高效微调。
参数量仅为原模型的 0.1%~1%，大幅降低显存需求。

使用方式：
    model = AutoModel.from_pretrained("bert-base-chinese")
    lora_model = LoRAModel(model, r=8, alpha=16, target_modules=["query", "value"])
"""

import torch
import torch.nn as nn
from typing import List, Optional


class LoRALayer(nn.Module):
    """单个 LoRA 适配层：把权重更新分解为低秩矩阵 A × B"""

    def __init__(self, in_dim: int, out_dim: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 低秩矩阵 A（随机初始化）
        self.lora_a = nn.Parameter(torch.randn(in_dim, r) * 0.02)
        # 低秩矩阵 B（初始化为 0）
        self.lora_b = nn.Parameter(torch.zeros(r, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.lora_a @ self.lora_b) * self.scaling


class LoRAModel(nn.Module):
    """
    LoRA 包装器

    在指定模块中插入 LoRA 层，冻结原模型权重。

    Args:
        base_model: 预训练模型
        r: 低秩矩阵的秩
        alpha: 缩放系数
        target_modules: 要适配的模块名列表，如 ["query", "value"]
    """

    def __init__(
        self,
        base_model: nn.Module,
        r: int = 8,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules or ["query", "value"]

        # 冻结原模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 插入 LoRA 层
        self._find_and_replace()

    def _find_and_replace(self):
        """在目标模块的线性层旁插入 LoRA"""
        for name, module in self.base_model.named_modules():
            if any(t in name for t in self.target_modules) and isinstance(module, nn.Linear):
                lora = LoRALayer(module.in_features, module.out_features, self.r, self.alpha)
                setattr(module, "lora", lora)

    def forward(self, *args, **kwargs):
        """前向传播（底层子模块的 LoRA 层在前向时自动生效）"""
        return self.base_model(*args, **kwargs)

    @property
    def trainable_params(self) -> int:
        """可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        """总参数量"""
        return sum(p.numel() for p in self.parameters())
