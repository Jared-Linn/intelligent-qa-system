"""
编码器模块

提供 Encoder 基类和预训练模型包装。
支持从零构建或在 BERT 等预训练模型上微调。
"""

import torch
import torch.nn as nn
from typing import Optional


class Encoder(nn.Module):
    """
    文本编码器

    支持两种模式：
    - from_scratch: 从零构建 Embedding + Transformer 编码器
    - pretrained:   使用预训练模型（BERT 等）作为编码器
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        from_scratch: bool = False,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.from_scratch = from_scratch
        self.hidden_size = hidden_size

        if from_scratch:
            self._build_from_scratch(vocab_size, hidden_size, num_layers,
                                     num_heads, max_len, dropout)
        else:
            self._load_pretrained(model_name)

    def _build_from_scratch(self, vocab_size, hidden_size, num_layers,
                            num_heads, max_len, dropout):
        """从零构建编码器（教学用途）"""
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_dim = hidden_size

    def _load_pretrained(self, model_name: str):
        """加载预训练模型"""
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.output_dim = self.backbone.config.hidden_size

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        if self.from_scratch:
            seq_len = input_ids.size(1)
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = self.embedding(input_ids) + self.pos_encoding(positions)
            x = self.dropout(x)
            return self.transformer(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

    def get_output_dim(self) -> int:
        return self.output_dim
