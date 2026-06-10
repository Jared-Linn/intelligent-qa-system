"""
检索模型

为检索任务设计的双编码器模型（Dual Encoder / Sentence-BERT）。
将问题和文档分别编码为向量，用于语义检索。

预留：第3课开始使用 FAISS 进行向量检索。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import Encoder


class RetrievalModel(nn.Module):
    """
    双编码器检索模型

    两个编码器（共享权重）分别编码 query 和 document，
    通过余弦相似度计算相关性得分。
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        from_scratch: bool = False,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        pooling: str = "cls",       # cls | mean | max
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            model_name=model_name,
            from_scratch=from_scratch,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.pooling = pooling
        output_dim = self.encoder.get_output_dim()
        self.projection = nn.Linear(output_dim, hidden_size)

    def _pool(self, hidden_states: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """池化：将序列表示聚合为向量"""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]  # [CLS] token
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return hidden_states.mean(dim=1)
        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states + (1 - mask) * (-1e9)
            return hidden_states.max(dim=1).values
        else:
            return hidden_states[:, 0, :]

    def encode_query(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码查询"""
        hidden = self.encoder(input_ids, attention_mask)
        pooled = self._pool(hidden, attention_mask)
        return F.normalize(self.projection(pooled), dim=-1)

    def encode_document(self, input_ids: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码文档（共享编码器）"""
        return self.encode_query(input_ids, attention_mask)

    def forward(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        doc_ids: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算 query 与 doc 的相似度得分"""
        q_vec = self.encode_query(query_ids, query_mask)
        d_vec = self.encode_document(doc_ids, doc_mask)
        return torch.mm(q_vec, d_vec.t())
