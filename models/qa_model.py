"""
问答模型

封装完整的问答模型，支持：
- 抽取式问答（预测答案起始/结束位置）
- 分类式问答（用于答案验证等）
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .encoder import Encoder


class QAModel(nn.Module):
    """
    问答模型

    - 编码器：BERT 或自建 Transformer
    - 输出头：起始位置预测 + 结束位置预测

    用途：抽取式阅读理解（如 CMRC 2018 任务）
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        from_scratch: bool = False,
        vocab_size: int = 30000,
        hidden_size: int = 768,
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
        output_dim = self.encoder.get_output_dim()

        # 抽取式问答头：预测起始和结束位置
        self.qa_outputs = nn.Linear(output_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            start_logits: [batch_size, seq_len]
            end_logits:   [batch_size, seq_len]
        """
        sequence_output = self.encoder(input_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
