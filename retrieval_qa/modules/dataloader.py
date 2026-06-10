"""
数据加载器

将预处理后的语料组装成训练所需的数据形式。
支持 Dataset + DataLoader 模式，动态 padding。
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class QADataset(Dataset):
    """问答数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 128,
        mode: str = "faq",
    ):
        """
        Args:
            data_path: 处理后的 JSON 数据路径
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
            mode: faq | qa_extractive
        """
        self.data = self._load(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def _load(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        question = item.get("question", "")
        answer = item.get("answer", "")

        if self.tokenizer:
            # 使用 HuggingFace tokenizer 编码
            encoding = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "question": question,
                "answer": answer,
            }
        else:
            return {"question": question, "answer": answer}


def create_dataloader(
    data_path: str,
    tokenizer=None,
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """创建 DataLoader"""
    dataset = QADataset(data_path, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
