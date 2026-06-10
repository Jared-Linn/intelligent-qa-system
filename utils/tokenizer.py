"""
分词工具模块

提供中文分词和基本文本处理功能。
支持 jieba 分词和 HuggingFace Tokenizer 两种方案。
"""

import re
from typing import List, Optional


class ChineseTokenizer:
    """中文分词器封装"""

    def __init__(self, method: str = "jieba"):
        self.method = method
        self._tokenizer = None

        if method == "jieba":
            import jieba
            self._tokenizer = jieba
        elif method == "char":
            pass  # 按字切分，不需要额外库
        else:
            raise ValueError(f"不支持的分词方法: {method}，可选: jieba, char")

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.method == "jieba":
            return list(self._tokenizer.cut(text))
        elif self.method == "char":
            return list(text.strip())
        return []

    def tokenize_with_pos(self, text: str) -> List[tuple]:
        """分词并返回词性标注（仅 jieba）"""
        if self.method == "jieba":
            return list(self._tokenizer.posseg.cut(text))
        return [(t, "x") for t in self.tokenize(text)]


# ── 文本预处理工具 ────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白、特殊字符"""
    text = re.sub(r"\s+", " ", text)  # 合并连续空白
    text = text.strip()
    return text


def extract_chinese(text: str) -> str:
    """仅保留中文字符和常见标点"""
    return re.sub(r"[^一-鿿＀-￯　-〿\w\s]", "", text)


def is_chinese_char(char: str) -> bool:
    """判断是否为中文字符"""
    return "一" <= char <= "鿿"


def get_text_stats(text: str) -> dict:
    """统计文本基本信息"""
    return {
        "char_count": len(text),
        "chinese_char_count": sum(1 for c in text if is_chinese_char(c)),
        "word_count": len(text.split()),
    }
