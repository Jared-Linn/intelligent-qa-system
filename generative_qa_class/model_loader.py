"""
模型加载器 - 单例模式，确保模型只加载一次
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    """单例模式模型加载器，整个应用只加载一次模型"""

    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        """单例模式：确保只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path):
        """
        加载模型（只会执行一次）

        参数:
            model_path (str): 合并后的模型路径
        """
        if self._model is None:
            print("正在加载模型（首次加载，请稍候）...")

            # 加载模型
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                attn_implementation="eager",
            )

            # 加载分词器
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

            # 设置 padding token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            print("模型加载完成！")

        return self._model, self._tokenizer

    def get_model(self):
        """获取模型实例"""
        return self._model

    def get_tokenizer(self):
        """获取分词器实例"""
        return self._tokenizer


# 全局模型加载器实例
model_loader = ModelLoader()