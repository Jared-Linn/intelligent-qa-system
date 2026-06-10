# Intelligent QA System · 智能问答系统

> 面向中文场景的智能问答系统 —— 问题理解 → 知识检索 → 答案生成 → 结果验证

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 系统架构

```
用户输入 → 问题理解 → 知识检索 → 答案生成 → 最终答案
                ↑           ↑           ↑
          问题分类    结构化检索    答案抽取
          关键词提取  非结构化检索  答案验证
```

### 核心模块

| 模块 | 功能 |
|------|------|
| **语料预处理** | 读取语料、分词、构建词典、拆分问答对 |
| **问题理解** | 问题分类、关键词/实体提取 |
| **知识检索** | 结构化检索 (SQL/知识图谱) + 非结构化检索 (BM25/向量) |
| **答案生成** | 答案抽取 (阅读理解) + 答案生成 (seq2seq/RAG) + 答案验证 |
| **模型训练** | 支持 BERT 微调、LoRA 等方案 |
| **模型评价** | Exact Match、F1、BLEU、ROUGE 等指标 |

### 项目结构

```
├── configs/          # 配置文件
├── data/             # 数据与数据加载器
├── models/           # 模型定义 (encoder, qa, retrieval, lora)
├── modules/          # 功能模块 (preprocess, qa, retrieval)
├── train.py          # 训练脚本
├── test.py           # 测试脚本
├── evaluate.py       # 评价脚本
└── predict.py        # 推理脚本
```

## 快速开始

```bash
# 1. 语料预处理
python modules/preprocess.py --input data/raw/corpus.json --output data/processed/

# 2. 训练模型
python train.py --config configs/train.yaml

# 3. 推理
python predict.py --question "什么是机器学习？" --model checkpoints/best.pt
```

## 技术栈

- **框架**: PyTorch 2.x, Transformers
- **分词**: jieba / HuggingFace Tokenizers
- **检索**: BM25, FAISS
- **模型**: BERT / RoBERTa / ChatGLM / Qwen (LoRA 微调)

## 项目来源

本科课程期末项目，基于项目思维导图（[mindmap.png](mindmap.png)）设计实现。

## License

MIT
