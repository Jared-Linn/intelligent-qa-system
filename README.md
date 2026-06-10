# 智能问答系统

> 面向中文场景的智能问答系统 —— 问题理解 → 知识检索 → 答案生成

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)

---

## 系统架构

```
用户输入 → 问题理解 → 知识检索 → 答案生成 → 最终答案
                ↑           ↑           ↑
          问题分类       TF-IDF/BM25   答案抽取
          关键词提取     语义检索(Q3)   答案验证
```

## 快速开始

```bash
# 激活环境（conda nlp）
conda activate nlp

# FAQ 交互模式
python predict.py -m faq

# 单次查询
python predict.py -q "什么是机器学习"

# 对话模式
python predict.py -m dialogue
```

## 项目结构

```
├── configs/          # 配置文件
├── data/             # 数据与数据加载器
│   ├── raw/          # 原始语料（FAQ + 对话）
│   └── dataloader.py # PyTorch DataLoader
├── models/           # 模型定义
│   ├── encoder.py    # 文本编码器（BERT/自建）
│   ├── qa_model.py   # 问答模型
│   ├── retrieval.py  # 双编码器检索模型
│   └── lora.py       # LoRA 适配层
├── modules/          # 核心功能模块
│   ├── preprocess.py           # 语料预处理
│   ├── question_understanding.py # 问题理解（分类+关键词）
│   ├── knowledge_retrieval.py   # 知识检索（TF-IDF/BM25）
│   └── answer_generation.py    # 答案生成
├── utils/            # 工具类（配置/指标/分词）
├── predict.py        # 推理入口
├── train.py          # 训练入口
├── test.py           # 测试入口
├── evaluate.py       # 评价入口
└── requirements.txt  # 依赖清单
```

## 学习路线

| 课程 | 内容 | 状态 |
|------|------|------|
| 第1课 | 项目骨架 + TF-IDF 检索 | ✅ |
| 第2课 | 问题理解模块 | ✅ |
| 第3课 | 语义检索与向量化 (FAISS) | 📅 |
| 第4课 | 检索增强生成 (RAG + LLM) | 📅 |
| 第5课 | 模型评价与优化 | 📅 |

## 技术栈

- **框架**: PyTorch 2.x, Transformers
- **分词**: jieba
- **检索**: TF-IDF / BM25
- **模型**: BERT / RoBERTa（预留 LoRA 微调）

## 数据

- `data/raw/faq_expanded.json` — 265 条 FAQ，覆盖 20+ 分类
- `data/raw/dialogues.json` — 35 个多轮对话场景

## License

MIT
