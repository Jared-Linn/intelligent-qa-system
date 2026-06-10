# 智能问答系统

> 面向中文场景的智能问答系统 —— 检索式 + 生成式双路线
>
> [![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
> [![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green)](https://fastapi.tiangolo.com)
> [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)

---

## 系统架构

```
用户输入 ──→ Web (FastAPI)
                │
        ┌───────┴───────┐
        ▼               ▼
  retrieval_qa/    generative_qa/
  (检索式问答)     (千问 LoRA 微调)
        │               │
        └───────┬───────┘
                ▼
            最终答案
```

### 两种问答模式

| 模式 | 方法 | 状态 |
|------|------|------|
| **检索式** | TF-IDF / BM25 检索 + 直接返回答案 | ✅ 可用 |
| **生成式** | Qwen2.5-3B-Instruct LoRA 微调 | 📅 待训练 |

## 快速开始

### 检索式问答

```bash
conda activate nlp

# CLI 单次查询
python retrieval_qa/predict.py -q "什么是机器学习"

# FAQ 交互模式
python retrieval_qa/predict.py -m faq

# 对话模式
python retrieval_qa/predict.py -m dialogue

# Web 界面
uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000
```

### 生成式问答（千问 LoRA 微调）

```bash
# 安装依赖
pip install torch transformers datasets accelerate peft trl

# 训练
python generative_qa/train_qwen_lora.py

# 推理
python generative_qa/predict.py -q "什么是机器学习"
```

## 项目结构

```
├── retrieval_qa/           # 检索式问答
│   ├── configs/            # 配置文件
│   ├── modules/            # 核心模块（预处理/问题理解/检索/答案生成）
│   ├── utils/              # 工具类（配置/指标/分词）
│   ├── predict.py          # 推理入口
│   ├── test.py / evaluate.py
│   └── requirements.txt
├── generative_qa/          # 生成式问答（千问 LoRA）
│   ├── configs/train.yaml  # 训练配置
│   ├── train_qwen_lora.py  # 微调训练
│   ├── predict.py          # 推理入口
│   └── requirements.txt
├── web/                    # FastAPI Web 服务
│   ├── main.py             # 应用入口
│   ├── routers/            # API 路由
│   │   ├── retrieval.py    # 检索式 API
│   │   └── generative.py   # 生成式 API（预留）
│   ├── static/             # 前端文件
│   └── templates/
├── data/                   # 共享数据
│   ├── raw/                # 原始语料（FAQ 265条 + 对话35场景）
│   └── processed/          # 预处理后数据
├── checkpoints/            # 模型权重
├── logs/                   # 日志
└── requirements.txt
```

## 技术栈

| 层 | 技术 |
|----|------|
| Web 框架 | FastAPI |
| 深度学习 | PyTorch 2.x, Transformers |
| 分词 | jieba |
| 检索 | TF-IDF / BM25 |
| 生成模型 | Qwen2.5-3B-Instruct + LoRA |
| 训练硬件 | RTX 3090 (24GB) |
| 环境管理 | conda (nlp) |

## API 文档

启动 Web 服务后访问:

- Swagger UI: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/retrieval/ask` | POST/GET | 检索式问答 |
| `/api/v1/generative/ask` | POST/GET | 生成式问答（预留） |
| `/api/v1/generative/status` | GET | 生成式服务状态 |
| `/health` | GET | 健康检查 |

## 学习路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| 第1课 | 项目骨架 + TF-IDF 检索 | ✅ |
| 第2课 | 问题理解模块（分类+关键词） | ✅ |
| 📍 **现在** | **FastAPI Web 服务 + 项目拆分** | ✅ |
| 第3课 | 千问 LoRA 微调训练 | 📅 |
| 第4课 | 语义检索 FAISS + 混合检索 | 📅 |
| 第5课 | 评价与调优 | 📅 |

## License

MIT
