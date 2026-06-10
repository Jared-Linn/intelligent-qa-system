# 检索式问答模块 — retrieval_qa

> TF-IDF / BM25 匹配预设问答对。查字典模式，一问一答，不支持上下文。
> 需要智能对话？用 generative_qa（千问 LoRA 微调）。

## 概述

基于 TF-IDF / BM25 的检索式问答系统。输入问题 → 问题理解（分类 + 关键词）→ 知识检索 → 答案生成。

**三步上手：**

```bash
conda activate nlp
pip install -r retrieval_qa/requirements.txt
python retrieval_qa/predict.py -m faq
```

---

## 目录

- [环境配置](#环境配置)
- [数据说明](#数据说明)
- [运行方式](#运行方式)
- [模块说明](#模块说明)
- [训练流程](#训练流程)
- [评估](#评估)
- [API 服务](#api-服务)
- [常见问题](#常见问题)

---

## 环境配置

### 硬件要求

无 GPU 要求，纯 CPU 即可运行。

### 依赖安装

```bash
# 创建 conda 环境
conda create -n qa python=3.10 -y
conda activate qa

# 或使用已有环境
conda activate nlp

# 安装依赖
pip install -r retrieval_qa/requirements.txt
```

`requirements.txt` 内容：

```
pyyaml>=6.0
jieba>=0.42.1
numpy>=1.24.0
scikit-learn>=1.2.0
scipy>=1.9.0
```

---

## 数据说明

### 数据格式

FAQ 数据放在 `data/raw/faq_expanded.json`，JSON 数组格式：

```json
[
  {
    "question": "什么是机器学习",
    "answer": "机器学习是人工智能的一个子领域...",
    "category": "ai_ml"
  }
]
```

多轮对话数据放在 `data/raw/dialogues.json`：

```json
[
  {
    "id": "dialogue_001",
    "title": "Python 入门学习咨询",
    "scenario": "学习咨询",
    "category": "programming_learning",
    "turns": [
      {"role": "user", "text": "你好，我想学编程"},
      {"role": "assistant", "text": "你好！推荐从 Python 开始..."}
    ]
  }
]
```

### 自定义数据集

按上述格式准备 JSON 文件，修改 `retrieval_qa/configs/default.yaml` 中的 `data.faq_path` 或 `data.dialogue_path`。

---

## 运行方式

### 1. FAQ 交互模式

```bash
python retrieval_qa/predict.py -m faq
```

启动后进入交互式问答，输入问题获取答案，输入 `exit` 退出。

### 2. 单次查询

```bash
python retrieval_qa/predict.py -q "什么是机器学习"
```

输出：
```
[问题] 什么是机器学习
[答案] 机器学习是人工智能的一个子领域...
[置信度] 0.9241
[来源] 什么是机器学习
[候选结果]:
  1. [1.0000] 什么是机器学习
  2. [0.4764] 什么是机器学习中的过拟合和欠拟合
```

### 3. 对话模式

```bash
python retrieval_qa/predict.py -m dialogue
```

多轮对话，支持上下文感知。输入 `reset` 重置对话。

### 4. 禁用问题理解

```bash
python retrieval_qa/predict.py -q "什么是机器学习" --no-understand
```

跳过问题分类和关键词提取，直接用原文本检索。

### 命令行参数

| 参数 | 简写 | 说明 | 默认 |
|------|------|------|------|
| `--question` | `-q` | 单次查询的问题 | 无 |
| `--mode` | `-m` | 运行模式: `faq` / `dialogue` | 自动判断 |
| `--config` | `-c` | 配置文件路径 | `configs/default.yaml` |
| `--top_k` | `-k` | 返回候选数 | 配置默认 3 |
| `--no-understand` | | 禁用问题理解 | false |

---

## 模块说明

```
retrieval_qa/
├── configs/default.yaml       # 配置文件
├── modules/
│   ├── preprocess.py          # 语料加载、清洗、词表构建
│   ├── question_understanding.py  # 问题分类 + 关键词提取
│   ├── knowledge_retrieval.py     # TF-IDF / BM25 / 混合检索
│   ├── answer_generation.py       # 答案生成 + 对话管理
│   └── dataloader.py              # PyTorch DataLoader（预留）
├── models/
│   ├── encoder.py             # 文本编码器（BERT/自建）
│   ├── qa_model.py            # 抽取式问答模型
│   ├── retrieval.py           # 双编码器检索模型
│   └── lora.py                # LoRA 适配层
├── utils/
│   ├── config.py              # YAML 配置加载
│   ├── metrics.py             # EM / F1 / Precision@K / Recall@K
│   └── tokenizer.py           # jieba 中文分词
├── predict.py                 # 推理入口
├── test.py                    # 测试脚本
├── evaluate.py                # 评价脚本
└── train.py                   # 训练骨架（预留）
```

### 核心流程

```
用户输入
   │
   ▼
问题理解 (QuestionUnderstanding)
   ├─ 问题分类（关键词规则匹配 → 返回分类标签）
   └─ 关键词提取（TF-IDF / TextRank → 关键词列表）
   │
   ▼
知识检索 (TfidfRetriever / Bm25Retriever)
   ├─ 构建 TF-IDF 向量索引（首次运行自动构建）
   ├─ 计算查询向量与所有文档的余弦相似度
   └─ 返回 Top-K 候选问答对
   │
   ▼
答案生成 (AnswerGenerator)
   ├─ 取最高分候选作为答案
   ├─ 低于阈值返回兜底回答
   └─ 附加置信度评分和来源信息
   │
   ▼
最终答案
```

### 配置项

参见 `retrieval_qa/configs/default.yaml`，关键配置：

| 配置路径 | 说明 | 可选值 |
|----------|------|--------|
| `retrieval.method` | 检索方法 | `tfidf` / `bm25` / `hybrid` |
| `retrieval.top_k` | 返回候选数 | 整数 |
| `answer.method` | 生成方法 | `direct` / `extractive` / `generative` |
| `answer.min_score` | 最低相似度阈值 | 0.0 ~ 1.0 |
| `question_understanding.classifier` | 分类方法 | `rule` / `tfidf` |
| `question_understanding.keyword_method` | 关键词方法 | `tfidf` / `textrank` |

---

## 训练流程

当前检索式问答采用无监督方法（TF-IDF / BM25），无需训练。

### "训练" = 构建索引

首次运行 `predict.py` 时自动完成：

```bash
# 首次运行会自动分词 → 构建词表 → 计算 TF-IDF → 保存索引
python retrieval_qa/predict.py -q "测试"
```

流程日志：
```
[数据] 加载了 264 条 FAQ 数据
[检索器] 未检测到缓存，重新构建索引...
[索引] 正在分词...
[索引] 分词完成，264 篇文档
[索引] 词表大小: 1010
[索引] 正在计算 TF-IDF 矩阵...
[索引] TF-IDF 矩阵形状: (264, 1010)
[检索器] 已保存到 data/processed/vectorizer.pkl
```

第二次启动加载缓存，不需重新构建：
```
[检索器] 检测到缓存，加载: data/processed/vectorizer.pkl
```

### 切换检索方法

编辑 `configs/default.yaml`：

```yaml
retrieval:
  method: "bm25"    # 改为 bm25 或 hybrid（混合检索）
```

删除缓存重建索引：

```bash
rm data/processed/vectorizer.pkl
python retrieval_qa/predict.py -q "测试"
```

### 使用自定义数据

1. 准备 JSON 文件（格式见上方数据说明）
2. 修改配置：
   ```yaml
   data:
     faq_path: "data/raw/my_data.json"
   ```
3. 删除旧缓存，运行 `predict.py` 自动构建新索引

---

## 评估

```bash
python retrieval_qa/evaluate.py
```

输出示例：
```
============================================================
  综合评价报告
============================================================
  总样本: 50
  整体 EM:  1.0000
  整体 F1:  1.0000
------------------------------------------------------------
  分类评估:
    ai_ml              n= 31  EM=1.0000  F1=1.0000
    cloud_computing    n=  8  EM=1.0000  F1=1.0000
    programming        n= 11  EM=1.0000  F1=1.0000
------------------------------------------------------------
  失败案例（EM=0 且 F1<0.3）:
  共 0 个失败案例
============================================================
```

> **注意：** 默认用训练数据本身做评估，EM/F1 为 1.0 是预期行为。
> 如需真实评估，用独立测试集替换 `evaluate.py` 中的数据路径。

```bash
# 测试推理性能
python retrieval_qa/test.py
```

---

## API 服务

检索式问答已封装为 FastAPI 端点，通过 Web 服务调用：

```bash
# 启动 Web 服务
uvicorn web.main:app --reload --host 0.0.0.0 --port 8000

# 请求示例
curl "http://localhost:8000/api/v1/retrieval/ask?q=什么是机器学习"

# 响应格式
{
  "question": "什么是机器学习",
  "answer": "机器学习是人工智能的一个子领域...",
  "confidence": 0.915,
  "source": "什么是机器学习",
  "category": "ai_ml",
  "keywords": ["机器学习", "什么"],
  "candidates": [
    {"question": "什么是机器学习", "score": 0.9765},
    ...
  ]
}
```

---

## 常见问题

### Q: 首次运行很慢？

首次需要加载 jieba 词典 + 分词语料 + 构建 TF-IDF 矩阵，约 3-5 秒。第二次启动加载缓存，1 秒内完成。

### Q: 答案不对或为空？

1. 检查 `answer.min_score` 阈值，默认 0.1，可适当降低
2. 确认问题关键词在数据集中存在
3. 运行 `--no-understand` 排除问题理解模块的影响

### Q: 如何增加新数据？

直接编辑 `data/raw/faq_expanded.json` 添加条目，删除 `data/processed/vectorizer.pkl`，重新运行。

### Q: 支持英文吗？

分词器（jieba）专为中文设计。英文问题会按字切分，效果较差。如需处理英文，可切换 `tokenizer.method` 为 `char` 或替换分词器。

### Q: 对话模式和 FAQ 模式有什么区别？

- FAQ 模式：每次独立回答问题，无上下文
- 对话模式：维护对话历史，匹配对话场景中的完整话轮

---

## 项目来源

本科课程期末项目，基于项目思维导图设计实现。

- 架构文档：`/智能问答系统/架构文档_精修版.md`
- 思维导图：`/智能问答系统/mindmap.png`
