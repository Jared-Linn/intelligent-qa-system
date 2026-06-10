# 智能问答系统

> 面向中文场景的智能问答平台 —— 检索式 + 生成式双路线，支持用户系统、模型管理、在线评测与排行榜。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)

---

## 快速体验

```bash
conda activate nlp
pip install -r requirements.txt
uvicorn web.main:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://localhost:8000` → 注册 → 创建模型 → 上传数据 → 开始问答。

---

## 两种模型

| 维度 | 检索式 (retrieval) | 生成式 (generative) |
|------|-------------------|-------------------|
| 原理 | TF-IDF 匹配预设问答对 | 千问模型 LoRA 微调生成 |
| 上传文件 | JSON 问答数据（question+answer） | LoRA 权重 zip |
| 上下文 | ❌ 一问一答 | ✅ 多轮对话 |
| 速度 | CPU，毫秒级 | GPU，秒级 |
| 一句话 | **查字典** | **真聊天** |

---

## Web 功能一览

| 功能 | 说明 |
|------|------|
| 用户系统 | 注册/登录 + JWT 无状态认证 |
| 模型管理 | 创建/上传/编辑/删除，两种模型类型 |
| 问答测试 | 微信风格聊天 UI，选模型即问即答 |
| 排行榜 | 综合/热度/检索式/生成式 Tab 切换 |
| 评分 | 用户可以对模型答案打分 |

## 快速开始

### 检索式

```bash
# CLI 单次查询
python retrieval_qa/predict.py -q "什么是机器学习"

# FAQ 交互模式
python retrieval_qa/predict.py -m faq

# Web 界面
uvicorn web.main:app --host 0.0.0.0 --port 8000
```

### 生成式（需要 GPU 训练）

```bash
# 安装依赖
pip install torch transformers datasets accelerate peft trl

# 训练（在 3090 上约 30 分钟）
python generative_qa/train_qwen_lora.py

# CLI 推理
python generative_qa/predict.py -q "什么是机器学习"
```

## 项目结构

```
├── retrieval_qa/           # 检索式问答模块
│   ├── modules/            #   预处理/问题理解/检索/答案生成
│   ├── utils/              #   配置/分词/指标
│   ├── models/             #   编码器/QA模型/LoRA
│   ├── predict.py          #   CLI 推理
│   └── configs/            #   配置文件
├── generative_qa/          # 生成式问答模块（千问 LoRA）
│   ├── train_qwen_lora.py  #   微调训练脚本
│   ├── predict.py          #   CLI 推理
│   └── configs/            #   训练配置
├── web/                    # FastAPI + SPA 前端
│   ├── main.py             #   应用入口
│   ├── database.py         #   SQLite ORM
│   ├── routers/            #   API 路由
│   │   ├── auth.py         #     认证（注册/登录/鉴权）
│   │   ├── models.py       #     模型 CRUD + 文件上传
│   │   ├── qa.py           #     问答 + 多模型对比
│   │   ├── ratings.py      #     评分
│   │   └── ranking.py      #     排行榜
│   ├── services/           #   业务逻辑
│   │   ├── auth.py         #     JWT + bcrypt
│   │   ├── file_manager.py #     文件上传/校验/解压
│   │   ├── infer_retrieval.py  # 检索式推理引擎
│   │   └── infer_generative.py # 生成式推理引擎
│   └── static/             #   SPA 前端
│       ├── index.html      #     入口
│       ├── style.css       #     样式
│       └── js/             #     JS 应用
│           ├── api.js      #     API 客户端
│           ├── auth.js     #     JWT 管理
│           └── router.js   #     SPA 路由 + 页面渲染
├── data/                   # 共享数据
│   ├── raw/                #   原始语料
│   └── processed/          #   处理后数据
├── checkpoints/            # 模型权重
└── docs/                   # 文档（架构设计等）
```

## API 端点

### 认证
| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 注册 |
| POST | `/api/auth/login` | 登录 → JWT |
| GET | `/api/auth/me` | 当前用户信息 |

### 模型管理（需登录）
| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/models` | 我的模型列表 |
| POST | `/api/models` | 创建模型 |
| PUT | `/api/models/{id}` | 更新模型（名称/描述/可见性） |
| POST | `/api/models/{id}/upload` | 上传文件 |
| DELETE | `/api/models/{id}` | 删除模型 |
| GET | `/api/models/public` | 公开模型广场 |

### 问答
| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/qa/ask` | 单模型问答 |
| POST | `/api/qa/compare` | 多模型对比 |

### 评分与排行榜
| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/ratings` | 提交评分 |
| GET | `/api/ratings/{model_id}` | 模型评分列表 |
| GET | `/api/ranking` | 排行榜 |

### 健康检查
| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/health` | 服务状态 |

## 技术栈

| 层 | 技术 |
|----|------|
| Web 框架 | FastAPI |
| 数据库 | SQLite |
| 认证 | JWT + bcrypt |
| 前端 | 纯 JS SPA（无框架） |
| 分词 | jieba |
| 检索 | TF-IDF / BM25（scikit-learn） |
| 生成模型 | Qwen2.5-3B-Instruct + PEFT LoRA |
| 训练硬件 | RTX 3090 (24GB) |

## 本地开发

```bash
# 启动
uvicorn web.main:app --reload --host 0.0.0.0 --port 8000

# 访问
# 前端:  http://localhost:8000
# API:   http://localhost:8000/docs (Swagger)
```

### Git 推送

本项目使用 SSH 推送 GitHub：

```bash
ssh -T git@github.com  # 验证连接
git push origin master
```

## License

MIT
