# 生成式问答模块 — generative_qa

> 基于 Qwen2.5 的 LoRA 微调生成式问答系统。

---

## 目录

1. [概述](#1-概述)
2. [环境准备](#2-环境准备)
3. [训练流程](#3-训练流程)
4. [推理测试](#4-推理测试)
5. [Web 对接](#5-web-对接)
6. [模型上传格式](#6-模型上传格式)
7. [部署架构](#7-部署架构)
8. [常见问题](#8-常见问题)

---

## 1. 概述

### 架构位置

```
用户请求 → Web (FastAPI) → QA 路由 → _do_inference()
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
              infer_retrieval.py               infer_generative.py
              (TF-IDF 检索)                     (Qwen + LoRA)
                        │                               │
                        ▼                               ▼
              data/raw/faq.json               checkpoints/uploads/{user}/{model}/
                                               (LoRA adapter weights)
```

### 技术栈

| 组件 | 选型 |
|------|------|
| 基座模型 | Qwen2.5-3B-Instruct |
| 微调方法 | LoRA (PEFT) |
| 训练框架 | HuggingFace TRL + Transformers |
| 推理硬件 | NVIDIA RTX 3090 (24GB) |
| 加载策略 | 即用即加载（方案A） |

### 两种调用方式

| 方式 | 路由 | 说明 |
|------|------|------|
| 通用问答 | `POST /api/qa/ask` | 通过模型 ID 自动分发（检索式/生成式） |
| 独立端点 | `POST /api/v1/generative/ask` | 专为生成式设计的独立接口（单模型/无用户系统） |

当前 Web 系统使用**通用问答**路由。`/api/v1/generative/ask` 为旧版预留端点。

---

## 2. 环境准备

### 硬件要求

- GPU: NVIDIA RTX 3090 (24GB) 或同等算力
- 显存: 训练 ~12GB / 推理 ~8GB
- 磁盘: 模型权重约 6GB

### 安装依赖

```bash
conda activate nlp

# 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 datasets>=2.14.0 accelerate>=0.28.0

# LoRA 微调
pip install peft>=0.10.0 trl>=0.8.0

# 可选
pip install tensorboard
```

### 验证安装

```bash
python -c "
import torch, transformers, peft
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
"
```

预期输出：
```
PyTorch: 2.x.x
CUDA: True
Transformers: 4.x.x
PEFT: 0.x.x
```

---

## 3. 训练流程

### 3.1 训练数据

使用已有 FAQ 数据 `data/raw/faq_expanded.json`，自动格式化为指令微调格式：

```json
{
    "instruction": "什么是机器学习",
    "output": "机器学习是人工智能的一个子领域..."
}
```

也可自定义数据，保持相同格式即可。

### 3.2 配置

编辑 `generative_qa/configs/train.yaml`：

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"    # 可从 1.5B/3B/7B 中选择

lora:
  r: 16                                 # LoRA 秩
  alpha: 32
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8        # 等效 batch size = 32
  learning_rate: 2e-4
  fp16: true
  gradient_checkpointing: true
```

### 3.3 运行训练

```bash
# 完整训练
python generative_qa/train_qwen_lora.py

# 参数覆盖
python generative_qa/train_qwen_lora.py \
    --epochs 5 \
    --batch_size 8 \
    --lr 3e-4

# 使用 7B 模型（需增加显存优化）
python generative_qa/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-7B-Instruct
```

### 3.4 训练输出

训练完成后，LoRA 权重保存在 `checkpoints/qwen_lora/`：

```
checkpoints/qwen_lora/
├── adapter_config.json      # LoRA 配置
├── adapter_model.safetensors  # LoRA 权重
├── tokenizer.json           # Tokenizer
└── tokenizer_config.json
```

整个目录约 50MB，压缩为 zip 后约 15MB。

### 3.5 显存优化（7B 模型）

如果要使用 Qwen2.5-7B 替代 3B：

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32
  fp16: true
  gradient_checkpointing: true

# 或使用 QLoRA（4-bit 量化）
# 需在 train_qwen_lora.py 中额外配置 BnBConfig
```

---

## 4. 推理测试

### 4.1 CLI 测试

```bash
# 使用训练好的 LoRA 权重
python generative_qa/predict.py -q "什么是机器学习"

# 交互模式
python generative_qa/predict.py -i

# 指定 LoRA 路径
python generative_qa/predict.py \
    -q "什么是人工智能" \
    --model_path checkpoints/qwen_lora/
```

### 4.2 Python API 测试

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web.services.infer_generative import GenerativeInferenceEngine

engine = GenerativeInferenceEngine()

# 模拟 Web 传入的模型数据
model = {
    "id": 1,
    "file_path": "checkpoints/uploads/1/1/",  # 用户上传的 LoRA 目录
}

result = engine.predict(model, "什么是机器学习")
print(result["answer"])
```

### 4.3 无 LoRA 回退

未训练或未上传 LoRA 时，引擎返回友好提示：

```json
{
    "answer": "[生成式问答] 模型未上传 LoRA 权重。请先上传。",
    "confidence": 0.0,
    "source": "no_lora"
}
```

---

## 5. Web 对接

### 5.1 对接架构

```
Web API 层                         推理引擎层
┌─────────────────┐    ┌──────────────────────────┐
│ /api/qa/ask      │───→│ _do_inference()          │
│ 请求体:           │    │   ├─ type == "retrieval" │
│   model_id       │    │   │   → RetrievalEngine  │
│   question       │    │   └─ type == "generative"│
└─────────────────┘    │       → GenerativeEngine  │
                       │           │               │
                       │       LoRA 加载策略        │
                       │       方案A: 即用即加载     │
                       │       1. 卸载当前 LoRA      │
                       │       2. torch.cuda.empty()│
                       │       3. PeftModel加载     │
                       │       4. 推理 → 返回       │
                       └──────────────────────────┘
```

### 5.2 调用示例

```bash
# 1. 上传 LoRA 权重到模型
curl -X POST http://localhost:8000/api/models/1/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@lora_weights.zip"

# 2. 问答测试
curl -X POST http://localhost:8000/api/qa/ask \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model_id": 1, "question": "什么是机器学习"}'
```

### 5.3 响应格式

```json
{
    "model_id": 1,
    "model_name": "我的千问模型",
    "answer": "机器学习是人工智能的一个子领域...",
    "confidence": 0.85,
    "latency_ms": 3200
}
```

### 5.4 路由分发逻辑

在 `web/routers/qa.py` 的 `_do_inference()` 中自动分发：

```python
if model["type"] == "retrieval":
    engine = RetrievalInferenceEngine()
    result = engine.predict(model, question)
else:  # generative
    engine = GenerativeInferenceEngine()
    result = engine.predict(model, question)
```

### 5.5 旧版独立端点

`web/routers/generative.py` 是一个独立的预留端点，当前返回 501。
如果不需要独立的生成式 API，可以删除此文件。

---

## 6. 模型上传格式

### 6.1 用户上传规范

Web 系统要求用户上传的 LoRA 权重为 **zip 压缩包**，包含：

```
lora_weights.zip
├── adapter_config.json        # 必需
├── adapter_model.safetensors  # 必需
├── tokenizer.json             # 建议包含
└── tokenizer_config.json      # 建议包含
```

### 6.2 从训练结果打包

```bash
# 训练后打包 LoRA 权重
cd checkpoints/qwen_lora/
zip -r ../../my_lora.zip adapter_config.json adapter_model.safetensors
```

### 6.3 上传后结构

服务器解压到 `checkpoints/uploads/{user_id}/{model_id}/`：

```
checkpoints/uploads/1/1/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

### 6.4 校验规则

上传时 `web/services/file_manager.py` 会校验：

| 校验项 | 说明 |
|--------|------|
| 是合法 zip | 非 zip 文件拒绝 |
| 包含 adapter_config.json | 缺少则拒绝 |
| 大小 ≤ 1GB | 超过则拒绝 |

---

## 7. 部署架构

### 7.1 单 GPU 方案（当前）

```
┌─────────────────────────────────────┐
│  FastAPI (uvicorn)                  │
│                                     │
│  GPU Memory:                        │
│  ┌──────────────────────┐           │
│  │ Qwen2.5-3B (~6GB)   │ 常驻       │
│  ├──────────────────────┤           │
│  │ LoRA_A 或 LoRA_B     │ 按需切换    │
│  │ (~50MB)             │           │
│  ├──────────────────────┤           │
│  │ 推理缓存 / 激活值     │           │
│  └──────────────────────┘           │
│                                     │
│  SQLite / 文件系统                   │
└─────────────────────────────────────┘
```

加载策略（方案A）：

```
请求到来 → 目标 LoRA 是否已在 GPU？
  ├─ 是 → 直接推理
  └─ 否 → 卸载当前 LoRA
          → torch.cuda.empty_cache()
          → PeftModel.from_pretrained()  耗时约 1-2s
          → 推理
          → 标记为"可卸载"
```

### 7.2 多 GPU 扩展（未来）

```
┌─ GPU 0: Qwen 基座 (常驻) ─┐
│  + LoRA 缓存槽 1           │
│  + LoRA 缓存槽 2           │
└───────────────────────────┘
┌─ GPU 1: Qwen 基座 (常驻) ─┐
│  + LoRA 缓存槽 3           │
│  + LoRA 缓存槽 4           │
└───────────────────────────┘
        ↑
  负载均衡器 (Nginx / 轮询)
```

### 7.3 启动脚本

```bash
# 生产启动
uvicorn web.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop asyncio \
    --log-level info

# 开发模式（热重载）
uvicorn web.main:app \
    --reload \
    --host 0.0.0.0 \
    --port 8000
```

> **注意**: 生成式推理有状态（GPU 内存），必须 `--workers 1`。

---

## 8. 常见问题

### Q: 训练时显存不足？

尝试以下优化（按效果排列）：

1. 减小 `batch_size` 到 1
2. 增大 `gradient_accumulation_steps`
3. 开启 `gradient_checkpointing: true`
4. 切换 3B 模型（而非 7B）
5. 使用 QLoRA（4-bit 量化）

### Q: 推理时显存不足？

- 方案A 每次只加载一个 LoRA，正常使用 ~8GB
- 如果 24GB 仍不足，检查是否其他进程占用了显存
- 回退方案：CPU 推理（`device_map="cpu"`），慢 10x 但可用

### Q: 切换 LoRA 太慢（1-2s）？

- 这是方案A 的代价。如果需要低延迟，考虑方案B（LRU 缓存常驻 2-3 个 LoRA）
- 在 `infer_generative.py` 中修改 `_current_lora` 为列表即可

### Q: 如何同时部署检索式和生成式？

完全不需要额外配置。`/api/qa/ask` 根据模型的 `type` 字段自动分发：

- `type: retrieval` → TF-IDF 检索（CPU，毫秒级）
- `type: generative` → Qwen + LoRA（GPU，秒级）

### Q: 如何备份训练好的 LoRA 权重？

```bash
# 训练后的权重在
checkpoints/qwen_lora/

# 打包
cd checkpoints/ && tar czf qwen_lora_backup.tar.gz qwen_lora/
```
