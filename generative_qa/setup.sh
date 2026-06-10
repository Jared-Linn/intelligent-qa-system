#!/usr/bin/env bash
# ============================================================================
# 生成式问答 — 远程服务器环境配置
# ============================================================================
# 用法: bash generative_qa/setup.sh
# 运行环境: Ubuntu 22.04, RTX 3090 (24GB), conda (base)
#
# 本脚本:
#   1. 安装 Python 依赖 (torch已存在则跳过)
#   2. 创建目录结构 (checkpoints, logs, HF cache)
#   3. 设置 HF_HOME 指向 /autodl-pub (防止30G root爆)
#   4. 验证环境
#   5. 可选: 预下载 Qwen2.5-3B-Instruct 模型
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_BASE="/root/miniconda3"

echo "========================================"
echo "  生成式问答 — 环境配置"
echo "========================================"
echo "  项目路径: $PROJECT_ROOT"
echo "  Conda:    $CONDA_BASE"
echo ""

# ── 1. Conda 基础 ─────────────────────────────────────────────────────────
export PATH="$CONDA_BASE/bin:$PATH"

# 创建 env (若不存在)
if conda env list | grep -q "qa_gen"; then
    echo "[Conda] qa_gen 已存在, 跳过创建"
else
    echo "[Conda] 创建环境: qa_gen (Python 3.12)..."
    conda create -y -n qa_gen python=3.12
fi

# ── 2. 安装核心依赖 ──────────────────────────────────────────────────────
echo ""
echo "[安装] Python 依赖 ..."

# 激活 env
source "$CONDA_BASE/bin/activate" qa_gen

# 检查 torch 已有 (可能预装)
if python -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null; then
    echo "[跳过] torch 已安装"
else
    echo "[安装] torch ..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# 训练依赖
pip install \
    transformers>=4.40.0 \
    datasets>=2.14.0 \
    accelerate>=0.28.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    evaluate \
    nltk \
    tensorboard \
    pyyaml

# 额外工具
pip install jupyter ipywidgets tqdm

echo "[完成] 依赖安装完成"

# ── 3. 目录结构 ──────────────────────────────────────────────────────────
echo ""
echo "[目录] 创建目录结构 ..."

# /autodl-pub 存在则用, 否则回退项目内
if [ -d "/autodl-pub" ]; then
    HF_CACHE_DIR="/autodl-pub/huggingface"
    CKPT_DIR="/autodl-pub/checkpoints/qwen_lora"
    LOG_DIR="/autodl-pub/logs/qwen_lora"
    MODEL_CACHE_DIR="/autodl-pub/models/qwen_cache"
else
    HF_CACHE_DIR="$PROJECT_ROOT/.huggingface"
    CKPT_DIR="$PROJECT_ROOT/checkpoints/qwen_lora"
    LOG_DIR="$PROJECT_ROOT/logs/qwen_lora"
    MODEL_CACHE_DIR="$PROJECT_ROOT/checkpoints/qwen_cache"
fi

mkdir -p "$HF_CACHE_DIR" "$CKPT_DIR" "$LOG_DIR" "$MODEL_CACHE_DIR"
echo "  HF_HOME:     $HF_CACHE_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo "  Logs:        $LOG_DIR"
echo "  Model cache: $MODEL_CACHE_DIR"

# ── 4. 环境变量配置 (写入 conda env) ────────────────────────────────────
echo ""
echo "[配置] 环境变量 ..."

# 写入 conda env vars (每次激活自动生效)
CONDA_ENV_DIR="$CONDA_BASE/envs/qa_gen"
mkdir -p "$CONDA_ENV_DIR/etc/conda/activate.d"
cat > "$CONDA_ENV_DIR/etc/conda/activate.d/env_vars.sh" << EOF
export HF_HOME=$HF_CACHE_DIR
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_HOME=$HF_CACHE_DIR/torch
EOF

chmod +x "$CONDA_ENV_DIR/etc/conda/activate.d/env_vars.sh"
echo "  已写入: $CONDA_ENV_DIR/etc/conda/activate.d/env_vars.sh"
echo "  下次 conda activate qa_gen 自动生效"

# 当前 session 生效
export HF_HOME=$HF_CACHE_DIR
export HF_ENDPOINT=https://hf-mirror.com

# ── 5. 验证 ─────────────────────────────────────────────────────────────
echo ""
echo "[验证] 环境测试 ..."

python -c "
import torch, transformers, peft, trl, accelerate, datasets
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA:        {torch.cuda.is_available()}')
print(f'GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'Transformers:{transformers.__version__}')
print(f'PEFT:        {peft.__version__}')
print(f'TRL:         {trl.__version__}')
print(f'Accelerate:  {accelerate.__version__}')
print(f'Datasets:    {datasets.__version__}')
print(f'HF_HOME:     {os.environ.get(\"HF_HOME\", \"\")}')
" 2>&1

echo ""
echo "[验证] ✅ 所有依赖正常"

# ── 6. 可选: 预下载模型 ───────────────────────────────────────────────
echo ""
echo "[可选] 预下载 Qwen2.5-3B-Instruct?"
echo "       下载约 6GB, 仅在首次需要."
echo "       跳过也没关系, 训练时会自动下载."
echo ""
read -p "  下载? (y/N): " DL_MODEL

if [ "$DL_MODEL" = "y" ] || [ "$DL_MODEL" = "Y" ]; then
    echo "[下载] Qwen/Qwen2.5-3B-Instruct ..."
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2.5-3B-Instruct'
print(f'下载 tokenizer: {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'下载模型: {model_name} (约 6GB, 耐心等待)')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    device_map='auto',
    trust_remote_code=True,
)
print('[下载] 完成!')
"
fi

# ── 7. 完成提示 ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  环境配置完成!"
echo "========================================"
echo ""
echo "  训练:"
echo "    conda activate qa_gen"
echo "    python generative_qa/train_qwen_lora.py"
echo ""
echo "  单次推理:"
echo "    python generative_qa/predict.py -q '问题'"
echo ""
echo "  评测:"
echo "    python generative_qa/evaluate.py"
echo ""
echo "  Web 服务 (需先训练):"
echo "    uvicorn web.main:app --host 0.0.0.0 --port 8000"
echo "========================================"
