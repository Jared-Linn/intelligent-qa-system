# 多轮对话微调训练脚本
# 使用 LoRA（Low-Rank Adaptation）对 Qwen3.5-0.8B 进行多轮对话微调。
# 核心流程：
# 1. 加载预训练模型（强制 FP32 + eager attention，适配 Windows CPU）
# 2. 加载分词器（设置 pad_token + right padding）
# 3. 配置 LoRA 适配器（仅微调注意力层，冻结原始权重）
# 4. 加载并预处理训练数据（分词 + label 掩码）
# 5. 配置训练参数并启动训练
# 6. 保存 LoRA 适配器权重
#
# 适用环境：Windows / Linux CPU，不适用于 GPU

import os
# 禁用 oneDNN（Intel 的 DNNL 加速库），避免 bf16 反向传播错误
os.environ["ONEDNN_ENABLED"] = "0"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# =============================================================================
# 超参数配置（集中管理，方便修改）
# =============================================================================
MODEL_PATH = r"D:\dataset\model\Qwen3.5-0.8B"   # 基础模型路径
OUTPUT_DIR = r"./outputs_multi_turn"           # 训练输出目录
DATA_PATH = r"./data/multi_turn_qa.json"       # 训练数据路径（JSON 格式）
MAX_SEQ_LENGTH = 1024                          # 最大序列长度

# LoRA 超参数
LORA_R = 16                                    # LoRA 秩（rank）
LORA_ALPHA = 32                                # LoRA 缩放系数（实际缩放 = alpha/r = 2）
LORA_DROPOUT = 0.1                             # Dropout 概率

# 训练超参数
LEARNING_RATE = 2e-4                           # 学习率
NUM_EPOCHS = 3                                 # 训练轮数
BATCH_SIZE = 1                                 # 每设备批次大小（CPU 内存有限）
GRADIENT_ACCUMULATION_STEPS = 4                # 梯度累积步数（等效批次 = 4）

print("=" * 60)
print("多轮对话微调训练")
print(f"模型路径：{MODEL_PATH}")
print(f"输出目录：{OUTPUT_DIR}")
print(f"数据路径：{DATA_PATH}")
print(f"最大序列长度：{MAX_SEQ_LENGTH}")
print("=" * 60)

# ==================== 1. 加载模型 ====================
print("\n1. 加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="eager",               # 强制使用 eager attention，避免 Windows 下的 bug
)
# 二次保险：逐参数强制转换为 FP32
model = model.to("float")
for param in model.parameters():
    param.data = param.data.to(torch.float32)
print("模型加载完成（已转换为 FP32）")

# ==================== 2. 加载分词器 ====================
print("\n2. 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("已设置 padding token = eos token")
tokenizer.padding_side = "right"               # 训练时使用右填充

# ==================== 3. 配置 LoRA ====================
print("\n3. 配置 LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== 4. 加载数据集 ====================
print("\n4. 加载训练数据...")
dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
print(f"数据集加载完成，共 {len(dataset)} 条对话")
print("\n数据示例：")
print(dataset[0]["text"][:300] + "...")

# ==================== 5. 分词处理（含 label 掩码） ====================
print("\n5. 数据预处理（分词 + label 掩码）...")

IM_START = "<im_start>"
IM_END = "</im_end>"

def tokenize_function(examples):
    """
    分词函数 - 多轮对话标签 label 掩码
    核心思想：只有 assistant 回复的 token 参与 loss 计算，用户输入和系统提示的 token 设为 -100 被忽略。
    """
    # 第一步：分词（不填充，后续由 DataCollator 动态填充）
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    # 第二步：获取特殊 token 和角色关键词的 ID
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    user_id = tokenizer.convert_tokens_to_ids("user")
    system_id = tokenizer.convert_tokens_to_ids("system")

    # 角色 ID 映射表（用于快速查找）
    role_to_id = {
        "assistant": assistant_id,
        "user": user_id,
        "system": system_id,
    }
    # 反向映射：token id -> 角色名（仅用于判断）
    id_to_role = {v: k for k, v in role_to_id.items()}

    # 第三步：为每个序列生成 labels
    all_labels = []
    for input_ids in tokens["input_ids"]:
        seq_len = len(input_ids)
        label_ids = [-100] * seq_len          # 默认全部掩码
        current_role = None                   # 当前所在角色块

        # 遍历 token 序列，根据 <im_start>role 切换角色状态
        i = 0
        while i < seq_len:
            # 检测 <im_start> 标记
            if input_ids[i] == im_start_id and i + 1 < seq_len:
                # 下一个 token 应该是角色名（assistant/user/system）
                role_token_id = input_ids[i + 1]
                role_name = id_to_role.get(role_token_id, None)
                if role_name:
                    current_role = role_name
                    # 保留 <im_start> 和角色名的 label（-100 或参与计算）
                    # 注意：<im_start> 和角色名本身也应被掩码或保留？通常只保留回复内容，
                    # 但为了简单，这里只标记 assistant 块内的所有 token（包括 <im_start>assistant 和 <im_end>）
                    # 我们将在后续循环中统一处理：如果 current_role == "assistant"，则全部保留。
                i += 2
                continue
            # 检测 <im_end> 标记（可选，用于结束当前角色块）
            if input_ids[i] == im_end_id:
                # 保留该 token（如果是 assistant 块内）
                # 然后退出当前角色块（下一个标记开始新角色）
                current_role = None
                i += 1
                continue
            # 普通 token：根据当前角色决定是否保留 label
            if current_role == "assistant":
                label_ids[i] = input_ids[i]   # 保留原始 token ID，参与 loss 计算
            # 否则 label_ids[i] 保持 -100（被忽略）
            i += 1
        all_labels.append(label_ids)

    tokens["labels"] = all_labels
    return tokens

# 批量应用分词函数
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)
print("分词处理完成（已对用户输入和系统提示做 label 掩码）")

# ==================== 6. 配置训练参数 ====================
print("\n6. 配置训练参数...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=5,
    logging_first_step=True,
    save_steps=50,
    save_total_limit=2,
    prediction_loss_only=True,
    report_to="none",
    data_loader_num_workers=0,      # Windows 下避免多进程问题
    remove_unused_columns=False,    # 必须为 False，因为手动添加了 "labels" 列
    use_cpu=True,
    fp16=False,
    bf16=False,
    tf32=False,
)

# ==================== 7. 初始化 Trainer ====================
print("\n7. 初始化 Trainer...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,                      # False = 因果语言模型（CLM），预测下一个 token
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ==================== 8. 开始训练 ====================
print("\n" + "=" * 60)
print("8. 开始训练...")
print("=" * 60)
print(f"训练轮数: {NUM_EPOCHS}")
print(f"每设备批次: {BATCH_SIZE}")
print(f"梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
print(f"学习率: {LEARNING_RATE}")
print("=" * 60)

try:
    train_result = trainer.train()
    print("\n训练完成！")

    metrics = train_result.metrics
    print(f"\n训练指标:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 打印 loss 历史
    log_history = trainer.state.log_history
    if log_history:
        print("\nloss 变化历史:")
        for log in log_history:
            if "loss" in log:
                print(f"  Step {log['step']:>4d} | Loss: {log['loss']:.4f}")
except Exception as e:
    print(f"\n训练失败: {e}")
    raise

# ==================== 9. 保存模型 ====================
print("\n9. 保存 LoRA 适配器...")
lora_path = f"{OUTPUT_DIR}/lora_adapter"
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"LoRA 适配器已保存至: {lora_path}")

print("\n" + "=" * 60)
print("多轮对话微调完成！")
print("=" * 60)