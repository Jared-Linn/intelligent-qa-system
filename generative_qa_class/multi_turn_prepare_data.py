"""
多轮对话数据准备脚本

【功能说明】
将 jiandanxinli（简单心理）心理咨询 QA 数据转换为 Qwen ChatML 格式。
该脚本负责：
  1. 读取原始 JSON 格式的心理咨询问答数据
  2. 将每条记录按回答者拆分为独立的多轮对话（system / user / assistant）
  3. 转换为 Qwen ChatML 文本格式，输出为 JSONL 文件供 LoRA 微调使用

【输入】jiandanxinli_qa_data_v1.0.json（原始咨询数据）
【输出】./data/multi_turn_qa.json（ChatML 格式训练数据）
"""

# json：用于读取原始 JSON 数据和写出 JSONL 格式的训练文件
import json
# os：用于创建输出目录（os.makedirs），确保 data/ 目录存在
import os

# ==================== 全局配置 ====================
# 输入数据文件路径：jiandanxinli（简单心理）心理咨询 QA 数据集（JSON 格式）
# 该数据集包含用户提问（标题+正文）和多个回答者的多轮对话记录
# 作用：作为脚本的唯一数据入口，指定待处理的原始文件
# 效果：修改此路径可切换为其他数据源，无需改动函数逻辑
INPUT_FILE = "jiandanxinli_qa_data_v1.0.json"

# 输出数据文件路径：转换后的 Qwen ChatML 格式数据，每行一条 JSON 记录
# 作用：指定处理后训练数据的存储位置
# 效果：输出目录 data/ 会在运行时自动创建，无需手动新建；
#       输出的 JSONL 文件可直接被 multi_turn_finetune_qwen_cpu.py 训练脚本读取
OUTPUT_FILE = "./data/multi_turn_qa.json"

# 系统提示词（System Prompt）：
#   作用 —— 设定模型在对话中的角色身份和行为准则
#   效果 —— 微调时模型会学习以「专业心理咨询师」的风格回复，
#           保持共情、温暖、非评判性的语气，避免生硬说教
SYSTEM_PROMPT = "你是一位专业、温暖、富有共情能力的心理咨询师。你会认真倾听来访者的困惑，给予理解和接纳，并从专业角度提供有建设性的建议。回复时请保持真诚、温和的语气，避免说教和评判。"


def load_raw_data(file_path):
    """
    从 JSON 文件加载原始心理咨询 QA 数据。
    
    该函数读取 jiandanxinli 数据集的完整 JSON 文件，每条记录包含：
      - question_title: 问题标题（简短描述）
      - question_content: 问题正文（详细描述）
      - answers: 回答者列表，每个回答者包含 dialogs（对话轮次列表）
    
    :param file_path: str — JSON 文件的绝对或相对路径，
                      文件内容为 JSON 数组，每个元素是一条咨询记录
    :return: list[dict] — 原始数据列表，每个字典对应一条完整的咨询记录
    :raises FileNotFoundError: 当指定路径的文件不存在时抛出
    :raises json.JSONDecodeError: 当文件内容不是合法 JSON 时抛出
    """
    # 以 UTF-8 编码打开文件，确保中文字符正常读写，避免 Windows 系统默认 GBK 编码导致的乱码
    with open(file_path, "r", encoding="utf-8") as f:
        # json.load() 将整个 JSON 数组一次性加载到内存
        # 适用场景：数据集规模较小（数千条级别），可一次性加载
        # 如果数据集极大（百万级），建议改用逐行读取的 JSONL 格式
        data = json.load(f)
    # 打印加载的数据条数，便于快速确认文件是否读取正确
    print(f"📂 已加载 {len(data)} 条原始数据")
    return data


def build_conversations(raw_data):
    """
    将 jiandanxinli 心理咨询数据转换为标准多轮对话列表。
    
    【数据映射规则】
    1. 每条记录的 question_title + question_content 拼接为首轮 user 消息
       - 如果两者都存在，用换行符拼接；如果只有其一，直接使用存在的部分
    2. 每条记录可能有多个回答者（answers），每个回答者拆分为一条独立对话
    3. 每个回答者的 dialogs 按时间顺序解析：
       - role 以 "answer_{answer_user_id}" 或 "answer_" 开头 → 映射为 assistant（咨询师回复）
       - role 以 "user_" 开头 → 映射为 user（提问者的追问/补充）
    4. 每条对话最前面插入 system 角色的系统提示词，定义模型行为
    
    【过滤规则】
    - 跳过没有问题文本的记录（question_title 和 question_content 均为空）
    - 跳过没有回答者（answers 为空）的记录
    - 跳过没有任何 assistant 回复的对话（至少需要一轮咨询师回复）
    
    :param raw_data: list[dict] — 原始 JSON 数据列表，每个元素包含以下字段：
                     - question_title (str): 问题标题
                     - question_content (str): 问题正文
                     - answers (list[dict]): 回答者列表，每个回答者包含：
                       - answer_user_id (str): 回答者用户 ID，用于匹配 role 前缀
                       - dialogs (list[dict]): 对话轮次，每个元素包含 role 和 content
    :return: list[dict] — 对话列表，每项结构为：
             {"conversations": [{"role": "system", "content": ...},
                                {"role": "user", "content": ...},
                                {"role": "assistant", "content": ...}, ...]}
    """
    conversations = []   # 存储所有生成的多轮对话，最终作为函数返回值供后续格式转换使用
    skipped = 0          # 计数器：因缺少问题文本或无回答而被跳过的记录数
                         #   作用：追踪无效数据量，帮助判断原始数据质量
                         #   效果：日志中展示跳过数，方便发现数据采集问题
    multi_turn = 0       # 计数器：包含用户追问（真正多轮交互）的对话数量
                         #   作用：统计多轮对话占比，评估数据集的多轮交互丰富度
                         #   效果：日志中展示多轮对话数，帮助判断是否需要补充多轮数据

    for item in raw_data:
        # 拼接问题标题和正文，作为首轮用户提问内容
        # 设计意图：标题通常是简短概括，正文包含更多细节，拼接后信息更完整
        # strip() 去除首尾空白，避免拼接时出现多余空格
        title = item.get("question_title", "").strip()
        content = item.get("question_content", "").strip()
        # 拼接策略：两者都有时用换行连接；只有一个时取非空值；都为空则 question_text 为空字符串
        question_text = f"{title}\n{content}" if title and content else (title or content)

        if not question_text:
            skipped += 1
            continue

        answers = item.get("answers", [])  # 回答者列表，每条记录可能有多个回答者
        if not answers:  # 如果该问题没有任何回答者，跳过并计数
            skipped += 1
            continue

        # 每个回答者生成一条独立的多轮对话
        # 设计意图：同一问题下不同回答者的对话风格和深度可能不同，
        #           拆分为独立对话可以让模型分别学习每种交互模式
        for answer in answers:
            # dialogs: 该回答者与提问者之间的对话轮次列表，按时间顺序排列
            # 作用：包含该回答者与用户的所有交互记录，是构建多轮对话的核心数据源
            # 效果：遍历 dialogs 可以按原始时间线还原完整的对话流程
            dialogs = answer.get("dialogs", [])
            if not dialogs:
                continue

            # answer_user_id: 回答者的用户标识，用于构建 role 前缀
            # 作用：精确匹配该回答者在 dialogs 中的消息（如 "answer_12345"）
            # 效果：避免多个回答者的消息混淆，确保每条消息正确归属到对应角色
            answer_user_id = answer.get("answer_user_id", "")
            # answer_role_prefix: 拼接完整的回答者 role 前缀，如 "answer_12345"
            # 用于后续与 dialogs 中每条消息的 role 字段进行 startswith 匹配
            answer_role_prefix = f"answer_{answer_user_id}"

            # 构建对话轮次列表，首轮固定为用户提问
            # chat_turns 仅包含 user 和 assistant 的消息，system 提示词后续单独插入
            chat_turns = [{"role": "user", "content": question_text}]
            # 标记是否存在用户追问，用于区分单轮对话 vs 多轮对话
            # 作用：统计多轮对话占比，帮助评估数据集的多轮交互丰富度
            # 效果：打印日志时展示多轮对话数量，便于判断数据质量是否满足微调需求
            has_user_followup = False

            # 遍历该回答者的所有对话轮次，按原始时间顺序逐条解析
            for d in dialogs:
                role_str = d.get("role", "")       # 原始角色标识，格式如 "answer_12345" 或 "user_67890"
                msg_content = d.get("content", "").strip()  # 消息文本内容，去除首尾空白
                # 跳过空消息：某些轮次可能 content 为空字符串或 None，对训练无意义
                if not msg_content:
                    continue

                # 角色匹配逻辑：
                #   优先匹配 answer_role_prefix（精确到具体回答者 ID），
                #   兜底匹配 "answer_" 前缀（处理 ID 缺失或不一致的情况）
                if role_str.startswith(answer_role_prefix) or role_str.startswith("answer_"):
                    # 回答者（咨询师）的消息 → 映射为 assistant 角色
                    # 效果：微调时模型学习以咨询师身份生成回复
                    chat_turns.append({"role": "assistant", "content": msg_content})
                elif role_str.startswith("user_"):
                    # 提问者的追问消息 → 映射为 user 角色
                    # 效果：模型学会根据用户追问做出针对性的后续回复
                    chat_turns.append({"role": "user", "content": msg_content})
                    has_user_followup = True

            # 过滤无效对话：至少要有一轮 assistant 回复才算有效训练样本
            # 作用：排除纯用户消息的无效对话，避免微调时模型学不到任何回复能力
            # 效果：保证输出数据集中每条对话都包含至少一轮咨询师回复，提升训练数据质量
            if not any(t["role"] == "assistant" for t in chat_turns):
                continue

            # 统计包含用户追问的多轮对话数量（用于输出数据质量报告）
            if has_user_followup:
                multi_turn += 1

            # 在对话最前面插入 system 角色提示词
            # 设计意图：Qwen ChatML 格式要求 system 消息位于对话开头，
            #           用于在每次推理时设定模型的行为模式和回复风格
            # 作用：将 SYSTEM_PROMPT 作为对话的上下文前缀，指导模型以心理咨询师身份回复
            # 效果：微调后模型会在所有对话场景中保持一致的专业、共情风格
            conv_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_turns
            # 将完整对话（含 system + user + assistant 轮次）加入结果列表
            # conv_messages 结构：[system消息, user首轮提问, assistant回复, user追问, assistant回复, ...]
            conversations.append({"conversations": conv_messages})

    print(f"🔄 共生成 {len(conversations)} 条对话（跳过 {skipped} 条无回答记录，其中 {multi_turn} 条包含用户追问的多轮对话）")
    return conversations


def format_conversation(conv):
    """
    将对话消息列表转换为 Qwen ChatML 格式的纯文本字符串。
    
    【ChatML 格式说明】
    每条消息格式化为: <|im_start|>role\n内容<|im_end|>\n
    其中 role 为 system / user / assistant 之一。
    这种格式是 Qwen 系列模型预训练时使用的对话模板，
    微调时保持一致可以让模型正确识别角色边界，生成符合预期的回复。
    
    【示例输出】
    <|im_start|>system\n你是一位心理咨询师...<|im_end|>\n
    <|im_start|>user\n我最近很焦虑...<|im_end|>\n
    <|im_start|>assistant\n我能理解你的感受...<|im_end|>\n
    
    :param conv: dict — 包含 conversations 键的字典，值为消息列表，
                 每条消息为 {"role": str, "content": str} 格式
    :return: dict — 包含 text 键的字典，值为拼接后的 ChatML 格式文本，
             可直接作为训练数据的输入特征（input_ids 由此文本 tokenize 得到）
    """
    # 角色到 ChatML 标签的映射表
    # 作用：将标准 role 名称映射为 Qwen ChatML 中使用的标签名，
    #       确保拼接的文本符合 Qwen 模型预训练时使用的对话模板格式
    # 效果：微调时模型能正确识别角色边界，在推理时按预期生成 assistant 回复
    # 说明：当前 system/user/assistant 映射后名称不变，保留映射表便于后续扩展
    #       （如增加 observation、tool 等角色时只需在此表中添加一行）
    role_map = {
        "system": "system",        # 系统提示词，设定模型行为
        "user": "user",            # 用户输入（提问者消息）
        "assistant": "assistant"   # 助手回复（咨询师消息）
    }
    # 逐条拼接所有消息为 ChatML 格式文本
    text = ""
    for msg in conv["conversations"]:
        role = msg["role"]                          # 当前消息的角色标识
        tag = role_map.get(role, role)               # 查找 ChatML 标签，未知角色直接使用原始值（容错处理）
        # 拼接单条消息：<|im_start|> 标记消息开始，<|im_end|> 标记消息结束
        text += f"<|im_start|>{tag}\n{msg['content']}<|im_end|>\n"
    return {"text": text}


def main():
    """
    数据准备主流程：加载原始数据 → 构建多轮对话 → 转换 ChatML 格式 → 保存文件。
    
    执行流程：
    1. 从 INPUT_FILE 加载 jiandanxinli 心理咨询原始 JSON 数据
    2. 调用 build_conversations() 将原始数据转换为标准多轮对话结构
    3. 调用 format_conversation() 将每条对话转换为 Qwen ChatML 格式文本
    4. 以 JSONL 格式（每行一个 JSON 对象）写入 OUTPUT_FILE，便于后续训练脚本逐行读取
    
    输出文件格式示例（每行）：
    {"text": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n..."}
    """
    # ---- 第1步：加载原始数据 ----
    raw_data = load_raw_data(INPUT_FILE)

    # ---- 第2步：构建多轮对话结构 ----
    # 将原始 QA 数据按「一个回答者 → 一条对话」的规则拆分，
    # 并映射 role 为标准的 system/user/assistant
    conversations = build_conversations(raw_data)

    # 安全检查：如果数据转换后为空，提前终止并给出提示
    if not conversations:
        print("⚠️ 没有生成任何对话数据，请检查输入文件")
        return

    # ---- 第3步：创建输出目录 ----
    # os.path.dirname 提取目录路径，exist_ok=True 表示目录已存在时不报错
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ---- 第4步：转换格式并写入文件 ----
    # 列表推导式对每条对话批量调用 format_conversation()，生成 ChatML 格式文本
    formatted_data = [format_conversation(conv) for conv in conversations]
    # 以 JSONL 格式写入（每行一个 JSON 对象）
    # 作用：JSONL 格式便于训练脚本逐行流式读取，降低内存占用
    # 效果：即使数据集很大，训练时也不需要一次性全部加载到内存
    # encoding="utf-8"：确保中文字符正确写入，避免 Windows 默认 GBK 编码导致乱码
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in formatted_data:
            # json.dumps 参数说明：
            #   ensure_ascii=False —— 作用：允许直接输出中文字符而非 \uXXXX 转义
            #                        效果：生成的 JSONL 文件可读性更好，方便人工检查数据质量
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- 第5步：输出统计信息与示例 ----
    print(f"✅ 数据准备完成！共 {len(formatted_data)} 条对话")
    print(f"📁 保存路径: {OUTPUT_FILE}")

    # 打印第一条数据的前 300 个字符，用于人工检查格式是否正确
    # 截取 300 字符是因为完整对话可能很长，示例只需确认格式无误即可
    print("\n📝 数据示例:")
    print(formatted_data[0]["text"][:300] + "...")


# Python 入口点保护：
#   作用 —— 确保脚本仅在直接运行时执行 main()，被 import 时不会自动执行
#   效果 —— 其他脚本可以安全 import 本文件中的函数（如 load_raw_data）而不触发数据处理流程
if __name__ == "__main__":
    main()