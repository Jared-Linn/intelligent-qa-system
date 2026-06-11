"""
多轮对话服务端 - FastAPI 实现

功能说明：
1. 加载 Qwen3.5-0.8B 模型 + LoRA 适配器
2. 提供多轮对话 API（自动记忆上下文，通过 session_id 区分不同用户）
3. 提供聊天界面（使用 Jinja2 模板渲染）
4. 支持多会话管理（内存字典存储历史）
5. 支持系统角色（System Prompt）设定，定义AI的角色和行为

设计原则：
- 先加载模型，再创建 FastAPI 应用
- 模型加载失败时不会启动服务
- 避免在应用启动后才加载模型导致的超时问题
"""

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from peft import PeftModel
from model_loader import model_loader
import os
import multiprocessing

# ==================== 配置参数 ====================
# 模型路径配置
BASE_MODEL_PATH = r"D:\dataset\model\Qwen3.5-0.8B"  # 基础预训练模型路径
LORA_PATH = "./outputs_multi_turn/lora_adapter"  # LoRA 微调适配器路径

# 文本生成参数配置
MAX_NEW_TOKENS = 256  # 最多生成的新 token 数，值越大回答越长但速度越慢
TEMPERATURE = 0.7  # 温度系数：越高回答越随机多样，越低越确定保守（范围 0.1-2.0）
TOP_P = 0.9  # 核采样：只从累计概率前90%的token中选择，平衡多样性和质量
REPETITION_PENALTY = 1.1  # 重复惩罚：大于1会惩罚重复出现的词，避免死循环（推荐1.05-1.2）

# 默认系统角色（System Prompt）
# 定义AI的身份、行为准则和回答风格
# 用户可以在请求中覆盖此设置
DEFAULT_SYSTEM_PROMPT = """你是一个友好、专业的AI助手，名叫小Q。
你的特点：
1. 回答简洁明了，易于理解
2. 对于不懂的问题，诚实地说"我不知道"
3. 保持积极、礼貌的语气
4. 回答问题时有条理，分点说明"""

# ==================== 第一步：加载模型 ====================
# 在启动 FastAPI 之前先加载模型，确保模型加载成功后再启动服务
print("=" * 50)
print("第一步：加载模型中...")
print("=" * 50)

# 1. 加载基础模型和分词器（通过自定义的 model_loader 单例）
#    model_loader 使用单例模式，确保模型只加载一次，避免重复加载浪费内存
base_model, tokenizer = model_loader.load_model(BASE_MODEL_PATH)

# 2. 加载 LoRA 适配器（微调得到的额外权重）
#    PeftModel 将 LoRA 权重附加到基础模型上，实现参数高效微调
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 3. 强制转换为 FP32 精度
#    避免 CPU 上出现 bf16/f16 精度不支持的错误（如 oneDNN 报错）
model = model.float()

print("✅ 模型加载完成！")
print("=" * 50)

# ==================== 第二步：创建 FastAPI 应用 ====================
# 模型加载成功后再创建 FastAPI 应用
app = FastAPI(title="多轮对话API")

# 配置 CORS（跨域资源共享）
# 允许前端页面（不同域名）调用此 API，解决浏览器跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带凭证（如 Cookie、Authorization 头）
    allow_methods=["*"],  # 允许所有 HTTP 方法（GET、POST、DELETE、PUT 等）
    allow_headers=["*"],  # 允许所有请求头
)

# ==================== 第三步：配置 Jinja2 模板 ====================
# 获取当前文件所在目录的绝对路径
# __file__ 是当前文件的完整路径，os.path.dirname 获取其所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 拼接模板目录路径（默认为当前目录下的 templates 文件夹）
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# 确保模板目录存在，如果不存在则自动创建
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# 创建 Jinja2 模板引擎实例
# Jinja2 是 Python 的模板引擎，用于将 HTML 模板与数据结合渲染
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ==================== 第四步：会话历史存储 ====================
# 使用内存字典存储所有会话的对话历史
# 数据结构: {
#     session_id: {
#         "system_prompt": "系统角色设定",
#         "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
#     }
# }
# - 键: session_id (字符串，客户端传入的会话标识)
# - 值: dict 包含系统提示词和对话历史列表
class SessionData:
    """会话数据结构"""

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT  # 优先使用传入的提示词，否则使用默认提示词
        self.history: List[dict] = []  # 初始化空对话历史列表


# 存储所有会话
sessions: Dict[str, SessionData] = {}


# ==================== 第五步：请求/响应数据模型 ====================
# 使用 Pydantic 定义请求体结构，FastAPI 会自动进行数据校验
class ChatRequest(BaseModel):
    """聊天请求的数据结构"""
    session_id: str = "default"  # 会话ID，相同ID共享对话历史，默认为 "default"
    question: str  # 用户问题（必填）
    system_prompt: Optional[str] = None  # 系统角色设定（可选，不传则使用会话已有的或默认值）


class ChatResponse(BaseModel):
    """聊天响应的数据结构"""
    answer: str  # 模型生成的回答
    session_id: str  # 返回会话ID，便于客户端确认当前会话


class SessionUpdateRequest(BaseModel):
    """更新会话配置的请求"""
    system_prompt: str  # 新的系统角色设定


class SessionInfoResponse(BaseModel):
    """会话信息响应"""
    session_id: str
    system_prompt: str
    history_length: int


# ==================== 第六步：辅助函数 ====================
def format_prompt(question: str, history: List[dict], system_prompt: str) -> str:
    """
    构造 Qwen 模型所需的对话 prompt（支持系统角色）

    Qwen 对话模板格式：
    <|im_start|>system\n系统角色设定<|im_end|>\n
    <|im_start|>user\n用户问题<|im_end|>\n
    <|im_start|>assistant\n助手回答<|im_end|>\n
    <|im_start|>user\n下一个问题<|im_end|>\n
    <|im_start|>assistant\n

    参数:
        question: 当前用户的问题
        history: 历史对话列表，每个元素包含 role 和 content
        system_prompt: 系统角色设定（定义AI的身份和行为）

    返回:
        格式化后的 prompt 字符串，可直接输入模型
    """
    prompt = ""

    # 添加系统角色设定（System Prompt）
    # 系统角色在对话的最开头，用于设定整个对话的基调
    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    # 遍历历史对话，按顺序添加
    for turn in history:
        if turn["role"] == "user":
            # 用户消息格式：<|im_start|>user\n内容<|im_end|>
            prompt += f"<|im_start|>user\n{turn['content']}<|im_end|>\n"
        else:
            # 助手消息格式：<|im_start|>assistant\n内容<|im_end|>
            prompt += f"<|im_start|>assistant\n{turn['content']}<|im_end|>\n"

    # 添加当前问题（最后一条用户消息）
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
    # 添加助手开始标记，模型会从这里开始生成回答
    prompt += f"<|im_start|>assistant\n"

    return prompt


def generate_answer(question: str, history: List[dict], system_prompt: str) -> str:
    """
    使用模型生成回答

    参数:
        question: 当前用户问题
        history: 对话历史（用于上下文记忆）
        system_prompt: 系统角色设定

    返回:
        模型生成的回答文本
    """
    # 1. 构造完整的 prompt（包含系统角色 + 历史对话 + 当前问题）
    prompt = format_prompt(question, history, system_prompt)

    # 2. 将文本编码为模型可理解的 token ID 序列
    #    return_tensors="pt" 表示返回 PyTorch 张量格式
    inputs = tokenizer(prompt, return_tensors="pt")

    # 3. 模型生成回答
    #    torch.no_grad() 禁用梯度计算，推理时不需要反向传播，可节省内存
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],  # 输入的 token ID 序列
            max_new_tokens=MAX_NEW_TOKENS,  # 最多生成的新 token 数
            temperature=TEMPERATURE,  # 温度系数（控制随机性）
            top_p=TOP_P,  # 核采样概率阈值
            repetition_penalty=REPETITION_PENALTY,  # 重复惩罚系数
            eos_token_id=tokenizer.eos_token_id,  # 结束标记 ID，遇到则停止生成
            pad_token_id=tokenizer.pad_token_id,  # 填充标记 ID，用于对齐长度
        )

    # 4. 将生成的 token ID 序列解码回文本
    #    skip_special_tokens=False 保留特殊标记（如 <|im_start|>），便于后续提取
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 5. 从完整响应中提取 assistant 的回答部分
    #    响应格式: "...<|im_start|>assistant\n实际回答内容<|im_end|>..."
    if "<|im_start|>assistant\n" in response:
        # 按 assistant 标记分割，取最后一部分（即助手回答）
        answer = response.split("<|im_start|>assistant\n")[-1]
        # 去掉结束标记
        answer = answer.split("<|im_end|>")[0].strip()
    else:
        # 容错处理：如果没有找到标记，直接返回整个响应
        answer = response.strip()

    return answer


def get_or_create_session(session_id: str, system_prompt: Optional[str] = None) -> SessionData:
    """
    获取或创建会话

    参数:
        session_id: 会话标识
        system_prompt: 系统角色设定（仅在创建会话时生效）

    返回:
        SessionData: 会话数据对象
    """
    if session_id not in sessions:
        # 创建新会话，使用传入的 system_prompt 或默认值
        sessions[session_id] = SessionData(system_prompt=system_prompt)
    elif system_prompt is not None:
        # 如果会话已存在但传入了新的 system_prompt，更新它
        sessions[session_id].system_prompt = system_prompt

    return sessions[session_id]


# ==================== 第七步：页面路由 ====================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    根路由：返回聊天界面

    使用 Jinja2 模板渲染 index.html
    request 参数必须传入，模板中可以使用 {{ request }}

    参数:
        request: FastAPI 请求对象，包含请求信息

    返回:
        渲染后的 HTML 页面
    """
    return templates.TemplateResponse(request, "index.html")


# ==================== 第八步：API 路由 ====================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    多轮对话接口

    请求方式: POST
    请求路径: /api/chat
    请求体格式: {
        "session_id": "用户标识",
        "question": "用户问题",
        "system_prompt": "系统角色设定（可选）"
    }

    功能：
    1. 根据 session_id 获取或创建对话历史
    2. 支持自定义系统角色（System Prompt）
    3. 使用模型生成回答
    4. 将本次对话保存到历史中
    5. 返回模型回答

    参数:
        request: 聊天请求对象（自动从 JSON 解析）

    返回:
        ChatResponse: 包含模型回答和会话ID
    """
    session_id = request.session_id

    # 获取或创建会话（如果提供了 system_prompt，新会话会使用它）
    session = get_or_create_session(session_id, request.system_prompt)
    history = session.history
    system_prompt = session.system_prompt

    # 使用模型生成回答
    answer = generate_answer(request.question, history, system_prompt)

    # 保存本次对话到历史
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})

    # 限制历史长度，避免超出模型上下文限制
    # 保留最近 20 条消息（即 10 轮对话），超出则删除最早的记录
    if len(history) > 20:
        session.history = history[-20:]

    return ChatResponse(answer=answer, session_id=session_id)


@app.put("/api/session/{session_id}/system")
async def update_system_prompt(session_id: str, request: SessionUpdateRequest):
    """
    更新会话的系统角色设定

    请求方式: PUT
    请求路径: /api/session/{session_id}/system
    请求体格式: {"system_prompt": "新的系统角色设定"}

    功能：修改指定会话的 System Prompt，后续对话将使用新的设定

    参数:
        session_id: 会话标识
        request: 包含新 system_prompt 的请求体

    返回:
        操作结果消息
    """
    session = get_or_create_session(session_id)
    session.system_prompt = request.system_prompt
    return {
        "message": f"会话 {session_id} 系统角色已更新",
        "system_prompt": session.system_prompt
    }


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """
    获取会话信息

    请求方式: GET
    请求路径: /api/session/{session_id}

    功能：查看会话的当前设置和历史记录数量

    参数:
        session_id: 会话标识

    返回:
        会话信息（系统角色、历史记录数等）
    """
    if session_id not in sessions:
        return {"message": f"会话 {session_id} 不存在", "exists": False}

    session = sessions[session_id]
    return {
        "session_id": session_id,
        "exists": True,
        "system_prompt": session.system_prompt,
        "history_length": len(session.history)
    }


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """
    清空指定会话的历史记录

    请求方式: DELETE
    请求路径: /api/history/{session_id}

    功能：删除该会话的所有对话历史，重新开始
    注意：系统角色设定会被保留

    参数:
        session_id: 会话标识

    返回:
        操作结果消息
    """
    if session_id in sessions:
        sessions[session_id].history = []  # 清空历史，保留 system_prompt
        return {"message": f"会话 {session_id} 历史已清空"}
    return {"message": f"会话 {session_id} 不存在"}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """
    删除整个会话

    请求方式: DELETE
    请求路径: /api/session/{session_id}

    功能：完全删除会话（包括系统角色和历史记录）

    参数:
        session_id: 会话标识

    返回:
        操作结果消息
    """
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"会话 {session_id} 已删除"}
    return {"message": f"会话 {session_id} 不存在"}


@app.get("/api/health")
async def health():
    """
    健康检查接口

    请求方式: GET
    请求路径: /api/health

    用途：监控服务状态，可用于容器健康检查

    返回:
        服务状态和模型加载情况
    """
    return {
        "status": "ok",  # 服务状态
        "model_loaded": model is not None  # 模型是否已加载
    }


@app.get("/api/sessions")
async def list_sessions():
    """
    列出所有活跃会话（调试接口）

    请求方式: GET
    请求路径: /api/sessions

    返回:
        所有会话ID列表
    """
    return {
        "sessions": list(sessions.keys()),
        "count": len(sessions)
    }


# ==================== 第九步：启动入口 ====================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🚀 启动多轮对话服务")
    print("📱 聊天界面: http://localhost:8000")
    print("📚 API 文档: http://localhost:8000/docs")
    print("=" * 50)
    print("\n📌 系统角色功能说明：")
    print("  - 每个会话可以单独设置 System Prompt")
    print("  - 默认系统角色：小Q（友好专业的AI助手）")
    print("  - 可通过 PUT /api/session/{id}/system 更新")
    print("=" * 50 + "\n")

    # 设置多进程启动方式为 spawn（Windows 兼容）
    # 解决 Windows 下 Event Loop 关闭的报错问题
    # spawn: 子进程启动时重新导入模块，避免 fork 导致的问题
    multiprocessing.set_start_method("spawn", force=True)

    # 启动 Uvicorn 服务器
    uvicorn.run(
        "multi_turn_app:app",  # 应用路径（模块名:应用实例名）
        host="0.0.0.0",  # 监听所有网络接口，允许外部访问
        port=8000,  # 监听端口
        reload=False,  # 是否热重载（生产环境设为 False）
        workers=1  # worker 进程数（模型只加载一次，必须为1）
    )