"""
认证路由 — 注册 / 登录 / 获取当前用户

使用 JWT 无状态认证。
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from web.models import UserRegister, UserLogin, UserResponse, TokenResponse
from web.database import create_user, get_user_by_username, get_user_by_id
from web.services.auth import hash_password, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/api/auth", tags=["认证"])
security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """依赖注入：从 JWT 获取当前用户"""
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    user = get_user_by_id(int(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


@router.post("/register", response_model=TokenResponse)
async def register(data: UserRegister):
    """注册新用户"""
    # 检查用户名是否已存在
    existing = get_user_by_username(data.username)
    if existing:
        raise HTTPException(status_code=409, detail="用户名已存在")

    # 创建用户
    hashed = hash_password(data.password)
    user = create_user(data.username, hashed)
    if user is None:
        raise HTTPException(status_code=409, detail="用户名已存在")

    # 生成令牌
    token = create_access_token(user["id"], user["username"])

    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user["id"], username=user["username"], created_at=""),
    )


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """用户登录"""
    user = get_user_by_username(data.username)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_access_token(user["id"], user["username"])

    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user["id"], username=user["username"], created_at=user["created_at"]),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return UserResponse(id=user["id"], username=user["username"], created_at=user["created_at"])
