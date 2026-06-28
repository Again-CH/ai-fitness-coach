import os as _os
from dotenv import load_dotenv
_env_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=True)

import json
import asyncio
import os
import sys
import jwt
import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from database import init_db, save_profile, get_profile, save_chat_message, get_user_by_id
from langgraph_brain import fitness_graph
from config import LLM_ENABLED, DEEPSEEK_MODEL, TENCENT_ENABLED, TENCENT_TTS_VOICE_TYPE
from safety_filter import check_user_input, SAFETY_BLOCK_MESSAGE

# ★ AI Agent 架构升级：引入 ReAct 控制循环（渐进式迁移，新旧并行）
from react_loop import react_graph
from memory.memory_manager import AgentMemoryManager

# ★ 引入点外卖意图检测（预检用）
from food_order_guard import detect_food_order_intent, classify_intent


# ============================================================
# JWT 配置
# ============================================================
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ai-fitness-coach-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # Token 有效期 7 天


def create_access_token(user_id: str, phone: str) -> str:
    """
    创建 JWT Access Token。
    
    参数:
        user_id: 用户 ID
        phone: 用户手机号
    
    返回:
        JWT Token 字符串
    """
    payload = {
        "user_id": user_id,
        "phone": phone,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def verify_token(token: str) -> Dict[str, Any]:
    """
    验证 JWT Token。
    
    参数:
        token: JWT Token 字符串
    
    返回:
        成功: {"user_id": 用户ID, "phone": 手机号}
        失败: 抛出 HTTPException
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {"user_id": payload["user_id"], "phone": payload["phone"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token 无效，请重新登录")


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    FastAPI 依赖项：从 Authorization Header 中提取并验证当前用户。
    
    使用方式：
        @app.get("/api/some_endpoint")
        async def some_endpoint(current_user: dict = Depends(get_current_user)):
            user_id = current_user["user_id"]
            ...
    
    返回:
        成功: {"user_id": 用户ID, "email": 邮箱}
        失败: 抛出 HTTPException (401)
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证 Token")
    
    # 解析 "Bearer <token>" 格式
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="认证格式错误，应为 'Bearer <token>'")
    
    token = parts[1]
    return verify_token(token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化数据库，打印当前 AI 模式"""
    print("[FastAPI] 应用启动中...")
    init_db()
    if LLM_ENABLED:
        print(f"[FastAPI] AI 模型: ✅ DeepSeek {DEEPSEEK_MODEL}（OpenAI 兼容接口）")
        print(f"[FastAPI] Agent 架构: ✅ ReAct 控制循环 + bind_tools 双模式")
        print(f"[FastAPI] 记忆系统: ✅ 三层架构（短期/结构化/摘要）")
    else:
        print(f"[FastAPI] AI 模型: 🔧 模拟模式（未配置 DEEPSEEK_API_KEY）")
        print(f"           请在 backend/.env 文件中设置 DEEPSEEK_API_KEY=sk-xxxxxxxx")

    # ★ 第十一步：多模态能力状态
    if TENCENT_ENABLED:
        print(f"[FastAPI] 语音能力: ✅ 腾讯云 ASR/TTS 已配置（音色: {TENCENT_TTS_VOICE_TYPE}）")
    else:
        print(f"[FastAPI] 语音能力: 🔧 未配置腾讯云密钥（请在 .env 中设置 TENCENT_SECRET_ID/KEY）")
    print(f"[FastAPI] 视觉能力: ✅ Qwen-VL 图像理解（通过 DashScope）")
    print("[FastAPI] 应用启动完成！")
    yield
    print("[FastAPI] 应用关闭")


app = FastAPI(
    title="AI 健身教练 API",
    description="基于 FastAPI + SQLite + LangGraph + DeepSeek 的 AI 健身教练后端（支持语音+图片多模态 + ReAct Agent）",
    version="4.0.0",
    lifespan=lifespan,
)

# CORS 跨域配置（允许前端从任意来源访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. 请求/响应数据模型（Pydantic）
# ============================================================
class RegisterRequest(BaseModel):
    """用户注册请求（发送验证码）"""
    phone: str


class LoginRequest(BaseModel):
    """用户登录请求（验证验证码）"""
    phone: str
    code: str


class ProfileRequest(BaseModel):
    """身体数据保存请求"""
    # user_id 不再从请求体中获取，而是从 JWT Token 中解析
    height: float
    weight: float
    age: int
    gender: str


class ChatRequest(BaseModel):
    """对话请求（★ 第十一步：新增 image 字段支持图片上传）
    
    ★ 新增字段：
        resume: 用于 LangGraph Human-in-the-loop 恢复。
                - "confirm": 用户确认了点外卖请求
                - "cancel": 用户取消了外卖请求
                - None: 正常对话
        mode: AI 模式（★ AI Agent 架构升级）
                - None/"default": 使用传统 bind_tools 模式
                - "react": 使用 ReAct 控制循环（新一代）
    """
    # user_id 不再从请求体中获取，而是从 JWT Token 中解析
    message: str
    image: Optional[str] = None  # Base64 编码的图片数据（可选）
    resume: Optional[str] = None  # LangGraph interrupt 恢复值（可选）
    mode: Optional[str] = None    # AI 模式："default" / "react"


class TTSRequest(BaseModel):
    """语音合成请求"""
    text: str
    voice_type: Optional[int] = None  # 音色 ID（可选，默认使用配置中的值）


# ============================================================
# 4. 健康检查接口
# ============================================================
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "llm_enabled": LLM_ENABLED,
        "llm_model": DEEPSEEK_MODEL if LLM_ENABLED else None,
        "tencent_enabled": TENCENT_ENABLED,
        "tts_voice_type": TENCENT_TTS_VOICE_TYPE,
        "agent_mode": "react + bind_tools",  # ★ AI Agent 架构升级
    }


# ============================================================
# 5. 用户鉴权接口
# ============================================================
@app.post("/api/auth/send_code")
async def api_send_code(req: RegisterRequest):
    """
    发送验证码接口（手机号登录/注册）。
    
    请求体：
        {"phone": "13800138000"}
    
    返回：
        {"success": True, "message": "验证码已发送"}
        或
        {"success": False, "error": "发送失败"}
    
    注意：
        - 生产环境应接入真实短信服务（阿里云、腾讯云短信等）
        - 当前为模拟模式：验证码打印到控制台，并返回给前端（仅用于开发）
    """
    from database import send_sms_code
    result = send_sms_code(req.phone)
    if result["success"]:
        # ★ 开发模式：返回验证码给前端（方便测试）
        # 生产环境应删除 code 字段
        return {
            "success": True,
            "message": "验证码已发送到手机号（请查看控制台）",
            "code": result["code"]  # ★ 仅开发环境
        }
    else:
        return JSONResponse(status_code=400, content=result)


@app.post("/api/auth/login")
async def api_login(req: LoginRequest):
    """
    用户登录接口（验证验证码）。
    
    请求体：
        {"phone": "13800138000", "code": "123456"}
    
    返回：
        {"success": True, "token": "jwt_token", "user_id": "1", "phone": "13800138000"}
        或
        {"success": False, "error": "验证码错误"}
    """
    from database import verify_sms_code
    result = verify_sms_code(req.phone, req.code)
    if not result["success"]:
        return JSONResponse(status_code=401, content=result)
    
    # 生成 JWT Token
    user_id = result["user_id"]
    phone = result["phone"]
    token = create_access_token(user_id, phone)
    
    print(f"[Main] 用户登录成功: id={user_id}, phone={phone}")
    return {
        "success": True,
        "token": token,
        "user_id": user_id,
        "phone": phone,
    }


@app.get("/api/auth/me")
async def api_get_current_user(current_user: dict = Depends(get_current_user)):
    """
    获取当前登录用户信息接口（需要鉴权）。
    
    Header：
        Authorization: Bearer <token>
    
    返回：
        {"success": True, "user_id": "1", "phone": "13800138000"}
    """
    return {
        "success": True,
        "user_id": current_user["user_id"],
        "phone": current_user["phone"],
    }


# ============================================================
# 6. POST /api/save_profile —— 保存用户身体数据（需要鉴权）
# ============================================================
@app.post("/api/save_profile")
async def api_save_profile(req: ProfileRequest, current_user: dict = Depends(get_current_user)):
    """保存用户身体数据到 SQLite（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    try:
        save_profile(
            user_id=user_id,
            height=req.height,
            weight=req.weight,
            age=req.age,
            gender=req.gender,
        )
        return {
            "success": True,
            "message": "身体数据保存成功",
            "user_id": user_id,
        }
    except Exception as e:
        return {"success": False, "message": f"保存失败: {str(e)}"}


# ============================================================
# 7. POST /api/chat/stream —— SSE 流式对话接口（需要鉴权，★ 核心）
# ============================================================
@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    SSE 流式对话接口（打字机效果 + 工具调用 + Human-in-the-loop 确认）

    ★ 新增：LangGraph interrupt() 支持
    - 当 AI 检测到点外卖意图时，通过 interrupt() 暂停对话
    - 前端收到 confirmation_required 事件后弹出确认弹窗
    - 用户确认/取消后，通过 resume 字段恢复对话

    记忆系统：通过 config 实现对话线程记忆
    """
    user_id = current_user["user_id"]
    
    async def sse_event_generator():
        config = {"configurable": {"thread_id": user_id}}

        # ========================================================
        # ★ 路径 A：Resume（用户已确认/取消外卖）
        # ========================================================
        if req.resume:
            print(f"[Main] 🔄 恢复对话: resume={req.resume}, user={user_id}", file=sys.stderr, flush=True)
            
            # 发送恢复状态
            thinking_data = json.dumps(
                {"thinking": "正在处理..."}, ensure_ascii=False
            )
            yield f"data: {thinking_data}\n\n"
            
            try:
                # 使用 Command(resume=...) 恢复中断的图
                result = await fitness_graph.ainvoke(
                    Command(resume=req.resume), config
                )
                
                # 提取 AI 回复
                messages = result.get("messages", [])
                full_response = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                        full_response = msg.content
                        break
                
                if not full_response:
                    full_response = "好的，明白了！有什么健身相关的问题可以继续问我。💪"
                
                # 流式输出
                for char in full_response:
                    token_data = json.dumps({"token": char}, ensure_ascii=False)
                    yield f"data: {token_data}\n\n"
                    await asyncio.sleep(0.02)
                
                # 保存回复
                try:
                    save_chat_message(user_id, "assistant", full_response)
                except Exception as e:
                    print(f"[Main] 保存 AI 回复失败: {e}")
                
                # ★ 发送美团跳转链接（用户确认后自动推送）
                meituan_url = "https://www.meituan.com/"
                meituan_event = json.dumps({
                    "meituan_url": meituan_url,
                    "label": "去美团看看",
                }, ensure_ascii=False)
                yield f"data: {meituan_event}\n\n"
                    
            except Exception as e:
                error_msg = f"抱歉，处理请求时出现问题: {str(e)}"
                token_data = json.dumps({"token": error_msg}, ensure_ascii=False)
                yield f"data: {token_data}\n\n"
            
            yield "data: [DONE]\n\n"
            return

        # ========================================================
        # ★ 路径 B：正常对话流程
        # ========================================================
        # ---- 图片处理 ----
        image_path = None
        user_message = req.message
        
        if req.image:
            from multimodal_service import save_base64_image
            try:
                image_path = save_base64_image(req.image, prefix="chat_upload")
                if not user_message.strip():
                    user_message = "请帮我分析这张图片。"
                print(f"[Main] 用户上传了图片: {image_path}")
            except Exception as e:
                print(f"[Main] 图片保存失败: {e}")
                error_data = json.dumps({"token": f"图片处理失败: {str(e)}"}, ensure_ascii=False)
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
                return
        
        graph_input = {
            "user_id": user_id,
            "messages": [HumanMessage(content=user_message)],
            "user_profile": None,
            "system_prompt": "",
            "image_path": image_path,
        }
        
        # 保存用户消息
        try:
            save_chat_message(user_id, "user", user_message)
        except Exception as e:
            print(f"[Main] 保存用户消息失败: {e}")
        
        # ---- 安全熔断预检（代码级硬拦截） ----
        safety_result = check_user_input(user_message)
        if safety_result["level"] == "block":
            trigger = safety_result.get("trigger", "未知")
            keyword = safety_result.get("keyword", "未知")
            print(f"[Main] 🔴 安全熔断：类型={trigger}, 关键词={keyword}", file=sys.stderr, flush=True)
            safety_msg = safety_result["message"]
            
            for char in safety_msg:
                token_data = json.dumps({"token": char}, ensure_ascii=False)
                yield f"data: {token_data}\n\n"
                await asyncio.sleep(0.02)
            
            try:
                save_chat_message(user_id, "assistant", safety_msg)
            except Exception as e:
                print(f"[Main] 保存安全提示失败: {e}", file=sys.stderr, flush=True)
            
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            
            yield "data: [DONE]\n\n"
            return

        # ---- ★ 意图分类预检：统一 Human-in-the-loop ----
        # classify_intent: "我饿了" / "我想点外卖" → action="interrupt" → 确认 → 调美团API
        intent_result = classify_intent(user_message)
        print(f"[Main] 意图分类: {intent_result}", file=sys.stderr, flush=True)

        if intent_result["action"] == "interrupt":
            order_level = intent_result["level"]
            print(f"[Main] 🛑 检测到外卖意图（{order_level}），进入 Human-in-the-loop 流程", file=sys.stderr, flush=True)
            
            # ★ 关键修复：用后台任务 + 轮询 代替 ainvoke 直接等待
            # 问题：LangGraph 的 interrupt() 在某些版本中会导致 ainvoke 无限挂起
            # 解决：在后台启动图执行，轮询 get_state 检测中断，不取消后台任务
            loop_task = asyncio.ensure_future(
                fitness_graph.ainvoke(graph_input, config)
            )
            
            interrupt_data = None
            for i in range(15):  # 15 × 0.2s = 3s 总等待
                await asyncio.sleep(0.2)
                try:
                    state = fitness_graph.get_state(config)
                    if state and hasattr(state, "interrupts") and state.interrupts:
                        for intr in state.interrupts:
                            if hasattr(intr, "value"):
                                interrupt_data = intr.value
                                break
                        if interrupt_data:
                            print(f"[Main] 第 {i+1} 次轮询发现 interrupt", file=sys.stderr, flush=True)
                            break
                except Exception as e:
                    print(f"[Main] get_state 轮询异常: {type(e).__name__}", file=sys.stderr, flush=True)
            
            if interrupt_data:
                # 成功获取中断数据，发送确认事件
                print(f"[Main] 📤 发送 confirmation_required 事件: {json.dumps(interrupt_data, ensure_ascii=False)[:200]}", file=sys.stderr, flush=True)
                confirmation_event = json.dumps(
                    {"confirmation_required": interrupt_data},
                    ensure_ascii=False,
                )
                yield f"data: {confirmation_event}\n\n"
                yield "data: [DONE]\n\n"
                # ★ 不取消 loop_task —— 图已在 interrupt 点暂停，等待 resume
                return
            
            # 回退：轮询超时未检测到中断，发送通用确认
            print(f"[Main] ⚠️ 轮询超时未检测到中断数据，发送通用确认（回退路径）", file=sys.stderr, flush=True)
            # 此时 ainvoke 可能已完成（无中断），或仍在运行——尝试取消
            if not loop_task.done():
                loop_task.cancel()
            from food_order_guard import get_confirmation_message
            fallback_msg = get_confirmation_message(order_level, user_message)
            confirmation_event = json.dumps({
                "confirmation_required": {
                    "type": "food_order_confirmation",
                    "intent_level": order_level,
                    "message": fallback_msg,
                    "user_input": user_message,
                }
            }, ensure_ascii=False)
            yield f"data: {confirmation_event}\n\n"
            yield "data: [DONE]\n\n"
            return

        if LLM_ENABLED:
            # ---- ★ ReAct 模式（AI Agent 架构升级） ----
            if req.mode == "react":
                async for event in _react_sse_generator(
                    graph_input, config, user_id, image_path, user_message
                ):
                    yield event
                # 注意：_react_sse_generator 内部已经 yield 了 [DONE]
                return

            # ---- LLM 模式（传统 bind_tools）：流式调用 LangGraph ----
            thinking_data = json.dumps(
                {"thinking": "正在分析图片内容..." if image_path else "正在分析您的问题..."},
                ensure_ascii=False,
            )
            yield f"data: {thinking_data}\n\n"
            
            has_content = False
            full_response = ""
            
            try:
                async for event in fitness_graph.astream_events(
                    graph_input, config=config, version="v2"
                ):
                    kind = event["event"]
                    if kind == "on_tool_start":
                        tool_name = event.get("name", "")
                        if "analyze_image" in tool_name:
                            thinking_data = json.dumps({"thinking": "正在用视觉模型分析图片..."}, ensure_ascii=False)
                        elif "search" in tool_name:
                            thinking_data = json.dumps({"thinking": "正在检索健身知识库..."}, ensure_ascii=False)
                        else:
                            thinking_data = json.dumps({"thinking": "正在为您进行科学计算..."}, ensure_ascii=False)
                        yield f"data: {thinking_data}\n\n"
                    elif kind == "on_tool_end":
                        thinking_data = json.dumps({"thinking": "分析完成，正在生成回复..."}, ensure_ascii=False)
                        yield f"data: {thinking_data}\n\n"
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            has_content = True
                            full_response += chunk.content
                            token_data = json.dumps({"token": chunk.content}, ensure_ascii=False)
                            yield f"data: {token_data}\n\n"
            
            except Exception as e:
                error_msg = f"抱歉，AI 服务出现了问题: {str(e)}"
                full_response = error_msg
                token_data = json.dumps({"token": error_msg}, ensure_ascii=False)
                yield f"data: {token_data}\n\n"
            
            if not has_content:
                full_response = "抱歉，处理遇到问题，请重试。"
                token_data = json.dumps({"token": full_response}, ensure_ascii=False)
                yield f"data: {token_data}\n\n"
            
            # 保存 AI 回复
            if full_response:
                try:
                    save_chat_message(user_id, "assistant", full_response)
                except Exception as e:
                    print(f"[Main] 保存 AI 回复失败: {e}", file=sys.stderr, flush=True)
            
            # 清理临时图片
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            
            yield "data: [DONE]\n\n"
        
        else:
            # ---- 模拟模式 ----
            # 模拟模式也支持中断检测
            try:
                result = await fitness_graph.ainvoke(graph_input, config)
                
                # 检查中断
                state = fitness_graph.get_state(config)
                if state and hasattr(state, "interrupts") and state.interrupts:
                    interrupt_data = None
                    for intr in state.interrupts:
                        if hasattr(intr, "value"):
                            interrupt_data = intr.value
                            break
                    
                    if interrupt_data:
                        confirmation_event = json.dumps(
                            {"confirmation_required": interrupt_data},
                            ensure_ascii=False,
                        )
                        yield f"data: {confirmation_event}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                
                ai_message = result["messages"][-1]
                full_response = ai_message.content
                
                try:
                    save_chat_message(user_id, "assistant", full_response)
                except Exception as e:
                    print(f"[Main] 保存 AI 回复失败: {e}")
                
                for char in full_response:
                    token_data = json.dumps({"token": char}, ensure_ascii=False)
                    yield f"data: {token_data}\n\n"
                    await asyncio.sleep(0.02)
                
            except Exception as e:
                error_str = str(e)
                if "interrupt" in error_str.lower() or "GraphInterrupt" in error_str:
                    # 模拟模式中断
                    try:
                        state = fitness_graph.get_state(config)
                        interrupt_data = None
                        if state and hasattr(state, "interrupts") and state.interrupts:
                            for intr in state.interrupts:
                                if hasattr(intr, "value"):
                                    interrupt_data = intr.value
                                    break
                        if interrupt_data:
                            confirmation_event = json.dumps(
                                {"confirmation_required": interrupt_data},
                                ensure_ascii=False,
                            )
                            yield f"data: {confirmation_event}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    except:
                        pass
            
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ============================================================
# 8. GET /api/profile —— 查询用户身体数据（需要鉴权）
# ============================================================
@app.get("/api/profile")
async def api_get_profile(current_user: dict = Depends(get_current_user)):
    """根据 JWT Token 中的 user_id 查询用户身体数据"""
    user_id = current_user["user_id"]
    profile = get_profile(user_id)
    if profile:
        return {"success": True, "data": profile}
    else:
        return {"success": False, "message": "未找到该用户的身体数据"}


# ============================================================
# 9. 记忆管理接口（第五步：多轮对话与记忆系统，需要鉴权）
# ============================================================
@app.get("/api/chat/memory")
async def api_get_memory(current_user: dict = Depends(get_current_user)):
    """获取指定用户的对话历史（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from langgraph_brain import get_memory_messages
    messages = get_memory_messages(user_id)
    return {"success": True, "messages": messages}


@app.delete("/api/chat/memory")
async def api_clear_memory(current_user: dict = Depends(get_current_user)):
    """清空指定用户的对话记忆（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from langgraph_brain import clear_memory
    clear_memory(user_id)
    return {"success": True, "message": "对话记忆已清空"}


# ============================================================
# ★ 10. POST /api/asr —— 语音识别接口（需要鉴权，第十一步：多模态）
# ============================================================
@app.post("/api/asr")
async def api_asr(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """
    语音识别接口 —— 调用腾讯云 ASR 将语音转为文字。

    前端使用 MediaRecorder API 录音，将音频文件上传到此接口。
    后端调用腾讯云一句话识别 (SentenceRecognition) API 返回文字。

    请求格式：multipart/form-data
        file: 音频文件（wav / mp3 / m4a / webm）

    返回：
        {"success": True, "text": "识别出的文字"}
        或
        {"success": False, "error": "错误描述"}
    """
    from multimodal_service import speech_to_text
    
    # 读取上传的音频文件
    audio_data = await file.read()
    
    # 从文件名推断音频格式
    filename = file.filename or "audio.wav"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
    
    print(f"[Main] ASR 请求：文件名={filename}, 格式={ext}, 大小={len(audio_data)} bytes")
    
    # 调用腾讯云 ASR
    result = speech_to_text(audio_data, audio_format=ext)
    return JSONResponse(content=result)


# ============================================================
# ★ 11. POST /api/tts —— 语音合成接口（需要鉴权，第十一步：多模态）
# ============================================================
@app.post("/api/tts")
async def api_tts(req: TTSRequest, current_user: dict = Depends(get_current_user)):
    """
    语音合成接口 —— 调用腾讯云 TTS 将文字转为语音。

    前端在 AI 回复完成后，将文字发送到此接口，获取 MP3 音频数据播放。
    默认使用活力男声（晓浩 101014），适合健身教练场景。

    请求格式：application/json
        {"text": "要合成的文字", "voice_type": 101014}

    返回：
        {"success": True, "audio": "base64编码的MP3", "format": "mp3"}
        或
        {"success": False, "error": "错误描述"}

    ★ 注意：单次最多合成 150 个汉字，超出会自动截断。
    """
    from multimodal_service import text_to_speech
    
    if not req.text.strip():
        return JSONResponse(content={"success": False, "error": "文本不能为空"})
    
    print(f"[Main] TTS 请求：文本长度={len(req.text)}, 音色={req.voice_type or TENCENT_TTS_VOICE_TYPE}")
    
    # 调用腾讯云 TTS
    result = text_to_speech(req.text, voice_type=req.voice_type)
    return JSONResponse(content=result)


# ============================================================
# 12. 前端页面 —— 由 FastAPI 直接提供
# ============================================================
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_index():
    """根路径返回前端页面 index.html（禁用缓存，始终获取最新版本）"""
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Expires": "0"}
    )


# ============================================================
# ★ 数据看板 API 接口（需要鉴权）
# ============================================================

# ---- 获取体重趋势 ----
@app.get("/api/dashboard/weight")
async def api_get_weight_logs(days: int = 30, current_user: dict = Depends(get_current_user)):
    """获取最近 N 天的体重记录（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import get_weight_logs
    data = get_weight_logs(user_id, days)
    return {"success": True, "data": data}


# ---- 保存体重记录 ----
@app.post("/api/dashboard/weight")
async def api_save_weight_log(
    weight: float = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """保存一条体重记录（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import save_weight_log
    save_weight_log(user_id, weight)
    return {"success": True, "message": "体重记录已保存"}


# ---- 获取运动统计 ----
@app.get("/api/dashboard/exercise")
async def api_get_exercise_logs(days: int = 7, current_user: dict = Depends(get_current_user)):
    """获取最近 N 天的运动记录（按日期聚合，从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import get_exercise_logs
    data = get_exercise_logs(user_id, days)
    return {"success": True, "data": data}


# ---- 保存运动记录 ----
@app.post("/api/dashboard/exercise")
async def api_save_exercise_log(
    exercise_type: str = Form(...),
    duration: int = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """保存一条运动记录（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import save_exercise_log
    save_exercise_log(user_id, exercise_type, duration)
    return {"success": True, "message": "运动记录已保存"}


# ---- 获取今日饮食记录 ----
@app.get("/api/dashboard/diet")
async def api_get_diet_logs(date: str = None, current_user: dict = Depends(get_current_user)):
    """获取某天的饮食记录（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import get_diet_logs
    result = get_diet_logs(user_id, date)
    return {"success": True, "data": result}


# ---- 保存饮食记录 ----
@app.post("/api/dashboard/diet")
async def api_save_diet_log(
    food_name: str = Form(...),
    calories: float = Form(...),
    protein: float = Form(0),
    carb: float = Form(0),
    fat: float = Form(0),
    current_user: dict = Depends(get_current_user)
):
    """保存一条饮食记录（含营养成分，从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import save_diet_log
    save_diet_log(user_id, food_name, calories, protein, carb, fat)
    return {"success": True, "message": "饮食记录已保存"}


# ---- 获取目标设定 ----
@app.get("/api/dashboard/goal")
async def api_get_goal(current_user: dict = Depends(get_current_user)):
    """获取用户的目标设定（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import get_goal_settings
    data = get_goal_settings(user_id)
    return {"success": True, "data": data}


# ---- 保存目标设定 ----
@app.post("/api/dashboard/goal")
async def api_save_goal(
    calorie_target: int = Form(...),
    exercise_target: int = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """保存用户的目标设定（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import save_goal_settings
    save_goal_settings(user_id, calorie_target, exercise_target)
    return {"success": True, "message": "目标设定已保存"}


# ---- 获取今日数据汇总（供看板首页使用） ----
@app.get("/api/dashboard/summary")
async def api_get_summary(current_user: dict = Depends(get_current_user)):
    """获取今日数据汇总：体重、今日摄入热量、本周运动时长、BMI（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import (
        get_profile, get_weight_logs, get_diet_logs, get_exercise_logs, get_goal_settings
    )
    import datetime
    
    profile = get_profile(user_id)
    today = datetime.date.today().isoformat()
    diet_today = get_diet_logs(user_id, today)
    exercise_week = get_exercise_logs(user_id, 7)
    weight_logs = get_weight_logs(user_id, 7)
    goal = get_goal_settings(user_id)
    
    # 计算今日运动时长
    today_exercise = 0
    for item in exercise_week:
        if item["date"] == today:
            today_exercise = item["duration"]
            break
    
    # 计算 BMI
    bmi = None
    if profile:
        height_m = profile["height"] / 100
        bmi = profile["weight"] / (height_m ** 2)
    
    # 计算健康状态
    health_status = "未知"
    if bmi:
        if bmi < 18.5:
            health_status = "偏瘦"
        elif bmi < 24:
            health_status = "正常"
        elif bmi < 28:
            health_status = "偏胖"
        else:
            health_status = "肥胖"
    
    # 获取打卡数据
    from database import get_checkin_streak, get_achievements
    streak = get_checkin_streak(user_id)
    achievements = get_achievements(user_id)
    
    return {
        "success": True,
        "data": {
            "bmi": round(bmi, 1) if bmi else None,
            "health_status": health_status,
            "today_calories": diet_today.get("total_calories", 0),
            "today_protein": diet_today.get("total_protein", 0),
            "today_carb": diet_today.get("total_carb", 0),
            "today_fat": diet_today.get("total_fat", 0),
            "calorie_target": goal.get("calorie_target", 2000),
            "protein_target": goal.get("protein_target", 120),
            "carb_target": goal.get("carb_target", 300),
            "fat_target": goal.get("fat_target", 65),
            "today_exercise": today_exercise,
            "week_exercise_total": sum(item["duration"] for item in exercise_week),
            "exercise_target": goal.get("exercise_target", 60),
            "current_weight": profile["weight"] if profile else None,
            "weight_logs": weight_logs,
            "exercise_week": exercise_week,
            "diet_today": diet_today.get("logs", []),
            "checkin_streak": streak["current_streak"],
            "checkin_total": streak["total_days"],
            "achievements": achievements,
        }
    }


# ============================================================
# ★ 打卡记录 API 接口（需要鉴权）
# ============================================================

# ---- 保存打卡记录 ----
@app.post("/api/dashboard/checkin")
async def api_save_checkin(
    checkin_type: str = Form("daily"),
    current_user: dict = Depends(get_current_user)
):
    """保存一条打卡记录（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import save_checkin
    save_checkin(user_id, checkin_type)
    return {"success": True, "message": "打卡成功！"}


# ---- 获取打卡记录 ----
@app.get("/api/dashboard/checkin")
async def api_get_checkin_logs(days: int = 30, current_user: dict = Depends(get_current_user)):
    """获取最近 N 天的打卡记录（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import get_checkin_logs
    data = get_checkin_logs(user_id, days)
    return {"success": True, "data": data}


# ---- 获取连续打卡天数 ----
@app.get("/api/dashboard/streak")
async def api_get_streak(current_user: dict = Depends(get_current_user)):
    """获取连续打卡天数和总打卡天数（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import get_checkin_streak
    data = get_checkin_streak(user_id)
    return {"success": True, "data": data}


# ---- 获取成就列表 ----
@app.get("/api/dashboard/achievements")
async def api_get_achievements(current_user: dict = Depends(get_current_user)):
    """获取用户已获得的成就列表（从 JWT Token 中获取 user_id）"""
    user_id = current_user["user_id"]
    from database import get_achievements
    data = get_achievements(user_id)
    return {"success": True, "data": data}


# ============================================================
# ★ ReAct 模式 SSE 生成器（AI Agent 架构升级）
# ============================================================
async def _react_sse_generator(graph_input, config, user_id, image_path, user_message):
    """
    ReAct 模式 SSE 事件流生成器。

    与旧版 bind_tools 模式的区别：
      - 发送 "thought" 事件（LLM 的推理过程，前端可显示 💭 气泡）
      - 发送 "meituan_url" 事件（open_meituan 工具结果）
      - 使用 react_graph（结构化 JSON 输出的 ReAct 循环）
      - 支持 interrupt() 确认机制
    """
    thinking_data = json.dumps(
        {"thinking": "正在分析图片内容..." if image_path else "正在分析您的问题..."},
        ensure_ascii=False,
    )
    yield f"data: {thinking_data}\n\n"

    full_response = ""
    llm_buffer = ""       # ★ 缓冲本轮 LLM 输出，用于解析 structured JSON

    try:
        async for event in react_graph.astream_events(
            graph_input, config=config, version="v2"
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                # ★ 工具调用开始 → 先提取 thought，再清空 buffer
                if llm_buffer.strip():
                    # 尝试从 buffer 中解析 thought
                    try:
                        raw = llm_buffer.strip()
                        if raw.startswith("```"):
                            lines_raw = raw.split("\n")
                            raw = "\n".join(lines_raw[1:] if lines_raw[-1].strip() != "```" else lines_raw[1:-1])
                        parsed_thought = json.loads(raw)
                        thought_text = parsed_thought.get("thought", "")
                        if thought_text:
                            thought_data = json.dumps({"thought": thought_text}, ensure_ascii=False)
                            yield f"data: {thought_data}\n\n"
                    except Exception:
                        pass  # 解析失败忽略，不影响主流程

                llm_buffer = ""

                tool_name = event.get("name", "")
                if "analyze_image" in tool_name:
                    thinking_data = json.dumps({"thinking": "正在用视觉模型分析图片..."}, ensure_ascii=False)
                elif "search" in tool_name:
                    thinking_data = json.dumps({"thinking": "正在检索健身知识库..."}, ensure_ascii=False)
                elif "open_meituan" in tool_name:
                    thinking_data = json.dumps({"thinking": "正在打开美团..."}, ensure_ascii=False)
                else:
                    thinking_data = json.dumps({"thinking": "正在为您进行科学计算..."}, ensure_ascii=False)
                yield f"data: {thinking_data}\n\n"

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                tool_output = event.get("data", {}).get("output", "")

                if "open_meituan" in tool_name:
                    try:
                        if isinstance(tool_output, ToolMessage):
                            result_str = tool_output.content
                            result = json.loads(result_str) if result_str else {}
                        elif isinstance(tool_output, dict):
                            result = tool_output
                        else:
                            result = {}

                        if result.get("action") == "open_meituan":
                            meituan_url = result.get("meituan_url", "")
                            keyword = result.get("keyword", "")
                            meituan_event = json.dumps({
                                "meituan_url": meituan_url,
                                "keyword": keyword,
                                "label": f"去美团{'搜「' + keyword + '」' if keyword else '看看'}",
                            }, ensure_ascii=False)
                            yield f"data: {meituan_event}\n\n"
                    except Exception as e:
                        print(f"[Main] 解析 open_meituan 结果失败: {e}", file=sys.stderr, flush=True)

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    llm_buffer += chunk.content

    except Exception as e:
        error_str = str(e)

        # 检测 GraphInterrupt（Human-in-the-loop）
        if "interrupt" in error_str.lower() or "GraphInterrupt" in error_str:
            print(f"[Main] ReAct 模式: 检测到 interrupt", file=sys.stderr, flush=True)
            try:
                state = react_graph.get_state(config)
                interrupt_data = None
                if state and hasattr(state, "interrupts") and state.interrupts:
                    for intr in state.interrupts:
                        if hasattr(intr, "value"):
                            interrupt_data = intr.value
                            break

                if interrupt_data:
                    confirmation_event = json.dumps(
                        {"confirmation_required": interrupt_data},
                        ensure_ascii=False,
                    )
                    yield f"data: {confirmation_event}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            except Exception as ex:
                print(f"[Main] 获取 interrupt 状态失败: {ex}", file=sys.stderr, flush=True)

        error_msg = f"抱歉，AI 服务出现了问题: {error_str[:200]}"
        full_response = error_msg
        token_data = json.dumps({"token": error_msg}, ensure_ascii=False)
        yield f"data: {token_data}\n\n"

    # ============================================================
    # ★ 处理最终的 LLM buffer —— 解析 structured JSON，只发送 response 文本
    # ============================================================
    if llm_buffer.strip():
        print(f"[Main] ReAct buffer 长度: {len(llm_buffer)} chars", file=sys.stderr, flush=True)
        try:
            # 提取 JSON（处理可能的 markdown 代码块标记）
            json_str = llm_buffer.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:] if lines[-1].strip() != "```" else lines[1:-1])

            parsed = json.loads(json_str)
            action = parsed.get("action", {})

            # ★ 发送 thought 事件（前端可展示 💭 气泡）
            thought = parsed.get("thought", "")
            if thought:
                thought_data = json.dumps({"thought": thought}, ensure_ascii=False)
                yield f"data: {thought_data}\n\n"

            # ★ 只发送 response 文本（不再泄漏 raw JSON）
            if action.get("type") == "final_response":
                response_text = action.get("response", "")
                if response_text:
                    full_response = response_text
                    for char in response_text:
                        token_data = json.dumps({"token": char}, ensure_ascii=False)
                        yield f"data: {token_data}\n\n"
                        await asyncio.sleep(0.02)
                else:
                    full_response = "抱歉，我需要更多信息来帮助您。"
                    token_data = json.dumps({"token": full_response}, ensure_ascii=False)
                    yield f"data: {token_data}\n\n"
            else:
                # 非 final_response（可能是未处理的 tool_call 格式）
                full_response = "抱歉，处理遇到问题，请重试。"
                token_data = json.dumps({"token": full_response}, ensure_ascii=False)
                yield f"data: {token_data}\n\n"

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[Main] ReAct JSON 解析失败: {e}", file=sys.stderr, flush=True)
            # 当作普通文本处理
            full_response = llm_buffer.strip()
            for char in full_response:
                token_data = json.dumps({"token": char}, ensure_ascii=False)
                yield f"data: {token_data}\n\n"
                await asyncio.sleep(0.02)
    else:
        full_response = "抱歉，处理遇到问题，请重试。"
        token_data = json.dumps({"token": full_response}, ensure_ascii=False)
        yield f"data: {token_data}\n\n"

    # 保存 AI 回复 + 自动更新结构化记忆
    if full_response:
        try:
            save_chat_message(user_id, "assistant", full_response)
            # ★ 自动更新结构化记忆
            try:
                memory = AgentMemoryManager(user_id)
                memory.auto_update_from_conversation(user_message, full_response)
            except Exception as e:
                print(f"[Main] 自动更新记忆失败: {e}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Main] 保存 AI 回复失败: {e}", file=sys.stderr, flush=True)

    # 清理临时图片
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except:
            pass

    yield "data: [DONE]\n\n"


# ============================================================
# ★ Phase 2: 营养报告 API
# ============================================================
@app.get("/api/nutrition/report")
async def api_nutrition_report(days: int = 7, current_user: dict = Depends(get_current_user)):
    """获取最近 N 天的营养摄入汇总报告"""
    user_id = current_user["user_id"]
    import datetime
    from database import get_diet_logs, get_goal_settings

    goal = get_goal_settings(user_id)
    report = []

    for i in range(days):
        date = (datetime.date.today() - datetime.timedelta(days=i)).isoformat()
        day_data = get_diet_logs(user_id, date)
        report.append({
            "date": date,
            "calories": day_data.get("total_calories", 0),
            "protein": day_data.get("total_protein", 0),
            "carb": day_data.get("total_carb", 0),
            "fat": day_data.get("total_fat", 0),
            "logs": day_data.get("logs", []),
        })

    report.reverse()  # 按日期升序

    return {
        "success": True,
        "data": {
            "report": report,
            "targets": {
                "calorie_target": goal.get("calorie_target", 2000),
                "protein_target": goal.get("protein_target", 120),
                "carb_target": goal.get("carb_target", 250),
                "fat_target": goal.get("fat_target", 65),
            }
        }
    }


# ============================================================
# ★ Phase 2: 训练日志 API
# ============================================================
@app.get("/api/training/log")
async def api_get_training_logs(days: int = 7, current_user: dict = Depends(get_current_user)):
    """获取最近 N 天的训练日志"""
    user_id = current_user["user_id"]
    from database import get_training_logs
    data = get_training_logs(user_id, days)
    return {"success": True, "data": data}


@app.post("/api/training/log")
async def api_save_training_log(
    exercise_name: str = Form(...),
    sets_done: int = Form(...),
    reps_done: str = Form(...),
    weight_kg: float = Form(0),
    muscle_group: str = Form(""),
    plan_name: str = Form(""),
    notes: str = Form(""),
    current_user: dict = Depends(get_current_user)
):
    """手动保存一条训练日志"""
    user_id = current_user["user_id"]
    from database import save_training_log
    save_training_log(user_id, exercise_name, sets_done, reps_done,
                      weight_kg, muscle_group, plan_name, notes)
    return {"success": True, "message": "训练日志已保存"}


# ============================================================
# ★ Phase 2: 训练计划 API
# ============================================================
@app.get("/api/training/plan")
async def api_get_training_plan(current_user: dict = Depends(get_current_user)):
    """获取用户当前激活的训练计划"""
    user_id = current_user["user_id"]
    from database import get_active_training_plan
    plan = get_active_training_plan(user_id)
    if plan:
        return {"success": True, "data": plan}
    else:
        return {"success": False, "message": "尚未生成训练计划，请告诉 AI 教练你的目标"}


@app.post("/api/training/plan")
async def api_save_training_plan_api(
    plan_name: str = Form(...),
    plan_json: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """保存训练计划"""
    user_id = current_user["user_id"]
    from database import save_training_plan
    save_training_plan(user_id, plan_name, plan_json)
    return {"success": True, "message": "训练计划已保存"}


@app.post("/api/training/generate")
async def api_generate_training_plan(
    goal: str = Form("减脂"),
    experience_level: str = Form("新手"),
    days_per_week: int = Form(3),
    session_minutes: int = Form(60),
    available_equipment: str = Form("哑铃"),
    focus_area: str = Form("全身"),
    current_user: dict = Depends(get_current_user)
):
    """一键生成训练计划（直接从用户画像 + 参数生成）"""
    import json
    user_id = current_user["user_id"]
    from training_plan import generate_workout_plan
    from database import save_training_plan, get_profile

    # 如果用户有画像，尝试从中提取目标
    profile = get_profile(user_id)
    if profile and profile.get("goal") and goal == "减脂":
        profile_goal = profile["goal"]
        # goal 可能是纯文本或 JSON
        if profile_goal.startswith("{"):
            try:
                parsed = json.loads(profile_goal)
                if isinstance(parsed, dict) and parsed.get("goal_type"):
                    goal = parsed["goal_type"]
            except Exception:
                pass
        else:
            goal_map = {
                "减脂": "减脂", "减肥": "减脂", "lose_weight": "减脂",
                "增肌": "增肌", "gain_muscle": "增肌",
                "塑形": "维持", "维持": "维持", "maintain": "维持",
                "力量": "力量", "strength": "力量",
            }
            for key, val in goal_map.items():
                if key in profile_goal:
                    goal = val
                    break

    plan = generate_workout_plan(
        goal=goal,
        experience_level=experience_level,
        days_per_week=days_per_week,
        session_minutes=session_minutes,
        available_equipment=available_equipment,
        focus_area=focus_area,
    )

    plan_json = json.dumps(plan, ensure_ascii=False)
    save_training_plan(user_id, plan["title"], plan_json)

    return {"success": True, "data": plan, "message": f"已生成「{plan['title']}」"}


# ============================================================
# ★ Phase 2: 进度洞察 API
# ============================================================
@app.get("/api/training/progress")
async def api_get_progress(current_user: dict = Depends(get_current_user)):
    """获取用户的进度洞察报告"""
    user_id = current_user["user_id"]
    from database import (
        get_profile, get_weight_logs, get_exercise_logs,
        get_goal_settings, get_checkin_streak, get_milestones
    )

    profile = get_profile(user_id)
    goal = get_goal_settings(user_id)
    weight_logs = get_weight_logs(user_id, 30)
    exercise_logs = get_exercise_logs(user_id, 7)
    streak = get_checkin_streak(user_id)
    milestones = get_milestones(user_id)

    return {
        "success": True,
        "data": {
            "profile": profile,
            "goal": goal,
            "weight_logs": weight_logs,
            "exercise_logs": exercise_logs,
            "streak": streak,
            "milestones": milestones,
        }
    }


# ============================================================
# ★ Phase 2: 里程碑 API
# ============================================================
@app.get("/api/milestones")
async def api_get_milestones(current_user: dict = Depends(get_current_user)):
    """获取用户达成的所有里程碑"""
    user_id = current_user["user_id"]
    from database import get_milestones
    data = get_milestones(user_id)
    return {"success": True, "data": data}


# ============================================================
# ★ Phase 2: 目标设置扩展 API（支持宏量营养素单独设置）
# ============================================================
@app.post("/api/dashboard/goal/nutrition")
async def api_save_nutrition_goal(
    calorie_target: int = Form(...),
    protein_target: int = Form(120),
    carb_target: int = Form(250),
    fat_target: int = Form(65),
    exercise_target: int = Form(60),
    current_user: dict = Depends(get_current_user)
):
    """保存详细的营养目标（含三大宏量营养素）"""
    user_id = current_user["user_id"]
    from database import save_nutrition_goals
    save_nutrition_goals(user_id, calorie_target, protein_target, carb_target, fat_target, exercise_target)
    return {"success": True, "message": "营养目标已保存"}


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

