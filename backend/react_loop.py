# -*- coding: utf-8 -*-
"""
react_loop.py —— ReAct Agent 控制循环（AI Agent 架构升级核心）
================================================================
ReAct (Reasoning + Acting) 控制循环 —— Agent 的核心大脑。

循环逻辑：
    1. Thought（思考）：LLM 分析当前状态，输出推理过程
    2. Action（行动）：LLM 选择一个工具调用 OR 生成最终回复
    3. Observation（观察）：如果是工具调用，执行工具并获取结果
    4. Loop：将观察结果反馈给 LLM，继续下一轮思考

与旧版 bind_tools() 的区别：
    - 旧版：LLM 隐式决策，tool_calls 和 response 混在一起，思考不可见
    - 新版：强制「先思考、后行动」的两步分离，思考过程对用户可见

设计参考：LangGraph ReAct Agent Pattern + 结构化 JSON 输出
"""

from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

import json
import sys
from functools import partial

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, LLM_ENABLED
from database import get_profile
from tools import fitness_tools, tool_map
from safety_filter import check_user_input
from memory.memory_manager import AgentMemoryManager


# ============================================================
# 1. 扩展的 Agent 状态
# ============================================================
class AgentState(TypedDict):
    """ReAct Agent 的完整状态"""
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: Optional[dict]
    memory_context: Optional[dict]       # ★ 新增：结构化记忆上下文
    system_prompt: str
    thought_log: Annotated[list[str], add_messages]  # ★ 思考日志（可观测）
    pending_confirmation: Optional[dict]  # ★ 待确认的操作
    image_path: Optional[str]             # 用户上传的图片路径


# ============================================================
# 2. 工具描述生成（供 ReAct Prompt 使用）
# ============================================================
def _build_tool_descriptions() -> str:
    """将所有工具的描述信息格式化为 ReAct Prompt 片段"""
    lines = []
    for tool in fitness_tools:
        name = tool.name
        desc = (tool.description or "").split("\n")[0][:200]
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


TOOL_DESCRIPTIONS = _build_tool_descriptions()


# ============================================================
# 3. 核心：ReAct 决策节点
# ============================================================
def agent_react_node(state: AgentState, llm) -> dict:
    """
    ReAct 决策节点 —— 执行一轮「思考 → 行动」循环。

    这个节点会被反复调用（Agent Loop），每次执行一轮：
      1. 构建包含「思考引导」的 prompt
      2. 调用 LLM，强制其先输出 Thought 再输出 Action
      3. 解析 LLM 输出，决定是调用工具还是回复用户

    关键创新：
      - 不使用 bind_tools() 的隐式 tool_calls
      - 而是用 system prompt 要求 LLM 输出结构化 JSON
      - 这样 Thought 和 Action 是分开的、可观测的
    """
    # ---- 构建 ReAct 专用 System Prompt ----
    base_prompt = state.get("system_prompt", "")

    react_instructions = f"""

# ★★★ ReAct 思考-行动协议（核心规则）★★★

你的每一次回复必须严格遵循以下格式。先思考，后行动。

## 输出格式（JSON）

请直接输出一个 JSON 对象（不要用 ```json 包裹）:

{{
  "thought": "你的推理过程：分析用户意图、结合上下文、决定下一步做什么",
  "action": {{
    "type": "tool_call 或 final_response",
    "tool_name": "工具名称（仅 tool_call 时需要）",
    "tool_args": {{"参数名": "参数值"}},
    "response": "最终回复文本（仅 final_response 时需要）"
  }}
}}

## 思考规则（必须遵守）

1. **第一步永远是思考**。不要跳过推理直接行动。
2. 思考要考虑：用户意图、已有的上下文、工具执行结果。
3. **外卖/饥饿处理流程（★ 强制执行）**:
   - ⚠️ 当用户说"饿了"、"想吃东西"、"点外卖"、"有什么好吃的"、"吃啥"、"叫外卖"等任何与吃/外卖相关的表达时：
     → **AI 禁止直接生成文字建议！必须优先调用 ask_food_delivery 工具！**
     → thought: "用户表达了饥饿感/外卖意图。根据规则，我不能直接建议，必须先调用 ask_food_delivery..."
     → action: 调用 ask_food_delivery(user_id="<用户ID>", user_message="<用户原话>")
   - 如果用户已经确认了外卖请求:
     → action: 调用 search_nearby_food(keyword="...", goal="...")
   - 如果用户说"打开美团":
     → action: 调用 open_meituan(user_id="<用户ID>", keyword="...")
4. **健身知识查询**: 回答具体健身问题前，必须先调用 search_fitness_knowledge
5. **营养计算**: 涉及 TDEE/热量/蛋白质时，先调用 calculate_tdee 或 generate_macro_plan
6. **图片分析**: 如果有图片，第一个动作必须是调用 analyze_image
7. **身体数据**: 用户提供身高体重等信息时，立即调用 save_user_profile
8. **打卡**: 用户表示完成训练时，调用 record_checkin

## 可用工具

{TOOL_DESCRIPTIONS}

## 常见场景示例

场景1: 用户说"我饿了"
{{
  "thought": "用户表达了饥饿感。根据规则，我不能直接给建议，必须先调用 ask_food_delivery 工具触发二次确认。",
  "action": {{"type": "tool_call", "tool_name": "ask_food_delivery", "tool_args": {{"user_id": "<user_id>", "user_message": "我饿了"}}}}
}}

场景2: 用户确认需要外卖推荐
{{
  "thought": "用户确认了需要帮助。他正在减脂期，我来搜索适合的健康低卡餐。",
  "action": {{"type": "tool_call", "tool_name": "search_nearby_food", "tool_args": {{"keyword": "轻食沙拉", "goal": "lose_weight"}}}}
}}

场景3: 用户说"打开美团"
{{
  "thought": "用户直接要求打开美团。",
  "action": {{"type": "tool_call", "tool_name": "open_meituan", "tool_args": {{"user_id": "<user_id>", "keyword": "沙拉"}}}}
}}

场景4: 用户问训练计划
{{
  "thought": "用户是新手，需要全身性训练建议。我先检索知识库获取专业内容。",
  "action": {{"type": "tool_call", "tool_name": "search_fitness_knowledge", "tool_args": {{"query": "新手训练计划"}}}}
}}

场景5: 工具结果返回后，直接回复用户
{{
  "thought": "知识库返回了新手训练建议，我来组织一个热情专业的回复。要口语化、接地气，像教练一样。",
  "action": {{"type": "final_response", "response": "没问题！对于新手来说，我建议从全身性训练开始..."}}
}}
"""

    full_system_prompt = base_prompt + react_instructions

    # ---- 组装消息列表 ----
    messages = [SystemMessage(content=full_system_prompt)]
    # 添加记忆上下文中的信息（作为系统注入，不显示给用户）
    if state.get("memory_context"):
        memory_msg = f"# 用户记忆信息\n{state['memory_context']}"
        messages.append(SystemMessage(content=memory_msg))
    messages.extend(state["messages"])

    # ---- 调用 LLM ----
    print(f"[ReAct] 🧠 调用 LLM 进行 Thought → Action 决策...", file=sys.stderr, flush=True)
    response = llm.invoke(messages)
    raw_content = response.content

    if hasattr(response, "content"):
        raw_content = response.content
    else:
        raw_content = str(response)

    print(f"[ReAct] 📥 LLM 原始输出({len(raw_content)} chars): {raw_content[:200]}...", file=sys.stderr, flush=True)

    # ---- 解析 LLM 的结构化输出 ----
    try:
        # 尝试提取 JSON（处理 LLM 输出中可能的额外文本）
        json_str = raw_content.strip()
        # 去掉可能的 markdown 代码块标记
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        parsed = json.loads(json_str)
        thought = parsed.get("thought", "")
        action = parsed.get("action", {})
        action_type = action.get("type", "final_response")

        # 记录思考日志（前端可展示）
        print(f"[ReAct] 💭 Thought: {thought[:150]}...", file=sys.stderr, flush=True)
        print(f"[ReAct] 🎬 Action: {action_type}", file=sys.stderr, flush=True)

        if action_type == "tool_call":
            tool_name = action.get("tool_name", "")
            tool_args = action.get("tool_args", {})

            # ★ 特殊处理：ask_food_delivery / confirm_food_order 需要 Human-in-the-loop
            if tool_name in ("ask_food_delivery", "confirm_food_order"):
                return _handle_confirmation(state, thought, tool_args)

            # 执行普通工具
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    result_str = str(result)
                    print(f"[ReAct] 🔧 工具 {tool_name} 执行成功，结果({len(result_str)} chars)", file=sys.stderr, flush=True)

                    # 特殊处理：open_meituan 的结果需要透传
                    is_meituan = tool_name == "open_meituan" and isinstance(result, dict)

                    return {
                        "thought_log": [f"💭 {thought}"],
                        "messages": [
                            AIMessage(content=json.dumps(parsed, ensure_ascii=False)),
                            ToolMessage(
                                content=result_str,
                                tool_call_id=f"react_{tool_name}",
                                name=tool_name,
                            )
                        ],
                        # 如果是 open_meituan，额外传递 meituan_url 给前端
                        **({"pending_confirmation": {
                            "type": "meituan_url",
                            "meituan_url": result.get("meituan_url", ""),
                            "keyword": result.get("keyword", ""),
                        }} if is_meituan and result.get("success") else {})
                    }
                except Exception as e:
                    error_msg = f"工具执行出错: {type(e).__name__}: {e}"
                    print(f"[ReAct] ❌ 工具 {tool_name} 执行失败: {e}", file=sys.stderr, flush=True)
                    return {
                        "thought_log": [f"💭 {thought}"],
                        "messages": [
                            AIMessage(content=json.dumps(parsed, ensure_ascii=False)),
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=f"react_{tool_name}",
                                name=tool_name,
                            )
                        ]
                    }
            else:
                print(f"[ReAct] ⚠️ 未知工具: {tool_name}", file=sys.stderr, flush=True)
                return {
                    "thought_log": [f"💭 {thought}"],
                    "messages": [AIMessage(content=f"抱歉，我暂时不支持 {tool_name} 功能。")]
                }

        else:  # final_response
            reply = action.get("response", "")
            if not reply:
                reply = "抱歉，我需要更多信息来帮助您。"
            print(f"[ReAct] ✅ 最终回复({len(reply)} chars)", file=sys.stderr, flush=True)
            return {
                "thought_log": [f"💭 {thought}"],
                "messages": [AIMessage(content=reply)]
            }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # LLM 未按格式输出，直接当作纯文本回复
        print(f"[ReAct] ⚠️ JSON 解析失败 ({type(e).__name__}): {e}", file=sys.stderr, flush=True)
        # 检查是否 content 本身就是合理的回复
        if raw_content and len(raw_content.strip()) > 10:
            return {
                "thought_log": ["💭 (直接回复)"],
                "messages": [AIMessage(content=raw_content.strip())]
            }
        else:
            return {
                "thought_log": ["💭 (兜底回复)"],
                "messages": [AIMessage(content="好的，让我来帮你。有什么健身相关的问题可以详细说说吗?")]
            }


# ============================================================
# 4. 确认处理器（Human-in-the-loop）
# ============================================================
def _handle_confirmation(state, thought, tool_args):
    """
    处理需要用户确认的操作（如 ask_food_delivery / confirm_food_order）。

    使用 LangGraph 的 interrupt() 暂停执行，
    等待前端用户确认后通过 Command(resume=...) 恢复。
    """
    # ask_food_delivery 没有 intent 字段，默认使用 "hunger"
    confirm_type = tool_args.get("intent", "hunger")
    user_message = tool_args.get("user_message", "")

    # 构建确认消息
    from food_order_guard import get_confirmation_message
    confirm_msg = get_confirmation_message(
        "hunger" if confirm_type in ("hunger", "") else "explicit",
        user_message,
    )

    # ★ 暂停 Agent，等待用户确认
    user_decision = interrupt({
        "type": "food_order_confirmation",
        "intent_type": confirm_type,
        "thought": thought,
        "message": confirm_msg,
        "user_input": user_message,
    })

    print(f"[ReAct] 🔄 interrupt 返回，用户决策: {user_decision}", file=sys.stderr, flush=True)

    if user_decision == "confirm":
        return {
            "thought_log": [
                f"💭 {thought}",
                f"✅ 用户已确认，继续搜索附近健康餐厅"
            ],
            "messages": [
                AIMessage(content=f"💭 {thought}"),
                AIMessage(content="好的! 用户已确认，让我来帮你找找看附近适合你的健康餐厅。"),
            ]
        }
    else:
        return {
            "thought_log": [f"💭 {thought}", "❌ 用户取消了操作"],
            "messages": [AIMessage(content="好的，没问题! 有任何健身相关的问题随时问我。")]
        }


# ============================================================
# 5. 循环条件判断
# ============================================================
def should_continue_react(state: AgentState) -> str:
    """
    判断 ReAct 循环是否应继续。

    规则：
      - 如果最后一条消息是 ToolMessage → 继续循环（回到 Agent）
      - 如果最后一条消息是 AIMessage（无 tool_call）→ 结束
      - 如果有 pending_confirmation → 结束（让 main.py 处理跳转）
    """
    messages = state.get("messages", [])
    if not messages:
        return END

    # 有 meituan_url 待处理 → 结束，交给 main.py
    if state.get("pending_confirmation", {}).get("type") == "meituan_url":
        return END

    last_msg = messages[-1]

    # 工具执行结果 → 继续思考
    if isinstance(last_msg, ToolMessage):
        return "Agent"

    # AI 直接回复 → 结束
    if isinstance(last_msg, AIMessage):
        return END

    return END


# ============================================================
# 6. Context_Loader 节点（从旧版迁移）
# ============================================================
def context_loader_node(state: AgentState) -> dict:
    """
    上下文加载节点 —— 从数据库加载用户数据并构建 System Prompt。

    核心逻辑：
        1. 从 state 中取出 user_id
        2. 从 SQLite 查询用户身体数据
        3. 主动检索健身知识库（RAG）
        4. 构建 System Prompt
    """
    user_id = state["user_id"]
    profile = get_profile(user_id)

    # ---- RAG：主动检索健身知识库 ----
    rag_context = ""
    messages = state.get("messages", [])
    if messages:
        latest_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_user_msg = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                latest_user_msg = msg.get("content", "")
                break

        if latest_user_msg:
            from tools import search_fitness_knowledge
            rag_result = search_fitness_knowledge.invoke({"query": latest_user_msg})
            if rag_result["total_matched"] > 0:
                rag_chunks = []
                for chunk in rag_result["knowledge_chunks"]:
                    rag_chunks.append(f"  [{chunk['topic']}] {chunk['content']}")
                rag_context = "\n# ★ 知识库检索结果（RAG 上下文）\n" + "\n".join(rag_chunks)
                print(f"[ReAct] RAG: 检索到 {rag_result['total_matched']} 条相关知识")

    # ---- 教练人设 + 规则 ----
    tool_rules = """
# ★★★ 安全熔断机制（最高优先级）★★★

【硬熔断规则】
触发条件：高血压、糖尿病、心脏病、哮喘、痛风、关节炎、骨折、术后恢复等疾病；
         剧烈疼痛、头晕、胸闷、心悸、呼吸困难等症状；
         孕妇、哺乳期、未成年人、60岁以上老人。
硬熔断时必须输出以下回复：
  "⚠️ 安全提示：检测到您的情况涉及医疗健康范畴。作为 AI 健身教练，我无法提供医疗诊断..."

【灰色地带规则】
用户轻微提及身体不适（如"腰酸"、"脖子酸"）时，必须在建议前加温馨提示。

# 身份与性格
你是 AI 智能教练——专业贴心的专属健身伙伴。性格热情、专业、有耐心且极具鼓励性。

# 规则
- 只回答健身、饮食、营养、运动恢复相关问题。
- 涉及营养计算时必须先调用工具，严禁凭空捏造数字。
- 如果用户没有录入身体数据，友好引导填写。
- 不要复述用户的原始身体数据（除非用户明确询问）。
- 口语化、接地气，多用"咱们"、"加油"等词汇。
- ⚠️【外卖铁律】当用户提到"饿"、"吃"、"外卖"、"点餐"等任何与饮食/外卖相关的意图时，
   必须调用 ask_food_delivery 工具触发二次确认，严禁直接生成文字建议！
"""

    # ---- 构建 System Prompt ----
    image_instruction = ""
    image_path = state.get("image_path")
    if image_path:
        image_instruction = f"\n\n用户上传了一张图片: {image_path}\n请先调用 analyze_image 工具分析。"
        print(f"[ReAct] Context_Loader: 检测到图片 {image_path}")

    if profile:
        height_m = profile["height"] / 100
        bmi = profile["weight"] / (height_m ** 2)
        focus_hint = "减脂" if bmi >= 24 else ("增肌" if bmi < 18.5 else "体能提升")

        system_prompt = f"""你是 AI 智能教练——专业贴心的专属健身伙伴。

当前客户数据（仅用于内部计算，不要复述）:
- 身高：{profile["height"]} cm
- 体重：{profile["weight"]} kg
- 年龄：{profile["age"]} 岁
- 性别：{profile["gender"]}
- BMI：{bmi:.1f}（建议方向：{focus_hint}）

{tool_rules}
{rag_context}
{image_instruction}

调用工具时使用这些参数: weight={profile["weight"]}, height={profile["height"]}, age={profile["age"]}, gender="{profile["gender"]}"""
    else:
        system_prompt = f"""你是 AI 智能教练——专业贴心的专属健身伙伴。
当前客户尚未录入身体数据，请友好引导填写。
{tool_rules}
{rag_context}
{image_instruction}"""

    return {
        "user_profile": profile,
        "system_prompt": system_prompt,
    }


# ============================================================
# 7. Memory_Loader 节点（★ 新增）
# ============================================================
def memory_loader_node(state: AgentState) -> dict:
    """
    Memory_Loader —— 记忆加载节点

    从结构化记忆存储（agent_memory 表）中加载用户的：
      1. 训练状态（当前阶段、最近训练）
      2. 饮食偏好（喜欢的食物、忌口）
      3. 上次对话摘要

    将记忆上下文注入 system_prompt。
    """
    user_id = state["user_id"]
    memory = AgentMemoryManager(user_id)
    memory_context = memory.get_memory_context()

    if memory_context:
        current_prompt = state.get("system_prompt", "")
        memory_instructions = """
# ★ 记忆使用规则
- 请自然地引用上述记忆中的信息，让用户感受到你记得他/她。
- 如果用户更新了训练状态或饮食偏好，请调用 save_user_profile 保存。
- 不要机械地复述记忆内容，要自然地融入回复中。
"""
        enhanced_prompt = current_prompt + "\n\n" + memory_context + memory_instructions
        print(f"[ReAct] Memory_Loader: 已加载用户 {user_id} 的结构化记忆")
        return {
            "system_prompt": enhanced_prompt,
            "memory_context": memory_context,
        }
    else:
        print(f"[ReAct] Memory_Loader: 用户 {user_id} 无结构化记忆")
        return {}


# ============================================================
# 8. Safety_Gate 节点
# ============================================================
def safety_gate_node(state: AgentState) -> dict:
    """
    Safety_Gate —— 安全熔断门控节点

    对用户最新消息进行安全预检。三级响应：
        1. BLOCK（硬熔断）：检测到疾病/严重症状 → 直接返回安全提示
        2. WARN（灰色地带）：检测到轻微不适 → 注入警告到 system_prompt
        3. SAFE（安全）：不做干预
    """
    messages = state.get("messages", [])
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_msg = msg.get("content", "")
            break

    if not latest_user_msg:
        return {}

    result = check_user_input(latest_user_msg)
    level = result["level"]

    if level == "block":
        print(f"[ReAct] Safety_Gate: 🔴 硬熔断! keyword={result.get('keyword')}")
        return {"messages": [AIMessage(content=result["message"])]}

    elif level == "warn":
        keyword = result.get("keyword", "")
        warning = result.get("warning", "")
        print(f"[ReAct] Safety_Gate: 🟡 灰色地带, keyword={keyword}")
        current_prompt = state.get("system_prompt", "")
        safety_instruction = (
            f"\n\n# ★ 灰色地带安全警告\n"
            f"检测到用户提及：{keyword}\n"
            f"你必须在回复的最前面加上以下温馨提示：\n"
            f"{warning}\n然后再给出建议。"
        )
        return {"system_prompt": current_prompt + safety_instruction}

    else:
        print(f"[ReAct] Safety_Gate: 🟢 安全放行")
        return {}


def route_after_safety(state: AgentState) -> str:
    """Safety_Gate 后的路由：熔断→END，否则继续"""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and not getattr(last_msg, "tool_calls", None):
            return END
    return "Agent"


# ============================================================
# 9. 构建完整 ReAct Agent 图
# ============================================================
def build_react_agent():
    """
    构建完整的 ReAct Agent 图。

    图结构：
        START → Context_Loader → Memory_Loader → Safety_Gate → Agent ⇄ END
                                                                   ↓
                                                              (有 ToolMessage 就继续)

    Agent 内部执行「Thought → Action」ReAct 循环。
    """
    # 创建 LLM 实例（使用 ChatOpenAI 连接 DeepSeek，不绑定工具，用结构化 JSON 替代）
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        streaming=True,
        temperature=0.3,  # 稍微降低温度以获得更稳定的 JSON 输出
    )

    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("Context_Loader", context_loader_node)
    graph.add_node("Memory_Loader", memory_loader_node)      # ★ 新增
    graph.add_node("Safety_Gate", safety_gate_node)
    graph.add_node("Agent", partial(agent_react_node, llm=llm))

    # 连接边
    graph.add_edge(START, "Context_Loader")
    graph.add_edge("Context_Loader", "Memory_Loader")
    graph.add_edge("Memory_Loader", "Safety_Gate")

    # Safety_Gate 后的条件路由
    graph.add_conditional_edges(
        "Safety_Gate",
        route_after_safety,
        {"Agent": "Agent", END: END},
    )

    # Agent 循环：如果有 ToolMessage → 回到 Agent 继续思考
    graph.add_conditional_edges(
        "Agent",
        should_continue_react,
        {"Agent": "Agent", END: END}
    )

    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    print("[ReAct] Agent 图构建完成: START -> Context_Loader -> Memory_Loader -> Safety_Gate -> Agent (loop)")
    return compiled


# ============================================================
# 10. 全局图实例
# ============================================================
if LLM_ENABLED:
    react_graph = build_react_agent()
else:
    # 未配置 API Key 时，回退到旧版图
    from langgraph_brain import fitness_graph as fallback_graph
    react_graph = fallback_graph
    print("[ReAct] ⚠️ 未配置 DEEPSEEK_API_KEY，使用旧版图作为 fallback")
