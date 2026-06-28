# AI 健身教练 → AI Agent 架构升级设计文档

> **目标**：将当前「关键词驱动 + 固定流程」的 LangGraph 管道，升级为真正的 **ReAct Agent 控制循环**。

---

## 一、当前架构 vs 目标架构

### 1.1 当前架构（存在的问题）

```
START → Context_Loader → Safety_Gate → FoodOrder_Guard → Agent(LLM) → Tools → Agent → END
```

| 问题 | 说明 |
|------|------|
| **无显式思考** | LLM 直接生成 tool_call 或回复，没有"思考→行动"的显式步骤 |
| **外卖逻辑硬编码** | `FoodOrder_Guard` 用关键词匹配 + `interrupt()`，不是 LLM 自主决策 |
| **工具粒度粗** | `search_nearby_food` 搜索+推荐一体化，缺少 `open_meituan()` 独立动作 |
| **记忆分散** | `MemorySaver` 只存对话消息，没有结构化的「训练状态」「饮食偏好」 |
| **不可观测** | 用户看不到 Agent 的思考过程，缺乏透明度 |

### 1.2 目标架构（ReAct Agent）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ReAct Agent 控制循环                          │
│                                                                     │
│  START → Context_Loader → Memory_Loader → Safety_Gate              │
│                                                    ↓                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  AGENT LOOP (可多轮)                                          │   │
│  │                                                               │   │
│  │   Step 1: THOUGHT ── LLM 显式输出推理过程                      │   │
│  │       │                                                       │   │
│  │       ▼                                                       │   │
│  │   Step 2: ACTION ── LLM 决定调用工具 OR 直接回复用户            │   │
│  │       │                                                       │   │
│  │       ├── 调用工具 → OBSERVATION (工具执行结果) → 回到 Step 1   │   │
│  │       │                                                       │   │
│  │       └── 回复用户 → END                                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**关键区别**：

| 维度 | 旧架构 | 新架构 |
|------|--------|--------|
| 决策模式 | 预定义流程 + 关键词触发 | LLM 自主推理 + 决策 |
| 思考可见性 | 无 | 前端展示 💭 Thinking |
| 外卖处理 | `FoodOrder_Guard` 硬编码检测 | LLM 推理后主动调用 `confirm_food_order()` 工具 |
| 美团跳转 | `main.py` 中硬编码 `meituan_url` SSE 事件 | LLM 调用 `open_meituan()` 工具 |
| 记忆 | 只存对话消息 | 结构化记忆：训练状态 + 饮食偏好 + 对话摘要 |

---

## 二、核心模块设计

### 2.1 ReAct 控制循环（react_loop.py 新增文件）

```python
# ============================================================
# react_loop.py —— ReAct Agent 控制循环
# ============================================================
"""
ReAct (Reasoning + Acting) 控制循环 —— Agent 的核心大脑。

循环逻辑：
    1. Thought（思考）：LLM 分析当前状态，输出推理过程
    2. Action（行动）：LLM 选择一个工具调用 OR 生成最终回复
    3. Observation（观察）：如果是工具调用，执行工具并获取结果
    4. Loop：将观察结果反馈给 LLM，继续下一轮思考

与旧版区别：
    - 旧版：LLM 一次性决定（tool_calls + response 混在一起）
    - 新版：强制「先思考、后行动」的两步分离
"""

from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langchain_community.chat_models import ChatTongyi

import json


# ---- 2.1.1 扩展的 Agent 状态 ----
class AgentState(TypedDict):
    """Agent 的完整状态"""
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: Optional[dict]
    memory_context: Optional[dict]       # ★ 新增：结构化记忆上下文
    system_prompt: str
    thought_log: Annotated[list[str], add_messages]  # ★ 思考日志（可观测）
    pending_confirmation: Optional[dict]  # ★ 待确认的操作


# ---- 2.1.2 核心：ReAct 决策节点 ----
def agent_react_node(state: AgentState, llm, tools, tool_map) -> dict:
    """
    ReAct 决策节点 —— 执行一轮「思考 → 行动」循环。
    
    这个节点会被反复调用（Agent Loop），每次执行一轮：
      1. 构建包含「思考引导」的 prompt
      2. 调用 LLM，强制其先输出 Thought 再输出 Action
      3. 解析 LLM 输出，决定是调用工具还是回复用户
    
    ★ 关键创新：
      - 不使用 bind_tools() 的隐式 tool_calls
      - 而是用 system prompt 要求 LLM 输出结构化 JSON
      - 这样 Thought 和 Action 是分开的、可观测的
    """
    
    # 构建 ReAct 专用 System Prompt
    react_prompt = f"""
{state["system_prompt"]}

# ★★★ ReAct 思考-行动协议（核心规则）★★★

你的每一次回复必须严格遵循以下格式。先思考，后行动。

## 输出格式（JSON）
```json
{{
  "thought": "你的推理过程：分析用户意图、结合上下文、决定下一步做什么",
  "action": {{
    "type": "tool_call | final_response",
    "tool_name": "工具名称（仅 tool_call 时）",
    "tool_args": {{"参数名": "参数值"}},
    "response": "最终回复文本（仅 final_response 时）"
  }}
}}
```

## 思考规则
1. **第一步永远是思考**。不要跳过推理直接行动。
2. 思考要考虑：用户意图、已有的上下文、工具执行结果
3. 如果用户说"我饿了"，思考后应调用 `confirm_food_order` 工具（请求确认），而不是直接搜索外卖
4. 如果用户已经确认了外卖请求，思考后应调用 `search_nearby_food` 工具
5. 如果搜索结果已返回，思考后应基于结果组织最终回复

## 可用工具
{tool_descriptions}
"""
    
    # 组装消息列表
    messages = [SystemMessage(content=react_prompt)] + state["messages"]
    
    # 调用 LLM
    response = llm.invoke(messages)
    
    # 解析 LLM 的结构化输出
    try:
        parsed = json.loads(response.content)
        thought = parsed.get("thought", "")
        action = parsed.get("action", {})
        action_type = action.get("type", "final_response")
        
        # 记录思考日志（前端可展示）
        print(f"[ReAct] 💭 Thought: {thought}")
        print(f"[ReAct] 🎬 Action: {action_type}")
        
        if action_type == "tool_call":
            tool_name = action.get("tool_name", "")
            tool_args = action.get("tool_args", {})
            
            # 特殊处理：confirm_food_order 需要 Human-in-the-loop
            if tool_name == "confirm_food_order":
                return _handle_confirmation(state, thought, tool_args)
            
            # 执行普通工具
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                return {
                    "thought_log": [f"💭 {thought}"],
                    "messages": [
                        AIMessage(content=response.content),  # 原始 LLM 输出
                        ToolMessage(
                            content=str(result),
                            tool_call_id=f"react_{tool_name}",
                            name=tool_name,
                        )
                    ]
                }
            else:
                return {
                    "thought_log": [f"💭 {thought}"],
                    "messages": [AIMessage(content=f"未知工具: {tool_name}")]
                }
        
        else:  # final_response
            reply = action.get("response", "抱歉，我需要更多信息来帮助您。")
            return {
                "thought_log": [f"💭 {thought}"],
                "messages": [AIMessage(content=reply)]
            }
    
    except json.JSONDecodeError:
        # LLM 未按格式输出，直接当作纯文本回复
        return {
            "thought_log": ["💭 (直接回复)"],
            "messages": [AIMessage(content=response.content)]
        }


# ---- 2.1.3 确认处理器（Human-in-the-loop） ----
def _handle_confirmation(state, thought, tool_args):
    """
    处理需要用户确认的操作（如 confirm_food_order）。
    
    使用 LangGraph 的 interrupt() 暂停执行，
    等待前端用户确认后通过 Command(resume=...) 恢复。
    """
    confirm_type = tool_args.get("confirm_type", "food_order")
    message = tool_args.get("message", "确认执行此操作？")
    
    # ★ 暂停 Agent，等待用户确认
    user_decision = interrupt({
        "type": confirm_type,
        "thought": thought,
        "message": message,
    })
    
    if user_decision == "confirm":
        return {
            "thought_log": [
                f"💭 {thought}",
                f"✅ 用户已确认，继续执行 {tool_args.get('next_tool', 'search_nearby_food')}"
            ],
            "messages": [
                AIMessage(content=f"💭 {thought}"),
                AIMessage(content=f"好的！用户已确认，让我来帮你找找看。🔍"),
            ]
        }
    else:
        return {
            "thought_log": [f"💭 {thought}", "❌ 用户取消了操作"],
            "messages": [AIMessage(content="好的，没问题！有任何健身相关的问题随时问我。💪")]
        }


# ---- 2.1.4 循环条件判断 ----
def should_continue_react(state: AgentState) -> str:
    """
    判断 ReAct 循环是否应继续。
    
    规则：
      - 如果最后一条消息是 ToolMessage → 继续循环（回到 Agent）
      - 如果最后一条消息是 AIMessage（无 tool_call）→ 结束
      - 如果有 pending_confirmation → 暂停（Human-in-the-loop）
    """
    messages = state.get("messages", [])
    if not messages:
        return END
    
    last_msg = messages[-1]
    
    # 工具执行结果 → 继续思考
    if isinstance(last_msg, ToolMessage):
        return "Agent"
    
    # AI 直接回复 → 结束
    if isinstance(last_msg, AIMessage):
        return END
    
    return END


# ---- 2.1.5 构建完整 Agent 图 ----
def build_react_agent(llm, tools, tool_map, checkpointer):
    """
    构建完整的 ReAct Agent 图。
    
    图结构：
        START → Context_Loader → Memory_Loader → Safety_Gate → Agent ⇄ END
                                                                   ↓
                                                              (如果有 ToolMessage)
    
    Agent 内部执行一轮「Thought → Action」循环。
    """
    from functools import partial
    
    graph = StateGraph(AgentState)
    
    # 注册节点
    graph.add_node("Context_Loader", context_loader_node)
    graph.add_node("Memory_Loader", memory_loader_node)     # ★ 新增
    graph.add_node("Safety_Gate", safety_gate_node)
    graph.add_node("Agent", partial(
        agent_react_node, llm=llm, tools=tools, tool_map=tool_map
    ))
    
    # 连接边
    graph.add_edge(START, "Context_Loader")
    graph.add_edge("Context_Loader", "Memory_Loader")        # ★ 新增
    graph.add_edge("Memory_Loader", "Safety_Gate")
    graph.add_edge("Safety_Gate", "Agent")
    
    # 条件循环：如果 Agent 执行了工具 → 回到 Agent 继续思考
    graph.add_conditional_edges(
        "Agent",
        should_continue_react,
        {"Agent": "Agent", END: END}
    )
    
    return graph.compile(checkpointer=checkpointer)
```

### 2.2 open_meituan() 工具（tools.py 新增）

```python
# ============================================================
# tools.py 新增工具：open_meituan
# ============================================================

@tool
def open_meituan(
    user_id: str,
    keyword: str = "",
    goal: str = "maintain",
) -> dict:
    """
    打开美团外卖，为用户跳转到美团外卖页面。

    ★ 这是"执行动作"工具，不是"搜索"工具。
    只有当用户在对话中明确确认（如回复"是的"、"确认"、"打开"）后，
    Agent 才应调用此工具。

    工作流程：
        1. 用户说"我饿了" → Agent 调用 confirm_food_order() → 等待确认
        2. 用户确认 → Agent 调用 search_nearby_food() → 获取推荐
        3. 用户说"打开美团" → Agent 调用 open_meituan() → 返回跳转链接

    参数说明：
        user_id: 用户唯一标识
        keyword: 搜索关键词（如"沙拉"、"鸡胸肉"），用于拼接跳转 URL
        goal: 用户健身目标，用于日志记录

    返回：
        包含 meituan_url 的字典，前端收到后自动跳转或打开弹窗
    """
    import urllib.parse
    from database import save_diet_log
    
    # 构建美团外卖跳转 URL
    base_url = "https://waimai.meituan.com/"
    
    # 如果有搜索关键词，拼接到 URL
    if keyword:
        encoded_keyword = urllib.parse.quote(keyword)
        url = f"{base_url}?keyword={encoded_keyword}"
    else:
        url = base_url
    
    # 记录用户的饮食偏好（供记忆系统使用）
    if keyword:
        save_diet_log(user_id, f"搜索外卖: {keyword}", 0, 0, 0, 0)
    
    print(f"[Tool] open_meituan: user={user_id}, keyword='{keyword}', url={url}")
    
    return {
        "success": True,
        "action": "open_meituan",
        "meituan_url": url,
        "keyword": keyword,
        "message": f"已为你打开美团外卖{'搜索「' + keyword + '」' if keyword else ''}！🏃",
    }


@tool
def confirm_food_order(
    user_id: str,
    intent: str = "",
    user_message: str = "",
) -> dict:
    """
    请求用户确认是否需要外卖推荐（Human-in-the-loop 触发工具）。

    当 Agent 检测到用户可能想点外卖时（如"我饿了"、"想吃东西"），
    应调用此工具而不是直接搜索外卖。此工具会触发中断确认流程。

    参数说明：
        user_id: 用户唯一标识
        intent: 意图描述（如"hunger"、"food_order"）
        user_message: 用户原始消息

    返回：
        确认结果，如果用户确认则包含下一步指令
    """
    # 这个工具本身不执行任何操作
    # 它的作用是让 Agent 显式表达"我需要确认"的意图
    # 实际的中断由 react_loop 中的 _handle_confirmation 处理
    
    return {
        "success": True,
        "action": "confirm_food_order",
        "requires_confirmation": True,
        "confirmation_message": (
            "🛑 检测到你饿了！作为健身教练，我可以帮你搜索附近适合你"
            "训练目标的健康餐厅。需要我帮你找找看吗？"
        ),
        "next_tool": "search_nearby_food",  # 确认后应执行的下一个工具
    }
```

### 2.3 记忆系统（memory/memory_manager.py 新增）

```python
# ============================================================
# memory_manager.py —— AI Agent 结构化记忆系统
# ============================================================
"""
记忆系统设计 —— 为 Agent 提供跨对话的上下文记忆。

★ 三层记忆架构：

  第 1 层：短期记忆（Short-term）
    - 来源：LangGraph MemorySaver（当前对话线程的消息历史）
    - 生命周期：单次会话 / 服务重启后丢失
    - 用途：当前对话中的上下文理解

  第 2 层：结构化记忆（Structured）★ 新增
    - 来源：SQLite 数据库（持久化存储）
    - 生命周期：永久
    - 用途：训练状态追踪、饮食偏好、用户习惯画像

  第 3 层：会话摘要（Session Summary）★ 新增
    - 来源：每次对话结束后的 LLM 摘要
    - 生命周期：持久化（直到下次生成新摘要）
    - 用途：新对话开始时的快速上下文恢复
"""

import sqlite3
import json
import datetime
from config import DB_PATH


# ----------------------------------------------------------
# 数据库表定义（在 init_db 中创建）
# ----------------------------------------------------------

CREATE_AGENT_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS agent_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- 'training_state' | 'diet_preference' | 'summary'
    memory_key  TEXT NOT NULL,  -- 记忆键（如 'current_phase', 'likes'）
    memory_value TEXT NOT NULL, -- JSON 序列化的记忆值
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, memory_type, memory_key)
)
"""


# ----------------------------------------------------------
# 记忆管理器核心类
# ----------------------------------------------------------

class AgentMemoryManager:
    """
    Agent 记忆管理器 —— 读写结构化记忆。
    
    使用方式：
        memory = AgentMemoryManager(user_id)
        
        # 写入记忆
        memory.set_training_state({"phase": "减脂期", "last_workout": "深蹲 5x5"})
        memory.set_diet_preference("likes", ["鸡胸肉", "西兰花", "糙米"])
        memory.set_diet_preference("dislikes", ["香菜", "苦瓜"])
        
        # 读取记忆
        context = memory.get_memory_context()
        # → 返回格式化的 Prompt 片段，可直接注入 System Prompt
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
    
    # ---- 训练状态记忆 ----
    
    def set_training_state(self, state: dict):
        """设置/更新训练状态（当前阶段、最近训练、本周计划等）"""
        self._upsert("training_state", "current", json.dumps(state, ensure_ascii=False))
    
    def get_training_state(self) -> dict:
        """获取训练状态"""
        return self._get("training_state", "current")
    
    # ---- 饮食偏好记忆 ----
    
    def set_diet_preference(self, key: str, value):
        """设置饮食偏好（喜欢的食物、忌口、过敏源等）"""
        self._upsert("diet_preference", key, json.dumps(value, ensure_ascii=False))
    
    def get_diet_preferences(self) -> dict:
        """获取所有饮食偏好"""
        prefs = {}
        rows = self._get_all("diet_preference")
        for row in rows:
            try:
                prefs[row["memory_key"]] = json.loads(row["memory_value"])
            except json.JSONDecodeError:
                prefs[row["memory_key"]] = row["memory_value"]
        return prefs
    
    # ---- 会话摘要 ----
    
    def set_summary(self, summary: str):
        """设置会话摘要"""
        self._upsert("summary", "latest", summary)
    
    def get_summary(self) -> str:
        """获取最新的会话摘要"""
        result = self._get("summary", "latest")
        return result.get("summary", "") if result else ""
    
    # ---- 构建记忆上下文（注入 System Prompt） ----
    
    def get_memory_context(self) -> str:
        """
        构建完整的记忆上下文，作为 Prompt 片段返回。
        
        返回格式：
            # ★ 你的记忆（跨会话持久化）
            
            ## 训练状态
            - 当前阶段：减脂期
            - 最近训练：2024-06-24 深蹲 5x5, 卧推 3x10
            
            ## 饮食偏好
            - 喜欢：鸡胸肉, 西兰花, 糙米
            - 不喜欢：香菜, 苦瓜
            
            ## 上次对话摘要
            用户询问了减脂期的蛋白质摄入量...
        """
        training_state = self.get_training_state()
        diet_prefs = self.get_diet_preferences()
        summary = self.get_summary()
        
        parts = ["\n# ★ 你的记忆（跨会话持久化）\n"]
        
        # 训练状态
        if training_state:
            parts.append("## 训练状态")
            for key, value in training_state.items():
                parts.append(f"- {key}: {value}")
            parts.append("")
        
        # 饮食偏好
        if diet_prefs:
            parts.append("## 饮食偏好")
            for key, values in diet_prefs.items():
                if isinstance(values, list):
                    parts.append(f"- {key}: {', '.join(str(v) for v in values)}")
                else:
                    parts.append(f"- {key}: {values}")
            parts.append("")
        
        # 会话摘要
        if summary:
            parts.append("## 上次对话摘要")
            parts.append(summary)
            parts.append("")
        
        if len(parts) == 1:  # 只有标题，无记忆
            return ""
        
        return "\n".join(parts)
    
    # ---- 自动记忆更新（由 Agent 在对话中触发） ----
    
    def auto_update_from_conversation(self, user_message: str, ai_response: str):
        """
        从对话中自动提取并更新记忆。
        
        识别规则（可扩展）：
          - 用户提到"今天练了X" → 更新训练状态
          - 用户提到"我不喜欢吃X" → 更新饮食偏好
          - 用户提到"我最近在X期" → 更新训练阶段
        """
        # 简单的关键词触发更新
        msg_lower = user_message.lower()
        
        # 检测训练状态更新
        training_keywords = {
            "减脂": "减脂期",
            "增肌": "增肌期", 
            "维持": "维持期",
            "今天练了": None,  # 提取具体训练内容
            "刚练完": None,
            "完成了": None,
        }
        
        for kw, phase in training_keywords.items():
            if kw in user_message:
                state = self.get_training_state() or {}
                if phase:
                    state["current_phase"] = phase
                state["last_activity"] = user_message[:100]
                state["last_updated"] = datetime.datetime.now().isoformat()
                self.set_training_state(state)
                break
        
        # 检测饮食偏好更新
        if "不喜欢" in msg_lower or "不吃" in msg_lower or "过敏" in msg_lower:
            dislikes = self.get_diet_preferences().get("dislikes", [])
            # 简单提取（生产环境可用 NER）
            dislikes.append(user_message[:50])
            self.set_diet_preference("dislikes", dislikes[:10])  # 保留最近 10 条
        
        if "喜欢吃" in msg_lower or "爱吃" in msg_lower or "喜欢" in msg_lower:
            likes = self.get_diet_preferences().get("likes", [])
            likes.append(user_message[:50])
            self.set_diet_preference("likes", likes[:10])
    
    # ---- 内部方法 ----
    
    def _upsert(self, memory_type: str, memory_key: str, memory_value: str):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO agent_memory (user_id, memory_type, memory_key, memory_value, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, memory_type, memory_key)
            DO UPDATE SET memory_value = excluded.memory_value, updated_at = CURRENT_TIMESTAMP
        """, (self.user_id, memory_type, memory_key, memory_value))
        conn.commit()
        conn.close()
    
    def _get(self, memory_type: str, memory_key: str) -> dict:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT memory_value FROM agent_memory WHERE user_id=? AND memory_type=? AND memory_key=?",
            (self.user_id, memory_type, memory_key)
        ).fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row["memory_value"])
            except json.JSONDecodeError:
                return row["memory_value"]
        return {}
    
    def _get_all(self, memory_type: str) -> list:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT memory_key, memory_value FROM agent_memory WHERE user_id=? AND memory_type=?",
            (self.user_id, memory_type)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
```

### 2.4 Memory_Loader 节点（langgraph_brain.py 修改）

```python
# ============================================================
# ★ Memory_Loader 节点（新增 —— 在 Context_Loader 之后执行）
# ============================================================

def memory_loader_node(state: FitnessState) -> dict:
    """
    Memory_Loader —— 记忆加载节点
    
    从结构化记忆存储（agent_memory 表）中加载用户的：
      1. 训练状态（当前阶段、最近训练、本周计划）
      2. 饮食偏好（喜欢的食物、忌口、过敏源）
      3. 上次对话摘要
    
    将记忆上下文注入 system_prompt，让 Agent 在对话中利用这些信息。
    
    执行时机：Context_Loader 之后、Safety_Gate 之前
    """
    from memory.memory_manager import AgentMemoryManager
    
    user_id = state["user_id"]
    memory = AgentMemoryManager(user_id)
    memory_context = memory.get_memory_context()
    
    if memory_context:
        # 将记忆上下文追加到 system_prompt
        current_prompt = state.get("system_prompt", "")
        enhanced_prompt = current_prompt + "\n\n" + memory_context + """
# ★ 记忆使用规则
- 请自然地引用上述记忆中的信息，让用户感受到你记得他/她。
- 如果用户更新了训练状态或饮食偏好，请调用 save_user_profile 和记忆更新工具。
- 不要机械地复述记忆内容，要自然地融入回复中。
"""
        print(f"[LangGraph] Memory_Loader: 已加载用户 {user_id} 的结构化记忆")
        return {
            "system_prompt": enhanced_prompt,
            "memory_context": memory_context,
        }
    else:
        print(f"[LangGraph] Memory_Loader: 用户 {user_id} 无结构化记忆")
        return {}
```

---

## 三、完整集成方案

### 3.1 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `backend/react_loop.py` | **新增** | ReAct 控制循环核心 |
| `backend/memory/memory_manager.py` | **新增** | 结构化记忆管理器 |
| `backend/langgraph_brain.py` | **修改** | 添加 Memory_Loader 节点、移除 FoodOrder_Guard |
| `backend/tools.py` | **修改** | 添加 `open_meituan()` 和 `confirm_food_order()` |
| `backend/main.py` | **修改** | 适配新的 Agent 循环（Thought 事件流） |
| `backend/database.py` | **修改** | 添加 `agent_memory` 表初始化 |
| `backend/static/index.html` | **修改** | 前端展示「💭 Thinking」气泡 |

### 3.2 新的 main.py SSE 事件流

```
旧版事件流:
  data: {"thinking": "正在分析..."}
  data: {"token": "好的"}
  data: {"token": "..."}
  data: {"confirmation_required": {...}}   ← 外卖确认
  data: {"meituan_url": "https://..."}     ← 美团跳转
  data: [DONE]

新版事件流:
  data: {"thought": "用户说饿了，作为健身教练我应该..."}  ← ★ 新增
  data: {"thinking": "正在确认..."}
  data: {"confirmation_required": {...}}                ← 确认弹窗
  data: [DONE]
  ↓ 用户确认后 resume
  data: {"thought": "用户已确认，搜索附近健康餐厅..."}    ← ★ 新增
  data: {"thinking": "正在搜索附近餐厅..."}
  data: {"token": "好的！为你找到..."}                   ← 流式回复
  data: {"meituan_url": "https://..."}                  ← open_meituan 结果
  data: [DONE]
```

### 3.3 用户交互流程对比

#### 旧版流程（硬编码）
```
用户: "我饿了"
  → FoodOrder_Guard 关键词匹配 "饿了"
  → 强制 interrupt() 弹确认窗
  → 用户点确认
  → 注入 FOOD_SEARCH_PROMPT
  → LLM 调用 search_nearby_food
  → main.py 硬编码发送 meituan_url
```

#### 新版流程（Agent 自主决策）
```
用户: "我饿了"
  → Agent Thought: "用户表达了饥饿感。作为健身教练，我需要先确认
     他是否想找吃的，然后根据他的健身目标推荐合适的食物。"
  → Agent Action: 调用 confirm_food_order(confirm_type="hunger")
  → react_loop 检测到确认请求 → interrupt()
  → 前端弹窗：需要帮你找找吃的吗？
  → 用户点确认 → resume
  → Agent Thought: "用户确认需要。他的目标是减脂，我来搜索低卡健康餐。"
  → Agent Action: 调用 search_nearby_food(keyword="", goal="lose_weight")
  → Agent Observation: 收到 5 家餐厅结果
  → Agent Thought: "有 5 家餐厅符合他的需求，我来组织回复..."
  → Agent Action: final_response（流式输出推荐列表）
  → 用户: "打开美团"
  → Agent Thought: "用户想打开美团外卖。"
  → Agent Action: 调用 open_meituan(keyword="沙拉轻食")
  → main.py 发送 meituan_url SSE 事件
```

---

## 四、实施路线图

### Phase 1：核心 Agent 循环（2-3 天）

1. 创建 `backend/react_loop.py`，实现基础 ReAct 循环
2. 创建 `backend/memory/memory_manager.py`，实现结构化记忆
3. 修改 `backend/langgraph_brain.py`：
   - 添加 Memory_Loader 节点
   - 将 Agent 节点改为调用 react_loop
   - 暂时保留 FoodOrder_Guard（作为兜底）
4. 修改 `backend/database.py`，添加 agent_memory 表
5. 测试基本对话是否正常

### Phase 2：工具系统升级（1-2 天）

6. 在 `backend/tools.py` 添加 `open_meituan()` 工具
7. 添加 `confirm_food_order()` 触发工具
8. 修改 `backend/main.py`：
   - 适配 ReAct 的 Thought 事件流
   - 处理 `open_meituan()` 工具结果 → 发送 meituan_url 事件

### Phase 3：前端适配（1-2 天）

9. 修改 `backend/static/index.html`：
   - 添加「💭 Thought」气泡展示
   - 处理新的 SSE 事件类型
   - 保持确认弹窗的兼容性

### Phase 4：迁移与清理（1 天）

10. 移除旧的 `FoodOrder_Guard` 节点（如果新版稳定）
11. 移除 `food_order_guard.py` 中的硬编码关键词逻辑
12. 全面回归测试

---

## 五、关键设计决策

### 5.1 为什么不用 bind_tools() 的隐式 tool_calls？

旧版使用 `ChatTongyi.bind_tools()` 让 LLM 自动决策工具调用。这在简单场景下很好用，但有两个致命问题：

1. **思考不可见**：LLM 的推理过程和工具调用决策混在一起，前端只能看到最终结果
2. **无法强制思考**：LLM 可能跳过推理直接调用工具（如直接 `search_nearby_food` 而不是先 `confirm_food_order`）

新版使用**结构化 JSON 输出**强制 LLM 先输出 Thought 再输出 Action，使思考过程对用户可见。

### 5.2 渐进式迁移策略

- **Phase 1-2**：新旧并行（保留 FoodOrder_Guard 作兜底）
- **Phase 3**：以 ReAct 为主，旧逻辑作为 fallback
- **Phase 4**：完全切换到 ReAct

这样可以在不破坏现有功能的前提下逐步验证新架构。

### 5.3 记忆系统的设计哲学

- **写入自动化**：Agent 在对话中自动更新记忆，用户无感知
- **读取即注入**：Memory_Loader 将记忆转为 Prompt 片段，LLM 自然利用
- **三层分离**：短期（MemorySaver）/ 结构化（SQLite）/ 摘要（LLM 生成），各司其职
