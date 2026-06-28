# -*- coding: utf-8 -*-
"""
memory_manager.py —— AI Agent 结构化记忆系统
================================================
为 Agent 提供跨对话的上下文记忆，实现"越聊越懂你"的能力。

★ 三层记忆架构：

  第 1 层：短期记忆（Short-term）
    - 来源：LangGraph MemorySaver（当前对话线程的消息历史）
    - 生命周期：单次会话 / 服务重启后丢失
    - 用途：当前对话中的上下文理解

  第 2 层：结构化记忆（Structured）★ 本模块实现
    - 来源：SQLite agent_memory 表（持久化存储）
    - 生命周期：永久
    - 用途：训练状态追踪、饮食偏好、用户习惯画像

  第 3 层：会话摘要（Session Summary）
    - 来源：每次对话结束后的 LLM 摘要
    - 生命周期：持久化（直到下次生成新摘要）
    - 用途：新对话开始时的快速上下文恢复

使用方式：
    memory = AgentMemoryManager(user_id)

    # 写入记忆
    memory.set_training_state({"phase": "减脂期", "last_workout": "深蹲 5x5"})
    memory.set_diet_preference("likes", ["鸡胸肉", "西兰花", "糙米"])
    memory.set_summary("用户上次询问了减脂期的蛋白质摄入量...")

    # 读取并构建上下文
    context = memory.get_memory_context()
    # → 返回格式化的 Prompt 片段，可直接注入 System Prompt
"""

import sqlite3
import json
import datetime
from config import DB_PATH


# ============================================================
# 记忆管理器核心类
# ============================================================

class AgentMemoryManager:
    """
    Agent 记忆管理器 —— 读写结构化记忆。

    所有读写操作都通过 agent_memory 表进行，
    支持 upsert（存在则更新，不存在则插入）。
    """

    def __init__(self, user_id: str):
        self.user_id = user_id

    # ---- 训练状态记忆 ----

    def set_training_state(self, state: dict):
        """设置/更新训练状态（当前阶段、最近训练、本周计划等）"""
        self._upsert("training_state", "current", json.dumps(state, ensure_ascii=False))

    def get_training_state(self) -> dict:
        """获取训练状态"""
        result = self._get("training_state", "current")
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return {}
        return result if result else {}

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
            except (json.JSONDecodeError, TypeError):
                prefs[row["memory_key"]] = row["memory_value"]
        return prefs

    # ---- 会话摘要 ----

    def set_summary(self, summary: str):
        """设置会话摘要"""
        self._upsert("summary", "latest", json.dumps({"summary": summary}, ensure_ascii=False))

    def get_summary(self) -> str:
        """获取最新的会话摘要"""
        result = self._get("summary", "latest")
        if isinstance(result, dict):
            return result.get("summary", "")
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return parsed.get("summary", "") if isinstance(parsed, dict) else result
            except (json.JSONDecodeError, TypeError):
                return result
        return ""

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
            - likes: 鸡胸肉, 西兰花, 糙米
            - dislikes: 香菜, 苦瓜

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
                if isinstance(value, list):
                    parts.append(f"- {key}: {', '.join(str(v) for v in value)}")
                else:
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

        if len(parts) == 1:  # 只有标题，无实际记忆
            return ""

        return "\n".join(parts)

    # ---- 自动记忆更新（由 Agent 在对话中触发） ----

    def auto_update_from_conversation(self, user_message: str, ai_response: str):
        """
        从对话中自动提取并更新记忆。

        识别规则：
          - 用户提到"今天练了X" / "刚练完" → 更新训练状态
          - 用户提到"减脂" / "增肌" / "维持" → 更新训练阶段
          - 用户提到"不喜欢吃X" / "爱吃X" → 更新饮食偏好

        ★ 注意：此方法在当前版本使用简单关键词匹配。
        未来可以接入 LLM 做更智能的意图提取。
        """
        msg_lower = user_message.lower()

        # 检测训练状态更新
        training_phase_map = {
            "减脂": "减脂期",
            "增肌": "增肌期",
            "维持": "维持期",
        }

        updated = False
        state = self.get_training_state() or {}

        for kw, phase in training_phase_map.items():
            if kw in user_message:
                state["current_phase"] = phase
                updated = True

        # 检测训练完成
        training_triggers = ["今天练了", "刚练完", "完成了", "打卡", "练完了", "已经跑了", "做了"]
        for trigger in training_triggers:
            if trigger in user_message:
                state["last_activity"] = user_message[:100]
                updated = True
                break

        if updated:
            state["last_updated"] = datetime.datetime.now().isoformat()
            self.set_training_state(state)
            print(f"[Memory] 自动更新训练状态: user={self.user_id}, phase={state.get('current_phase', 'N/A')}")

        # 检测饮食偏好更新
        if "不喜欢" in msg_lower or "不吃" in msg_lower or "过敏" in msg_lower:
            dislikes = self.get_diet_preferences().get("dislikes", [])
            food_mention = user_message[:80].strip()
            if food_mention and food_mention not in dislikes:
                dislikes.append(food_mention)
                self.set_diet_preference("dislikes", dislikes[:10])  # 保留最近 10 条
                print(f"[Memory] 自动更新饮食偏好(dislikes): user={self.user_id}")

        if "喜欢吃" in msg_lower or "爱吃" in msg_lower or "喜欢" in msg_lower:
            likes = self.get_diet_preferences().get("likes", [])
            food_mention = user_message[:80].strip()
            if food_mention and food_mention not in likes:
                likes.append(food_mention)
                self.set_diet_preference("likes", likes[:10])
                print(f"[Memory] 自动更新饮食偏好(likes): user={self.user_id}")

    # ---- 内部数据库方法 ----

    def _upsert(self, memory_type: str, memory_key: str, memory_value: str):
        """插入或更新一条记忆记录"""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO agent_memory (user_id, memory_type, memory_key, memory_value, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, memory_type, memory_key)
            DO UPDATE SET memory_value = excluded.memory_value, updated_at = CURRENT_TIMESTAMP
        """, (self.user_id, memory_type, memory_key, memory_value))
        conn.commit()
        conn.close()

    def _get(self, memory_type: str, memory_key: str):
        """获取一条记忆记录的值"""
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
            except (json.JSONDecodeError, TypeError):
                return row["memory_value"]
        return {}

    def _get_all(self, memory_type: str) -> list:
        """获取某一类型的所有记忆记录"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT memory_key, memory_value FROM agent_memory WHERE user_id=? AND memory_type=?",
            (self.user_id, memory_type)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
