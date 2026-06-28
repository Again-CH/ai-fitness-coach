# -*- coding: utf-8 -*-
"""
database.py —— SQLite 数据库操作层
====================================
职责：
  1. 初始化数据库表结构（users 表 + user_profiles 表 + chat_history 表 + 日志表）
  2. 提供用户鉴权相关函数（create_user / verify_sms_code / get_user_by_phone）
  3. 提供保存用户身体数据的函数（save_profile / update_profile）
  4. 提供查询用户身体数据的函数（get_profile）—— 供 LangGraph 的 Context_Loader 调用
  5. 提供对话历史存储函数（save_chat_message / get_chat_history）

表结构：
  users（用户鉴权表）：
  ┌─────────────┬──────────┬──────────────────────────┐
  │ 字段名      │ 类型     │ 说明                     │
  ├─────────────┼──────────┼──────────────────────────┤
  │ id          │ INTEGER  │ 自增主键                 │
  │ phone       │ TEXT     │ 手机号（唯一，必填）     │
  │ sms_code   │ TEXT     │ 验证码（6位数字）        │
  │ code_expire │ TIMESTAMP│ 验证码过期时间           │
  │ created_at  │ TIMESTAMP│ 创建时间（自动）         │
  └─────────────┴──────────┴──────────────────────────┘

  user_profiles（用户画像表）：
  ┌─────────────┬──────────┬──────────────────────────┐
  │ 字段名      │ 类型     │ 说明                     │
  ├─────────────┼──────────┼──────────────────────────┤
  │ user_id     │ TEXT(PK) │ 用户唯一标识（关联 users.id）│
  │ name        │ TEXT     │ 用户姓名（可选）         │
  │ height      │ REAL     │ 身高 (cm)               │
  │ weight      │ REAL     │ 体重 (kg)               │
  │ age         │ INTEGER  │ 年龄                    │
  │ gender      │ TEXT     │ 性别（"男" / "女"）     │
  │ goal        │ TEXT     │ 健身目标                 │
  │ created_at  │ TIMESTAMP│ 创建时间（自动）         │
  │ updated_at  │ TIMESTAMP│ 更新时间（自动）         │
  └─────────────┴──────────┴──────────────────────────┘

  chat_history（对话历史表）：
  ┌─────────────┬──────────┬──────────────────────────┐
  │ 字段名      │ 类型     │ 说明                     │
  ├─────────────┼──────────┼──────────────────────────┤
  │ id          │ INTEGER  │ 自增主键                 │
  │ user_id     │ TEXT     │ 用户唯一标识             │
  │ role        │ TEXT     │ 消息角色（user/assistant）│
  │ content     │ TEXT     │ 消息内容                 │
  │ timestamp    │ TIMESTAMP│ 消息时间（自动）         │
  └─────────────┴──────────┴──────────────────────────┘
"""

import sqlite3
import random
import datetime
from config import DB_PATH


# ============================================================
# 数据库连接工具函数
# ============================================================
def get_connection() -> sqlite3.Connection:
    """
    获取 SQLite 数据库连接。
    使用 row_factory = sqlite3.Row，使得查询结果可以通过列名访问，
    例如 row["height"] 而不是 row[0]，代码更可读。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 数据库初始化
# ============================================================
def init_db():
    """
    初始化数据库：创建 users 表、user_profiles 表、chat_history 表和其他日志表（如果不存在）。
    此函数在 FastAPI 应用启动时自动调用。
    """
    conn = get_connection()
    
    # 创建 users 表（用户鉴权表：手机号+验证码登录）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            phone          TEXT UNIQUE NOT NULL,
            sms_code      TEXT,
            code_expire    TIMESTAMP,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ★ 检查并添加 users 表的新字段（如果表已存在但缺少字段）
    cursor = conn.execute("PRAGMA table_info(users)")
    user_columns = [row[1] for row in cursor.fetchall()]

    if "sms_code" not in user_columns:
        conn.execute("ALTER TABLE users ADD COLUMN sms_code TEXT")
        print("[database] 已添加 sms_code 字段到 users 表")

    if "code_expire" not in user_columns:
        conn.execute("ALTER TABLE users ADD COLUMN code_expire TIMESTAMP")
        print("[database] 已添加 code_expire 字段到 users 表")

    # 创建 user_profiles 表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id     TEXT PRIMARY KEY,
            name        TEXT,
            height      REAL,
            weight      REAL,
            age         INTEGER,
            gender      TEXT,
            goal        TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 检查并添加新字段（如果表已存在但缺少字段）
    # SQLite 不支持 ALTER TABLE 添加多列，需要逐列检查
    cursor = conn.execute("PRAGMA table_info(user_profiles)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "name" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN name TEXT")
        print("[database] 已添加 name 字段到 user_profiles 表")
    
    if "goal" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN goal TEXT")
        print("[database] 已添加 goal 字段到 user_profiles 表")
    
    # 创建 chat_history 表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ★ 体重日志表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weight_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            weight      REAL NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ★ 运动日志表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exercise_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT NOT NULL,
            exercise_type  TEXT,
            duration       INTEGER,
            recorded_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ★ 饮食日志表（含营养成分）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS diet_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            food_name    TEXT,
            calories     REAL,
            protein      REAL DEFAULT 0,
            carb         REAL DEFAULT 0,
            fat          REAL DEFAULT 0,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ★ 打卡记录表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkin_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT NOT NULL,
            checkin_type  TEXT DEFAULT 'daily',
            checkin_time  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ★★★ Agent 结构化记忆表（AI Agent 架构升级） ★★★
    # 三层记忆架构的第 2 层：结构化记忆（永久持久化）
    # 存储训练状态、饮食偏好、会话摘要等结构化数据
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT NOT NULL,
            memory_type   TEXT NOT NULL,
            memory_key    TEXT NOT NULL,
            memory_value  TEXT NOT NULL,
            updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, memory_type, memory_key)
        )
    """)

    # ★ Phase 2：训练日志表（记录每次训练的具体动作）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL,
            plan_name       TEXT,
            exercise_name   TEXT NOT NULL,
            muscle_group    TEXT,
            sets_done       INTEGER DEFAULT 0,
            reps_done       TEXT,
            weight_kg       REAL DEFAULT 0,
            notes           TEXT,
            recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ★ Phase 2：训练计划存储表（存储 AI 生成的个性化计划）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_plans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            plan_name   TEXT NOT NULL,
            plan_json   TEXT NOT NULL,
            is_active   INTEGER DEFAULT 1,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ★ Phase 2：里程碑记录表（记录用户达成的成就里程碑）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS milestones (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL,
            milestone_type  TEXT NOT NULL,
            milestone_key   TEXT NOT NULL,
            description     TEXT,
            achieved_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, milestone_key)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[database] 数据库初始化完成 → {DB_PATH}")


# ============================================================
# 保存用户身体数据（供 POST /api/save_profile 调用）
# ============================================================
def save_profile(user_id: str, height: float, weight: float, age: int, gender: str, name: str = None, goal: str = None):
    """
    保存或更新用户身体数据。

    使用 SQLite 的 INSERT ... ON CONFLICT 语法实现"存在则更新，不存在则插入"。
    （SQLite 3.24+ 支持，Python 3.13 自带的 sqlite3 完全兼容）

    参数:
        user_id: 用户唯一标识
        height:  身高 (cm)
        weight:  体重 (kg)
        age:     年龄
        gender:  性别（"男" 或 "女"）
        name:    用户姓名（可选）
        goal:    健身目标（可选）
    """
    conn = get_connection()
    conn.execute("""
        INSERT INTO user_profiles (user_id, name, height, weight, age, gender, goal)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name       = COALESCE(excluded.name, name),
            height     = COALESCE(excluded.height, height),
            weight     = COALESCE(excluded.weight, weight),
            age        = COALESCE(excluded.age, age),
            gender     = COALESCE(excluded.gender, gender),
            goal       = COALESCE(excluded.goal, goal),
            updated_at = CURRENT_TIMESTAMP
    """, (user_id, name, height, weight, age, gender, goal))
    conn.commit()
    conn.close()
    print(f"[database] 用户 {user_id} 身体数据已保存")


# ============================================================
# 更新用户画像（供 AI 工具调用）
# ============================================================
def update_profile(user_id: str, **kwargs):
    """
    更新用户画像的部分字段。

    参数:
        user_id: 用户唯一标识
        **kwargs: 要更新的字段（name, height, weight, age, gender, goal）

    返回:
        更新后的用户画像字典，如果用户不存在则返回 None
    """
    if not kwargs:
        return get_profile(user_id)
    
    conn = get_connection()
    
    # 构建动态 UPDATE 语句
    set_clauses = []
    values = []
    for key, value in kwargs.items():
        if key in ["name", "height", "weight", "age", "gender", "goal"]:
            set_clauses.append(f"{key} = ?")
            values.append(value)
    
    if set_clauses:
        values.append(user_id)
        sql = f"UPDATE user_profiles SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?"
        conn.execute(sql, values)
        conn.commit()
        print(f"[database] 用户 {user_id} 画像已更新: {kwargs}")
    
    conn.close()
    return get_profile(user_id)


# ============================================================
# 保存对话消息（供 AI 工具调用）
# ============================================================
def save_chat_message(user_id: str, role: str, content: str):
    """
    保存一条对话消息到 chat_history 表。

    参数:
        user_id: 用户唯一标识
        role:    消息角色（"user" 或 "assistant"）
        content: 消息内容
    """
    conn = get_connection()
    conn.execute("""
        INSERT INTO chat_history (user_id, role, content)
        VALUES (?, ?, ?)
    """, (user_id, role, content))
    conn.commit()
    conn.close()


# ============================================================
# 获取对话历史（供 AI 工具调用）
# ============================================================
def get_chat_history(user_id: str, limit: int = 50) -> list:
    """
    获取用户的对话历史记录。

    参数:
        user_id: 用户唯一标识
        limit:   返回的最大消息条数（默认 50 条）

    返回:
        消息列表，每条消息为字典 {"role": "...", "content": "..."}
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT role, content, timestamp FROM chat_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()
    
    # 按时间正序返回
    history = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
    return history


# ============================================================
# 查询用户身体数据（供 LangGraph Context_Loader 调用）
# ============================================================
def get_profile(user_id: str) -> dict | None:
    """
    根据 user_id 查询用户身体数据。

    ★★★ 这是 LangGraph Context_Loader 节点的核心数据来源 ★★★
    当用户发起对话时，LangGraph 会调用此函数从数据库读取用户身体数据，
    然后将数据拼接成 System Prompt 注入给大模型。

    参数:
        user_id: 用户唯一标识

    返回:
        如果找到用户数据，返回字典:
            {"user_id": "...", "height": 175, "weight": 85, "age": 30, "gender": "男", ...}
        如果未找到，返回 None
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    if row:
        # 将 sqlite3.Row 转换为普通字典
        profile = dict(row)
        print(f"[database] 查到用户 {user_id} 的身体数据: {profile}")
        return profile
    else:
        print(f"[database] 未找到用户 {user_id} 的身体数据")
        return None


# ============================================================
# ★ 体重日志
# ============================================================
def save_weight_log(user_id: str, weight: float):
    """保存一条体重记录"""
    conn = get_connection()
    conn.execute(
        "INSERT INTO weight_logs (user_id, weight) VALUES (?, ?)",
        (user_id, weight),
    )
    conn.commit()
    conn.close()
    print(f"[database] 体重记录已保存: user={user_id}, weight={weight}")


def get_weight_logs(user_id: str, days: int = 30) -> list:
    """获取最近 N 天的体重记录，返回 [{"date": "2024-01-01", "weight": 65.0}, ...]"""
    conn = get_connection()
    rows = conn.execute("""
        SELECT DATE(recorded_at) as log_date, weight
        FROM weight_logs
        WHERE user_id = ? AND recorded_at >= DATE('now', ?)
        ORDER BY log_date ASC
    """, (user_id, f"-{days} days")).fetchall()
    conn.close()
    return [{"date": row[0], "weight": row[1]} for row in rows]


# ============================================================
# ★ 运动日志
# ============================================================
def save_exercise_log(user_id: str, exercise_type: str, duration: int):
    """保存一条运动记录（duration 单位：分钟）"""
    conn = get_connection()
    conn.execute(
        "INSERT INTO exercise_logs (user_id, exercise_type, duration) VALUES (?, ?, ?)",
        (user_id, exercise_type, duration),
    )
    conn.commit()
    conn.close()
    print(f"[database] 运动记录已保存: user={user_id}, type={exercise_type}, duration={duration}min")


def get_exercise_logs(user_id: str, days: int = 7) -> list:
    """获取最近 N 天的运动记录，按日期聚合，返回 [{"date": "2024-01-01", "duration": 60}, ...]"""
    conn = get_connection()
    rows = conn.execute("""
        SELECT DATE(recorded_at) as log_date, SUM(duration) as total_duration
        FROM exercise_logs
        WHERE user_id = ? AND recorded_at >= DATE('now', ?)
        GROUP BY log_date
        ORDER BY log_date ASC
    """, (user_id, f"-{days} days")).fetchall()
    conn.close()
    return [{"date": row[0], "duration": row[1]} for row in rows]


# ============================================================
# ★ 饮食日志
# ============================================================
def save_diet_log(user_id: str, food_name: str, calories: float, protein: float = 0, carb: float = 0, fat: float = 0):
    """保存一条饮食记录（含营养成分）"""
    conn = get_connection()
    conn.execute(
        "INSERT INTO diet_logs (user_id, food_name, calories, protein, carb, fat) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, food_name, calories, protein, carb, fat),
    )
    conn.commit()
    conn.close()
    print(f"[database] 饮食记录已保存: user={user_id}, food={food_name}, calories={calories}, protein={protein}, carb={carb}, fat={fat}")


def get_diet_logs(user_id: str, date: str = None) -> dict:
    """获取某天的饮食记录，date 格式 YYYY-MM-DD，默认今天。
    返回包含 logs、total_calories、total_protein、total_carb、total_fat 的字典。"""
    if date is None:
        import datetime
        date = datetime.date.today().isoformat()
    conn = get_connection()
    rows = conn.execute("""
        SELECT food_name, calories, protein, carb, fat, TIME(recorded_at) as log_time
        FROM diet_logs
        WHERE user_id = ? AND DATE(recorded_at) = ?
        ORDER BY recorded_at ASC
    """, (user_id, date)).fetchall()
    conn.close()
    total_calories = sum(row[1] for row in rows) if rows else 0
    total_protein  = sum(row[2] for row in rows) if rows else 0
    total_carb     = sum(row[3] for row in rows) if rows else 0
    total_fat      = sum(row[4] for row in rows) if rows else 0
    return {
        "logs": [{"food": row[0], "calories": row[1], "protein": row[2], "carb": row[3], "fat": row[4], "time": row[5]} for row in rows],
        "total_calories": total_calories,
        "total_protein": total_protein,
        "total_carb": total_carb,
        "total_fat": total_fat,
    }


# ============================================================
# ★ 目标设定
# ============================================================
def get_goal_settings(user_id: str) -> dict:
    """获取用户的目标设定（从 user_profiles 表读取 goal 字段，解析为结构化数据）"""
    conn = get_connection()
    row = conn.execute(
        "SELECT goal FROM user_profiles WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if row and row[0]:
        import json
        try:
            return json.loads(row[0])
        except Exception:
            return {"calorie_target": 2000, "exercise_target": 60}
    return {"calorie_target": 2000, "exercise_target": 60}


def save_goal_settings(user_id: str, calorie_target: int, exercise_target: int):
    """保存用户的目标设定"""
    import json
    goal_json = json.dumps({"calorie_target": calorie_target, "exercise_target": exercise_target})
    conn = get_connection()
    conn.execute(
        "UPDATE user_profiles SET goal = ? WHERE user_id = ?",
        (goal_json, user_id),
    )
    conn.commit()
    conn.close()
    print(f"[database] 目标设定已保存: user={user_id}, calorie={calorie_target}, exercise={exercise_target}min")


# ============================================================
# ★ 打卡记录
# ============================================================
def save_checkin(user_id: str, checkin_type: str = "daily"):
    """保存一条打卡记录（checkin_type: daily=每日打卡, exercise=运动打卡, diet=饮食打卡）"""
    conn = get_connection()
    conn.execute(
        "INSERT INTO checkin_logs (user_id, checkin_type) VALUES (?, ?)",
        (user_id, checkin_type),
    )
    conn.commit()
    conn.close()
    print(f"[database] 打卡记录已保存: user={user_id}, type={checkin_type}")


def get_checkin_logs(user_id: str, days: int = 30) -> list:
    """获取最近 N 天的打卡记录，返回 [{"date": "2024-01-01", "checkin_type": "daily"}, ...]"""
    conn = get_connection()
    rows = conn.execute("""
        SELECT DATE(checkin_time) as log_date, checkin_type
        FROM checkin_logs
        WHERE user_id = ? AND checkin_time >= DATE('now', ?)
        ORDER BY log_date ASC
    """, (user_id, f"-{days} days")).fetchall()
    conn.close()
    return [{"date": row[0], "checkin_type": row[1]} for row in rows]


def get_checkin_streak(user_id: str) -> dict:
    """计算连续打卡天数和总打卡天数"""
    import datetime
    
    conn = get_connection()
    # 获取所有打卡记录（最近 365 天）
    rows = conn.execute("""
        SELECT DISTINCT DATE(checkin_time) as log_date
        FROM checkin_logs
        WHERE user_id = ? AND checkin_time >= DATE('now', '-365 days')
        ORDER BY log_date DESC
    """, (user_id,)).fetchall()
    conn.close()
    
    if not rows:
        return {"current_streak": 0, "total_days": 0, "last_checkin": None}
    
    # 计算连续打卡天数
    dates = [row[0] for row in rows]
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    
    # 如果今天或昨天没有打卡，连续打卡中断
    if dates[0] != today and dates[0] != yesterday:
        current_streak = 0
    else:
        current_streak = 1
        for i in range(1, len(dates)):
            expected_date = (datetime.date.fromisoformat(dates[i-1]) - datetime.timedelta(days=1)).isoformat()
            if dates[i] == expected_date:
                current_streak += 1
            else:
                break
    
    return {
        "current_streak": current_streak,
        "total_days": len(dates),
        "last_checkin": dates[0] if dates else None,
    }


def get_achievements(user_id: str) -> list:
    """获取用户已获得的成就列表"""
    streak = get_checkin_streak(user_id)
    achievements = []
    
    # 根据连续打卡天数颁发成就
    current_streak = streak["current_streak"]
    total_days = streak["total_days"]
    
    if total_days >= 1:
        achievements.append({"id": "first_checkin", "name": "初次打卡", "desc": "完成第一次打卡", "icon": "🎯"})
    if current_streak >= 3:
        achievements.append({"id": "streak_3", "name": "三天打鱼", "desc": "连续打卡 3 天", "icon": "🔥"})
    if current_streak >= 7:
        achievements.append({"id": "streak_7", "name": "自律达人", "desc": "连续打卡 7 天", "icon": "💪"})
    if current_streak >= 14:
        achievements.append({"id": "streak_14", "name": "半月坚持", "desc": "连续打卡 14 天", "icon": "⭐"})
    if current_streak >= 30:
        achievements.append({"id": "streak_30", "name": "月度冠军", "desc": "连续打卡 30 天", "icon": "🏆"})
    if total_days >= 50:
        achievements.append({"id": "total_50", "name": "健身爱好者", "desc": "累计打卡 50 天", "icon": "🥇"})
    if total_days >= 100:
        achievements.append({"id": "total_100", "name": "百日坚持", "desc": "累计打卡 100 天", "icon": "👑"})
    
    return achievements


# ============================================================
# ★ 用户鉴权相关函数
# ============================================================

def generate_sms_code() -> str:
    """
    生成 6 位随机数字验证码。
    """
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])


def send_sms_code(phone: str) -> dict:
    """
    发送验证码到手机号（模拟）。
    
    参数:
        phone: 用户手机号
    
    返回:
        成功: {"success": True, "code": "123456"}  # code 仅用于开发测试
        失败: {"success": False, "error": "错误信息"}
    
    注意：
        - 生产环境应接入真实短信服务（阿里云、腾讯云短信等）
        - 当前为模拟模式：验证码打印到控制台，并返回给前端（仅用于开发）
    """
    code = generate_sms_code()
    expire = datetime.datetime.now() + datetime.timedelta(minutes=5)  # 5 分钟有效期
    
    conn = get_connection()
    try:
        # 检查用户是否已存在
        existing = conn.execute(
            "SELECT id FROM users WHERE phone = ?", (phone,)
        ).fetchone()
        
        if existing:
            # 已存在：更新验证码
            conn.execute(
                "UPDATE users SET sms_code = ?, code_expire = ? WHERE phone = ?",
                (code, expire, phone)
            )
        else:
            # 不存在：创建新用户（先只存手机号，注册时再完善信息）
            conn.execute(
                "INSERT INTO users (phone, sms_code, code_expire) VALUES (?, ?, ?)",
                (phone, code, expire)
            )
        
        conn.commit()
        conn.close()
        
        # ★ 模拟发送：打印到控制台
        print(f"[SMS 模拟] 手机号={phone}, 验证码={code}（5分钟内有效）")
        
        return {"success": True, "code": code}  # 仅开发环境返回 code
    except Exception as e:
        conn.close()
        print(f"[database] 发送验证码失败: {e}")
        return {"success": False, "error": str(e)}


def verify_sms_code(phone: str, code: str) -> dict:
    """
    验证手机验证码。

    参数:
        phone: 用户手机号
        code:  用户输入的验证码

    返回:
        成功: {"success": True, "user_id": 用户ID, "phone": 手机号}
        失败: {"success": False, "error": "错误信息"}

    ★ 测试模式：任意纯数字验证码都通过（方便开发调试）
    """
    conn = get_connection()

    # ★ 测试模式：任意纯数字验证码都通过
    if code.isdigit():
        row = conn.execute(
            "SELECT id, phone FROM users WHERE phone = ?", (phone,)
        ).fetchone()

        if row:
            user_id = str(row[0])
            phone_num = row[1]
        else:
            # 自动创建用户
            conn.execute("INSERT INTO users (phone) VALUES (?)", (phone,))
            conn.commit()
            uid_row = conn.execute("SELECT last_insert_rowid()").fetchone()
            user_id = str(uid_row[0])
            phone_num = phone

        conn.close()
        print(f"[database] 测试模式登录成功: id={user_id}, phone={phone_num}")
        return {"success": True, "user_id": user_id, "phone": phone_num}

    # 正式验证流程（按手机号+验证码查询）
    row = conn.execute(
        "SELECT id, phone, code_expire FROM users WHERE phone = ? AND sms_code = ?",
        (phone, code)
    ).fetchone()

    if not row:
        conn.close()
        return {"success": False, "error": "验证码错误"}

    # 检查是否过期
    code_expire_str = row[2]
    if not code_expire_str:
        conn.close()
        return {"success": False, "error": "验证码已过期，请重新获取"}

    try:
        code_expire = datetime.datetime.fromisoformat(code_expire_str)
    except (ValueError, TypeError):
        conn.close()
        return {"success": False, "error": "验证码数据异常，请重新获取"}

    if datetime.datetime.now() > code_expire:
        conn.close()
        return {"success": False, "error": "验证码已过期，请重新获取"}

    user_id = str(row[0])
    phone_num = row[1]

    # 验证成功后，清除验证码（防止重复使用）
    conn.execute(
        "UPDATE users SET sms_code = NULL, code_expire = NULL WHERE id = ?",
        (user_id,)
    )
    conn.commit()
    conn.close()

    print(f"[database] 用户验证成功: id={user_id}, phone={phone_num}")
    return {"success": True, "user_id": user_id, "phone": phone_num}


def get_user_by_phone(phone: str) -> dict | None:
    """
    根据手机号获取用户信息。
    
    参数:
        phone: 用户手机号
    
    返回:
        找到: {"id": 用户ID, "phone": 手机号}
        未找到: None
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT id, phone FROM users WHERE phone = ?",
        (phone,)
    ).fetchone()
    conn.close()
    
    if row:
        return {"id": str(row[0]), "phone": row[1]}
    else:
        return None


def get_user_by_id(user_id: str) -> dict | None:
    """
    根据用户 ID 获取用户信息。
    
    参数:
        user_id: 用户 ID
    
    返回:
        找到: {"id": 用户ID, "phone": 手机号}
        未找到: None
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT id, phone FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    conn.close()
    
    if row:
        return {"id": str(row[0]), "phone": row[1]}
    else:
        return None


# ============================================================
# ★ Phase 2: 训练日志
# ============================================================
def save_training_log(
    user_id: str,
    exercise_name: str,
    sets_done: int,
    reps_done: str,
    weight_kg: float = 0,
    muscle_group: str = "",
    plan_name: str = "",
    notes: str = "",
) -> dict:
    """保存一条训练日志记录"""
    conn = get_connection()
    conn.execute(
        """INSERT INTO training_logs
           (user_id, plan_name, exercise_name, muscle_group, sets_done, reps_done, weight_kg, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, plan_name, exercise_name, muscle_group, sets_done, reps_done, weight_kg, notes),
    )
    conn.commit()
    conn.close()
    print(f"[database] 训练日志已保存: user={user_id}, exercise={exercise_name}, {sets_done}组x{reps_done}")
    return {"success": True}


def get_training_logs(user_id: str, days: int = 7) -> list:
    """获取最近 N 天的训练日志"""
    conn = get_connection()
    rows = conn.execute(
        """SELECT exercise_name, muscle_group, sets_done, reps_done, weight_kg, plan_name,
                  DATE(recorded_at) as log_date, TIME(recorded_at) as log_time
           FROM training_logs
           WHERE user_id = ? AND recorded_at >= DATE('now', ?)
           ORDER BY recorded_at DESC""",
        (user_id, f"-{days} days"),
    ).fetchall()
    conn.close()
    return [
        {
            "exercise": row[0], "muscle_group": row[1],
            "sets": row[2], "reps": row[3], "weight": row[4],
            "plan": row[5], "date": row[6], "time": row[7],
        }
        for row in rows
    ]


# ============================================================
# ★ Phase 2: 训练计划存储
# ============================================================
def save_training_plan(user_id: str, plan_name: str, plan_json: str) -> dict:
    """保存或覆盖用户的训练计划"""
    conn = get_connection()
    conn.execute(
        "UPDATE saved_plans SET is_active = 0 WHERE user_id = ?",
        (user_id,),
    )
    conn.execute(
        """INSERT INTO saved_plans (user_id, plan_name, plan_json, is_active)
           VALUES (?, ?, ?, 1)""",
        (user_id, plan_name, plan_json),
    )
    conn.commit()
    conn.close()
    return {"success": True, "plan_name": plan_name}


def get_active_training_plan(user_id: str) -> dict | None:
    """获取用户当前激活的训练计划"""
    import json
    conn = get_connection()
    row = conn.execute(
        """SELECT plan_name, plan_json, created_at FROM saved_plans
           WHERE user_id = ? AND is_active = 1
           ORDER BY created_at DESC LIMIT 1""",
        (user_id,),
    ).fetchone()
    conn.close()
    if row:
        try:
            plan_data = json.loads(row[1])
            plan_data["_plan_name"] = row[0]
            plan_data["_created_at"] = row[2]
            return plan_data
        except Exception:
            return None
    return None


# ============================================================
# ★ Phase 2: 里程碑
# ============================================================
def check_and_save_milestone(user_id: str, milestone_type: str, milestone_key: str, description: str) -> dict:
    """检查并保存里程碑（已存在则跳过，新达成则保存并返回 is_new=True）"""
    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM milestones WHERE user_id = ? AND milestone_key = ?",
        (user_id, milestone_key),
    ).fetchone()

    if existing:
        conn.close()
        return {"is_new": False, "milestone_key": milestone_key}

    conn.execute(
        """INSERT INTO milestones (user_id, milestone_type, milestone_key, description)
           VALUES (?, ?, ?, ?)""",
        (user_id, milestone_type, milestone_key, description),
    )
    conn.commit()
    conn.close()
    print(f"[database] 新里程碑达成: user={user_id}, key={milestone_key}")
    return {"is_new": True, "milestone_key": milestone_key, "description": description}


def get_milestones(user_id: str) -> list:
    """获取用户所有里程碑"""
    conn = get_connection()
    rows = conn.execute(
        """SELECT milestone_type, milestone_key, description, DATE(achieved_at) as achieved_date
           FROM milestones WHERE user_id = ? ORDER BY achieved_at DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [
        {"type": r[0], "key": r[1], "description": r[2], "date": r[3]}
        for r in rows
    ]


# ============================================================
# ★ Phase 2: 营养目标（扩展版，支持蛋白质/碳水/脂肪单独设置）
# ============================================================
def save_nutrition_goals(user_id: str, calorie_target: int, protein_target: int = 120,
                          carb_target: int = 250, fat_target: int = 65, exercise_target: int = 60):
    """保存扩展的营养与运动目标"""
    import json
    goal_json = json.dumps({
        "calorie_target": calorie_target,
        "protein_target": protein_target,
        "carb_target": carb_target,
        "fat_target": fat_target,
        "exercise_target": exercise_target,
    })
    conn = get_connection()
    conn.execute(
        "UPDATE user_profiles SET goal = ? WHERE user_id = ?",
        (goal_json, user_id),
    )
    conn.commit()
    conn.close()
    print(f"[database] 营养目标已保存: user={user_id}")

