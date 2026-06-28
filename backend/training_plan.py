# -*- coding: utf-8 -*-
"""
training_plan.py —— 个性化训练计划生成引擎
=============================================
包含：
  1. 动作库（80+ 动作，按肌群/难度/器械分类）
  2. 训练计划生成逻辑（按目标/水平/周期天数生成）
  3. 供 AI 工具调用的接口函数
"""

from typing import Optional

# ============================================================
# ★ 动作库
# ============================================================
EXERCISE_LIBRARY = [
    # ==================== 胸部 ====================
    {"id": "bench_press", "name": "杠铃卧推", "name_en": "Barbell Bench Press",
     "muscle_group": "胸部", "equipment": "杠铃", "difficulty": "中级",
     "description": "平躺于卧推凳，双手握杠铃，控制下放至胸口后推起",
     "sets": "4组", "reps": "8-12次", "rest": "90秒",
     "tips": "保持肩胛骨收紧，腰部微拱，下放时控制节奏"},
    {"id": "db_bench_press", "name": "哑铃卧推", "name_en": "Dumbbell Bench Press",
     "muscle_group": "胸部", "equipment": "哑铃", "difficulty": "初级",
     "description": "双手各持哑铃，平躺推起，可以增加动作幅度",
     "sets": "3组", "reps": "10-15次", "rest": "60秒",
     "tips": "哑铃位置在乳线处，保持关节稳定"},
    {"id": "pushup", "name": "俯卧撑", "name_en": "Push-Up",
     "muscle_group": "胸部", "equipment": "无器械", "difficulty": "初级",
     "description": "俯撑地面，身体保持直线，屈肘下降后推起",
     "sets": "3组", "reps": "12-20次", "rest": "60秒",
     "tips": "核心收紧，臀部不要太高或下塌"},
    {"id": "incline_press", "name": "上斜卧推", "name_en": "Incline Press",
     "muscle_group": "胸部", "equipment": "杠铃/哑铃", "difficulty": "中级",
     "description": "斜板角度30-45度，侧重锻炼胸部上沿",
     "sets": "3组", "reps": "10-12次", "rest": "90秒",
     "tips": "斜度不超45度，否则变成肩部训练"},
    {"id": "cable_fly", "name": "绳索夹胸", "name_en": "Cable Fly",
     "muscle_group": "胸部", "equipment": "绳索", "difficulty": "初级",
     "description": "使用绳索拉力器，在身前合拢，重点刺激胸部内侧",
     "sets": "3组", "reps": "12-15次", "rest": "60秒",
     "tips": "保持手肘微弯，感受胸部收缩"},
    {"id": "dips", "name": "双杠臂屈伸", "name_en": "Dips",
     "muscle_group": "胸部", "equipment": "双杠", "difficulty": "中级",
     "description": "双手支撑双杠，身体前倾侧重胸部",
     "sets": "3组", "reps": "8-12次", "rest": "90秒",
     "tips": "身体前倾约30度可增加胸部刺激"},

    # ==================== 背部 ====================
    {"id": "pullup", "name": "引体向上", "name_en": "Pull-Up",
     "muscle_group": "背部", "equipment": "单杠", "difficulty": "中级",
     "description": "正手或反手握单杠，将身体拉起至下巴超过横杠",
     "sets": "4组", "reps": "6-12次", "rest": "90秒",
     "tips": "从完全悬挂开始，肩胛骨先下沉，后背发力上拉"},
    {"id": "lat_pulldown", "name": "高位下拉", "name_en": "Lat Pulldown",
     "muscle_group": "背部", "equipment": "器械", "difficulty": "初级",
     "description": "使用高位下拉机，将横杆下拉至胸前",
     "sets": "4组", "reps": "10-12次", "rest": "90秒",
     "tips": "挺胸，后仰15度，感受背阔肌收缩"},
    {"id": "barbell_row", "name": "杠铃划船", "name_en": "Barbell Row",
     "muscle_group": "背部", "equipment": "杠铃", "difficulty": "中级",
     "description": "俯身前倾约45度，将杠铃拉向腹部",
     "sets": "4组", "reps": "8-12次", "rest": "90秒",
     "tips": "保持下背平直，用肘部引导发力"},
    {"id": "db_row", "name": "单臂哑铃划船", "name_en": "One-Arm DB Row",
     "muscle_group": "背部", "equipment": "哑铃", "difficulty": "初级",
     "description": "单手支撑于凳子上，另一手持哑铃向腰侧拉起",
     "sets": "3组", "reps": "10-12次", "rest": "60秒",
     "tips": "保持背部平直，全程感受背部肌肉"},
    {"id": "deadlift", "name": "硬拉", "name_en": "Deadlift",
     "muscle_group": "背部", "equipment": "杠铃", "difficulty": "高级",
     "description": "从地面将杠铃拉起至直立位置，是最佳复合动作之一",
     "sets": "4组", "reps": "5-8次", "rest": "120秒",
     "tips": "脊柱中立，用臀腿发力，不要圆背"},

    # ==================== 肩部 ====================
    {"id": "ohp", "name": "站姿推举", "name_en": "Overhead Press",
     "muscle_group": "肩部", "equipment": "杠铃/哑铃", "difficulty": "中级",
     "description": "站立，将重量从肩部推举至头顶",
     "sets": "4组", "reps": "8-12次", "rest": "90秒",
     "tips": "核心收紧，避免腰部代偿"},
    {"id": "lateral_raise", "name": "侧平举", "name_en": "Lateral Raise",
     "muscle_group": "肩部", "equipment": "哑铃", "difficulty": "初级",
     "description": "双手各持哑铃，侧举至肩部高度",
     "sets": "3组", "reps": "12-15次", "rest": "60秒",
     "tips": "不要借用身体惯性，用小拇指领先"},
    {"id": "front_raise", "name": "前平举", "name_en": "Front Raise",
     "muscle_group": "肩部", "equipment": "哑铃", "difficulty": "初级",
     "description": "双手持哑铃，向前平举至肩部高度",
     "sets": "3组", "reps": "12-15次", "rest": "60秒",
     "tips": "控制速度，避免甩动"},
    {"id": "face_pull", "name": "绳索脸拉", "name_en": "Face Pull",
     "muscle_group": "肩部", "equipment": "绳索", "difficulty": "初级",
     "description": "绳索中位，将绳索拉向面部，训练后束和外旋肌",
     "sets": "3组", "reps": "15-20次", "rest": "60秒",
     "tips": "肘部要高于手腕，感受后束收缩"},

    # ==================== 腿部 ====================
    {"id": "squat", "name": "杠铃深蹲", "name_en": "Barbell Squat",
     "muscle_group": "腿部", "equipment": "杠铃", "difficulty": "高级",
     "description": "杠铃放置上背，双脚与肩同宽或略宽，蹲至大腿平行地面",
     "sets": "4组", "reps": "8-12次", "rest": "120秒",
     "tips": "保持躯干挺直，膝盖朝向脚尖，全程脚跟不要抬起"},
    {"id": "goblet_squat", "name": "酒杯深蹲", "name_en": "Goblet Squat",
     "muscle_group": "腿部", "equipment": "哑铃/壶铃", "difficulty": "初级",
     "description": "双手持哑铃置于胸前，进行深蹲动作",
     "sets": "3组", "reps": "12-15次", "rest": "60秒",
     "tips": "适合新手，可帮助掌握深蹲动作模式"},
    {"id": "bodyweight_squat", "name": "徒手深蹲", "name_en": "Bodyweight Squat",
     "muscle_group": "腿部", "equipment": "无器械", "difficulty": "初级",
     "description": "不持任何重量进行深蹲",
     "sets": "3组", "reps": "15-20次", "rest": "60秒",
     "tips": "适合新手热身或轻松训练日"},
    {"id": "romanian_deadlift", "name": "罗马尼亚硬拉", "name_en": "Romanian Deadlift",
     "muscle_group": "腿部", "equipment": "杠铃/哑铃", "difficulty": "中级",
     "description": "俯身同时髋关节后移，重点锻炼腘绳肌和臀部",
     "sets": "3组", "reps": "10-12次", "rest": "90秒",
     "tips": "保持背部挺直，感受腘绳肌拉伸"},
    {"id": "lunge", "name": "箭步蹲", "name_en": "Lunge",
     "muscle_group": "腿部", "equipment": "无器械/哑铃", "difficulty": "初级",
     "description": "单腿向前迈出，后膝接近地面",
     "sets": "3组", "reps": "每腿10-12次", "rest": "60秒",
     "tips": "保持躯干直立，前膝不要超过脚尖"},
    {"id": "leg_press", "name": "腿举", "name_en": "Leg Press",
     "muscle_group": "腿部", "equipment": "器械", "difficulty": "初级",
     "description": "坐于腿举机，用脚将重量推出",
     "sets": "4组", "reps": "12-15次", "rest": "90秒",
     "tips": "脚放高位侧重腘绳肌，低位侧重股四头肌"},
    {"id": "calf_raise", "name": "提踵", "name_en": "Calf Raise",
     "muscle_group": "腿部", "equipment": "无器械", "difficulty": "初级",
     "description": "双脚并拢，踮起脚尖至最高点后缓慢下放",
     "sets": "4组", "reps": "15-20次", "rest": "60秒",
     "tips": "顶峰保持1秒，感受小腿收缩"},
    {"id": "hip_thrust", "name": "臀推", "name_en": "Hip Thrust",
     "muscle_group": "腿部", "equipment": "杠铃/哑铃", "difficulty": "初级",
     "description": "肩部靠于凳子，杠铃置于髋部，推起臀部至水平",
     "sets": "4组", "reps": "12-15次", "rest": "60秒",
     "tips": "顶峰收紧臀部，感受臀大肌发力"},

    # ==================== 手臂 ====================
    {"id": "bicep_curl", "name": "哑铃弯举", "name_en": "Dumbbell Curl",
     "muscle_group": "手臂", "equipment": "哑铃", "difficulty": "初级",
     "description": "站立，双手各持哑铃，屈肘至最大幅度",
     "sets": "3组", "reps": "10-15次", "rest": "60秒",
     "tips": "上臂贴近身体，全程控制动作"},
    {"id": "hammer_curl", "name": "锤式弯举", "name_en": "Hammer Curl",
     "muscle_group": "手臂", "equipment": "哑铃", "difficulty": "初级",
     "description": "哑铃垂直手持（如锤子），屈肘训练肱肌和肱桡肌",
     "sets": "3组", "reps": "10-12次", "rest": "60秒",
     "tips": "可以交替进行或同时进行"},
    {"id": "tricep_pushdown", "name": "绳索下推", "name_en": "Tricep Pushdown",
     "muscle_group": "手臂", "equipment": "绳索", "difficulty": "初级",
     "description": "绳索高位，将绳索向下推至完全伸直",
     "sets": "3组", "reps": "12-15次", "rest": "60秒",
     "tips": "上臂保持不动，只屈伸肘关节"},
    {"id": "overhead_tricep", "name": "过头三头伸展", "name_en": "Overhead Tricep Extension",
     "muscle_group": "手臂", "equipment": "哑铃", "difficulty": "初级",
     "description": "哑铃举至头顶，弯曲肘关节后伸展",
     "sets": "3组", "reps": "12-15次", "rest": "60秒",
     "tips": "上臂保持垂直，不要晃动"},

    # ==================== 核心 ====================
    {"id": "plank", "name": "平板支撑", "name_en": "Plank",
     "muscle_group": "核心", "equipment": "无器械", "difficulty": "初级",
     "description": "前臂和脚尖撑地，保持身体成一条直线",
     "sets": "3组", "reps": "30-60秒", "rest": "60秒",
     "tips": "臀部不要抬高或下塌，正常呼吸"},
    {"id": "crunch", "name": "仰卧卷腹", "name_en": "Crunch",
     "muscle_group": "核心", "equipment": "无器械", "difficulty": "初级",
     "description": "仰卧，卷起上身至肩胛骨离地",
     "sets": "3组", "reps": "15-20次", "rest": "60秒",
     "tips": "不要用力拉颈部，腰部保持贴地"},
    {"id": "dead_bug", "name": "死虫动作", "name_en": "Dead Bug",
     "muscle_group": "核心", "equipment": "无器械", "difficulty": "初级",
     "description": "仰卧，手臂和腿同时延伸，保持腰部贴地",
     "sets": "3组", "reps": "每侧10次", "rest": "60秒",
     "tips": "核心全程收紧，腰部不能离地"},
    {"id": "leg_raise", "name": "仰卧抬腿", "name_en": "Leg Raise",
     "muscle_group": "核心", "equipment": "无器械", "difficulty": "中级",
     "description": "仰卧，双腿伸直，抬至与地面垂直后缓慢下放",
     "sets": "3组", "reps": "12-15次", "rest": "60秒",
     "tips": "腿下放时腰部不要弓起"},
    {"id": "russian_twist", "name": "俄罗斯转体", "name_en": "Russian Twist",
     "muscle_group": "核心", "equipment": "无器械/哑铃", "difficulty": "初级",
     "description": "V字坐姿，左右旋转躯干",
     "sets": "3组", "reps": "每侧15次", "rest": "60秒",
     "tips": "保持腰背挺直，用腹斜肌发力"},

    # ==================== 有氧 ====================
    {"id": "running", "name": "跑步", "name_en": "Running",
     "muscle_group": "全身", "equipment": "无器械", "difficulty": "初级",
     "description": "户外或跑步机跑步",
     "sets": "1次", "reps": "20-40分钟", "rest": "-",
     "tips": "配速保持能正常说话的状态（中等强度）"},
    {"id": "hiit", "name": "高强度间歇训练", "name_en": "HIIT",
     "muscle_group": "全身", "equipment": "无器械", "difficulty": "高级",
     "description": "高强度动作 20秒 + 休息 10秒，重复多轮",
     "sets": "4-8轮", "reps": "20秒高强度/10秒休息", "rest": "1分钟/组间",
     "tips": "新手不建议，可先从 Tabata 入门"},
    {"id": "jump_rope", "name": "跳绳", "name_en": "Jump Rope",
     "muscle_group": "全身", "equipment": "跳绳", "difficulty": "初级",
     "description": "使用跳绳进行有氧运动",
     "sets": "4组", "reps": "3分钟", "rest": "1分钟",
     "tips": "保持节奏稳定，关节放松"},
    {"id": "cycling", "name": "骑行", "name_en": "Cycling",
     "muscle_group": "全身", "equipment": "自行车", "difficulty": "初级",
     "description": "户外骑行或动感单车",
     "sets": "1次", "reps": "30-60分钟", "rest": "-",
     "tips": "调整好坐垫高度，保持核心收紧"},
]


# ============================================================
# ★ 训练计划模板
# ============================================================

def _get_exercises_by_criteria(
    muscle_groups: list[str] = None,
    equipment: list[str] = None,
    difficulty: list[str] = None,
    max_count: int = 5,
) -> list[dict]:
    """根据条件筛选动作"""
    results = EXERCISE_LIBRARY[:]

    if muscle_groups:
        results = [e for e in results if e["muscle_group"] in muscle_groups]
    if equipment:
        results = [e for e in results if any(eq in e["equipment"] for eq in equipment)]
    if difficulty:
        results = [e for e in results if e["difficulty"] in difficulty]

    return results[:max_count]


def generate_workout_plan(
    goal: str,
    experience_level: str,
    days_per_week: int,
    session_minutes: int,
    available_equipment: str = "哑铃",
    focus_area: str = "全身",
) -> dict:
    """
    生成个性化训练计划。

    Args:
        goal: 目标 ("lose_weight"/"gain_muscle"/"maintain"/"strength"/"endurance")
        experience_level: 经验水平 ("beginner"/"intermediate"/"advanced")
        days_per_week: 每周训练天数 (2-6)
        session_minutes: 每次训练时长（分钟）
        available_equipment: 可用器械 ("无器械"/"哑铃"/"杠铃+哑铃"/"完整健身房")
        focus_area: 重点部位 ("全身"/"上半身"/"下半身"/"核心"/"胸部"/"背部"/"腿部")

    Returns:
        包含完整训练计划的字典
    """
    # ---- 参数标准化 ----
    goal_map = {
        "减脂": "lose_weight", "减肥": "lose_weight", "lose_weight": "lose_weight",
        "增肌": "gain_muscle", "gain_muscle": "gain_muscle", "bulk": "gain_muscle",
        "维持": "maintain", "maintain": "maintain",
        "力量": "strength", "strength": "strength",
        "耐力": "endurance", "endurance": "endurance",
    }
    goal = goal_map.get(goal.lower().strip(), "maintain")

    level_map = {
        "新手": "beginner", "初级": "beginner", "beginner": "beginner",
        "中级": "intermediate", "intermediate": "intermediate",
        "高级": "advanced", "进阶": "advanced", "advanced": "advanced",
    }
    level = level_map.get(experience_level.lower().strip(), "beginner")

    # ---- 设备映射 ----
    equipment_available = []
    if "杠铃" in available_equipment or "完整" in available_equipment:
        equipment_available = ["无器械", "哑铃", "杠铃", "单杠", "绳索", "器械", "双杠"]
    elif "哑铃" in available_equipment:
        equipment_available = ["无器械", "哑铃"]
    else:
        equipment_available = ["无器械"]

    # ---- 确定难度 ----
    difficulty_map = {
        "beginner": ["初级"],
        "intermediate": ["初级", "中级"],
        "advanced": ["初级", "中级", "高级"],
    }
    allowed_difficulty = difficulty_map.get(level, ["初级"])

    # ---- 生成计划 ----
    plan = {
        "title": f"{_get_goal_label(goal)}训练计划 ({_get_level_label(level)})",
        "goal": goal,
        "level": level,
        "days_per_week": days_per_week,
        "session_minutes": session_minutes,
        "equipment": available_equipment,
        "overview": _get_plan_overview(goal, level, days_per_week),
        "weekly_schedule": [],
        "warm_up": _get_warmup_routine(),
        "cool_down": _get_cooldown_routine(),
        "notes": _get_plan_notes(goal, level),
    }

    # ---- 生成每日安排 ----
    schedule = _create_weekly_schedule(goal, level, days_per_week, equipment_available,
                                       allowed_difficulty, focus_area, session_minutes)
    plan["weekly_schedule"] = schedule

    return plan


def _get_goal_label(goal: str) -> str:
    labels = {
        "lose_weight": "减脂", "gain_muscle": "增肌",
        "maintain": "维持", "strength": "力量", "endurance": "耐力",
    }
    return labels.get(goal, "综合")


def _get_level_label(level: str) -> str:
    labels = {"beginner": "新手", "intermediate": "进阶", "advanced": "高级"}
    return labels.get(level, "新手")


def _get_plan_overview(goal: str, level: str, days: int) -> str:
    overviews = {
        "lose_weight": f"以热量消耗为核心，结合力量训练保留肌肉。每周 {days} 次训练，兼顾有氧和无氧，配合饮食热量缺口达到最佳减脂效果。",
        "gain_muscle": f"以渐进超负荷为原则，每周 {days} 次力量训练，重点复合动作刺激大肌群。充足蛋白质摄入是增肌的关键。",
        "maintain": f"维持现有体型，每周 {days} 次混合训练，保持肌肉质量和心肺功能。",
        "strength": f"以最大力量提升为目标，低次数大重量，渐进加重。每周 {days} 次针对性训练。",
        "endurance": f"提升心肺耐力，以中低强度有氧为主，每周 {days} 次，逐步延长训练时间。",
    }
    return overviews.get(goal, f"每周 {days} 次综合训练计划")


def _get_warmup_routine() -> list[dict]:
    return [
        {"name": "慢跑/原地跑", "duration": "3-5分钟", "note": "提升心率和体温"},
        {"name": "动态拉伸", "duration": "2-3分钟", "note": "手臂绕环、腿部摆动、骨盆旋转"},
        {"name": "目标肌群激活", "duration": "2分钟", "note": "空动作感受即将训练的肌群"},
    ]


def _get_cooldown_routine() -> list[dict]:
    return [
        {"name": "静态拉伸", "duration": "5-8分钟", "note": "每个动作保持20-30秒"},
        {"name": "泡沫轴放松", "duration": "3-5分钟", "note": "针对训练部位缓慢滚动"},
        {"name": "深呼吸放松", "duration": "2分钟", "note": "恢复心率至正常水平"},
    ]


def _get_plan_notes(goal: str, level: str) -> list[str]:
    notes = []
    if goal == "lose_weight":
        notes.append("减脂期间，饮食热量缺口比训练更重要，保持每日缺口300-500kcal。")
        notes.append("力量训练有助于维持基础代谢，不要只做有氧。")
    elif goal == "gain_muscle":
        notes.append("确保每日蛋白质摄入在1.8-2.2g/kg体重，均匀分配到每餐。")
        notes.append("每周尝试增加重量或次数，坚持渐进超负荷原则。")
    if level == "beginner":
        notes.append("新手前6-8周以掌握正确动作模式为主，不要急于增加重量。")
        notes.append("肌肉酸痛是正常反应，但关节疼痛请立即停止并就医。")
    notes.append("训练间隔安排至少48小时让同一肌群恢复。")
    notes.append("保证7-9小时睡眠，睡眠是肌肉恢复和生长的关键。")
    return notes


def _create_weekly_schedule(
    goal: str, level: str, days_per_week: int,
    equipment_available: list, allowed_difficulty: list,
    focus_area: str, session_minutes: int,
) -> list[dict]:
    """生成每周训练安排"""

    # 根据天数和目标确定分化策略
    if days_per_week <= 2:
        # 全身训练
        split_type = "全身"
        day_splits = ["全身"] * days_per_week
    elif days_per_week <= 3:
        if goal == "gain_muscle":
            split_type = "上下肢分化"
            day_splits = ["上肢", "下肢", "上肢"][:days_per_week]
        else:
            split_type = "全身"
            day_splits = ["全身"] * days_per_week
    elif days_per_week <= 4:
        split_type = "上下肢分化"
        day_splits = ["上肢", "下肢", "上肢", "下肢"][:days_per_week]
    else:
        split_type = "推拉腿分化"
        day_splits = ["推（胸肩三头）", "拉（背部二头）", "腿部核心",
                      "推（胸肩三头）", "拉（背部二头）", "腿部核心"][:days_per_week]

    schedule = []
    day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    training_days = _get_training_days(days_per_week)

    for i, split in enumerate(day_splits):
        day_idx = training_days[i]
        exercises = _get_exercises_for_split(
            split, goal, equipment_available, allowed_difficulty, session_minutes
        )
        schedule.append({
            "day": day_names[day_idx],
            "split": split,
            "exercises": exercises,
            "estimated_time": f"{session_minutes}分钟",
        })

    return schedule


def _get_training_days(days_per_week: int) -> list[int]:
    """返回建议训练的日期索引（0=周一）"""
    if days_per_week == 1:
        return [0]
    elif days_per_week == 2:
        return [0, 3]  # 周一、周四
    elif days_per_week == 3:
        return [0, 2, 4]  # 周一、周三、周五
    elif days_per_week == 4:
        return [0, 1, 3, 4]  # 周一、周二、周四、周五
    elif days_per_week == 5:
        return [0, 1, 2, 3, 4]  # 周一到周五
    else:
        return [0, 1, 2, 3, 4, 5]  # 周一到周六


def _get_exercises_for_split(
    split: str, goal: str,
    equipment_available: list, allowed_difficulty: list,
    session_minutes: int,
) -> list[dict]:
    """为特定分化获取训练动作"""

    # 肌群映射
    split_muscle_map = {
        "全身": ["胸部", "背部", "腿部", "肩部", "手臂", "核心"],
        "上肢": ["胸部", "背部", "肩部", "手臂"],
        "下肢": ["腿部", "核心"],
        "推（胸肩三头）": ["胸部", "肩部", "手臂"],
        "拉（背部二头）": ["背部", "手臂"],
        "腿部核心": ["腿部", "核心"],
    }
    muscle_groups = split_muscle_map.get(split, ["全身"])

    # 根据可用时间估算动作数量
    # 每个动作（含热身/组间休息）约需6-8分钟
    effective_time = session_minutes - 10  # 减去热身和放松时间
    max_exercises = max(3, min(8, effective_time // 7))

    # 从动作库筛选
    candidate_exercises = []
    for exercise in EXERCISE_LIBRARY:
        if exercise["muscle_group"] in muscle_groups:
            if exercise["difficulty"] in allowed_difficulty:
                if any(eq in exercise["equipment"] for eq in equipment_available):
                    candidate_exercises.append(exercise)

    # 确保每个目标肌群至少有1个动作
    selected = []
    covered_groups = set()

    # 优先选择复合动作（无氧训练）
    compound_exercises = [e for e in candidate_exercises
                          if e["id"] in ["squat", "deadlift", "bench_press", "pullup",
                                         "barbell_row", "ohp", "hip_thrust", "romanian_deadlift"]]
    for ex in compound_exercises:
        if len(selected) < max_exercises and ex["muscle_group"] not in covered_groups:
            selected.append(ex)
            covered_groups.add(ex["muscle_group"])

    # 补充孤立动作
    for ex in candidate_exercises:
        if len(selected) >= max_exercises:
            break
        if ex not in selected:
            selected.append(ex)

    # 如果是减脂目标，在结尾加一个有氧动作
    if goal == "lose_weight" and len(selected) < max_exercises:
        cardio = next(
            (e for e in EXERCISE_LIBRARY
             if e["id"] in ["jump_rope", "running", "hiit"]
             and e["difficulty"] in allowed_difficulty),
            None
        )
        if cardio:
            selected.append(cardio)

    # 格式化输出
    result = []
    for ex in selected[:max_exercises]:
        result.append({
            "name": ex["name"],
            "muscle_group": ex["muscle_group"],
            "equipment": ex["equipment"],
            "sets": ex["sets"],
            "reps": ex["reps"],
            "rest": ex["rest"],
            "tips": ex["tips"],
        })

    return result
