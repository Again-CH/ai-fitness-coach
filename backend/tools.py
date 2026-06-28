# -*- coding: utf-8 -*-
"""
tools.py —— AI 健身教练专业计算工具 & RAG 知识库
=====================================================
本模块定义了以下工具，供 LangGraph Agent 调用：

计算工具：
    1. calculate_tdee             —— 计算每日总能量消耗（BMR + TDEE）
    2. generate_macro_plan        —— 生成宏量营养素计划（蛋白质/脂肪/碳水）

用户画像工具：
    3. save_user_profile          —— 保存/更新用户画像到数据库
    4. get_user_profile           —— 从数据库获取用户画像
    5. save_chat                  —— 保存对话消息到数据库

RAG 知识库工具：
    6. search_fitness_knowledge   —— 检索健身知识库（检索增强生成）

工作原理：
    - 使用 @tool 装饰器将普通 Python 函数注册为 LangChain Tool
    - ChatTongyi.bind_tools() 将工具绑定到 LLM
    - LLM 根据用户意图自动决定是否调用工具、传什么参数
    - 工具执行结果会回传给 LLM，LLM 据此生成最终回复

公式来源：
    - Mifflin-St Jeor 公式（目前最准确的 BMR 估算公式之一）
    - 活动系数参考 Harris-Benedict 修订版
    - 蛋白质建议参考 ACSM（美国运动医学会）指南
"""

from langchain_core.tools import tool
from database import get_profile, update_profile, save_chat_message, get_chat_history


# ============================================================
# ★ 健身知识库（RAG 数据源）
# ============================================================
# 这是 AI 教练的"专业参考资料库"。
# 每条记录包含 content（知识内容）和 topic（主题分类）。
# search_fitness_knowledge 工具会基于关键词匹配检索这些条目，
# 将相关片段作为上下文提供给 LLM，实现"检索增强生成"（RAG）。
#
# 后续可以扩展为向量数据库（FAISS / Chroma）实现语义检索，
# 目前使用关键词匹配 + 评分机制，轻量且无额外依赖。

fitness_knowledge = [
    {
        "content": "减脂的核心是制造热量缺口，建议每日减少 300-500 大卡摄入。热量缺口不宜过大，否则会导致肌肉流失和基础代谢下降。合理的减脂速度为每周减重 0.5-1 公斤。",
        "topic": "减脂原理",
        "keywords": ["减脂", "减重", "热量缺口", "减肥", "瘦身", "fat loss", "lose weight", "卡路里赤字"],
    },
    {
        "content": "增肌期蛋白质摄入量建议为每公斤体重 1.6g - 2.2g。蛋白质应均匀分配到每餐（每餐 20-40g），以提高肌肉蛋白质合成率。优质蛋白来源包括鸡胸肉、牛肉、鱼虾、鸡蛋、乳清蛋白等。",
        "topic": "增肌饮食",
        "keywords": ["增肌", "蛋白质", "muscle gain", "bulk", "肌肥大", "力量增长", "增重"],
    },
    {
        "content": "新手建议采用全身性训练或上下肢分化训练，每周 3-4 次。每次训练聚焦复合动作（深蹲、硬拉、卧推、引体向上），每组 8-12 次，做 3-4 组。新手应先掌握正确动作模式，再逐步增加负重。",
        "topic": "新手训练",
        "keywords": ["新手", "初学者", "训练计划", "训练安排", "分化训练", "全身训练", "beginner", "workout plan"],
    },
    {
        "content": "运动后补充碳水化合物和蛋白质有助于恢复，黄金窗口期约为运动后 30-60 分钟。建议碳水与蛋白质比例为 3:1 至 4:1。充足睡眠（7-9 小时）是恢复的关键因素，睡眠不足会显著影响肌肉修复和激素分泌。",
        "topic": "运动恢复",
        "keywords": ["恢复", "休息", "睡眠", "运动后", "补充", "recovery", "rest", "黄金窗口", "肌肉修复"],
    },
    {
        "content": "有氧运动建议每周进行 150-300 分钟中等强度或 75-150 分钟高强度有氧运动。可选择跑步、游泳、骑行、跳绳等。有氧运动前应进行 5-10 分钟动态热身，运动后进行静态拉伸。",
        "topic": "有氧运动",
        "keywords": ["有氧", "跑步", "游泳", "骑行", "心肺", "cardio", "耐力", "跳绳", "燃脂运动"],
    },
    {
        "content": "力量训练的渐进超负荷原则：通过逐步增加训练重量、次数或组数来持续刺激肌肉生长。建议每周增重不超过 2.5%，或每周增加 1-2 次重复次数。记录训练日志有助于追踪进度。",
        "topic": "力量训练",
        "keywords": ["力量训练", "渐进超负荷", "负重", "深蹲", "硬拉", "卧推", "strength", "progressive overload"],
    },
    {
        "content": "水分摄入建议：普通人每天饮水 1.5-2 升，运动人群建议每天 2.5-3 升。训练期间每 15-20 分钟补充 150-250ml 水。长时间高强度运动（>1 小时）应补充含电解质的运动饮料。",
        "topic": "水分补充",
        "keywords": ["喝水", "水分", "饮水", "脱水", "电解质", "hydration", "water", "补水"],
    },
    {
        "content": "拉伸分为动态拉伸和静态拉伸。动态拉伸适合训练前热身（如高抬腿、臂绕环），每个动作 8-10 次；静态拉伸适合训练后放松，每个动作保持 15-30 秒。避免在肌肉未热开时进行静态拉伸。",
        "topic": "拉伸热身",
        "keywords": ["拉伸", "热身", "放松", "柔韧性", "stretching", "warm up", "动态拉伸", "静态拉伸"],
    },
    {
        "content": "减脂期应保留力量训练以维持肌肉量，建议每周至少 2-3 次力量训练。有氧运动可安排在力量训练后或单独安排。空腹有氧不一定比餐后有氧减脂效果更好，关键是全天总热量摄入。",
        "topic": "减脂训练",
        "keywords": ["减脂训练", "空腹有氧", "减脂期", "保留肌肉", "有氧力量结合", "fat loss training"],
    },
    {
        "content": "常见训练误区：1) 只做有氧不做力量，导致肌肉流失；2) 训练前不做热身，容易受伤；3) 盲目追求大重量，动作变形；4) 忽视核心训练；5) 过度训练导致恢复不足。合理安排训练与休息同样重要。",
        "topic": "训练误区",
        "keywords": ["误区", "错误", "受伤", "过度训练", "常见错误", "mistake", "误区避免"],
    },
]


# ============================================================
# 工具 1：计算每日总能量消耗（TDEE）
# ============================================================
@tool
def calculate_tdee(
    weight: float,
    height: float,
    age: int,
    gender: str,
    activity_level: str,
) -> dict:
    """
    计算每日总能量消耗（TDEE）和基础代谢率（BMR）。

    使用 Mifflin-St Jeor 公式计算 BMR，再乘以活动系数得出 TDEE。
    当用户询问"每天消耗多少热量"、"基础代谢"、"TDEE"时调用此工具。

    Args:
        weight: 体重，单位 kg
        height: 身高，单位 cm
        age: 年龄
        gender: 性别，"male" 或 "female"（也接受 "男"/"女"）
        activity_level: 活动水平，可选值：
            - "sedentary"：久坐不动（办公室工作，很少运动）
            - "light"：轻度活动（每周运动 1-3 次）
            - "moderate"：中度活动（每周运动 3-5 次）
            - "active"：高度活动（每周运动 6-7 次）
            - "very_active"：极度活动（体力工作或每天训练两次）

    Returns:
        包含 BMR、TDEE 和活动系数的字典，单位均为 kcal
    """
    # ---- 性别标准化 ----
    gender_lower = gender.lower().strip()
    is_male = gender_lower in ("male", "男", "m")

    # ---- Mifflin-St Jeor 公式 ----
    # 男性: BMR = 10×体重(kg) + 6.25×身高(cm) - 5×年龄 + 5
    # 女性: BMR = 10×体重(kg) + 6.25×身高(cm) - 5×年龄 - 161
    bmr = 10 * weight + 6.25 * height - 5 * age
    bmr += 5 if is_male else -161
    bmr = round(bmr, 1)

    # ---- 活动系数映射 ----
    activity_multipliers = {
        "sedentary": 1.2,       # 久坐不动
        "light": 1.375,         # 轻度活动
        "moderate": 1.55,       # 中度活动
        "active": 1.725,        # 高度活动
        "very_active": 1.9,     # 极度活动
    }

    # 容错：尝试匹配用户可能输入的各种写法
    level_key = activity_level.lower().strip().replace("-", "_").replace(" ", "_")
    if level_key not in activity_multipliers:
        # 模糊匹配
        for key in activity_multipliers:
            if key in level_key or level_key in key:
                level_key = key
                break
        else:
            level_key = "moderate"  # 默认中等活动

    multiplier = activity_multipliers[level_key]
    tdee = round(bmr * multiplier, 1)

    result = {
        "bmr": bmr,
        "tdee": tdee,
        "activity_level": level_key,
        "activity_multiplier": multiplier,
        "formula": "Mifflin-St Jeor",
        "unit": "kcal/day",
    }

    print(f"[Tool] calculate_tdee: BMR={bmr}, TDEE={tdee}, "
          f"activity={level_key}({multiplier})")
    return result


# ============================================================
# 工具 2：生成宏量营养素计划
# ============================================================
@tool
def generate_macro_plan(
    tdee: float,
    goal: str,
    weight: float,
) -> dict:
    """
    根据每日总消耗和目标，生成三大宏量营养素的每日建议摄入量。

    当用户询问"每天吃多少蛋白质"、"营养配比"、"饮食计划"、"宏量营养素"时调用此工具。

    Args:
        tdee: 每日总能量消耗，单位 kcal（由 calculate_tdee 工具计算得出）
        goal: 目标，可选值：
            - "lose_weight"：减脂（热量缺口 20%，蛋白质 1.8g/kg）
            - "gain_muscle"：增肌（热量盈余 10%，蛋白质 2.0g/kg）
            - "maintain"：维持（TDEE 不变，蛋白质 1.4g/kg）
        weight: 体重，单位 kg（用于计算每公斤体重的蛋白质摄入量）

    Returns:
        包含蛋白质、脂肪、碳水化合物克数及目标热量的字典
    """
    # ---- 目标标准化 ----
    goal_key = goal.lower().strip().replace("-", "_").replace(" ", "_")

    # 模糊匹配
    if "lose" in goal_key or "减" in goal or "cut" in goal_key:
        goal_key = "lose_weight"
    elif "gain" in goal_key or "增" in goal or "bulk" in goal_key:
        goal_key = "gain_muscle"
    elif "maintain" in goal_key or "维持" in goal or "keep" in goal_key:
        goal_key = "maintain"
    else:
        goal_key = "maintain"  # 默认维持

    # ---- 目标热量计算 ----
    goal_config = {
        "lose_weight": {
            "calorie_factor": 0.8,       # 热量缺口 20%
            "protein_per_kg": 1.8,       # 蛋白质 1.8g/kg
            "fat_pct": 0.25,             # 脂肪占目标热量 25%
            "label": "减脂",
        },
        "gain_muscle": {
            "calorie_factor": 1.1,       # 热量盈余 10%
            "protein_per_kg": 2.0,       # 蛋白质 2.0g/kg
            "fat_pct": 0.25,             # 脂肪占目标热量 25%
            "label": "增肌",
        },
        "maintain": {
            "calorie_factor": 1.0,       # 维持热量
            "protein_per_kg": 1.4,       # 蛋白质 1.4g/kg
            "fat_pct": 0.25,             # 脂肪占目标热量 25%
            "label": "维持",
        },
    }

    config = goal_config[goal_key]
    target_calories = round(tdee * config["calorie_factor"], 0)

    # ---- 蛋白质 ----
    # 按体重计算：蛋白质g = protein_per_kg × weight
    protein_g = round(config["protein_per_kg"] * weight, 1)
    protein_cal = protein_g * 4  # 1g 蛋白质 = 4 kcal

    # ---- 脂肪 ----
    # 按热量占比计算：脂肪g = (target_calories × fat_pct) / 9
    fat_cal = target_calories * config["fat_pct"]
    fat_g = round(fat_cal / 9, 1)  # 1g 脂肪 = 9 kcal

    # ---- 碳水化合物 ----
    # 剩余热量全部给碳水：碳水g = (target_calories - protein_cal - fat_cal) / 4
    carbs_cal = target_calories - protein_cal - fat_cal
    carbs_g = round(carbs_cal / 4, 1)  # 1g 碳水 = 4 kcal

    result = {
        "goal": goal_key,
        "goal_label": config["label"],
        "target_calories": int(target_calories),
        "protein_g": protein_g,
        "protein_kcal": int(protein_cal),
        "fat_g": fat_g,
        "fat_kcal": int(fat_cal),
        "carbs_g": carbs_g,
        "carbs_kcal": int(carbs_cal),
        "protein_per_kg": config["protein_per_kg"],
        "weight": weight,
    }

    print(f"[Tool] generate_macro_plan: goal={goal_key}, "
          f"calories={int(target_calories)}, "
          f"P={protein_g}g/F={fat_g}g/C={carbs_g}g")
    return result


# ============================================================
# 工具 3：保存用户画像
# ============================================================
@tool
def save_user_profile(
    user_id: str,
    name: str = None,
    height: float = None,
    weight: float = None,
    age: int = None,
    gender: str = None,
    goal: str = None,
) -> dict:
    """
    保存或更新用户画像数据到数据库。

    当用户在对话中提供了身体数据或个人目标时，调用此工具将数据持久化到数据库。
    这样即使用户关闭浏览器，下次打开时 AI 仍能记住他的数据。

    参数说明：
        user_id: 用户唯一标识（必填）
        name:    用户姓名（可选）
        height:  身高，单位 cm（可选）
        weight:  体重，单位 kg（可选）
        age:     年龄（可选）
        gender:  性别，"男" 或 "女"（可选）
        goal:    健身目标，如 "减脂"、"增肌"、"维持"（可选）

    返回：
        更新后的用户画像字典
    """
    from database import update_profile
    result = update_profile(user_id, **{
        "name": name,
        "height": height,
        "weight": weight,
        "age": age,
        "gender": gender,
        "goal": goal,
    })
    return result


# ============================================================
# 工具 4：获取用户画像
# ============================================================
@tool
def get_user_profile(
    user_id: str,
) -> dict:
    """
    从数据库获取用户的完整画像数据。

    在每次对话开始时，AI 可以调用此工具检查是否有该用户的历史数据。
    如果有，可以用亲切的语气打招呼并提及他的目标。

    参数说明：
        user_id: 用户唯一标识（必填）

    返回：
        用户画像字典，包含 name, height, weight, age, gender, goal 等字段。
        如果用户不存在，返回 None。
    """
    from database import get_profile
    profile = get_profile(user_id)
    return profile


# ============================================================
# 工具 5：保存对话消息
# ============================================================
@tool
def save_chat(
    user_id: str,
    role: str,
    content: str,
) -> dict:
    """
    保存一条对话消息到数据库，用于持久化对话历史。

    参数说明：
        user_id: 用户唯一标识（必填）
        role:    消息角色，"user" 或 "assistant"（必填）
        content: 消息内容（必填）

    返回：
        {"success": True, "message": "消息已保存"}
    """
    from database import save_chat_message
    save_chat_message(user_id, role, content)
    return {"success": True, "message": "消息已保存到数据库"}


# ============================================================
# ★ 工具 6：检索健身知识库（RAG Retriever）
# ============================================================
@tool
def search_fitness_knowledge(
    query: str,
) -> dict:
    """
    检索健身知识库，返回与用户问题相关的专业知识片段。

    这是 RAG（检索增强生成）的核心工具。当用户提问涉及具体的训练计划、
    饮食细节、运动恢复、拉伸热身等健身专业问题时，必须先调用此工具检索
    相关知识，然后基于检索到的内容回答。

    检索机制：
        - 基于关键词匹配 + 评分排序
        - 对知识库中每条记录的 content、topic、keywords 字段进行匹配
        - 返回得分最高的前 3 条相关知识

    参数说明：
        query: 用户的搜索关键词或问题摘要（必填）
               例如："减脂怎么吃"、"新手训练计划"、"运动后恢复"

    返回：
        包含匹配到的知识片段列表的字典，每条包含 content 和 topic。
        如果没有匹配到任何内容，返回空列表。
    """
    query_lower = query.lower().strip()
    # 将查询拆分为关键词（按空格和常见标点分割）
    query_terms = []
    for term in query_lower.replace(",", " ").replace("，", " ").replace("?", " ").replace("？", " ").split():
        if len(term) >= 1:
            query_terms.append(term)

    scored_results = []

    for entry in fitness_knowledge:
        score = 0
        content_lower = entry["content"].lower()
        topic_lower = entry["topic"].lower()
        keywords_lower = [kw.lower() for kw in entry["keywords"]]

        # 1. 关键词字段精确匹配（权重最高：5 分/次）
        for kw in keywords_lower:
            if kw in query_lower:
                score += 5

        # 2. 查询词在 content 中出现（权重：3 分/次）
        for term in query_terms:
            if term in content_lower:
                score += 3

        # 3. 查询词在 topic 中出现（权重：4 分/次）
        for term in query_terms:
            if term in topic_lower:
                score += 4

        # 4. 知识库关键词在查询中出现（权重：2 分/次）
        for kw in keywords_lower:
            for term in query_terms:
                if kw in term or term in kw:
                    score += 2

        if score > 0:
            scored_results.append({
                "content": entry["content"],
                "topic": entry["topic"],
                "score": score,
            })

    # 按得分降序排序，取前 3 条
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_results[:3]

    # 清理 score 字段（不需要返回给 LLM）
    clean_results = [{"topic": r["topic"], "content": r["content"]} for r in top_results]

    print(f"[Tool] search_fitness_knowledge: query='{query}', "
          f"匹配到 {len(clean_results)} 条知识")

    return {
        "query": query,
        "total_matched": len(clean_results),
        "knowledge_chunks": clean_results,
    }


# ============================================================
# ★ 工具 7：图像分析（Qwen-VL 视觉模型）
# ============================================================
@tool
def analyze_image(
    image_path: str,
    question: str = "",
) -> dict:
    """
    分析用户上传的图片内容（食物或运动动作）。

    当用户上传了照片时调用此工具。工具会调用 Qwen-VL 视觉大模型：
    - 如果是食物照片：估算热量和营养成分（蛋白质/碳水/脂肪）
    - 如果是运动动作照片：评估动作是否标准，指出问题并给改进建议

    参数说明：
        image_path: 图片文件的本地路径（由系统在用户上传时自动保存）
        question:   用户的附加问题或分析方向提示（可选）

    返回：
        包含图像分析结果的字典。如果分析失败，返回错误信息。
    """
    from multimodal_service import analyze_image_with_vlm

    result = analyze_image_with_vlm(image_path, question)

    if result["success"]:
        print(f"[Tool] analyze_image: 分析成功，路径={image_path}")
        return {
            "image_path": image_path,
            "analysis": result["analysis"],
        }
    else:
        print(f"[Tool] analyze_image: 分析失败 - {result.get('error', '未知错误')}")
        return {
            "image_path": image_path,
            "error": result.get("error", "图像分析失败"),
        }


# ============================================================
# 工具 8：获取用户看板数据（供 AI 引导用户查看 Dashboard）
# ============================================================
@tool
def get_user_dashboard_data(
    user_id: str,
) -> dict:
    """
    获取用户的数据看板汇总数据，包括体重、BMI、今日饮食/运动情况。

    当用户询问"我今天的进度如何"、"我最近怎么样"、"我的数据"等
    与进度/数据相关的问题时，调用此工具获取数据，然后简短总结，
    并引导用户点击底部的"我的数据"Tab 查看详细图表。

    参数说明：
        user_id: 用户唯一标识（必填）

    返回：
        包含 BMI、今日摄入热量、今日运动时长、本周运动汇总、
        体重趋势、饮食记录等数据的字典。
    """
    from database import (
        get_profile, get_weight_logs, get_diet_logs,
        get_exercise_logs, get_goal_settings
    )
    import datetime

    profile = get_profile(user_id)
    today = datetime.date.today().isoformat()
    diet_today = get_diet_logs(user_id, today)
    exercise_week = get_exercise_logs(user_id, 7)
    weight_logs = get_weight_logs(user_id, 7)
    goal = get_goal_settings(user_id)

    # 计算 BMI
    bmi = None
    health_status = "未知"
    if profile:
        height_m = profile["height"] / 100
        bmi = profile["weight"] / (height_m ** 2)
        if bmi < 18.5:
            health_status = "偏瘦"
        elif bmi < 24:
            health_status = "正常"
        elif bmi < 28:
            health_status = "偏胖"
        else:
            health_status = "肥胖"

    # 计算今日运动时长
    today_exercise = 0
    for item in exercise_week:
        if item["date"] == today:
            today_exercise = item["duration"]
            break

    # 获取打卡数据
    from database import get_checkin_streak, get_achievements
    streak = get_checkin_streak(user_id)
    achievements = get_achievements(user_id)

    result = {
        "bmi": round(bmi, 1) if bmi else None,
        "health_status": health_status,
        "current_weight": profile["weight"] if profile else None,
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
        "weight_logs": weight_logs,
        "exercise_week": exercise_week,
        "diet_today_count": len(diet_today.get("logs", [])),
        "checkin_streak": streak["current_streak"],
        "checkin_total": streak["total_days"],
        "achievements": achievements,
    }

    print(f"[Tool] get_user_dashboard_data: user={user_id}, "
          f"calories={result['today_calories']}/{result['calorie_target']}, "
          f"exercise={result['today_exercise']}min, "
          f"streak={result['checkin_streak']}days")
    return result


# ============================================================
# 工具 9：记录打卡
# ============================================================
@tool
def record_checkin(
    user_id: str,
    checkin_type: str = "daily",
) -> dict:
    """
    记录用户打卡（每日打卡、运动打卡、饮食打卡等）。

    当用户表示"打卡"、"完成了"、"今天练完了"、"我已经跑了"等
    表示完成某项健身活动时，调用此工具记录打卡。

    参数说明：
        user_id: 用户唯一标识（必填）
        checkin_type: 打卡类型（可选，默认 "daily"）
            - "daily": 每日打卡
            - "exercise": 运动打卡
            - "diet": 饮食打卡

    返回：
        包含打卡结果、连续打卡天数、获得的成就等信息的字典。
    """
    from database import save_checkin, get_checkin_streak, get_achievements
    
    # 保存打卡记录
    save_checkin(user_id, checkin_type)
    
    # 获取打卡数据
    streak = get_checkin_streak(user_id)
    achievements = get_achievements(user_id)
    
    # 检查是否获得新成就
    new_achievements = []
    if streak["current_streak"] == 3:
        new_achievements.append({"id": "streak_3", "name": "三天打鱼", "icon": "🔥"})
    elif streak["current_streak"] == 7:
        new_achievements.append({"id": "streak_7", "name": "自律达人", "icon": "💪"})
    elif streak["current_streak"] == 14:
        new_achievements.append({"id": "streak_14", "name": "半月坚持", "icon": "⭐"})
    elif streak["current_streak"] == 30:
        new_achievements.append({"id": "streak_30", "name": "月度冠军", "icon": "🏆"})
    
    result = {
        "success": True,
        "message": "打卡成功！",
        "checkin_type": checkin_type,
        "current_streak": streak["current_streak"],
        "total_days": streak["total_days"],
        "achievements": achievements,
        "new_achievements": new_achievements,
    }

    print(f"[Tool] record_checkin: user={user_id}, type={checkin_type}, "
          f"streak={streak['current_streak']}days")
    return result


# ============================================================
# ★ 工具 10：搜索附近餐厅（美团 API）
# ============================================================
@tool
def search_nearby_food(
    keyword: str = "",
    goal: str = "",
) -> dict:
    """
    搜索附近的餐厅/外卖，返回适合用户的餐厅推荐列表。

    当用户说"我饿了"、"想吃东西"、"附近有什么好吃的"、
    "推荐个外卖"、"刚练完吃啥"时，调用此工具获取推荐。

    工具会根据用户的健身目标自动匹配搜索关键词：
        - 减脂 → 沙拉、低卡、轻食
        - 增肌 → 高蛋白、鸡胸肉、牛肉
        - 维持 → 均衡、营养、家常

    参数说明：
        keyword: 用户想吃的具体类型（如"沙拉"、"鸡胸肉"、"日料"等），留空则按目标推荐
        goal: 用户健身目标（"lose_weight" / "gain_muscle" / "maintain"），用于自动匹配

    返回：
        包含餐厅列表的字典，每家餐厅含名称、评分、最低消费、配送费、标签等。
        如果美团 API 调用失败，返回模拟数据。
    """
    from meituan_client import search_healthy_restaurants, search_nearby_restaurants

    try:
        if goal:
            result = search_healthy_restaurants(goal=goal, keyword=keyword)
        else:
            result = search_nearby_restaurants(keyword=keyword)

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "搜索失败"),
                "suggestion": "可以告诉我你想吃什么类型的？我帮你找找看。",
            }

        restaurants = result.get("data", {}).get("poi_list", [])
        # 格式化返回数据
        formatted = []
        for r in restaurants[:5]:  # 最多返回 5 家
            formatted.append({
                "name": r.get("name", ""),
                "score": r.get("wm_poi_score", 0),
                "min_price": r.get("min_price", 0),
                "shipping_fee": r.get("shipping_fee", 0),
                "delivery_time": r.get("avg_delivery_time", 0),
                "distance": r.get("distance", 0),
                "tags": r.get("tags", []),
                "categories": r.get("categories", []),
                "month_sales": r.get("month_sales", 0),
                "poi_id": r.get("wm_poi_id", ""),
            })

        return {
            "success": True,
            "total": len(formatted),
            "restaurants": formatted,
            "source": "mock" if result.get("data", {}).get("mock") else "meituan",
        }

    except Exception as e:
        print(f"[Tool] search_nearby_food 异常: {e}")
        return {
            "success": False,
            "error": f"搜索异常: {str(e)}",
            "suggestion": "抱歉，暂时无法连接到外卖服务。你可以告诉我想吃什么类型的，我来给你推荐。",
        }


# ============================================================
# ★ 工具 11：打开美团（AI Agent 架构升级 — 执行动作）
# ============================================================
@tool
def open_meituan(
    user_id: str,
    keyword: str = "",
    goal: str = "",
) -> dict:
    """
    打开美团，为用户跳转到美团页面。

    这是"执行动作"工具，不是"搜索"工具。
    只有当用户在对话中明确确认需要打开美团后，Agent 才应调用此工具。

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

    # 构建美团跳转 URL
    base_url = "https://www.meituan.com/"

    if keyword:
        encoded_keyword = urllib.parse.quote(keyword)
        url = f"{base_url}?keyword={encoded_keyword}"
    else:
        url = base_url

    # 记录用户的饮食偏好
    if keyword:
        try:
            save_diet_log(user_id, f"搜索外卖: {keyword}", 0, 0, 0, 0)
        except Exception as e:
            print(f"[Tool] open_meituan: 保存饮食日志失败: {e}")

    print(f"[Tool] open_meituan: user={user_id}, keyword='{keyword}', url={url}")

    return {
        "success": True,
        "action": "open_meituan",
        "meituan_url": url,
        "keyword": keyword,
        "message": f"已为你打开美团{'搜索「' + keyword + '」' if keyword else ''}!",
    }


# ============================================================
# ★ 工具 12：询问外卖意图（新增 — AI 首选入口）
# ============================================================
@tool
def ask_food_delivery(
    user_id: str,
    user_message: str = "",
) -> dict:
    """
    当用户明确表达饥饿、想吃东西、想点外卖等意图时，AI 必须优先调用此工具，
    而不是直接生成文字建议。此工具会触发前端二次确认弹窗，让用户选择是否需要帮助。

    【强制触发条件】用户消息中出现以下任一关键词或意图时，AI 必须先调用此工具：
      - "饿了"、"好饿"、"想吃"、"吃点什么"、"有什么吃的"
      - "外卖"、"点外卖"、"叫外卖"、"订餐"
      - "附近有什么好吃的"、"推荐吃的"、"找餐厅"
      - 任何与"吃"+"不知道吃什么"相关的表达
      - "刚练完吃啥"、"练完吃什么"、"健身餐推荐"

    参数说明：
        user_id: 用户唯一标识
        user_message: 用户原始消息文本

    返回：
        确认提示信息，前端将弹出二次确认弹窗。
    """
    confirm_msg = (
        "检测到你可能想找点吃的！作为你的健身教练，"
        "我可以帮你推荐附近适合你训练目标的健康餐厅和外卖。"
        "需要我帮你找找看吗？"
    )

    return {
        "success": True,
        "action": "ask_food_delivery",
        "requires_confirmation": True,
        "confirmation_message": confirm_msg,
        "next_tool": "search_nearby_food",
    }


# ============================================================
# ★ 工具 13：确认外卖请求（Human-in-the-loop 触发器）
# ============================================================
@tool
def confirm_food_order(
    user_id: str,
    intent: str = "",
    user_message: str = "",
) -> dict:
    """
    请求用户确认是否需要外卖推荐。

    当 Agent 检测到用户可能想点外卖时（如"我饿了"、"想吃东西"），
    应调用此工具而不是直接搜索外卖。此工具会触发中断确认流程。

    参数说明：
        user_id: 用户唯一标识
        intent: 意图描述（如"hunger"、"food_order"）
        user_message: 用户原始消息

    返回：
        确认提示信息。实际的中断由 react_loop 中的 _handle_confirmation 处理。
    """
    confirm_msg = (
        "检测到你饿了! 作为健身教练，我可以帮你搜索附近适合你"
        "训练目标的健康餐厅。需要我帮你找找看吗?"
    )

    return {
        "success": True,
        "action": "confirm_food_order",
        "requires_confirmation": True,
        "confirmation_message": confirm_msg,
        "next_tool": "search_nearby_food",
    }


# ============================================================
# ★ Phase 2 工具 13：通过自然语言描述记录饮食
# ============================================================
@tool
def log_food_by_description(
    user_id: str,
    food_description: str,
    meal_type: str = "snack",
) -> dict:
    """
    根据用户的自然语言饮食描述，智能解析食物名称和分量，并记录到数据库。

    当用户说"我刚吃了一碗米饭和两个鸡蛋"、"午餐吃了鸡胸肉沙拉"、
    "喝了一杯牛奶"等描述时，调用此工具自动识别食物并计算营养成分。

    工具会：
        1. 从本地食物数据库（200+食物）中匹配对应的食物
        2. 估算摄入量（如未指定，使用典型分量）
        3. 计算热量、蛋白质、碳水、脂肪
        4. 保存到数据库

    参数说明：
        user_id: 用户唯一标识（必填）
        food_description: 饮食描述文字，如"一碗米饭两个鸡蛋"、"鸡胸肉150g"
        meal_type: 餐次类型（可选），可选值：
            - "breakfast": 早餐
            - "lunch": 午餐
            - "dinner": 晚餐
            - "snack": 零食/加餐（默认）

    返回：
        包含已识别食物、营养成分汇总、记录结果的字典
    """
    from food_db import parse_food_description, search_food
    from database import save_diet_log

    # ---- 分割多种食物（按顿号、"和"、逗号分割）----
    import re
    parts = re.split(r"[，,、和以及&+]", food_description)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        parts = [food_description]

    recognized_foods = []
    total_nutrition = {"calories": 0, "protein": 0, "carb": 0, "fat": 0}
    unrecognized = []

    for part in parts:
        if not part:
            continue
        parsed = parse_food_description(part)

        if parsed["matched"]:
            nutrition = parsed["nutrition"]
            food_name = f"{parsed['food_name']}({parsed['quantity_g']:.0f}g)"

            # 保存到数据库
            try:
                save_diet_log(
                    user_id,
                    food_name,
                    nutrition["calories"],
                    nutrition["protein"],
                    nutrition["carb"],
                    nutrition["fat"],
                )
            except Exception as e:
                print(f"[Tool] log_food_by_description: 保存失败 {e}")

            recognized_foods.append({
                "input": part,
                "matched_name": parsed["food_name"],
                "quantity_g": parsed["quantity_g"],
                "nutrition": nutrition,
            })

            for key in total_nutrition:
                total_nutrition[key] += nutrition.get(key, 0)
        else:
            unrecognized.append(part)
            # 未匹配食物也以估算值保存（避免漏记）
            try:
                save_diet_log(user_id, part, 150, 8, 20, 5)  # 估算值
            except Exception:
                pass

    # 四舍五入
    for key in total_nutrition:
        total_nutrition[key] = round(total_nutrition[key], 1)

    print(f"[Tool] log_food_by_description: user={user_id}, "
          f"identified={len(recognized_foods)}, calories={total_nutrition['calories']}")

    return {
        "success": True,
        "meal_type": meal_type,
        "recognized_foods": recognized_foods,
        "unrecognized": unrecognized,
        "total_nutrition": total_nutrition,
        "message": (
            f"已记录 {len(recognized_foods)} 种食物，"
            f"本次摄入热量约 {total_nutrition['calories']} kcal，"
            f"蛋白质 {total_nutrition['protein']}g，"
            f"碳水 {total_nutrition['carb']}g，"
            f"脂肪 {total_nutrition['fat']}g。"
        ),
    }


# ============================================================
# ★ Phase 2 工具 14：生成个性化训练计划
# ============================================================
@tool
def generate_training_plan(
    user_id: str,
    goal: str,
    experience_level: str,
    days_per_week: int = 3,
    session_minutes: int = 60,
    available_equipment: str = "哑铃",
    focus_area: str = "全身",
) -> dict:
    """
    为用户生成个性化训练计划并保存到数据库。

    当用户询问"帮我制定训练计划"、"给我一个减脂计划"、"我想增肌，怎么练"等
    与训练计划相关的问题时，调用此工具。

    参数说明：
        user_id: 用户唯一标识（必填）
        goal: 训练目标，可选：
            - "减脂" / "lose_weight"
            - "增肌" / "gain_muscle"
            - "维持" / "maintain"
            - "力量" / "strength"
            - "耐力" / "endurance"
        experience_level: 经验水平，可选：
            - "新手" / "beginner"：0-6个月经验
            - "中级" / "intermediate"：6个月-2年经验
            - "高级" / "advanced"：2年以上经验
        days_per_week: 每周训练天数（默认3天，建议2-5天）
        session_minutes: 每次训练时长（分钟，默认60分钟）
        available_equipment: 可用器械：
            - "无器械"：只能徒手训练
            - "哑铃"：有哑铃
            - "杠铃+哑铃"：有完整自由重量
            - "完整健身房"：所有器械均可用
        focus_area: 重点训练部位（默认"全身"）：
            可选 "全身"/"上半身"/"下半身"/"胸部"/"背部"/"腿部"/"核心"

    返回：
        完整的训练计划字典，包含每日安排、动作说明、训练建议等
    """
    import json
    from training_plan import generate_workout_plan
    from database import save_training_plan

    # 生成训练计划
    plan = generate_workout_plan(
        goal=goal,
        experience_level=experience_level,
        days_per_week=days_per_week,
        session_minutes=session_minutes,
        available_equipment=available_equipment,
        focus_area=focus_area,
    )

    # 保存到数据库
    try:
        plan_json = json.dumps(plan, ensure_ascii=False)
        save_training_plan(user_id, plan["title"], plan_json)
    except Exception as e:
        print(f"[Tool] generate_training_plan: 保存计划失败 {e}")

    print(f"[Tool] generate_training_plan: user={user_id}, goal={goal}, "
          f"level={experience_level}, {days_per_week}days/week")

    return {
        "success": True,
        "plan": plan,
        "message": f"已为你生成「{plan['title']}」，包含每周 {days_per_week} 次训练安排。计划已保存，你可以在数据看板查看详情。",
    }


# ============================================================
# ★ Phase 2 工具 15：生成进度洞察报告
# ============================================================
@tool
def generate_progress_insight(
    user_id: str,
) -> dict:
    """
    分析用户最近的训练和饮食数据，生成个性化的进度洞察报告和激励建议。

    当用户询问"我最近进展如何"、"帮我分析一下"、"我有没有在进步"等
    与进度分析相关的问题时调用此工具。

    工具会分析：
        - 最近7天的饮食合规率（热量是否达标）
        - 最近7天的运动频率和时长
        - 体重变化趋势
        - 连续打卡情况
        - 是否有新的里程碑达成

    参数说明：
        user_id: 用户唯一标识（必填）

    返回：
        包含数据分析结果、趋势判断、激励建议的字典
    """
    from database import (
        get_profile, get_weight_logs, get_diet_logs,
        get_exercise_logs, get_goal_settings, get_checkin_streak,
        check_and_save_milestone,
    )
    import datetime

    profile = get_profile(user_id)
    goal = get_goal_settings(user_id)
    weight_logs = get_weight_logs(user_id, days=14)
    streak = get_checkin_streak(user_id)

    # ---- 最近7天饮食分析 ----
    calorie_target = goal.get("calorie_target", 2000)
    protein_target = goal.get("protein_target", 120)
    diet_compliance_days = 0
    total_protein_7d = 0
    diet_days_count = 0

    for i in range(7):
        date = (datetime.date.today() - datetime.timedelta(days=i)).isoformat()
        day_diet = get_diet_logs(user_id, date)
        day_calories = day_diet.get("total_calories", 0)
        if day_calories > 0:
            diet_days_count += 1
            total_protein_7d += day_diet.get("total_protein", 0)
            # 合规：热量在目标的 80%-110% 之间
            if calorie_target * 0.8 <= day_calories <= calorie_target * 1.1:
                diet_compliance_days += 1

    # ---- 最近7天运动分析 ----
    exercise_logs = get_exercise_logs(user_id, days=7)
    exercise_target = goal.get("exercise_target", 60)
    exercise_compliance_days = sum(
        1 for day in exercise_logs if day["duration"] >= exercise_target
    )
    total_exercise_mins = sum(day["duration"] for day in exercise_logs)

    # ---- 体重趋势 ----
    weight_trend = "无数据"
    weight_change = 0
    if len(weight_logs) >= 2:
        earliest = weight_logs[0]["weight"]
        latest = weight_logs[-1]["weight"]
        weight_change = round(latest - earliest, 1)
        if weight_change < -0.5:
            weight_trend = "下降"
        elif weight_change > 0.5:
            weight_trend = "上升"
        else:
            weight_trend = "稳定"

    # ---- 里程碑检测 ----
    new_milestones = []
    current_streak = streak["current_streak"]

    milestone_checks = [
        ("打卡", f"streak_{current_streak}", f"连续打卡 {current_streak} 天！"),
        ("运动", f"exercise_7d_{total_exercise_mins // 60}h", f"本周累计运动 {total_exercise_mins // 60} 小时"),
    ]

    if current_streak in [3, 7, 14, 30, 60, 100]:
        result = check_and_save_milestone(
            user_id, "打卡",
            f"streak_{current_streak}",
            f"连续打卡 {current_streak} 天！"
        )
        if result["is_new"]:
            new_milestones.append(result)

    # ---- 生成建议 ----
    suggestions = []

    if diet_days_count < 3:
        suggestions.append("📊 最近记录饮食次数较少，坚持每天记录可以帮助更好地控制热量。")
    elif diet_compliance_days / max(diet_days_count, 1) >= 0.7:
        suggestions.append("✅ 饮食执行得很好！保持合理的热量摄入是成功的关键。")

    if exercise_compliance_days == 0:
        suggestions.append("🏋️ 本周还没有完成目标运动时长的记录，今天就开始吧！")
    elif exercise_compliance_days >= 3:
        suggestions.append(f"💪 本周已完成 {exercise_compliance_days} 天达标运动，保持这个节奏！")

    if weight_trend == "下降" and profile and profile.get("goal") in ["减脂", "lose_weight"]:
        suggestions.append(f"🎉 体重下降了 {abs(weight_change)}kg，减脂效果显著！")
    elif weight_trend == "上升" and profile and profile.get("goal") in ["增肌", "gain_muscle"]:
        suggestions.append(f"📈 体重增加了 {weight_change}kg，注意监控体脂率，确保增的是肌肉。")

    if current_streak >= 7:
        suggestions.append(f"🔥 你已经连续打卡 {current_streak} 天，坚持就是胜利！")

    print(f"[Tool] generate_progress_insight: user={user_id}, "
          f"diet_compliance={diet_compliance_days}/{diet_days_count}, "
          f"exercise={total_exercise_mins}min, streak={current_streak}")

    return {
        "success": True,
        "period": "最近7天",
        "diet_analysis": {
            "recorded_days": diet_days_count,
            "compliant_days": diet_compliance_days,
            "compliance_rate": f"{int(diet_compliance_days / max(diet_days_count, 1) * 100)}%",
            "avg_protein": round(total_protein_7d / max(diet_days_count, 1), 1),
            "protein_target": protein_target,
        },
        "exercise_analysis": {
            "training_days": len(exercise_logs),
            "compliant_days": exercise_compliance_days,
            "total_minutes": total_exercise_mins,
            "daily_target": exercise_target,
        },
        "weight_trend": {
            "trend": weight_trend,
            "change": weight_change,
            "data_points": len(weight_logs),
        },
        "checkin": {
            "current_streak": current_streak,
            "total_days": streak["total_days"],
        },
        "new_milestones": new_milestones,
        "suggestions": suggestions,
    }


# ============================================================
# 工具列表（供 langgraph_brain.py 导入）
# ============================================================
fitness_tools = [
    calculate_tdee,
    generate_macro_plan,
    save_user_profile,
    get_user_profile,
    save_chat,
    search_fitness_knowledge,
    analyze_image,
    get_user_dashboard_data,
    record_checkin,
    ask_food_delivery,             # ★ 外卖意图询问（AI 首选入口）
    search_nearby_food,            # ★ 美团搜索
    open_meituan,                  # ★ AI Agent: 打开美团
    confirm_food_order,            # ★ AI Agent: 确认外卖请求（兼容保留）
    log_food_by_description,       # ★ Phase 2: 自然语言饮食记录
    generate_training_plan,        # ★ Phase 2: 个性化训练计划生成
    generate_progress_insight,     # ★ Phase 2: 进度洞察分析
]

# 工具名 → 工具对象的映射（供手动 ToolNode 使用）
tool_map = {t.name: t for t in fitness_tools}
