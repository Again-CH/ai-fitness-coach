# -*- coding: utf-8 -*-
"""
food_order_guard.py —— AI 点外卖安全守卫模块
================================================
本模块是 AI 健身教练的「外卖守门人」，在用户消息中检测点外卖意图，
并通过 LangGraph interrupt() 机制暂停对话、要求用户确认。

设计理念：意图识别 → 主动确认 → 安全引导
    第一层（本模块）：关键词 + 语义检测，识别外卖意图
    第二层（LangGraph interrupt）：强制暂停，必须用户确认后才继续
    第三层（LLM 回复）：确认后在健身教练身份下给予营养引导

为什么需要这个？
    - AI 健身教练不能直接帮用户点外卖（角色越界 + 安全风险）
    - 但用户可能在对话中自然流露出外卖需求
    - 正确做法：识别意图 → 确认 → 在健身框架内提供营养建议

★ 关键：使用 LangGraph 的 interrupt() 实现 Human-in-the-loop，
   不是简单的关键词匹配后返回固定话术。
"""

from typing import Optional, Dict, Any


# ============================================================
# ★ 点外卖意图关键词
# ============================================================

# 外卖平台 / 行为关键词（触发第一层检测）
ORDER_ACTION_KEYWORDS = [
    # 外卖平台
    "美团", "饿了么", "大众点评",
    # 外卖动作
    "点外卖", "叫外卖", "点餐", "外卖", "下单", "叫餐",
    "帮我点", "帮点", "帮我叫", "帮我订", "帮我买",
    "叫一份", "来一份", "来一单", "订一份", "买一份",
    "帮我点个", "叫个外卖", "订个外卖",
    # 常见外卖品牌 / 品类（健身教练场景下可能有冲突）
    "麦当劳", "肯德基", "汉堡王", "必胜客", "赛百味",
    "星巴克", "瑞幸", "喜茶", "奈雪", "茶颜悦色",
    "奶茶", "炸鸡", "烧烤", "麻辣烫", "火锅外卖",
    "轻食外卖", "沙拉外卖", "健康餐外卖",
]

# 上下文确认词（提高召回率，降低误触发）
# 当用户同时出现这些词和外卖动作词时，才触发中断
# （单独出现这类词不应该触发，例如"我想吃沙拉"是正常饮食咨询）
CONTEXT_CONFIRM_KEYWORDS = [
    "外卖", "点", "叫", "订", "下单", "配送", "送餐", "骑手",
]


# ============================================================
# ★ 饥饿/想吃意图关键词（AI 主动感知，触发 API 调用）
# ============================================================

HUNGER_INTENT_KEYWORDS = [
    # 饥饿表达
    "我饿了", "好饿", "饿了", "饿死", "肚子饿", "饿扁",
    # 想吃表达
    "想吃", "馋了", "嘴馋", "想吃点", "想吃什么", "吃啥",
    "吃点啥", "推荐吃的", "有什么吃的", "附近有什么",
    "有啥好吃的", "有什么好吃的",
    # 问推荐
    "推荐个外卖", "推荐外卖", "推荐个餐厅", "点个什么",
    "有什么推荐", "帮我看看", "帮我查查",
    # 场景
    "刚练完", "练完吃", "运动完吃", "训练后吃",
]

HUNGER_EXCLUDE_KEYWORDS = [
    # 排除：不是真的饿了/想吃，而是咨询
    "减脂餐", "怎么吃", "吃什么好",  # 饮食咨询，不触发外卖
]


# ============================================================
# ★ 中断确认话术模板
# ============================================================

CONFIRMATION_MESSAGE = """🛑 **检测到您可能想点外卖，稍等一下！**

我是你的健身教练，我来帮你搜索附近适合你的健康餐厅。

> 需要我帮你搜一下附近有什么健康好吃的吗？"""

HUNGER_CONFIRMATION_MESSAGE = """🍔 **检测到你饿了！需要帮你找点吃的吗？**

作为你的健身教练，我可以帮你搜索附近适合你训练目标的健康餐厅。

> 要帮你看看附近有什么好吃的吗？"""


# ============================================================
# ★ 确认后 LLM 引导 Prompt（注入到对话上下文）
# ============================================================

# ★ 统一外卖搜索指令（用户确认后，LLM 调用美团 API）
FOOD_SEARCH_PROMPT = """
# ★ 外卖搜索指令（用户已确认需要外卖推荐）

用户确认需要外卖/餐厅推荐。你必须立即采取以下行动：

1. **调用 search_nearby_food 工具**获取附近餐厅数据
   - keyword 参数：用户提到的食物类型（如"沙拉""鸡胸""牛排"等），没提到就传空字符串
   - goal 参数：从 user_profile 获取用户的健身目标（减脂/增肌/维持），无目标默认"维持"
2. 用热情、教练式的语气呈现推荐结果
3. 每条推荐都要提到：热量范围、蛋白质含量、为什么适合用户的健身目标
4. 如果用户的目标是减脂，优先推荐低卡选项；增肌则优先高蛋白选项
5. 帮助用户做出健康的选择，但不要苛责

语气要点：
- 热情鼓励（"练完了来份高蛋白的，补充一下！💪"）
- 在健身框架内给出专业建议
- 尊重用户的最终选择
"""

NUTRITION_GUIDANCE_PROMPT = FOOD_SEARCH_PROMPT  # 向后兼容别名


# ============================================================
# ★ 意图检测函数
# ============================================================

def detect_food_order_intent(user_input: str) -> tuple[bool, str]:
    """
    检测用户消息中是否包含点外卖意图。

    检测策略（多层递进）：
        1. 精确匹配：外卖动作词 + 外卖平台/品牌词 → 高置信度
        2. 宽松匹配：外卖动作词 + 食物相关上下文 → 中置信度
        3. 单独食物提及（无外卖上下文）→ 不触发

    Args:
        user_input: 用户输入的原始文本

    Returns:
        (是否触发, 置信度描述)
        - (True, "explicit"): 明确的外卖意图
        - (True, "likely"): 疑似外卖意图
        - (False, ""): 无外卖意图
    """
    if not user_input or not user_input.strip():
        return False, ""

    text = user_input.strip()

    # ---- 第一层：精确匹配（外卖动作 + 明确的外卖对象） ----
    has_order_action = False
    matched_action = ""
    for kw in ORDER_ACTION_KEYWORDS:
        if kw in text:
            has_order_action = True
            matched_action = kw
            break

    if not has_order_action:
        return False, ""

    # 有外卖动作词，判断置信度
    # 高置信度：包含明确的外卖平台名
    high_conf_platforms = ["美团", "饿了么", "大众点评"]
    for kw in high_conf_platforms:
        if kw in text:
            print(f"[FoodOrder] 检测到明确外卖意图：平台={kw}, 动作={matched_action}")
            return True, "explicit"

    # 高置信度：包含外卖品牌名
    high_conf_brands = ["麦当劳", "肯德基", "汉堡王", "必胜客", "星巴克", "瑞幸"]
    for kw in high_conf_brands:
        if kw in text:
            print(f"[FoodOrder] 检测到明确外卖意图：品牌={kw}, 动作={matched_action}")
            return True, "explicit"

    # 中置信度：有外卖动作词但无明确平台/品牌
    # 检查是否有"轻食"、"健康餐"等健身友好词——这种情况下可能只是咨询
    fitness_friendly = ["轻食", "沙拉", "健康餐", "减脂餐", "健身餐"]
    for kw in fitness_friendly:
        if kw in text:
            print(f"[FoodOrder] 疑似外卖意图（健身友好型），动作={matched_action}")
            return True, "likely"

    # 一般中置信度
    print(f"[FoodOrder] 疑似外卖意图（中等置信度），动作={matched_action}")
    return True, "likely"


# ============================================================
# ★ 确认问题生成（根据置信度选择不同模板）
# ============================================================

def get_confirmation_message(detected_intent: str, user_input: str = "") -> str:
    """
    根据检测到的意图类型生成不同的确认消息。

    Args:
        detected_intent: "explicit" | "likely" | "hunger"
        user_input: 用户原始消息（用于个性化回复）

    Returns:
        确认话术字符串
    """
    if detected_intent == "hunger":
        return HUNGER_CONFIRMATION_MESSAGE
    elif detected_intent == "explicit":
        return CONFIRMATION_MESSAGE
    else:
        return CONFIRMATION_MESSAGE  # likely 也用统一话术


# ============================================================
# ★ 饥饿意图检测（AI 主动感知）
# ============================================================

def detect_hunger_intent(user_input: str) -> tuple[bool, str]:
    """
    检测用户是否表达了「饥饿/想吃东西」的意图。

    与 detect_food_order_intent 的区别：
        - detect_food_order_intent：检测"点外卖"行为意图 → 触发 Human-in-the-loop 确认
        - detect_hunger_intent：检测"饿了/想吃"状态 → 触发 AI 主动调用美团 API 推荐

    策略：
        1. 匹配饥饿/想吃关键词
        2. 排除纯粹的饮食咨询（"怎么吃"、"吃什么好"）

    Args:
        user_input: 用户输入文本

    Returns:
        (是否触发, 意图类型)
        - (True, "hunger"): 饥饿意图，AI 应主动推荐
        - (False, ""): 无饥饿意图
    """
    if not user_input or not user_input.strip():
        return False, ""

    text = user_input.strip()

    # 排除饮食咨询
    for kw in HUNGER_EXCLUDE_KEYWORDS:
        if kw in text:
            return False, ""

    # 匹配饥饿关键词
    for kw in HUNGER_INTENT_KEYWORDS:
        if kw in text:
            print(f"[FoodOrder] 检测到饥饿/想吃意图：关键词={kw}")
            return True, "hunger"

    return False, ""


# ============================================================
# ★ 综合意图分类（供 LangGraph 节点使用）
# ============================================================

def classify_intent(user_input: str) -> Dict[str, Any]:
    """
    综合分类用户意图，返回意图类型和详细信息。

    ★ 统一流程：外卖下单 / 饥饿想吃 都走 Human-in-the-loop 确认
       确认后 LLM 调用 search_nearby_food 工具获取美团餐厅数据

    优先级：
        1. 外卖确认（explicit/likely） → Human-in-the-loop
        2. 饥饿想吃（hunger） → Human-in-the-loop（统一确认后调 API）
        3. 正常对话 → 放行

    Returns:
        {
            "intent_type": "food_order" | "hunger" | "normal",
            "level": "explicit" | "likely" | "hunger" | "",
            "action": "interrupt" | "pass",
        }
    """
    # 先检测外卖意图（优先级更高）
    order_triggered, order_level = detect_food_order_intent(user_input)
    if order_triggered:
        return {
            "intent_type": "food_order",
            "level": order_level,
            "action": "interrupt",
        }

    # 再检测饥饿意图 → 同样走 interrupt
    hunger_triggered, hunger_level = detect_hunger_intent(user_input)
    if hunger_triggered:
        return {
            "intent_type": "hunger",
            "level": hunger_level,
            "action": "interrupt",
        }

    return {
        "intent_type": "normal",
        "level": "",
        "action": "pass",
    }


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    test_cases = [
        # 明确外卖意图
        ("帮我点个麦当劳的巨无霸套餐", True, "explicit"),
        ("帮我在美团上点一份沙拉", True, "explicit"),
        ("能不能帮我叫个肯德基外卖", True, "explicit"),
        ("用饿了么帮我点奶茶", True, "explicit"),
        ("帮我订个外卖", True, "likely"),
        ("叫一份炸鸡", True, "likely"),
        ("帮我点个轻食外卖", True, "likely"),
        # 正常健身咨询（不应触发）
        ("帮我制定减脂计划", False, ""),
        ("深蹲标准动作是什么？", False, ""),
        ("我今天的蛋白质够了吗？", False, ""),
        ("推荐一下减脂餐", False, ""),
        ("我想吃沙拉，有什么推荐", False, ""),
        # 边界情况
        ("我想吃麦当劳但我知道不健康", True, "explicit"),
        ("外卖平台有哪些健康的选择", True, "likely"),
    ]

    print("=" * 60)
    print("点外卖意图检测自测")
    print("=" * 60)
    for text, expected_trigger, expected_level in test_cases:
        triggered, level = detect_food_order_intent(text)
        status = "✅" if (triggered == expected_trigger and level == expected_level) else "❌"
        print(f"{status} 输入: {text}")
        print(f"   预期: trigger={expected_trigger}, level={expected_level}")
        print(f"   实际: trigger={triggered}, level={level}")
        if triggered:
            msg = get_confirmation_message(level, text)
            print(f"   话术: {msg[:60]}...")
        print()

    print("\n" + "=" * 60)
    print("饥饿意图检测自测")
    print("=" * 60)
    hunger_tests = [
        ("我饿了", True, "hunger"),
        ("好饿啊想吃东西", True, "hunger"),
        ("刚练完吃啥", True, "hunger"),
        ("附近有什么好吃的", True, "hunger"),
        ("帮我看看有什么推荐的", True, "hunger"),
        ("减脂餐怎么吃", False, ""),  # 咨询，不触发
        ("今天的训练计划", False, ""),  # 正常对话
        ("怎么吃更健康", False, ""),  # 咨询，不触发
    ]
    for text, expected_trigger, expected_level in hunger_tests:
        triggered, level = detect_hunger_intent(text)
        status = "✅" if (triggered == expected_trigger and level == expected_level) else "❌"
        print(f"{status} 输入: {text} → trigger={triggered}, level={level}")

    print("\n" + "=" * 60)
    print("综合分类自测")
    print("=" * 60)
    classify_tests = [
        ("帮我点个麦当劳", "food_order", "explicit", "interrupt"),
        ("我想点个外卖", "food_order", "likely", "interrupt"),
        ("我饿了", "hunger", "hunger", "interrupt"),
        ("刚练完吃啥", "hunger", "hunger", "interrupt"),
        ("帮我制定减脂计划", "normal", "", "pass"),
    ]
    for text, exp_type, exp_level, exp_action in classify_tests:
        result = classify_intent(text)
        status = "✅" if (result["intent_type"] == exp_type and result["level"] == exp_level and result["action"] == exp_action) else "❌"
        print(f"{status} 输入: {text} → {result}")
