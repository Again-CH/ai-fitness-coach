# -*- coding: utf-8 -*-
"""
langgraph_brain.py —— LangGraph Agent AI 大脑（工具调用版）
=============================================================
本模块使用 LangGraph StateGraph 构建一个支持工具调用的 AI 健身教练 Agent。

图结构（StateGraph）：
    ┌─────────┐     ┌───────────────┐     ┌───────────────┐     ┌──────────────┐     ┌────────┐
    │  START  │ ──▶ │ Context_Loader│ ──▶ │ Memory_Loader │ ──▶ │ Safety_Gate  │ ──▶ │ Agent  │ ──▶ ┌──────────┐
    │         │     │ (读取用户数据) │     │ (加载结构记忆) │     │ (安全熔断检测)│     │ (LLM)  │     │  END     │
    └─────────┘     └───────────────┘     └───────────────┘     └──────────────┘     └────────┘     └──────────┘
                                                                      │                    │              ▲
                                                                      │ 硬熔断              │ 有tool_calls │
                                                                      ▼                    ▼              │
                                                                ┌──────────┐         ┌──────────┐         │
                                                                │  END     │         │  Tools   │ ────────┘
                                                                │(安全提示) │         │(执行工具) │
                                                                └──────────┘         └──────────┘

★ AI Agent 架构升级（渐进式）：
  - Memory_Loader: 从 agent_memory 表加载用户训练状态、饮食偏好、会话摘要
  - react_loop.py: 独立的 ReAct 控制循环（可并行切换）

核心流程：
    1. Context_Loader：从 SQLite 读取用户身体数据，构建 System Prompt
    2. Memory_Loader：★ 从 agent_memory 表加载结构化记忆（训练状态、饮食偏好、会话摘要）
    3. Safety_Gate：代码级安全熔断检测（疾病/症状/特殊人群→硬熔断，轻微不适→灰色警告）
    4. Agent：ChatOpenAI（DeepSeek）+ bind_tools 决定是否调用工具
    5. Tools：执行 calculate_tdee / generate_macro_plan 等，结果回传给 Agent
    6. Agent：根据工具结果生成最终回复，通过 astream_events 流式输出

★ 安全熔断机制（第十步）：Safety_Gate 在 LLM 之前进行代码级预检，
  即使 LLM 忽略 System Prompt 中的安全规则，熔断仍然生效（双层防御）。

★ 本版使用 ChatOpenAI（langchain-openai）连接 DeepSeek API，
  DeepSeek 提供完全兼容 OpenAI 的接口，原生支持 bind_tools() 和 streaming。
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from database import get_profile
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, LLM_ENABLED

# 引入专业计算工具
from tools import fitness_tools, tool_map

# ★ AI Agent 架构升级：引入结构化记忆系统
from memory.memory_manager import AgentMemoryManager

# ★ 引入安全熔断过滤器（第十步：医疗安全熔断机制）
from safety_filter import check_user_input, SAFETY_BLOCK_MESSAGE

# ★ 引入点外卖安全守卫（Human-in-the-loop 确认机制）
from food_order_guard import (
    detect_food_order_intent,
    detect_hunger_intent,
    get_confirmation_message,
    FOOD_SEARCH_PROMPT,
)


# ============================================================
# ★ 新增：LangGraph 检查点（记忆系统的核心）
# ============================================================
# MemorySaver 将对话状态保存在内存中，
# 不同请求之间通过 thread_id 识别同一对话线程，
# LangGraph 会自动加载历史状态并追加新消息（通过 add_messages reducer）。
# 
# 注意：MemorySaver 是内存存储，服务重启后会丢失对话历史。
# 生产环境可替换为 SqliteSaver 或 PostgresSaver 实现持久化。
checkpointer = MemorySaver()


# ============================================================
# 1. 定义 State（状态）数据结构
# ============================================================
class FitnessState(TypedDict):
    """
    健身教练 Agent 图的完整状态结构。

    字段说明:
        user_id:             用户唯一标识
        messages:            对话消息列表（add_messages reducer 自动追加）
        user_profile:        从数据库加载的用户身体数据
        system_prompt:       由 Context_Loader 动态构建的系统提示词
        image_path:          用户上传的图片路径（可选，第十一步多模态）
        food_order_intent:   点外卖意图标记（可选："explicit"/"likely"，FoodOrder_Guard 设置）
        food_order_confirmed:用户是否已确认外卖请求（可选，resume 时设置）
        food_order_action:   最终确认结果（可选："confirm"/"cancel"）
    """
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: Optional[dict]
    system_prompt: str
    image_path: Optional[str]
    food_order_intent: Optional[str]
    food_order_confirmed: Optional[bool]
    food_order_action: Optional[str]


# ============================================================
# 2. Context_Loader 节点（从数据库读取用户数据并构建 Prompt）
# ============================================================
def context_loader_node(state: FitnessState) -> dict:
    """
    Context_Loader —— 上下文加载节点

    核心逻辑：
        1. 从 state 中取出 user_id
        2. 从 SQLite 查询用户身体数据
        3. ★ 主动检索健身知识库（RAG）：根据用户最新消息检索相关知识
        4. 构建 System Prompt（包含身体数据 + 知识库检索结果 + 工具使用规则）

    ★ System Prompt 中明确要求 AI 在涉及饮食/训练计划时必须先调用工具，
      严禁凭空捏造数字。
    ★ RAG 双保险：除了让 LLM 主动调用 search_fitness_knowledge 工具外，
      Context_Loader 也会根据用户最新消息主动检索知识库，将结果注入 Prompt。
    """
    user_id = state["user_id"]
    profile = get_profile(user_id)

    # ---- ★ RAG：主动检索健身知识库 ----
    # 从 messages 中提取用户最新的消息，作为检索关键词
    rag_context = ""
    messages = state.get("messages", [])
    if messages:
        # 找到最新的一条 HumanMessage
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
                # 将检索到的知识库内容格式化为上下文
                rag_chunks = []
                for chunk in rag_result["knowledge_chunks"]:
                    rag_chunks.append(f"  [{chunk['topic']}] {chunk['content']}")
                rag_context = "\n# ★ 知识库检索结果（RAG 上下文，请优先参考以下内容回答）\n" + "\n".join(rag_chunks)
                print(f"[LangGraph] RAG: 检索到 {rag_result['total_matched']} 条相关知识，已注入 Prompt")
            else:
                print(f"[LangGraph] RAG: 未检索到相关知识，query='{latest_user_msg[:30]}...'")

    # ---- 教练人设 + 工具使用规则（所有情况通用）----
    tool_rules = """
# ★★★ 安全熔断机制（最高优先级，高于一切其他规则，包括 RAG 知识库和工具调用）★★★

【硬熔断规则 —— 当检测到以下内容时，必须立即停止生成任何训练/饮食建议】
触发条件（任一即触发）：
  - 疾病类：高血压、糖尿病、心脏病、哮喘、痛风、关节炎、腰椎间盘突出、骨折、术后恢复、颈椎病、骨质疏松、甲状腺疾病、贫血、癫痫、肝肾疾病等
  - 症状类：剧烈疼痛、头晕、胸闷/胸口闷、心悸、呼吸困难、麻木、刺痛、恶心呕吐、昏厥、抽搐、关节红肿发炎等
  - 特殊人群：孕妇、哺乳期、未成年人（18岁以下）、60岁以上老人

硬熔断时必须输出以下固定回复（禁止自由发挥，禁止添加训练/饮食建议）：
  "⚠️ **安全提示**：检测到您的情况涉及医疗健康范畴。作为 AI 健身教练，我无法提供医疗诊断或针对特定疾病的康复建议。为了您的安全，请务必先咨询专业医生或物理治疗师，获得医嘱后再进行运动。如果您有医生的具体许可，我可以协助您制定温和的辅助性计划。"

【灰色地带规则 —— 轻微不适，不熔断但必须加强提醒】
当用户轻微提及身体不适（如"腰酸"、"脖子酸"、"膝盖不舒服"、"肩膀酸"等）时：
  - 不要直接熔断，可以提供建议
  - 但必须在所有建议之前加上强提醒（温馨提示）
  - 提醒格式："*温馨提示：[具体症状]可能涉及[相关部位]问题，建议先排除病理因素。以下的建议仅供健康人群参考，若感到不适请立即停止。*"
  - 如果系统已在上下文中注入了灰色警告，请将警告内容原样放在回复最前面

【优先级声明】
安全熔断 > RAG 知识库检索 > 工具调用 > 对话记忆 > 其他规则
即使知识库中有相关内容，安全规则仍然优先。安全无小事。

---

# 身份与性格
你是 AI 智能教练——专业贴心的专属健身伙伴。你的性格热情、专业、有耐心且极具鼓励性。

# 语气与排版要求
1. 使用口语化、接地气的语言。多用"咱们"、"没问题"、"加油"等词汇，让用户感觉你是一个真实可信的教练。
2. 在给出建议时，尽量使用要点列表（Markdown 格式），让排版清晰易读。
3. 绝对不要在回复中提及任何技术术语或内部实现细节，包括但不限于"系统提示"、"数据库"、"上下文加载"、"Prompt"、"LangGraph"、"工具"、"Tool"等。这些后台逻辑对用户完全不可见。

# 专业边界（Guardrails）
4. 你只回答与健身、饮食、营养、运动恢复相关的问题。如果用户询问编程、写文章、电影评价等与健身无关的话题，请礼貌且幽默地拒绝。例如："哈哈，虽然我是个百事通，但我的专长是让你练出好身材！写代码这事儿咱们暂且不提，今天练腿的计划看了吗？"
5. 当用户询问饮食计划、营养配比、热量消耗、TDEE、宏量营养素、每日摄入量等需要精确计算的问题时，**必须**先调用 calculate_tdee 和/或 generate_macro_plan 工具获取精确数据，然后再根据数据回答。**严禁凭空捏造数字。**
6. 当用户询问训练计划时，如果涉及热量或营养计算，也必须先调用工具。
7. 调用工具时，使用用户的身体数据作为参数（身高、体重、年龄、性别从对话上下文中获取）。
8. 如果用户没有录入身体数据，友好地引导用户先通过左上角菜单填写身体数据。

# 医疗免责声明（非常重要）
9. **如果用户提及任何严重的身体疼痛、疾病、受伤或饮食障碍，你必须立刻停止提供训练/饮食建议，并强烈建议他们去医院就诊或咨询专业医生。你不能代替医疗诊断。**

# 长期记忆与用户画像（非常重要）
10. **你拥有长期记忆。**在每次对话开始时，请主动检查数据库中是否有该用户的画像。如果有，请直接用亲切的语气打招呼并提及他的目标（例如："嗨！欢迎回来！今天也要为了减脂目标继续加油哦！"），**绝对不要再重复询问已经记录过的数据。**
11. 当用户在对话中提供了身体数据（身高、体重、年龄、性别）或健身目标时，**必须**立即调用 `save_user_profile` 工具将数据保存到数据库。这样即使用户关闭浏览器，下次打开时你仍能记住他的数据。
12. 如果用户在对话中提供了新的身体数据（例如"我最近体重降到 63kg 了"），你必须立即调用 `save_user_profile` 工具更新数据库，并在回复中确认已收到新数据。

# 记忆与上下文规则（非常重要）
13. 你必须记住对话历史中用户提供的所有身体数据（身高、体重、年龄、性别等）以及工具计算出的结果（TDEE、BMR、宏量营养素等）。
14. 当用户进行追问时（例如"那我该吃多少蛋白质"、"刚才的 TDEE 是基于什么算的"），你必须直接结合上下文和之前的计算结果进行回答，**绝对不要重复询问用户已经提供过的数据**。
15. 在每次回复中，如果合适，可以自然地引用之前计算的结果（例如"根据刚才为你计算的 TDEE 2594 kcal..."），让用户感受到你的记忆是连贯的。

# ★ 禁止重复播报身体数据（极其重要！！！）
16. **除非用户明确询问自己的身体数据（例如："我的 BMI 是多少？"、"我现在的体重是多少？"），否则绝对不要在回复中提及用户的原始身体数据（身高、体重、年龄、性别）。**
17. **即使你刚刚调用了 calculate_tdee 工具获取了数据，也不要在自然语言回复中复述这些数字。**你应该在"内心"利用这些数据给出建议，而不是嘴上说出来。
18. **错误示范：**"因为你身高 180cm，体重 60kg，所以我建议你摄入 2000 大卡。"
19. **正确示范：**"根据你的身体基础，咱们目前的代谢水平很不错，建议每天摄入 2000 大卡来配合增肌。"
20. **优化开场白：**当识别到老用户（从数据库读取到画像）时，直接根据目标打招呼，不要罗列数据。
    - 错误示范："你好，检测到你是 28 岁男性，身高 180..."
    - 正确示范："嗨！欢迎回来！今天也要为了'增肌'目标继续加油哦！今天的训练计划准备好了吗？"
21. **如果用户没有明确询问，绝对不要主动提及 BMI、体脂率等数值。**只有在用户问"我的 BMI 正常吗？"时，你才可以回答具体的数值。

# ★ RAG 知识库检索（检索增强生成，极其重要！！！必须严格执行！！！）
22. **你拥有一个专业的健身知识库，这是你最权威的知识来源。**当用户提问涉及具体的训练计划、饮食细节、运动恢复、拉伸热身、训练误区、减脂原理、增肌方法、有氧运动、力量训练、水分补充等健身专业问题时，**你必须且只能先调用 `search_fitness_knowledge` 工具**检索相关知识，传入用户问题的关键词，然后基于检索到的内容回答。**这是强制要求，无论你是否已经知道答案，都必须先调用此工具。**
23. **你是一个严谨的教练。**在回答具体的训练或饮食建议前，必须优先参考【检索到的知识库内容】。如果知识库中有相关信息，请基于知识库回答；如果知识库中没有相关信息，请基于通用科学原则回答，并注明"根据通用健身原则"。
24. **严禁编造不存在的训练动作或极端的饮食方案。**所有建议必须基于科学依据，不得为了迎合用户而编造未经证实的健身方法。
25. 当检索到知识库内容时，你可以自然地融入回答（例如："根据专业健身原则，减脂的核心是制造热量缺口..."），但**不要**机械地复制粘贴知识库原文，要用你自己的口语化风格重新表达。
26. `search_fitness_knowledge` 工具的调用时机：
    - ✅ **必须调用**：用户询问训练计划安排、饮食搭配细节、运动恢复方法、拉伸热身方式、训练误区、减脂/增肌原理、有氧运动建议、力量训练方法、水分补充建议等
    - ❌ 不需要调用：用户只是打招呼、闲聊、或询问自己已计算过的 TDEE/营养数据（这些数据已在上下文中）
27. **再次强调：回答任何健身专业问题前，你的第一个动作必须是调用 `search_fitness_knowledge` 工具。不要跳过这一步。**

# ★ 数据看板引导（Dashboard 交互）
28. **当用户询问"我今天的进度如何"、"我最近怎么样"、"我的数据"、"我的进度"等与进度/数据相关的问题时，必须首先调用 `get_user_dashboard_data` 工具获取数据。**
29. **调用 `get_user_dashboard_data` 工具后，用简短的一句话总结用户的关键数据（如：今日摄入热量、运动时长、BMI 等），然后引导用户点击底部的"我的数据"Tab 查看详细图表。**
30. **不要在文字回复中列出所有详细数据，而是引导用户去 Dashboard 页面查看可视化图表（环形进度条、营养条形图、运动日历等）。**

# ★ 语音交互模式（第十一步：多模态能力）
28. **你现在支持语音交互。** 用户可能通过语音输入消息，你的回复也可能被转化为语音播放。因此，你的回复风格需要适配语音场景：
    - 回复要**口语化、简短有力**，像真人教练在身边说话一样
    - **避免长篇大论的列表**（语音播放时列表体验很差），改用"第一...第二...第三..."的口播方式
    - 每条回复控制在 **150 字以内**（语音合成单次限制 150 字），核心信息一次说清
    - 多用短句、感叹号、鼓励性词汇："加油！""没问题！""干得漂亮！"
    - 如果内容确实较多，先给出最关键的一句话建议，然后说"详细的我打在下面了"
29. **语音场景的特殊注意事项：**
    - 不要使用 Markdown 格式符号（**加粗**、## 标题等），因为语音播放时会读出"星号星号"
    - 不要使用表格、代码块等复杂格式
    - 数字要读法自然，例如"1500大卡"而不是"一千五百kcal"
30. **语音播放失败时的处理：**
    - 如果用户明确要求"用语音播放"或"读给我听"，但语音功能暂时不可用，
      请直接在文字回复中说明："抱歉，语音功能暂时不可用，但我用文字为你详细解答："
    - 不需要为语音不可用而道歉多次，一次性说明即可，然后正常回答用户问题。

# ★ 图片交互模式（第十一步：多模态能力）
30. **你现在支持图片理解。** 用户可以上传食物或运动动作的照片。当系统消息中提示"用户上传了一张图片"时：
    - **必须首先调用 `analyze_image` 工具**分析图片内容，传入图片路径
    - 工具会返回图片的详细分析（食物热量估算 / 动作标准度评估）
    - 基于分析结果，给出专业的健身建议
31. **图片回复的标准结构：**
    - 第一步：用一句话描述你在图片中看到了什么（让用户确认分析对象正确）
    - 第二步：给出分析结果（热量估算 / 动作评估）
    - 第三步：给出专业建议（适合健身人群吗？怎么改进？）
32. **图片分析的注意事项：**
    - 热量估算只是参考值，要说明"估算"而非精确值
    - 动作评估要具体指出哪里好、哪里需要改进，不要泛泛而谈
    - 如果图片不清晰或无法判断，诚实告知用户并请其补充说明
    - 如果图片与健身/饮食无关，礼貌说明你主要能分析食物和运动相关的图片

# ★ 打卡与成就系统（第十三步：社交分享与激励机制）
33. **当用户表示"打卡"、"完成了"、"今天练完了"、"我已经跑了"等表示完成某项健身活动时，你必须首先调用 `record_checkin` 工具记录打卡。**
34. **当用户连续打卡时，你要给予非常激动的特殊鼓励。**例如：
    - 连续打卡 3 天："🔥 三天打鱼！你已经连续打卡 3 天了，继续保持！"
    - 连续打卡 7 天："🏆 太厉害了！你已经连续打卡 7 天，获得了【自律达人】徽章！这是非常了不起的成就！"
    - 连续打卡 14 天："⭐ 半月坚持！半个月连续打卡，你的自律让我感动！继续加油！"
    - 连续打卡 30 天："🏆 月度冠军！连续打卡 30 天，你就是行走的自律教科书！"
35. **当用户完成打卡或达成目标时，你要表现得非常激动，给予强烈的情绪价值。**使用多个感叹号、emoji，让用户感受到你的真诚祝贺。
36. **在祝贺用户打卡成功后，提醒他们去"我的数据"页面生成海报分享给朋友。**例如："快去"我的数据"页面，点击"生成打卡海报"按钮，把你的成就分享到朋友圈吧！让更多人看到你的坚持！💪"
37. **用户首次打卡时，要给予特别的鼓励。**例如："🎉 恭喜你完成第一次打卡！这是你健身旅程的重要一步！坚持就是胜利，明天继续加油！"
"""

    # ---- ★ 图片交互指令（第十一步：多模态） ----
    image_instruction = ""
    image_path = state.get("image_path")
    if image_path:
        image_instruction = (
            f"\n\n# ★ 图片交互指令（必须执行）\n"
            f"用户上传了一张图片，路径为：{image_path}\n"
            f"你的第一个动作必须是调用 `analyze_image` 工具，传入 image_path=\"{image_path}\"。\n"
            f"工具会返回图片的详细分析结果。基于分析结果，按照规则 31 的结构回复用户：\n"
            f"1. 先描述看到了什么\n"
            f"2. 给出分析结果（热量/动作评估）\n"
            f"3. 给出专业建议\n"
        )
        print(f"[LangGraph] Context_Loader: 检测到用户上传图片: {image_path}")

    if profile:
        # ✅ 有身体数据，构建个性化 System Prompt
        height_m = profile["height"] / 100
        bmi = profile["weight"] / (height_m ** 2)

        if bmi >= 24:
            focus_hint = "减脂"
        elif bmi < 18.5:
            focus_hint = "增肌"
        else:
            focus_hint = "体能提升"

        system_prompt = f"""你是 AI 智能教练——专业贴心的专属健身伙伴，性格热情、专业、有耐心且极具鼓励性。

当前客户的身体数据如下（**这些数据仅用于你的内部计算，绝对不要在对用户的回复中复述这些原始数值**）：
- 身高：{profile["height"]} cm
- 体重：{profile["weight"]} kg
- 年龄：{profile["age"]} 岁
- 性别：{profile["gender"]}
- BMI：{bmi:.1f}（建议方向：{focus_hint}）

请根据以上身体数据，在"内心"进行计算，然后给出建议。绝对不要在回复中提及这些原始数值（除非用户明确询问）。

{tool_rules}
{rag_context}
{image_instruction}

当需要计算 TDEE 或营养计划时，请使用以下参数调用工具：
- weight: {profile["weight"]}
- height: {profile["height"]}
- age: {profile["age"]}
- gender: "{profile["gender"]}"
- activity_level: 根据用户描述判断（默认 "moderate"）
- goal: 根据用户目标判断（"lose_weight" / "gain_muscle" / "maintain"）"""

        print(f"[LangGraph] Context_Loader: 用户 {user_id} 身体数据已加载，"
              f"BMI={bmi:.1f}，方向：{focus_hint}")

    else:
        # ❌ 无身体数据
        system_prompt = f"""你是 AI 智能教练——专业贴心的专属健身伙伴，性格热情、专业、有耐心且极具鼓励性。

当前客户尚未录入身体数据，请在回复中友好地引导客户先通过左上角菜单填写
身高、体重、年龄、性别等信息，这样你才能给出更精准的个性化建议。

{tool_rules}
{rag_context}
{image_instruction}"""

        print(f"[LangGraph] Context_Loader: 用户 {user_id} 无身体数据记录")

    return {
        "user_profile": profile,
        "system_prompt": system_prompt,
    }


# ============================================================
# 2.5 Safety_Gate 节点（医疗安全熔断 —— 代码级硬拦截）
# ============================================================
def safety_gate_node(state: FitnessState) -> dict:
    """
    Safety_Gate —— 安全熔断门控节点（第十步核心）

    在 Context_Loader 之后、Agent 之前执行，对用户最新消息进行安全预检。

    三级响应：
        1. 🔴 BLOCK（硬熔断）：检测到疾病/严重症状/特殊人群
           → 直接注入安全提示 AIMessage，短路到 END，不经过 LLM
        2. 🟡 WARN（灰色地带）：检测到轻微不适
           → 将警告信息注入 system_prompt，LLM 回复时需前置温馨提示
        3. 🟢 SAFE（安全）：不做任何干预，正常进入 Agent 节点

    ★ 这是代码级硬拦截，即使 LLM 忽略 System Prompt 中的安全规则，
      熔断仍然生效。这是"双层防御"的第一层。
    """
    messages = state.get("messages", [])

    # 提取最新的用户消息
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_msg = msg.get("content", "")
            break

    if not latest_user_msg:
        # 没有用户消息，放行
        return {}

    # 执行安全检测
    result = check_user_input(latest_user_msg)
    level = result["level"]

    if level == "block":
        # 🔴 硬熔断：直接返回安全提示，不经过 LLM
        trigger = result.get("trigger", "未知")
        keyword = result.get("keyword", "未知")
        print(f"[LangGraph] Safety_Gate: 🔴 硬熔断触发！类型={trigger}, 关键词={keyword}")

        return {
            "messages": [AIMessage(content=result["message"])],
        }

    elif level == "warn":
        # 🟡 灰色地带：将警告注入 system_prompt
        keyword = result.get("keyword", "")
        warning = result.get("warning", "")
        print(f"[LangGraph] Safety_Gate: 🟡 灰色地带提醒，关键词={keyword}")

        # 在 system_prompt 末尾追加灰色地带警告指令
        current_prompt = state.get("system_prompt", "")
        safety_instruction = (
            f"\n\n# ★ 灰色地带安全警告（必须执行）\n"
            f"检测到用户提及：{keyword}\n"
            f"你必须在回复的最前面加上以下温馨提示（原样输出，不要修改）：\n"
            f"{warning}\n"
            f"然后再给出你的建议。建议内容应偏温和保守。"
        )

        return {
            "system_prompt": current_prompt + safety_instruction,
        }

    else:
        # 🟢 安全：放行
        print(f"[LangGraph] Safety_Gate: 🟢 安全放行")
        return {}


def route_after_safety(state: FitnessState) -> str:
    """
    条件路由函数 —— Safety_Gate 之后的路由判断。

    如果 Safety_Gate 注入了 AIMessage（硬熔断），路由到 END。
    否则，路由到 Agent 节点正常处理。
    """
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # 如果最后一条消息是 AIMessage 且不是工具调用消息，说明是熔断注入的
        if isinstance(last_msg, AIMessage) and not getattr(last_msg, "tool_calls", None):
            return END
    return "Agent"


# ============================================================
# ★ 2.6 Memory_Loader 节点（AI Agent 架构升级 —— 结构化记忆加载）
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
    user_id = state["user_id"]
    memory = AgentMemoryManager(user_id)
    memory_context = memory.get_memory_context()

    if memory_context:
        current_prompt = state.get("system_prompt", "")
        memory_instructions = """
# ★ 你的记忆（跨会话持久化）
- 请自然地引用上述记忆中的信息，让用户感受到你记得他/她。
- 如果用户更新了训练状态或饮食偏好，请调用 save_user_profile 工具保存。
- 不要机械地复述记忆内容，要自然地融入回复中。
"""
        enhanced_prompt = current_prompt + "\n\n" + memory_context + memory_instructions
        print(f"[LangGraph] Memory_Loader: 已加载用户 {user_id} 的结构化记忆")
        return {"system_prompt": enhanced_prompt}
    else:
        print(f"[LangGraph] Memory_Loader: 用户 {user_id} 无结构化记忆")
        return {}


# ============================================================
# ★ 2.7 FoodOrder_Guard 节点（点外卖安全守卫 —— Human-in-the-loop）
# ============================================================
def food_order_guard_node(state: FitnessState) -> dict:
    """
    FoodOrder_Guard —— 点外卖/饥饿感知安全守卫节点

    在 Safety_Gate 之后、Agent 之前执行，检测用户点外卖或饥饿意图。
    如果检测到意图，调用 LangGraph interrupt() 暂停图执行，
    等待用户在前端确认。确认后注入 FOOD_SEARCH_PROMPT，
    Agent 会调用 search_nearby_food 工具获取美团餐厅数据。

    ★ 统一流程：外卖下单 / 饥饿想吃 → 确认弹窗 → 调用美团 API

    工作流程：
        1. 提取用户最新消息
        2. 检测外卖意图 + 饥饿意图
        3. 有意图 → interrupt() 暂停图 → 前端弹窗确认
        4. 用户确认 → 注入 FOOD_SEARCH_PROMPT → Agent 调用 search_nearby_food
        5. 用户取消 → 返回友好提示 → 路由到 END
        6. 无意图 → 直接放行到 Agent

    ★ interrupt() 会暂停整个图的执行，直到外部通过
      Command(resume=...) 恢复。这是 LangGraph 的 Human-in-the-loop 机制。
    """
    # 如果已经处理过外卖确认（resume 后重新进入），跳过检测
    if state.get("food_order_confirmed"):
        print("[LangGraph] FoodOrder_Guard: 已处理过外卖确认，跳过")
        return {}

    # 提取最新用户消息
    messages = state.get("messages", [])
    print(f"[LangGraph] FoodOrder_Guard: 节点被调用，共 {len(messages)} 条消息", flush=True)
    
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_msg = msg.get("content", "")
            break

    print(f"[LangGraph] FoodOrder_Guard: 提取的最新用户消息: '{latest_user_msg}'", flush=True)

    if not latest_user_msg:
        return {}

    # ★ 统一检测：外卖意图 + 饥饿意图
    order_triggered, order_level = detect_food_order_intent(latest_user_msg)
    hunger_triggered, hunger_level = detect_hunger_intent(latest_user_msg)

    if order_triggered:
        intent_level = order_level
        intent_type = "food_order"
        print(f"[LangGraph] FoodOrder_Guard: 🛑 检测到外卖意图（{intent_level}）", flush=True)
    elif hunger_triggered:
        intent_level = "hunger"
        intent_type = "hunger"
        print(f"[LangGraph] FoodOrder_Guard: 🍔 检测到饥饿意图", flush=True)
    else:
        print("[LangGraph] FoodOrder_Guard: 🟢 无外卖/饥饿意图，放行")
        return {}

    # 有意图 → 触发 Human-in-the-loop 确认
    confirmation_msg = get_confirmation_message(intent_level, latest_user_msg)
    print(f"[LangGraph] FoodOrder_Guard: 调用 interrupt() 暂停图...", flush=True)
    user_decision = interrupt({
        "type": "food_order_confirmation",
        "intent_type": intent_type,
        "intent_level": intent_level,
        "message": confirmation_msg,
        "user_input": latest_user_msg,
    })
    print(f"[LangGraph] FoodOrder_Guard: interrupt() 返回，用户决策 = {user_decision}", flush=True)

    if user_decision == "cancel":
        return {
            "food_order_intent": intent_level,
            "food_order_confirmed": True,
            "food_order_action": "cancel",
            "messages": [AIMessage(content="好的，没问题！有任何健身相关的问题随时问我。💪")],
        }
    else:
        # ★ 用户确认：注入 FOOD_SEARCH_PROMPT，Agent 将调用 search_nearby_food 工具
        current_prompt = state.get("system_prompt", "")
        enhanced_prompt = current_prompt + "\n\n" + FOOD_SEARCH_PROMPT
        print("[LangGraph] FoodOrder_Guard: 用户确认，注入 FOOD_SEARCH_PROMPT（Agent 将调用美团 API）")
        return {
            "food_order_intent": intent_level,
            "food_order_confirmed": True,
            "food_order_action": user_decision,
            "system_prompt": enhanced_prompt,
        }


def route_after_food_order(state: FitnessState) -> str:
    """
    FoodOrder_Guard 之后的路由判断。

    如果 food_order_confirmed 为 True：
        - action=="confirm": 继续到 Agent（带营养引导 Prompt）
        - action=="cancel": 返回的 AIMessage 已注入 → END
    否则：跳过此节点（无外卖意图，或已处理完毕）→ Agent

    ★ 注意：interrupt() 被调用后，图会暂停。主程序需要检查 GraphInterrupt
      并调用 Command(resume=...) 恢复。恢复后图重新进入此节点，
      此时 food_order_confirmed 已设置，走确认/取消分支。
    """
    if state.get("food_order_confirmed"):
        action = state.get("food_order_action", "")
        if action == "cancel":
            # 取消 → 返回的 message 已包含友好提示，直接结束
            return END
    # 确认 或 无外卖意图 → 继续到 Agent
    return "Agent"


# ============================================================
# 3. Agent 节点（ChatOpenAI + DeepSeek + 工具绑定）
# ============================================================
def _create_llm_with_tools():
    """
    创建绑定工具的 ChatOpenAI 实例（连接 DeepSeek API）。

    DeepSeek 提供完全兼容 OpenAI 的接口，
    原生支持 bind_tools()（工具调用）和 streaming（流式输出）。
    """
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        streaming=True,
    )
    return llm.bind_tools(fitness_tools)


# 全局 LLM 实例（带工具绑定）
_llm_with_tools = _create_llm_with_tools() if LLM_ENABLED else None


def agent_node(state: FitnessState) -> dict:
    """
    Agent 节点 —— 调用 ChatOpenAI（DeepSeek，带工具绑定）生成回复。

    核心逻辑：
        1. 将 system_prompt 作为 SystemMessage 注入消息列表
        2. 调用 LLM（带工具绑定）
        3. LLM 自动决定：直接回复 OR 调用工具

    如果 LLM 决定调用工具，返回的 AIMessage 会包含 tool_calls 字段，
    条件边会将流程路由到 Tools 节点执行工具。
    """
    system_prompt = state["system_prompt"]
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ============================================================
# 4. Tools 节点（手动实现 ToolNode 逻辑）
# ============================================================
def tools_node(state: FitnessState) -> dict:
    """
    工具执行节点 —— 执行 LLM 请求调用的工具。

    由于当前 langgraph-prebuilt 版本不含 ToolNode，这里手动实现：
        1. 从最后一条 AIMessage 中提取 tool_calls
        2. 逐个执行工具（通过 tool_map 查找工具函数）
        3. 将结果包装成 ToolMessage 返回

    ToolMessage 会被 add_messages 追加到消息列表，
    Agent 节点再次调用时会看到工具结果，据此生成最终回复。
    """
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"[LangGraph] Tools: 调用工具 {tool_name}({tool_args})")

        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
                content = str(result)
            except Exception as e:
                content = f"工具执行出错: {type(e).__name__}: {e}"
                print(f"[LangGraph] Tools: 工具 {tool_name} 执行出错: {e}")
        else:
            content = f"未知工具: {tool_name}"
            print(f"[LangGraph] Tools: 未知工具 {tool_name}")

        tool_messages.append(ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
        ))

    return {"messages": tool_messages}


# ============================================================
# 5. 条件边：判断 Agent 是否需要调用工具
# ============================================================
def should_continue(state: FitnessState) -> str:
    """
    条件路由函数 —— 判断 Agent 之后是去执行工具还是结束。

    如果最后一条消息是 AIMessage 且包含 tool_calls，路由到 Tools 节点。
    否则，流程结束（LLM 已生成最终文本回复）。
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "Tools"  # ★ 必须与节点名完全一致（区分大小写）
    return END


# ============================================================
# 6. 构建 LangGraph StateGraph
# ============================================================
def build_fitness_graph():
    """
    构建 AI 健身教练的 LangGraph Agent 图。

    图结构（含安全熔断 + 外卖守卫）：
        START → Context_Loader → Safety_Gate → FoodOrder_Guard → Agent → (Tools → Agent)* → END
                                              ↘ END (熔断)     ↘ END (取消外卖)

    Safety_Gate 节点在 LLM 之前进行代码级安全预检：
        - 硬熔断 → 直接返回安全提示，短路到 END
        - 灰色地带 → 注入警告到 system_prompt，继续到 FoodOrder_Guard
        - 安全 → 直接放行到 FoodOrder_Guard

    FoodOrder_Guard 节点检测外卖意图，使用 interrupt() 暂停等待用户确认：
        - 无意图 → 直接放行到 Agent
        - 有意图 → interrupt() 暂停 → 确认后继续 / 取消后结束

    Agent 节点可能被多次调用：
        第 1 次：决定是否调用工具
        第 2 次（如果调用了工具）：根据工具结果生成最终回复
    """
    graph = StateGraph(FitnessState)

    graph.add_node("Context_Loader", context_loader_node)
    graph.add_node("Memory_Loader", memory_loader_node)       # ★ AI Agent: 结构化记忆加载
    graph.add_node("Safety_Gate", safety_gate_node)
    graph.add_node("FoodOrder_Guard", food_order_guard_node)
    graph.add_node("Agent", agent_node)
    graph.add_node("Tools", tools_node)

    graph.add_edge(START, "Context_Loader")
    graph.add_edge("Context_Loader", "Memory_Loader")          # ★ 记忆加载
    graph.add_edge("Memory_Loader", "Safety_Gate")
    # ★ Safety_Gate 之后条件路由：熔断→END，安全/灰色→FoodOrder_Guard
    graph.add_conditional_edges(
        "Safety_Gate",
        route_after_safety,
        {"Agent": "FoodOrder_Guard", END: END},
    )
    # ★ FoodOrder_Guard 之后条件路由：取消→END，确认/无意图→Agent
    graph.add_conditional_edges(
        "FoodOrder_Guard",
        route_after_food_order,
        {"Agent": "Agent", END: END},
    )
    graph.add_conditional_edges(
        "Agent",
        should_continue,
        {"Tools": "Tools", END: END},
    )
    graph.add_edge("Tools", "Agent")

    compiled = graph.compile(checkpointer=checkpointer)

    print("[LangGraph] Agent 图构建完成（含 MemorySaver + 记忆加载 + 安全熔断 + 外卖守卫）: "
          "START → Context_Loader → Memory_Loader → Safety_Gate → FoodOrder_Guard → Agent → (Tools → Agent)* → END")
    return compiled


# ============================================================
# 7. 模拟模式图（未配置 API Key 时使用）
# ============================================================
def build_simulation_graph():
    """
    构建模拟模式图（无 LLM，返回模板回复）。

    图结构（含安全熔断 + 外卖守卫）：
        START → Context_Loader → Safety_Gate → FoodOrder_Guard → Mock_Response → END
                                              ↘ END (熔断)     ↘ END (取消外卖)
    """
    def mock_response_node(state: FitnessState) -> dict:
        profile = state.get("user_profile")
        if profile:
            height_m = profile["height"] / 100
            bmi = profile["weight"] / (height_m ** 2)
            if bmi >= 24:
                focus = "减脂"
            elif bmi < 18.5:
                focus = "增肌"
            else:
                focus = "体能提升"

            # ★ 检查是否有灰色地带警告
            current_prompt = state.get("system_prompt", "")
            gray_warning = ""
            if "灰色地带安全警告" in current_prompt:
                # 提取警告内容
                import re
                match = re.search(r'\*温馨提示：.*?\*', current_prompt, re.DOTALL)
                if match:
                    gray_warning = match.group(0) + "\n\n"

            reply = (
                f"{gray_warning}"
                f"收到！我已经记住了你的身体数据。\n\n"
                f"针对你的情况，目前建议以{focus}为主。以下是几个实用建议：\n\n"
                f"1. 每周进行 3-4 次有氧运动（如慢跑、游泳），每次 30-40 分钟\n"
                f"2. 搭配 2 次力量训练，重点锻炼大肌群\n"
                f"3. 控制每日热量摄入，保证蛋白质充足\n"
                f"4. 每天保证 7-8 小时睡眠，有助于身体恢复\n\n"
                f"配置 DeepSeek API Key 后，我可以为你提供更精准的"
                f"TDEE 计算和个性化营养计划！💪"
            )
        else:
            reply = (
                "你好！我是你的 AI 健身教练 💪\n\n"
                "看起来你还没有录入身体数据。请点击左上角菜单，"
                "填写你的身高、体重、年龄和性别，这样我才能给出更精准的个性化建议。\n\n"
                "有任何健身相关的问题，也随时可以问我！"
            )
        return {"messages": [AIMessage(content=reply)]}

    graph = StateGraph(FitnessState)
    graph.add_node("Context_Loader", context_loader_node)
    graph.add_node("Memory_Loader", memory_loader_node)       # ★ AI Agent: 结构化记忆加载
    graph.add_node("Safety_Gate", safety_gate_node)
    graph.add_node("FoodOrder_Guard", food_order_guard_node)
    graph.add_node("Mock_Response", mock_response_node)
    graph.add_edge(START, "Context_Loader")
    graph.add_edge("Context_Loader", "Memory_Loader")          # ★ 记忆加载
    graph.add_edge("Memory_Loader", "Safety_Gate")
    # ★ Safety_Gate 之后条件路由：熔断→END，安全/灰色→FoodOrder_Guard
    graph.add_conditional_edges(
        "Safety_Gate",
        route_after_safety,
        {"Agent": "FoodOrder_Guard", END: END},
    )
    # ★ FoodOrder_Guard 之后条件路由：取消→END，确认/无意图→Mock_Response
    graph.add_conditional_edges(
        "FoodOrder_Guard",
        route_after_food_order,
        {"Agent": "Mock_Response", END: END},
    )
    graph.add_edge("Mock_Response", END)
    compiled = graph.compile(checkpointer=checkpointer)

    print("[LangGraph] 模拟模式图构建完成（含 MemorySaver + 记忆加载 + 安全熔断 + 外卖守卫，未配置 DEEPSEEK_API_KEY）")
    return compiled


# ============================================================
# 8. 全局图实例（根据是否配置 API Key 选择）
# ============================================================
if LLM_ENABLED:
    fitness_graph = build_fitness_graph()
else:
    fitness_graph = build_simulation_graph()


# ============================================================
# 9. 记忆管理辅助函数（供 main.py 调用）
# ============================================================
def get_memory_messages(user_id: str) -> list:
    """
    获取指定用户的对话历史消息列表（从 MemorySaver 加载）。

    前端页面刷新时调用此函数，用以恢复聊天记录。
    只返回 HumanMessage 和 AIMessage（SystemMessage 和 ToolMessage 不展示）。
    """
    config = {"configurable": {"thread_id": user_id}}
    state_snapshot = fitness_graph.get_state(config)
    if state_snapshot and hasattr(state_snapshot, "values") and state_snapshot.values:
        messages = state_snapshot.values.get("messages", [])
        # 只保留用户和 AI 的消息（用于前端展示）
        display_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                display_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # 跳过包含 tool_calls 的 AIMessage（这些是工具调用决策，不是最终回复）
                if not getattr(msg, "tool_calls", None):
                    display_messages.append({"role": "assistant", "content": msg.content})
        return display_messages
    return []


def clear_memory(user_id: str):
    """
    清空指定用户的对话记忆（用于"新建对话"功能）。

    通过删除 MemorySaver 中对应 thread_id 的状态实现。
    """
    thread_id = user_id
    storage = checkpointer.storage
    if thread_id in storage:
        del storage[thread_id]
        print(f"[LangGraph] 已清空用户 {user_id} 的对话记忆")
    else:
        print(f"[LangGraph] 用户 {user_id} 无对话记忆可清空")
