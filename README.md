# AI Fitness Coach Agent

基于大语言模型的智能健身顾问系统，支持自然语言对话、个性化训练计划生成、饮食方案计算及多模态交互。

## 技术栈

- **Agent 框架**: LangGraph (StateGraph, Checkpointer, interrupt/resume)
- **LLM**: DeepSeek (OpenAI 兼容接口)
- **后端**: FastAPI + Uvicorn
- **数据库**: SQLite (8 张表)
- **鉴权**: JWT (PyJWT)
- **多模态**: 腾讯云 ASR/TTS + Qwen-VL
- **流式传输**: SSE (Server-Sent Events)

## 核心功能

### 1. LangGraph 多节点 Agent 流水线

```
START → ContextLoader → MemoryLoader → SafetyGate → Agent → Tools → END
                                  │
                                  └─ 安全熔断 → END（硬拦截）
```

### 2. 双层安全熔断

- **代码级硬拦截**: 疾病/症状/特殊人群关键词检测，命中直接截断 LLM 调用
- **Prompt 软防御**: System Prompt 中嵌入安全规则，双层防御即使 LLM 忽略指令仍生效

### 3. Human-in-the-loop

基于 LangGraph `interrupt()` / `Command(resume=...)` 实现，高风险操作（如点外卖建议）需用户二次确认。

### 4. 三层记忆系统

| 层级 | 实现 | 作用 |
|------|------|------|
| 短期记忆 | LangGraph Checkpointer (`add_messages`) | 当前对话上下文 |
| 结构化记忆 | SQLite `agent_memory` 表 | 用户训练状态、饮食偏好 |
| 会话摘要 | LLM 生成摘要存入数据库 | 跨会话上下文连贯 |

### 5. RAG 检索增强

构建健身知识库，通过关键词匹配检索相关条目注入 System Prompt；架构支持平滑迁移至向量检索（FAISS/Chroma）。

### 6. 多模态交互

- **图片理解**: 上传图片 → Qwen-VL 分析 → 融入对话上下文
- **语音对话**: 录音 → 腾讯云 ASR 转写 → 对话 → TTS 播报

### 7. SSE 流式输出

基于 `astream_events` 接口向前端推送打字机效果回复，工具调用期间展示 Thinking 状态提示。

## 项目结构

```
ai-fitness-coach/
├── backend/
│   ├── main.py                  # FastAPI 后端入口
│   ├── langgraph_brain.py      # LangGraph Agent 核心（StateGraph）
│   ├── react_loop.py           # ReAct 控制循环（双模式 Agent）
│   ├── tools.py                # @tool 工具定义（TDEE/宏量营养素/RAG）
│   ├── memory/                 # 三层记忆系统
│   ├── safety_filter.py        # 安全熔断检测器
│   ├── food_order_guard.py     # Human-in-the-loop 守卫
│   ├── multimodal_service.py   # 多模态服务（ASR/TTS/VL）
│   ├── database.py             # SQLite 操作层
│   ├── config.py               # 配置管理
│   ├── static/index.html       # 前端页面
│   └── requirements.txt        # Python 依赖
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── docs/
    └── agent-architecture-design.md
```

## 快速启动

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（复制模板后填写 API Key）
cp backend/.env.example backend/.env

# 4. 启动后端
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000` 即可使用。

## Agent 架构亮点（面试要点）

- **为什么选 LangGraph**：状态可暂停/恢复（Checkpointer）、条件路由（`add_conditional_edges`）、Human-in-the-loop（`interrupt()`）是其他框架（CrewAI/AutoGen）难以优雅实现的
- **ReAct vs Tool Calling**：本项目同时实现了两种模式，bind_tools 用于生产，ReAct 用于调试和可观测性
- **副作用隔离**：工具函数纯计算或无状态 DB 读取，不修改全局状态，便于测试和回放
- **SSE 流式输出**：`astream_events` 的 `on_chat_model_stream` 事件驱动前端渲染，工具调用时注入 Thinking 提示

## 证书

MIT License
