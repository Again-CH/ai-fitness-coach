# -*- coding: utf-8 -*-
"""
config.py —— 全局配置管理
====================================
集中管理 LLM 大模型配置和数据库路径。
所有配置通过环境变量读取，优先从项目根目录的 .env 文件加载。

★ 重要：本项目的 AI 对话使用 DeepSeek 模型（OpenAI 兼容接口）
  请在 backend/.env 文件中填入你的 DEEPSEEK_API_KEY。
"""

import os
from dotenv import load_dotenv

# 显式加载 backend 目录下的 .env 文件
# override=True 确保覆盖系统中已有同名环境变量
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=True)


# ============================================================
# DeepSeek 配置（OpenAI 兼容接口）
# ============================================================
# DeepSeek API Key
# 获取方式：https://platform.deepseek.com/api_keys
# 请在 backend/.env 文件中设置 DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# DeepSeek API Base URL（OpenAI 兼容接口）
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# 模型名称
# 常用选项：deepseek-chat（V3）、deepseek-reasoner（R1）
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


# ============================================================
# 腾讯云配置（ASR 语音识别 + TTS 语音合成）
# ============================================================
# 获取方式：https://console.cloud.tencent.com/cam/capi
# 请在 backend/.env 文件中设置 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY
TENCENT_SECRET_ID = os.environ.get("TENCENT_SECRET_ID", "")
TENCENT_SECRET_KEY = os.environ.get("TENCENT_SECRET_KEY", "")
TENCENT_APP_ID = os.environ.get("TENCENT_APP_ID", "")

# TTS 语音合成音色（默认 101014 晓浩-活力男声，适合健身教练场景）
# 可选值：101014(晓浩-男) / 101012(晓峰-男) / 101003(智美-女) / 101004(智云-男)
TENCENT_TTS_VOICE_TYPE = int(os.environ.get("TENCENT_TTS_VOICE_TYPE", "101014"))

# 通义千问视觉模型名称（用于图像理解）
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL", "qwen-vl-plus")

# ★ 向后兼容：multimodal_service.py 的图像分析仍使用 DashScope Qwen-VL
# 默认复用 DEEPSEEK_API_KEY（如果用户有独立的 DashScope key，可在 .env 中单独设置 DASHSCOPE_API_KEY）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", DEEPSEEK_API_KEY)
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")

# 便捷属性：判断是否已配置腾讯云密钥
TENCENT_ENABLED = bool(TENCENT_SECRET_ID and TENCENT_SECRET_KEY)


# ============================================================
# 数据库配置
# ============================================================
# SQLite 数据库文件路径，默认放在 backend 目录下
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitness.db"),
)


# ============================================================
# 便捷属性：判断是否已配置 DeepSeek API Key
# ============================================================
# 如果 DEEPSEEK_API_KEY 为空，后端进入模拟模式
LLM_ENABLED = bool(DEEPSEEK_API_KEY)
