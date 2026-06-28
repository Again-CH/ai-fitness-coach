# -*- coding: utf-8 -*-
"""
multimodal_service.py —— 多模态服务模块（语音 + 视觉）
=========================================================
本模块封装了 AI 健身教练的"耳朵"和"眼睛"：

1. 语音识别 (ASR) —— 腾讯云一句话识别
   用户在前端点击麦克风说话 → 录音 → 发送到后端 → 调用腾讯云 ASR → 返回文字

2. 语音合成 (TTS) —— 腾讯云语音合成
   AI 生成的文字回复 → 调用腾讯云 TTS → 返回 MP3 音频 → 前端播放

3. 图像理解 (Vision) —— 通义千问 Qwen-VL
   用户上传食物/动作照片 → 调用 Qwen-VL 视觉模型 → 返回分析结果
   - 食物照片：估算热量和营养成分
   - 动作照片：判断动作是否标准

★ 为什么 Vision 用 Qwen-VL 而不是腾讯云？
  腾讯云的图像 API 侧重于分类/检测（"这是一碗面"），
  而 Qwen-VL 是多模态大模型，可以理解图片内容并用自然语言分析
  （"这是一碗牛肉面，面条约200g，牛肉约50g，估算热量500-600大卡"），
  更适合健身场景的智能分析。

依赖：
  - tencentcloud-sdk-python（腾讯云 SDK）
  - dashscope（通义千问 SDK，已在项目中使用）
"""

import base64
import os
import time
import uuid
import tempfile

from config import (
    TENCENT_SECRET_ID,
    TENCENT_SECRET_KEY,
    TENCENT_APP_ID,
    TENCENT_TTS_VOICE_TYPE,
    QWEN_VL_MODEL,
    DASHSCOPE_API_KEY,
    TENCENT_ENABLED,
)


# ============================================================
# 1. 语音识别 (ASR) —— 腾讯云一句话识别
# ============================================================
def speech_to_text(audio_data: bytes, audio_format: str = "wav") -> dict:
    """
    调用腾讯云 ASR 一句话识别 API，将语音转为文字。

    使用场景：
        用户在前端按住麦克风说话 → MediaRecorder 录音为 webm/wav →
        上传到后端 → 本函数调用腾讯云 ASR → 返回识别文字

    Args:
        audio_data:  音频文件的二进制数据（wav / mp3 / m4a / webm 等）
        audio_format: 音频格式，如 "wav" / "mp3" / "m4a" / "webm"

    Returns:
        {
            "success": True,
            "text": "识别出的文字内容",
        }
        或
        {
            "success": False,
            "error": "错误描述",
        }

    技术细节：
        - 腾讯云 ASR 一句话识别 (SentenceRecognition) 支持 60 秒以内的音频
        - 音频数据通过 Base64 编码上传
        - 支持中文普通话 (16k) 和英文
        - 引擎模型：16k_zh（16kHz 中文普通话）
    """
    if not TENCENT_ENABLED:
        return {
            "success": False,
            "error": "腾讯云密钥未配置，请在 .env 文件中设置 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY",
        }

    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.asr.v20190614 import asr_client, models as asr_models

        # ---- 构建腾讯云客户端 ----
        cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
        http_profile = HttpProfile(endpoint="asr.tencentcloudapi.com")
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client = asr_client.AsrClient(cred, "", client_profile)

        # ---- 音频格式映射 ----
        # 腾讯云 ASR 支持的格式：wav / pcm / ogg-opus / speex / m4a / mp3
        # webm 格式需要转换为 wav（浏览器 MediaRecorder 默认输出 webm）
        format_map = {
            "wav": "wav",
            "mp3": "mp3",
            "m4a": "m4a",
            "pcm": "pcm",
            "ogg": "ogg-opus",
            "webm": "mp3",  # webm 暂时映射为 mp3（前端应转为 wav 再上传）
        }
        eng_format = format_map.get(audio_format.lower(), "wav")

        # ---- Base64 编码音频数据 ----
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        # ---- 构建请求参数 ----
        req = asr_models.SentenceRecognitionRequest()
        req.ProjectId = int(TENCENT_APP_ID) if TENCENT_APP_ID else 0
        req.SubServiceType = 2       # 2 = 一句话识别
        req.EngSerViceType = "16k_zh"  # 16kHz 中文普通话引擎
        req.SourceType = 1           # 1 = 语音数据直接上传（Base64）
        req.VoiceFormat = eng_format
        req.Data = audio_b64
        req.DataLen = len(audio_data)
        # 可选参数
        req.UsrAudioKey = "fitness_coach_user"

        # ---- 发起请求 ----
        resp = client.SentenceRecognition(req)
        result_text = resp.Result

        print(f"[Multimodal] ASR 识别成功：{result_text[:50]}...")
        return {
            "success": True,
            "text": result_text,
        }

    except Exception as e:
        print(f"[Multimodal] ASR 识别失败: {e}")
        return {
            "success": False,
            "error": f"语音识别失败: {str(e)}",
        }


# ============================================================
# 2. 语音合成 (TTS) —— 腾讯云语音合成
# ============================================================
def text_to_speech(text: str, voice_type: int = None) -> dict:
    """
    调用腾讯云 TTS 语音合成 API，将文字转为 MP3 音频。
    ...
    """
    if not TENCENT_ENABLED:
        # 未配置腾讯云密钥：返回特定错误码，前端会降级到浏览器 Web Speech API
        print(f"[Multimodal] TTS：腾讯云密钥未配置，返回降级提示")
        return {
            "success": False,
            "error": "TENCENT_NOT_CONFIGURED",
            "message": "语音服务暂时不可用，正在使用浏览器语音播放",
        }

    # 超长文本截断（腾讯云基础 TTS 限制 150 字/次）
    max_chars = 150
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"[Multimodal] TTS 文本超过 {max_chars} 字，已截断")

    # 使用传入的音色或默认配置
    vt = voice_type if voice_type is not None else TENCENT_TTS_VOICE_TYPE

    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.tts.v20190823 import tts_client, models as tts_models

        # ---- 构建腾讯云客户端 ----
        cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
        http_profile = HttpProfile(endpoint="tts.tencentcloudapi.com")
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client = tts_client.TtsClient(cred, "", client_profile)

        # ---- 构建请求参数 ----
        req = tts_models.TextToVoiceRequest()
        req.Text = text
        req.SessionId = "fitness_coach_tts"
        req.Volume = 5          # 音量：0-10，5 为适中
        req.Speed = 1           # 语速：-2~6，1 为略快（更有活力）
        req.VoiceType = vt      # 音色 ID
        req.PrimaryLanguage = 1 # 1 = 中文
        req.SampleRate = 16000  # 16kHz
        req.Codec = "mp3"       # 输出格式

        # ---- 发起请求 ----
        resp = client.TextToVoice(req)

        # resp.Audio 是 Base64 编码的音频数据
        print(f"[Multimodal] TTS 合成成功：文本长度={len(text)}，音色={vt}")
        return {
            "success": True,
            "audio": resp.Audio,  # Base64 编码的 MP3
            "format": "mp3",
        }

    except Exception as e:
        print(f"[Multimodal] TTS 合成失败: {e}")
        return {
            "success": False,
            "error": f"语音合成失败: {str(e)}",
        }


# ============================================================
# 3. 图像理解 (Vision) —— 通义千问 Qwen-VL
# ============================================================
def analyze_image_with_vlm(image_path: str, question: str = "") -> dict:
    """
    调用通义千问 Qwen-VL 视觉模型，分析图片内容。

    使用场景：
        用户上传食物照片 → 估算热量和营养成分
        用户上传动作照片 → 判断动作是否标准

    本函数是 tools.py 中 analyze_image 工具的底层实现。

    Args:
        image_path: 图片文件路径（本地路径）
        question:   用户的附加问题或分析指令

    Returns:
        {
            "success": True,
            "analysis": "Qwen-VL 返回的分析文本",
        }
        或
        {
            "success": False,
            "error": "错误描述",
        }

    技术细节：
        - 使用 dashscope.MultiModalConversation 调用 qwen-vl-plus 模型
        - 图片通过本地文件路径传入（file:// 协议）
        - Prompt 中包含食物热量估算和动作评估的专业指令
    """
    if not DASHSCOPE_API_KEY:
        return {
            "success": False,
            "error": "DashScope API Key 未配置，无法调用 Qwen-VL",
        }

    # ---- 构建专业的分析 Prompt ----
    default_prompt = (
        "你是一个专业的健身营养教练。请仔细分析这张图片，判断它是食物还是运动动作。\n\n"
        "如果是食物：\n"
        "1. 描述食物的名称和主要食材\n"
        "2. 估算大致的份量（克数）\n"
        "3. 估算总热量（大卡）和主要营养成分（蛋白质/碳水/脂肪，克数）\n"
        "4. 给出适合健身人群的食用建议\n\n"
        "如果是运动动作：\n"
        "1. 描述动作名称\n"
        "2. 评估动作是否标准（如脊柱中立、膝盖方向、核心收紧等）\n"
        "3. 指出可能存在的问题和改进建议\n"
        "4. 提醒注意事项\n\n"
        "请用简洁清晰的中文回答，适合语音朗读。"
    )

    user_prompt = question if question.strip() else default_prompt

    try:
        import dashscope
        from dashscope import MultiModalConversation

        # ---- 构建 Qwen-VL 消息 ----
        # Qwen-VL 的消息格式：content 是一个列表，包含 image 和 text 类型的消息
        messages = [{
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": user_prompt},
            ],
        }]

        # ---- 调用 Qwen-VL ----
        response = MultiModalConversation.call(
            model=QWEN_VL_MODEL,
            messages=messages,
            api_key=DASHSCOPE_API_KEY,
        )

        if response.status_code == 200:
            # 提取回复文本
            result_text = ""
            if response.output and response.output.choices:
                for choice in response.output.choices:
                    if choice.message and choice.message.content:
                        for item in choice.message.content:
                            if "text" in item:
                                result_text += item["text"]

            print(f"[Multimodal] Qwen-VL 图像分析成功：{result_text[:80]}...")
            return {
                "success": True,
                "analysis": result_text,
            }
        else:
            error_msg = f"Qwen-VL 返回错误: code={response.code}, msg={response.message}"
            print(f"[Multimodal] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
            }

    except Exception as e:
        print(f"[Multimodal] Qwen-VL 图像分析失败: {e}")
        return {
            "success": False,
            "error": f"图像分析失败: {str(e)}",
        }


# ============================================================
# 4. 辅助函数：保存 Base64 图片到临时文件
# ============================================================
def save_base64_image(base64_data: str, prefix: str = "upload") -> str:
    """
    将前端上传的 Base64 编码图片保存为临时文件，返回文件路径。

    前端通过 FileReader.readAsDataURL 获取的 Base64 格式为：
        data:image/jpeg;base64,/9j/4AAQ...

    本函数会去掉 data URI 前缀，解码并保存为临时文件。

    Args:
        base64_data: Base64 编码的图片数据（可能包含 data URI 前缀）
        prefix:      临时文件名前缀

    Returns:
        临时图片文件路径（如 /tmp/upload_xxx.jpg）

    ★ 调用方应在使用完毕后删除临时文件。
    """
    # 去掉 data URI 前缀（如 "data:image/jpeg;base64,"）
    if "," in base64_data and base64_data.startswith("data:"):
        header, base64_data = base64_data.split(",", 1)
        # 从 header 推断图片格式
        if "jpeg" in header or "jpg" in header:
            ext = "jpg"
        elif "png" in header:
            ext = "png"
        elif "webp" in header:
            ext = "webp"
        elif "gif" in header:
            ext = "gif"
        else:
            ext = "jpg"
    else:
        ext = "jpg"

    # 解码 Base64
    image_bytes = base64.b64decode(base64_data)

    # 保存到临时文件
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.{ext}")

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    print(f"[Multimodal] 图片已保存到临时文件: {file_path} ({len(image_bytes)} bytes)")
    return file_path
