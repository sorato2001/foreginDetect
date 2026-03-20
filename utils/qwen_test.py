# temp_url_llm_call_mcp.py
import os
import asyncio
from typing import Any, Dict, Optional, Literal
from concurrent.futures import ThreadPoolExecutor

import dashscope
# from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()
# mcp = FastMCP("TempUrlLLMCallServer")
executor = ThreadPoolExecutor(max_workers=4)

MediaType = Literal["image", "video"]


def _extract_text(resp: Any) -> str:
    """
    尽量从 DashScope 响应里抽取 assistant 文本，兼容对象/字典两种返回。
    """
    # 对象形态：resp.output.choices[0].message.content -> [{"text": "..."}]
    try:
        content = resp.output.choices[0].message.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
        if isinstance(content, str):
            return content
    except Exception:
        pass

    # 字典形态
    try:
        content = resp["output"]["choices"][0]["message"]["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
        if isinstance(content, str):
            return content
    except Exception:
        pass

    return str(resp)


def _call_multimodal(
    temp_url: str,
    prompt: str,
    model: str,
    media_type: MediaType,
    api_key: Optional[str],
) -> Dict[str, Any]:
    
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return {"status": "error", "msg": "请设置环境变量 DASHSCOPE_API_KEY 或传入 api_key 参数"}

    if not temp_url.startswith("oss://"):
        return {"status": "error", "msg": "temp_url 需要是 oss:// 开头的临时URL"}

    # DashScope 多模态对话消息结构：{"image": "..."} 或 {"video": "..."} + {"text": "..."}
    media_item = {media_type: temp_url}
    messages = [
        {
            "role": "user",
            "content": [
                media_item,
                {"text": prompt},
            ],
        }
    ]

    try:
        resp = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
        )
        return {
            "status": "success",
            "model": model,
            "media_type": media_type,
            "temp_url": temp_url,
            "prompt": prompt,
            "answer": _extract_text(resp),
            "raw": resp,
        }
    except Exception as e:
        return {
            "status": "error",
            "msg": str(e),
            "model": model,
            "media_type": media_type,
            "temp_url": temp_url,
        }


# @mcp.tool()
# async def llm_detect_with_temp_url(
#     temp_url: str,
#     prompt: str = "请描述内容，并给出关键目标/异常点。",
#     model: str = "qwen-vl-plus",
#     media_type: MediaType = "image",
#     api_key: Optional[str] = None,
# ) -> dict:
#     """
#     仅一个工具：用 DashScope 多模态大模型直接读取 oss:// 临时URL（图片/视频）并输出识别/检测结果。

#     参数：
#     - temp_url: oss://... 临时URL
#     - prompt: 提示词
#     - model: 模型名（需与上传临时URL时绑定的模型一致）
#     - media_type: "image" 或 "video"
#     - api_key: 可选；不填则读取环境变量 DASHSCOPE_API_KEY
#     """
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(
#         executor, _call_multimodal, temp_url, prompt, model, media_type, api_key
#     )


# if __name__ == "__main__":
#     mcp.run(transport="stdio")
