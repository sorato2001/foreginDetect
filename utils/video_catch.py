# camera_clip_upload_opencv_mcp.py
import os
import time
import uuid
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests
# from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

# mcp = FastMCP("CameraClipUploadServer")
executor = ThreadPoolExecutor(max_workers=1)

DASHSCOPE_UPLOAD_API = "https://dashscope.aliyuncs.com/api/v1/uploads"


# ----------------------------
# DashScope 临时文件上传（getPolicy + OSS表单上传）
# ----------------------------
def get_upload_policy(api_key: str, model_name: str, max_retries: int = 5) -> Dict[str, Any]:
    """
    获取文件上传凭证（getPolicy）
    注意：该接口有限流（按 主账号+模型 维度），这里做简单重试退避。:contentReference[oaicite:1]{index=1}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {"action": "getPolicy", "model": model_name}

    backoff = 0.5
    last_err = None
    for _ in range(max_retries):
        try:
            resp = requests.get(DASHSCOPE_UPLOAD_API, headers=headers, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                if "data" not in data:
                    raise RuntimeError(f"Unexpected response: {data}")
                return data["data"]

            # 429/5xx 做退避重试
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"{resp.status_code} {resp.text}"
                time.sleep(backoff)
                backoff = min(backoff * 2, 5)
                continue

            raise RuntimeError(f"Failed to get upload policy: {resp.status_code} {resp.text}")
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(backoff)
            backoff = min(backoff * 2, 5)

    raise RuntimeError(f"Failed to get upload policy after retries: {last_err}")


def upload_file_to_oss(policy_data: Dict[str, Any], file_path: str) -> str:
    """将文件上传到临时存储 OSS，返回 oss://... 临时URL。:contentReference[oaicite:2]{index=2}"""
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"

    with open(file_path, "rb") as f:
        files = {
            "OSSAccessKeyId": (None, policy_data["oss_access_key_id"]),
            "Signature": (None, policy_data["signature"]),
            "policy": (None, policy_data["policy"]),
            "x-oss-object-acl": (None, policy_data["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, policy_data["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (file_name, f),
        }
        resp = requests.post(policy_data["upload_host"], files=files, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to upload file: {resp.status_code} {resp.text}")

    return f"oss://{key}"


def upload_file_and_get_url(api_key: str, model_name: str, file_path: str) -> str:
    policy = get_upload_policy(api_key, model_name)
    return upload_file_to_oss(policy, file_path)


# ----------------------------
# OpenCV 截取视频（仅 OpenCV）
# ----------------------------
def run_clip_opencv(stream_url: str, duration_sec: int = 10, out_dir: str = ".") -> Dict[str, Any]:
    """
    从 RTSP/HTTP 拉流，截取指定时长，写成 MP4（逐帧写入）。
    """
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return {"status": "error", "msg": f"cannot open video stream: {stream_url}"}

    task_id = str(uuid.uuid4())[:8]
    out_path = str(Path(out_dir) / f"clip_{task_id}.mp4")

    # 读流信息（部分流会返回 0，兜底）
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 20.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # MP4 写入常用 fourcc：mp4v（能不能成功取决于你本机 OpenCV/编解码支持）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        cap.release()
        return {
            "status": "error",
            "msg": "cannot open VideoWriter (mp4v). "
                   "Your OpenCV build may lack MP4 encoder support. "
                   "Try rebuilding OpenCV with FFmpeg/GStreamer, or write AVI with XVID.",
        }

    frames = 0
    start = time.time()

    while time.time() - start < duration_sec:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # 有些流分辨率会变，简单兜底
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)
        frames += 1

    cap.release()
    writer.release()

    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    return {
        "status": "success",
        "video_path": out_path,
        "duration_sec": duration_sec,
        "frames": frames,
        "fps": float(fps),
        "width": w,
        "height": h,
        "file_bytes": size,
        "mode": "opencv",
    }


def clip_then_upload_opencv(
    stream_url: str,
    duration_sec: int,
    model_name: str,
    cleanup_local: bool = False,
) -> Dict[str, Any]:
    
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return {"status": "error", "msg": "Missing DASHSCOPE_API_KEY env var."}

    clip = run_clip_opencv(stream_url, duration_sec, out_dir=".")
    if clip.get("status") != "success":
        return clip

    video_path = clip["video_path"]
    oss_url = upload_file_and_get_url(api_key, model_name, video_path)

    # 文档说明临时 URL 有效期 48 小时:contentReference[oaicite:3]{index=3}
    expire_time = datetime.now() + timedelta(hours=48)

    result = {
        "status": "success",
        "clip_meta": clip,
        "temp_url": oss_url,  # oss://...
        "expires_at": expire_time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "note": (
            "If you call model via raw HTTP (curl/Postman), add header "
            "'X-DashScope-OssResourceResolve: enable'."
        ),
    }

    if cleanup_local:
        try:
            os.remove(video_path)
            result["video_deleted"] = True
        except Exception as e:
            result["video_deleted"] = False
            result["delete_error"] = str(e)

    return result


# ----------------------------
# MCP Tool：一键截取并上传（仅 OpenCV）
# ----------------------------
# @mcp.tool()
# async def camera_clip_and_upload(
#     url: str,
#     duration_sec: int = 10,
#     model_name: str = "qwen-vl-plus",
#     cleanup_local: bool = False,
# ) -> Dict[str, Any]:
#     """
#     用 OpenCV 从摄像头/视频流截取一段 MP4，然后上传到百炼临时存储，返回 oss:// 临时URL（48小时有效）。:contentReference[oaicite:4]{index=4}
#     """
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(
#         executor,
#         clip_then_upload_opencv,
#         url,
#         duration_sec,
#         model_name,
#         cleanup_local,
#     )


# if __name__ == "__main__":
#     mcp.run(transport="stdio")
