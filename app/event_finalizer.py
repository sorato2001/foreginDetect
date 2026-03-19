"""
Event finalizer module for composing recorded video segments into event videos.
Converts time-windowed segment collections into final MP4 files.
"""

import os
import logging
import subprocess
import threading
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from ultralytics import YOLO
import dashscope

from .models import ActiveEvent, EventRecord, EventStatus
from .utils import find_ts_files_in_range, generate_concat_file, get_video_output_path


logger = logging.getLogger(__name__)

DASHSCOPE_UPLOAD_API = "https://dashscope.aliyuncs.com/api/v1/uploads"


def get_upload_policy(api_key: str, model_name: str, max_retries: int = 5) -> Dict[str, Any]:
    """Get file upload policy from DashScope."""
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
    """Upload file to OSS and return temporary URL."""
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
    """Upload file and get OSS URL."""
    policy = get_upload_policy(api_key, model_name)
    return upload_file_to_oss(policy, file_path)


def extract_text(resp: Any) -> str:
    """Extract text from VLM response."""
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


def call_multimodal(
    temp_url: str,
    prompt: str,
    model: str,
    api_key: str,
) -> Dict[str, Any]:
    """Call Qwen VLM API with video URL."""
    if not temp_url.startswith("oss://"):
        return {"status": "error", "msg": "temp_url must start with oss://"}

    messages = [
        {
            "role": "user",
            "content": [
                {"video": temp_url},
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
            "answer": extract_text(resp),
            "raw": resp,
        }
    except Exception as e:
        return {
            "status": "error",
            "msg": str(e),
        }


class EventFinalizer:
    """Handles video composition and event finalization."""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize event finalizer.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path
        self.thread_pool: dict[str, threading.Thread] = {}
        
        logger.info("Initialized EventFinalizer")
    
    def finalize_event(
        self,
        active_event: ActiveEvent,
        cache_dir: str,
        output_dir: str,
        on_complete: Optional[callable] = None,
    ) -> None:
        """
        Finalize an event by composing video segments (async).
        
        Args:
            active_event: ActiveEvent to finalize
            cache_dir: Cache directory containing .ts files
            output_dir: Directory to save final video
            on_complete: Optional callback when finalization complete
        """
        # Use a unique thread key per camera
        thread_key = f"finalize_{active_event.camera_id}_{active_event.alarm_time.timestamp()}"
        
        # Check if already running
        if thread_key in self.thread_pool:
            existing_thread = self.thread_pool[thread_key]
            if existing_thread.is_alive():
                logger.warning(f"Finalization already running for {thread_key}")
                return
        
        # Start finalization in background thread
        thread = threading.Thread(
            target=self._finalize_sync,
            args=(active_event, cache_dir, output_dir, on_complete),
            daemon=True,
            name=thread_key,
        )
        thread.start()
        self.thread_pool[thread_key] = thread
        
        logger.info(f"Started async finalization for {active_event.camera_id}")
    
    def _finalize_sync(
        self,
        active_event: ActiveEvent,
        cache_dir: str,
        output_dir: str,
        on_complete: Optional[callable] = None,
    ) -> None:
        """
        Synchronous finalization of a single event.
        
        Args:
            active_event: ActiveEvent to finalize
            cache_dir: Cache directory containing .ts files
            output_dir: Directory to save final video
            on_complete: Optional callback when done
        """
        try:
            logger.info(f"Finalizing event for {active_event.camera_id}: "
                       f"{active_event.event_start} -> {active_event.record_until}")
            
            # 1. Find ts files in time range
            ts_files = find_ts_files_in_range(
                cache_dir,
                active_event.event_start,
                active_event.record_until,
            )
            
            if not ts_files:
                logger.warning(f"No ts files found for event {active_event.camera_id} "
                             f"in range [{active_event.event_start}, {active_event.record_until}]")
                # Even if no files, create record to avoid re-processing
                if on_complete:
                    on_complete(active_event, None, "no_segments")
                return
            
            # Extract just the file paths
            ts_file_paths = [fp for fp, _ in ts_files]
            
            # 2. Generate concat file
            concat_file = os.path.join(
                Path(output_dir).parent,  # Temp location
                f"concat_{active_event.camera_id}_{int(active_event.alarm_time.timestamp())}.txt"
            )
            
            try:
                generate_concat_file(ts_file_paths, concat_file)
            except Exception as e:
                logger.error(f"Failed to generate concat file: {e}")
                if on_complete:
                    on_complete(active_event, None, "concat_failed")
                return
            
            # 3. Generate output video path
            output_video = get_video_output_path(
                output_dir,
                active_event.camera_id,
                active_event.alarm_time,
            )
            
            # 4. Compose video with ffmpeg
            try:
                video_path = self._compose_video(concat_file, output_video)
                
                # Clean up concat file
                try:
                    os.remove(concat_file)
                except OSError:
                    pass
                
                logger.info(f"Successfully finalized event for {active_event.camera_id}: "
                           f"{video_path} ({os.path.getsize(video_path)} bytes)")
                
                active_event.video_path = video_path
                
                # 5. Run YOLO and VLM analysis (in background)
                logger.info(f"Starting analysis for {active_event.camera_id}")
                try:
                    analyzer = get_event_analyzer()
                    analysis_result = analyzer.analyze_video(video_path)
                    active_event.analysis_result = analysis_result
                    logger.info(f"Analysis result: {analysis_result}")
                except Exception as e:
                    logger.error(f"Failed to analyze video: {e}", exc_info=True)
                    active_event.analysis_result = {"status": "error", "message": str(e)}
                
                if on_complete:
                    on_complete(active_event, video_path, "success")
                
            except Exception as e:
                logger.error(f"Failed to compose video: {e}")
                if on_complete:
                    on_complete(active_event, None, "compose_failed")
        
        except Exception as e:
            logger.error(f"Unexpected error in event finalization: {e}", exc_info=True)
            if on_complete:
                on_complete(active_event, None, "error")
    
    def _compose_video(self, concat_file: str, output_path: str) -> str:
        """
        Use ffmpeg to compose video from concat file.
        
        Args:
            concat_file: Path to ffmpeg concat demuxer file
            output_path: Path for output MP4
            
        Returns:
            Path to output video file
            
        Raises:
            subprocess.CalledProcessError: If ffmpeg fails
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",              # No re-encoding
            "-y",                      # Overwrite
            output_path,
        ]
        
        logger.debug(f"Running ffmpeg compose: {' '.join(cmd)}")
        logger.info(f"Concat file path: {concat_file}")
        
        # Read concat file content for debugging
        try:
            with open(concat_file, 'r') as f:
                concat_content = f.read()
                logger.debug(f"Concat file content:\n{concat_content[:500]}")  # First 500 chars
        except Exception as e:
            logger.warning(f"Could not read concat file: {e}")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        # Log FFmpeg output
        if result.stdout:
            logger.debug(f"FFmpeg stdout:\n{result.stdout}")
        if result.stderr:
            logger.debug(f"FFmpeg stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            logger.error(f"FFmpeg compose failed with code {result.returncode}")
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"FFmpeg did not create output file: {output_path}")
        
        return output_path
    
    def wait_all(self, timeout: float = 600) -> None:
        """
        Wait for all pending finalization threads to complete.
        
        Args:
            timeout: Maximum seconds to wait
        """
        logger.info(f"Waiting for {len(self.thread_pool)} finalization threads...")
        
        for thread_key, thread in self.thread_pool.items():
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"Finalization thread {thread_key} did not complete within timeout")
        
        logger.info("All finalization threads completed")


class EventAnalyzer:
    """Handles YOLO detection and Qwen VLM analysis for event videos."""

    def __init__(self, yolo_weights: str = "../yolov8n.pt", qwen_model: str = "qwen-vl-plus"):
        """
        Initialize event analyzer.

        Args:
            yolo_weights: Path to YOLO model weights
            qwen_model: Qwen model name for VLM analysis
        """
        try:
            import torch
            import sys
            
            # Fix PyTorch 2.6 weights_only issue by patching torch.load
            original_load = torch.load
            
            def patched_load(f, *args, **kwargs):
                """Patched torch.load that disables weights_only checks."""
                try:
                    # Try with weights_only=False if it fails with defaults
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return original_load(f, *args, **kwargs)
                except Exception as e:
                    if 'weights_only' in str(e):
                        logger.debug(f"Retrying load with weights_only=False: {e}")
                        kwargs['weights_only'] = False
                        return original_load(f, *args, **kwargs)
                    raise
            
            # Monkey patch torch.load for this session
            torch.load = patched_load
            
            logger.info(f"Loading YOLO model from {yolo_weights}")
            self.yolo = YOLO(yolo_weights)
            
            # Restore original torch.load
            torch.load = original_load
        except Exception as e:
            logger.warning(f"Failed to load YOLO model {yolo_weights}, will download fresh model: {e}")
            try:
                # Try again with patched load for fresh download
                import torch
                original_load = torch.load
                
                def patched_load(f, *args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return original_load(f, *args, **kwargs)
                
                torch.load = patched_load
                self.yolo = YOLO("yolov8n.pt")
                torch.load = original_load
            except Exception as e2:
                logger.error(f"Failed to load fresh YOLO model: {e2}")
                raise RuntimeError(f"Could not load YOLO model: {e2}")
        
        self.qwen_model = qwen_model
        load_dotenv()  # Load environment variables for API keys

        logger.info("Initialized EventAnalyzer with YOLO and Qwen VLM")

    def analyze_video(self, video_path: str) -> dict:
        """
        Analyze video with YOLO for person detection and Qwen VLM for description.

        Args:
            video_path: Path to the video file to analyze

        Returns:
            Dictionary containing analysis results
        """
        try:
            # 1. Run YOLO detection on video
            logger.info(f"Running YOLO detection on {video_path}")

            # Use streaming inference to avoid storing all results in memory
            # and reduce peak memory usage by resizing frames.
            stream_args = {
                "stream": True,
                "imgsz": 640,
                "verbose": False,
            }

            person_detections = []
            frame_idx = 0

            try:
                for result in self.yolo.predict(source=video_path, **stream_args):
                    frame_idx += 1

                    if result.boxes is None:
                        continue

                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        if cls_id == 0:  # person class in COCO
                            person_detections.append({
                                "confidence": conf,
                                "bbox": box.xyxy[0].tolist(),
                                "frame_index": frame_idx,
                            })

            except MemoryError as me:
                logger.warning(
                    "YOLO inference ran out of memory; retrying with smaller resolution",
                    exc_info=True,
                )
                # Retry with even smaller image size
                stream_args["imgsz"] = 480
                person_detections = []
                frame_idx = 0
                for result in self.yolo.predict(source=video_path, **stream_args):
                    frame_idx += 1
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        if cls_id == 0:
                            person_detections.append({
                                "confidence": conf,
                                "bbox": box.xyxy[0].tolist(),
                                "frame_index": frame_idx,
                            })

            # 2. Upload video to DashScope for VLM analysis
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                logger.error("DASHSCOPE_API_KEY not found in environment")
                return {"status": "error", "message": "Missing API key"}

            logger.info("Uploading video to DashScope for VLM analysis")
            temp_url = upload_file_and_get_url(api_key, self.qwen_model, video_path)

            # 3. Call Qwen VLM for analysis
            prompt = (
                "你是安防巡检助手。请基于这段监控视频判断是否存在异常事件。\n"
                "重点关注：是否有人闯入/徘徊、是否有可疑行为、是否有摔倒/打斗等。\n"
                "请输出：\n"
                "1) 是否异常(是/否)\n"
                "2) 异常类型\n"
                "3) 关键证据（画面描述 + 发生在视频的哪个时间段）\n"
                "4) 建议处置\n"
                f"\n检测到的人员数量: {len(person_detections)}"
            )

            vlm_result = call_multimodal(
                temp_url=temp_url,
                prompt=prompt,
                model=self.qwen_model,
                api_key=api_key
            )

            analysis = {
                "status": "success",
                "yolo_detections": {
                    "person_count": len(person_detections),
                    "detections": person_detections
                },
                "vlm_analysis": vlm_result,
                "temp_url": temp_url
            }

            logger.info(f"Analysis complete for {video_path}: {len(person_detections)} persons detected")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze video {video_path}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


# Global analyzer instance
_event_analyzer = None

def get_event_analyzer() -> EventAnalyzer:
    """Get or create the global event analyzer instance."""
    global _event_analyzer
    if _event_analyzer is None:
        _event_analyzer = EventAnalyzer()
    return _event_analyzer
