"""
Shared YOLO + VLM analyzers for FTP alarm images and finalized event videos.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import cv2
import requests
from dotenv import load_dotenv


MediaType = Literal["image", "video"]

logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

DASHSCOPE_UPLOAD_API = "https://dashscope.aliyuncs.com/api/v1/uploads"


def get_upload_policy(api_key: str, model_name: str, max_retries: int = 5) -> Dict[str, Any]:
    """Get temporary upload credentials for DashScope OSS."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {"action": "getPolicy", "model": model_name}

    backoff = 0.5
    last_err = None
    for _ in range(max_retries):
        try:
            resp = requests.get(
                DASHSCOPE_UPLOAD_API,
                headers=headers,
                params=params,
                timeout=20,
            )
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

            raise RuntimeError(
                f"Failed to get upload policy: {resp.status_code} {resp.text}"
            )
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 5)

    raise RuntimeError(f"Failed to get upload policy after retries: {last_err}")


def upload_file_to_oss(policy_data: Dict[str, Any], file_path: str) -> str:
    """Upload file to OSS and return temporary oss:// URL."""
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"

    with open(file_path, "rb") as file_obj:
        files = {
            "OSSAccessKeyId": (None, policy_data["oss_access_key_id"]),
            "Signature": (None, policy_data["signature"]),
            "policy": (None, policy_data["policy"]),
            "x-oss-object-acl": (None, policy_data["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, policy_data["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (file_name, file_obj),
        }
        resp = requests.post(policy_data["upload_host"], files=files, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to upload file: {resp.status_code} {resp.text}")

    return f"oss://{key}"


def upload_file_and_get_url(api_key: str, model_name: str, file_path: str) -> str:
    """Upload a local file and return DashScope temporary URL."""
    policy = get_upload_policy(api_key, model_name)
    return upload_file_to_oss(policy, file_path)


def extract_text(resp: Any) -> str:
    """Extract text from DashScope multimodal response."""
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
    media_type: MediaType,
    api_key: Optional[str],
) -> Dict[str, Any]:
    """Call DashScope multimodal model for image or video analysis."""
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
        if not api_key:
            return {"status": "error", "msg": "Missing DASHSCOPE_API_KEY"}

        if not temp_url.startswith("oss://"):
            return {"status": "error", "msg": "temp_url must start with oss://"}

        import dashscope

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
            "answer": extract_text(resp),
            "raw": resp,
        }
    except Exception as exc:
        logger.error("Error calling multimodal model: %s", exc, exc_info=True)
        return {
            "status": "error",
            "msg": str(exc),
            "model": model,
            "media_type": media_type,
            "temp_url": temp_url,
        }


def summarize_analysis_status(vlm_result: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """Convert VLM result into a top-level analyzer status."""
    vlm_status = vlm_result.get("status")
    if vlm_status == "success":
        return "success", None
    if vlm_status == "error":
        return "partial_success", vlm_result.get("msg") or vlm_result.get("message")
    return "success", None


class VisionAnalyzer:
    """Shared analyzer for event videos and FTP alarm images."""

    def __init__(self, yolo_weights: str = "../yolov8n.pt", qwen_model: str = "qwen-vl-plus"):
        self.yolo = self._load_yolo_model(yolo_weights)
        self.qwen_model = qwen_model
        logger.info("Initialized VisionAnalyzer with YOLO and Qwen VLM")

    def _load_yolo_model(self, yolo_weights: str) -> Any:
        """Load YOLO model with a compatibility workaround for torch.load."""
        try:
            from ultralytics import YOLO
            import torch

            original_load = torch.load

            def patched_load(file_obj, *args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return original_load(file_obj, *args, **kwargs)

            torch.load = patched_load
            logger.info("Loading YOLO model from %s", yolo_weights)
            model = YOLO(yolo_weights)
            torch.load = original_load
            return model
        except Exception as exc:
            logger.warning(
                "Failed to load YOLO model %s, retrying with fresh download: %s",
                yolo_weights,
                exc,
            )
            from ultralytics import YOLO
            import torch

            original_load = torch.load

            def patched_load(file_obj, *args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return original_load(file_obj, *args, **kwargs)

            torch.load = patched_load
            model = YOLO("yolov8n.pt")
            torch.load = original_load
            return model

    def analyze_image(
        self,
        image_path: str,
        output_dir: str,
        save_annotated_images: bool = False,
    ) -> Dict[str, Any]:
        """Analyze a single FTP image using YOLO and VLM."""
        try:
            logger.info("Running YOLO detection on image %s", image_path)
            results = self.yolo.predict(source=image_path, imgsz=640, verbose=False)
            if not results:
                raise RuntimeError(f"No YOLO result returned for image: {image_path}")

            result = results[0]
            detections = []
            person_count = 0
            detected_reason = "none"

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    bbox = box.xyxy[0].tolist()
                    if cls_id == 0:
                        person_count += 1
                        detections.append({
                            "confidence": conf,
                            "bbox": bbox,
                        })

            if person_count > 0:
                detected_reason = "YOLO: person_detected"

            annotated_image_path = None
            if save_annotated_images:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                source_path = Path(image_path)
                annotated_image_path = str(
                    Path(output_dir) / f"{source_path.stem}_yolo{source_path.suffix}"
                )
                annotated_frame = result.plot()
                if annotated_frame is not None:
                    saved = cv2.imwrite(annotated_image_path, annotated_frame)
                    if saved:
                        logger.info("Saved YOLO annotated image to %s", annotated_image_path)
                    else:
                        logger.warning("Failed to save YOLO annotated image to %s", annotated_image_path)
                        annotated_image_path = None
            else:
                logger.info("Skipping annotated image output because save_annotated_images is disabled")

            vlm_result, temp_url = self._run_vlm(
                file_path=image_path,
                media_type="image",
                prompt=(
                    "你是安防巡检助手。请基于这张监控抓拍图片判断是否存在异常事件。\n"
                    "重点关注：是否有人闯入、徘徊、可疑行为、摔倒、打斗等。\n"
                    "请输出：\n"
                    "1) 是否异常(是/否)\n"
                    "2) 异常类型\n"
                    "3) 关键证据（画面描述）\n"
                    "4) 建议处置\n"
                    f"\n检测到的人员数量: {person_count}"
                    f"\n触发原因: {detected_reason}"
                ),
            )

            analysis_status, analysis_message = summarize_analysis_status(vlm_result)
            vlm_answer = vlm_result.get("answer")
            if vlm_answer:
                logger.info("VLM image analysis summary: %s", vlm_answer[:300])
            elif analysis_message:
                logger.warning("VLM image analysis unavailable: %s", analysis_message)

            return {
                "status": analysis_status,
                "message": analysis_message,
                "annotated_image_path": annotated_image_path,
                "yolo_detections": {
                    "person_count": person_count,
                    "total_person_detections": len(detections),
                    "detections": detections,
                    "detected_reason": detected_reason,
                },
                "vlm_analysis": vlm_result,
                "temp_url": temp_url,
            }
        except Exception as exc:
            logger.error("Failed to analyze image %s: %s", image_path, exc, exc_info=True)
            return {"status": "error", "message": str(exc)}

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze a composed event video using YOLO and VLM."""
        try:
            logger.info("Running YOLO detection on video %s", video_path)
            stream_args = {
                "stream": True,
                "imgsz": 640,
                "verbose": False,
            }

            source_path = Path(video_path)
            yolo_video_path = str(source_path.with_name(f"{source_path.stem}_yolo{source_path.suffix}"))

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video for YOLO overlay: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            cap.release()

            yolo_writer = cv2.VideoWriter(
                yolo_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (width, height),
            )
            if not yolo_writer.isOpened():
                raise RuntimeError(f"cannot open VideoWriter for yolo output: {yolo_video_path}")

            detected_reason = "none"

            def run_yolo_pass(args):
                nonlocal detected_reason
                total_person_detections = []
                max_persons_in_single_frame = 0
                person_frames = 0

                for frame_index, result in enumerate(
                    self.yolo.predict(source=video_path, **args),
                    start=1,
                ):
                    frame_persons = 0

                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            if cls_id == 0:
                                frame_persons += 1
                                total_person_detections.append({
                                    "confidence": conf,
                                    "bbox": box.xyxy[0].tolist(),
                                    "frame_index": frame_index,
                                })

                    if frame_persons > 0:
                        person_frames += 1
                        detected_reason = "YOLO: person_detected"

                    max_persons_in_single_frame = max(
                        max_persons_in_single_frame,
                        frame_persons,
                    )

                    try:
                        annotated_frame = result.plot()
                        if annotated_frame is not None:
                            frame_height, frame_width = annotated_frame.shape[:2]
                            if (frame_width, frame_height) != (width, height):
                                annotated_frame = cv2.resize(annotated_frame, (width, height))
                            yolo_writer.write(annotated_frame)
                    except Exception as exc:
                        logger.warning(
                            "Failed to write YOLO overlay frame %s: %s",
                            frame_index,
                            exc,
                        )

                return total_person_detections, max_persons_in_single_frame, person_frames

            try:
                total_person_detections, max_persons_in_single_frame, person_frames = run_yolo_pass(
                    stream_args
                )
            except MemoryError:
                logger.warning(
                    "YOLO inference ran out of memory; retrying with smaller resolution",
                    exc_info=True,
                )
                stream_args["imgsz"] = 480
                detected_reason = "none"
                total_person_detections, max_persons_in_single_frame, person_frames = run_yolo_pass(
                    stream_args
                )
            finally:
                yolo_writer.release()
            logger.info("Saved YOLO annotated video to %s", yolo_video_path)

            vlm_result, temp_url = self._run_vlm(
                file_path=video_path,
                media_type="video",
                prompt=(
                    "你是安防巡检助手。请基于这段监控视频判断是否存在异常事件。\n"
                    "重点关注：是否有人闯入、徘徊、可疑行为、摔倒、打斗等。\n"
                    "请输出：\n"
                    "1) 是否异常(是/否)\n"
                    "2) 异常类型\n"
                    "3) 关键证据（画面描述 + 发生时间段）\n"
                    "4) 建议处置\n"
                    f"\n检测到的最大人员数量: {max_persons_in_single_frame}"
                    f"\n触发原因: {detected_reason}"
                ),
            )

            analysis_status, analysis_message = summarize_analysis_status(vlm_result)
            vlm_answer = vlm_result.get("answer")
            if vlm_answer:
                logger.info("VLM video analysis summary: %s", vlm_answer[:300])
            elif analysis_message:
                logger.warning("VLM video analysis unavailable: %s", analysis_message)

            return {
                "status": analysis_status,
                "message": analysis_message,
                "yolo_video_path": yolo_video_path,
                "yolo_detections": {
                    "person_count": max_persons_in_single_frame,
                    "total_person_detections": len(total_person_detections),
                    "person_frames": person_frames,
                    "detected_reason": detected_reason,
                },
                "vlm_analysis": vlm_result,
                "temp_url": temp_url,
            }
        except Exception as exc:
            logger.error("Failed to analyze video %s: %s", video_path, exc, exc_info=True)
            return {"status": "error", "message": str(exc)}

    def _run_vlm(
        self,
        file_path: str,
        media_type: MediaType,
        prompt: str,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Upload file and run VLM analysis."""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return {"status": "error", "message": "Missing DASHSCOPE_API_KEY"}, None

        temp_url = upload_file_and_get_url(api_key, self.qwen_model, file_path)
        result = call_multimodal(
            temp_url=temp_url,
            prompt=prompt,
            model=self.qwen_model,
            media_type=media_type,
            api_key=api_key,
        )
        return result, temp_url


_vision_analyzer: Optional[VisionAnalyzer] = None


def get_vision_analyzer() -> VisionAnalyzer:
    """Get or create a shared analyzer instance."""
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer
