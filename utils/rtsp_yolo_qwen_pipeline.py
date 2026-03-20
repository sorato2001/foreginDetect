import os
import time
import threading
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

import cv2
from ultralytics import YOLO

# ===== 复用你现有脚本能力 =====
# 来自 video_catch.py: 上传临时OSS
from video_catch import upload_file_and_get_url  # :contentReference[oaicite:4]{index=4}
# 来自 qwen_test.py: 调用多模态VLM
from qwen_test import _call_multimodal           # :contentReference[oaicite:5]{index=5}

load_dotenv()

# ----------------------------
# 画 YOLO 框（给可视化视频用）
# ----------------------------
def draw_yolo_boxes(frame, result, names):
    if result is None or result.boxes is None:
        return frame

    for b in result.boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return frame


# ----------------------------
# 触发规则：示例“有人经过”
# 你可以按需求扩展为：入侵区域/徘徊/摔倒等
# ----------------------------
def should_trigger_person(
    result,
    frame_w: int,
    frame_h: int,
    conf_thres: float = 0.5,
    area_ratio_thres: float = 0.02,
    person_cls_id: int = 0,  # COCO: person=0（你自定义模型请改）
) -> bool:
    img_area = float(frame_w * frame_h)
    if result is None or result.boxes is None:
        return False

    for b in result.boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        if cls_id != person_cls_id or conf < conf_thres:
            continue

        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / img_area >= area_ratio_thres:
            return True

    return False


@dataclass
class EventMeta:
    event_ts: str
    reason: str
    raw_path: str
    yolo_path: str
    fps: float
    width: int
    height: int


class RtspEventRecorder:
    """
    单路 RTSP：持续读帧 + pre-roll 缓冲
    触发时保存：pre_sec + post_sec 的 raw/yolo 两份视频
    """
    def __init__(
        self,
        save_dir: str = "./events",
        pre_sec: int = 3,
        post_sec: int = 7,
        cooldown_sec: int = 30,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self.cooldown_sec = cooldown_sec

        self._last_trigger_ts = 0.0
        self._lock = threading.Lock()

        self._fps: float = 20.0
        self._w: int = 640
        self._h: int = 480

        self._pre_raw: Optional[deque] = None
        self._pre_yolo: Optional[deque] = None

        self._recording = False
        self._post_frames_left = 0
        self._event_frames_raw = []
        self._event_frames_yolo = []
        self._event_ts = ""
        self._reason = ""

    def init_buffers(self, w: int, h: int, fps: float):
        self._w, self._h = w, h
        self._fps = fps if fps and fps > 1 else 20.0
        pre_n = int(self.pre_sec * self._fps)
        self._pre_raw = deque(maxlen=pre_n)
        self._pre_yolo = deque(maxlen=pre_n)

    def in_cooldown(self) -> bool:
        with self._lock:
            return (time.time() - self._last_trigger_ts) < self.cooldown_sec

    def trigger(self, reason: str) -> bool:
        if self._pre_raw is None or self._pre_yolo is None:
            raise RuntimeError("Recorder buffers not initialized")

        with self._lock:
            if (time.time() - self._last_trigger_ts) < self.cooldown_sec:
                return False
            self._last_trigger_ts = time.time()

        self._event_ts = time.strftime("%Y%m%d_%H%M%S")
        self._reason = reason

        # 把 pre-roll 复制进事件帧列表
        self._event_frames_raw = list(self._pre_raw)
        self._event_frames_yolo = list(self._pre_yolo)

        self._recording = True
        self._post_frames_left = int(self.post_sec * self._fps)
        return True

    def _write_video(self, path: Path, frames):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(self._fps), (self._w, self._h))
        if not writer.isOpened():
            raise RuntimeError(
                "cannot open VideoWriter (mp4v). "
                "Your OpenCV build may lack MP4 encoder support; "
                "try rebuilding OpenCV with FFmpeg/GStreamer or switch to AVI."
            )
        for f in frames:
            writer.write(f)
        writer.release()

    def _flush_event(self) -> EventMeta:
        raw_path = self.save_dir / f"raw_{self._event_ts}.mp4"
        yolo_path = self.save_dir / f"yolo_{self._event_ts}.mp4"

        self._write_video(raw_path, self._event_frames_raw)
        self._write_video(yolo_path, self._event_frames_yolo)

        self._recording = False
        self._post_frames_left = 0

        return EventMeta(
            event_ts=self._event_ts,
            reason=self._reason,
            raw_path=str(raw_path),
            yolo_path=str(yolo_path),
            fps=float(self._fps),
            width=self._w,
            height=self._h,
        )

    def step(self, frame, yolo_result=None) -> Optional[EventMeta]:
        if self._pre_raw is None or self._pre_yolo is None:
            raise RuntimeError("Recorder buffers not initialized")

        raw = frame
        yolo_vis = frame.copy()
        if yolo_result is not None:
            # 注意：names 在不同版本 ultralytics 里可能是 dict 或 list，这里兼容处理
            names = getattr(yolo_result, "names", None) or {}
            yolo_vis = draw_yolo_boxes(yolo_vis, yolo_result, names)

        # 更新 pre-roll
        self._pre_raw.append(raw.copy())
        self._pre_yolo.append(yolo_vis.copy())

        # 事件录制中：追加 post-roll
        if self._recording:
            self._event_frames_raw.append(raw.copy())
            self._event_frames_yolo.append(yolo_vis.copy())
            self._post_frames_left -= 1
            if self._post_frames_left <= 0:
                return self._flush_event()

        return None


class Pipeline:
    def __init__(
        self,
        rtsp_url: str,
        yolo_weights: str = "yolov8n.pt",
        qwen_model_name: str = "qwen-vl-plus",
        save_dir: str = "./events",
        pre_sec: int = 3,
        post_sec: int = 7,
        cooldown_sec: int = 30,
        detect_every_n: int = 2,
        conf_thres: float = 0.5,
        area_ratio_thres: float = 0.02,
        max_workers: int = 2,
    ):
        self.rtsp_url = rtsp_url
        self.yolo = YOLO(yolo_weights)
        self.qwen_model_name = qwen_model_name

        self.detect_every_n = max(1, detect_every_n)
        self.conf_thres = conf_thres
        self.area_ratio_thres = area_ratio_thres

        self.rec = RtspEventRecorder(save_dir=save_dir, pre_sec=pre_sec, post_sec=post_sec, cooldown_sec=cooldown_sec)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def _upload_and_vlm(self, meta: EventMeta) -> Dict[str, Any]:
        """
        事件落盘 -> 上传 raw -> 调 Qwen-VLM 分析（media_type=video）
        """
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return {"status": "error", "msg": "Missing DASHSCOPE_API_KEY env var."}

        # 1) 上传 raw 视频到 DashScope 临时OSS（返回 oss://...）
        temp_url = upload_file_and_get_url(api_key, self.qwen_model_name, meta.raw_path)  # :contentReference[oaicite:6]{index=6}

        # 2) 调 Qwen-VLM 分析
        prompt = (
            "你是安防巡检助手。请基于这段监控视频判断是否存在异常事件。\n"
            "重点关注：是否有人闯入/徘徊、是否有可疑行为、是否有摔倒/打斗等。\n"
            "请输出：\n"
            "1) 是否异常(是/否)\n"
            "2) 异常类型\n"
            "3) 关键证据（画面描述 + 发生在视频的哪个时间段）\n"
            "4) 建议处置\n"
            f"\n触发原因: {meta.reason}"
        )

        vlm_ret = _call_multimodal(  # :contentReference[oaicite:7]{index=7}
            temp_url=temp_url,
            prompt=prompt,
            model=self.qwen_model_name,
            media_type="video",
            api_key=api_key,
        )

        return {
            "status": "success",
            "event": meta.__dict__,
            "temp_url": temp_url,
            "vlm": vlm_ret,
        }

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {self.rtsp_url}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 20.0  # 兜底

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.rec.init_buffers(w, h, float(fps))

        print("[INFO] RTSP pipeline started.")
        print(f"[INFO] stream: {self.rtsp_url}")
        print(f"[INFO] w/h/fps: {w}/{h}/{fps}")
        print("[INFO] press Ctrl+C to stop.")

        frame_idx = 0
        last_yolo_result = None

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue

            frame_idx += 1

            # YOLO 推理：每 N 帧做一次
            if frame_idx % self.detect_every_n == 0:
                results = self.yolo.predict(frame, verbose=False)
                last_yolo_result = results[0] if results else None

                # 触发规则示例：检测到 person
                if last_yolo_result is not None and should_trigger_person(
                    last_yolo_result,
                    frame_w=w,
                    frame_h=h,
                    conf_thres=self.conf_thres,
                    area_ratio_thres=self.area_ratio_thres,
                ):
                    if self.rec.trigger(reason="YOLO: person_detected"):
                        print("[TRIGGER] person_detected -> recording event clip...")

            # 每帧写入 recorder（raw + yolo 可视化）
            meta = self.rec.step(frame, last_yolo_result)
            if meta:
                print(f"[SAVED] raw={meta.raw_path} yolo={meta.yolo_path} reason={meta.reason}")

                # 上传 + VLM 分析放线程池，避免阻塞主循环
                self.pool.submit(self._handle_vlm_result, meta)

    def _handle_vlm_result(self, meta: EventMeta):
        ret = self._upload_and_vlm(meta)
        if ret.get("status") != "success":
            print("[VLM] error:", ret)
            return

        vlm = ret["vlm"]
        print("\n========== VLM ANALYSIS ==========")
        print("event_ts:", meta.event_ts)
        print("temp_url:", ret["temp_url"])
        print("answer:", vlm.get("answer"))
        print("==================================\n")


if __name__ == "__main__":
    load_dotenv()
    rtsp = os.getenv("RTSP_URL", "rtsp://admin:vge2024211865@192.168.120.33:554/Streaming/Channels/101")
    pipe = Pipeline(
        rtsp_url=rtsp,
        yolo_weights=os.getenv("YOLO_WEIGHTS", "yolov8n.pt"),
        qwen_model_name=os.getenv("QWEN_MODEL", "qwen-vl-plus"),
        save_dir=os.getenv("SAVE_DIR", "./events"),
        pre_sec=int(os.getenv("PRE_SEC", "3")),
        post_sec=int(os.getenv("POST_SEC", "7")),
        cooldown_sec=int(os.getenv("COOLDOWN_SEC", "30")),
        detect_every_n=int(os.getenv("DETECT_EVERY_N", "2")),
        conf_thres=float(os.getenv("CONF", "0.5")),
        area_ratio_thres=float(os.getenv("AREA_RATIO", "0.02")),
    )
    pipe.run()
