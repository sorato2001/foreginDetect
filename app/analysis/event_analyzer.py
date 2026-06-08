"""Unified railway perimeter event analyzer.

EventAnalyzer is the public entry point used by the scheduler, finalizer, REST
helpers, and demo scripts. It produces a stable JSON schema suitable for storage
in SQLite and direct frontend rendering.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .frame_sampler import FrameSampler
from .region_rules import RISK_ORDER, RegionRules
from .vlm_reviewer import VlmReviewer
from .yolo_detector import YoloDetector

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "railway_security_v1"


class EventAnalyzer:
    """Coordinate YOLO detection, region rules, VLM review, and final decision."""

    def __init__(self, default_config: dict[str, Any] | None = None) -> None:
        self.default_config = default_config or {}
        self._detectors: dict[tuple[str, float, tuple[str, ...]], YoloDetector] = {}

    @staticmethod
    def _get_security_config(camera_config: Any | None) -> dict[str, Any]:
        if camera_config is None:
            return {}
        if isinstance(camera_config, dict):
            return camera_config.get("railway_security") or {}
        return getattr(camera_config, "railway_security", None) or {}

    def _detector_for(self, security_config: dict[str, Any]) -> YoloDetector | None:
        yolo_cfg = security_config.get("yolo") or {}
        if not yolo_cfg.get("enabled", True):
            return None
        model_path = str(yolo_cfg.get("model_path", "yolov8n.pt"))
        conf = float(yolo_cfg.get("conf_threshold", 0.35))
        targets = tuple(yolo_cfg.get("target_classes") or [])
        key = (model_path, conf, targets)
        if key not in self._detectors:
            self._detectors[key] = YoloDetector(model_path, conf, targets or None)
        return self._detectors[key]

    def analyze_image(
        self,
        image_path: str,
        camera_id: str,
        event_time: datetime | str | None = None,
        camera_config: Any | None = None,
        save_visualization_path: str | None = None,
    ) -> dict[str, Any]:
        """Analyze one FTP alarm image and return the unified result JSON."""
        event_time_iso = self._iso(event_time)
        security_config = self._get_security_config(camera_config) or self.default_config
        if not security_config.get("enabled", True):
            return self._disabled_pipeline_result(camera_id, event_time_iso)

        detections: list[dict[str, Any]] = []
        try:
            detector = self._detector_for(security_config)
            detections = detector.detect(image_path) if detector else []
        except Exception as exc:
            logger.error("YOLO stage failed for %s: %s", image_path, exc, exc_info=True)

        rules = RegionRules(security_config.get("regions") or {})
        detections = rules.annotate_detections(detections)
        rule_result = rules.evaluate(detections)

        vlm_cfg = security_config.get("vlm") or {}
        vlm_result = VlmReviewer(vlm_cfg).review(
            [image_path] if image_path else [],
            detections,
            rule_result,
            camera_id,
            event_time_iso,
        )
        final_result = self._fuse(detections, rule_result, vlm_result)
        result = {
            "camera_id": camera_id,
            "event_time": event_time_iso,
            "pipeline_version": PIPELINE_VERSION,
            "source": {"type": "image", "image_path": image_path},
            "detections": detections,
            "rule_result": rule_result,
            "vlm_result": vlm_result,
            # Alias for frontends/users that expect explicit large-model wording.
            "llm_review": vlm_result,
            "final_result": final_result,
        }
        if save_visualization_path:
            vis_path = self.save_visualization(image_path, result, security_config, save_visualization_path)
            if vis_path:
                result.setdefault("artifacts", {})["visualization_path"] = vis_path
        logger.info(
            "Railway image analysis summary: camera=%s detections=%s rule=%s vlm_mode=%s final=%s alarm=%s",
            camera_id,
            len(detections),
            rule_result.get("risk_level"),
            vlm_result.get("mode"),
            final_result.get("risk_level"),
            final_result.get("is_alarm"),
        )
        return result

    def analyze_event(
        self,
        video_path: str | None,
        camera_id: str,
        event_time: datetime | str | None = None,
        camera_config: Any | None = None,
        alarm_image_path: str | None = None,
        output_dir: str | None = None,
        previous_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze a finalized event video by sampling key frames.

        If video sampling fails, the analyzer falls back to the alarm image or the
        previous image-level result, keeping the event pipeline robust.
        """
        event_time_iso = self._iso(event_time)
        security_config = self._get_security_config(camera_config) or self.default_config
        if not security_config.get("enabled", True):
            return self._disabled_pipeline_result(camera_id, event_time_iso)

        frame_paths: list[str] = []
        if video_path:
            out = output_dir or str(Path(video_path).with_suffix("")) + "_frames"
            frame_paths = FrameSampler(max_frames=5).sample_video(
                video_path,
                out,
                prefix=f"{camera_id}_{self._safe_time(event_time_iso)}",
                include_paths=[alarm_image_path] if alarm_image_path else None,
            )
        elif alarm_image_path:
            frame_paths = [alarm_image_path]

        if not frame_paths and previous_result:
            result = dict(previous_result)
            result.setdefault("source", {})["video_path"] = video_path
            result.setdefault("artifacts", {})["key_frames"] = []
            return result
        if not frame_paths:
            return self._empty_result(camera_id, event_time_iso, video_path, "未能抽取关键帧，无法进行视频复核。")

        # Analyze the most relevant frame (alarm image if present, otherwise first sampled frame).
        base = self.analyze_image(
            frame_paths[0],
            camera_id=camera_id,
            event_time=event_time_iso,
            camera_config=camera_config,
        )
        base["source"] = {"type": "event_video", "video_path": video_path, "primary_frame": frame_paths[0]}
        base.setdefault("artifacts", {})["key_frames"] = frame_paths
        if previous_result:
            base["previous_image_analysis"] = previous_result
            base["final_result"] = self._merge_previous_and_current(previous_result, base)
        return base

    @staticmethod
    def _iso(value: datetime | str | None) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str) and value:
            return value
        return datetime.now().isoformat()

    @staticmethod
    def _safe_time(value: str) -> str:
        return value.replace(":", "").replace("-", "").replace(".", "")[:15]

    @staticmethod
    def _disabled_pipeline_result(camera_id: str, event_time_iso: str) -> dict[str, Any]:
        return {
            "camera_id": camera_id,
            "event_time": event_time_iso,
            "pipeline_version": PIPELINE_VERSION,
            "detections": [],
            "rule_result": {"risk_level": "normal", "risk_score": 0.0, "reason": "railway_security 未启用。"},
            "vlm_result": {"enabled": False, "is_valid_alarm": False, "risk_level": "normal", "target_type": "none", "behavior": "unknown", "reason": "未启用", "confidence": 0.0, "false_alarm_reason": "未启用"},
            "final_result": {"is_alarm": False, "risk_level": "normal", "alarm_title": "铁路周界误报", "alarm_reason": "智能复核未启用。", "recommended_action": "无需处理", "needs_review": False},
        }

    @staticmethod
    def _empty_result(camera_id: str, event_time_iso: str, video_path: str | None, reason: str) -> dict[str, Any]:
        return {
            "camera_id": camera_id,
            "event_time": event_time_iso,
            "pipeline_version": PIPELINE_VERSION,
            "source": {"type": "event_video", "video_path": video_path},
            "detections": [],
            "rule_result": {"risk_level": "normal", "risk_score": 0.0, "reason": reason},
            "vlm_result": {"enabled": False, "is_valid_alarm": False, "risk_level": "normal", "target_type": "none", "behavior": "unknown", "reason": reason, "confidence": 0.0, "false_alarm_reason": reason},
            "final_result": {"is_alarm": False, "risk_level": "normal", "alarm_title": "铁路周界误报", "alarm_reason": reason, "recommended_action": "无需处理", "needs_review": True},
        }

    @staticmethod
    def _fuse(detections: list[dict[str, Any]], rule_result: dict[str, Any], vlm_result: dict[str, Any]) -> dict[str, Any]:
        rule_level = str(rule_result.get("risk_level", "normal"))
        vlm_level = str(vlm_result.get("risk_level", rule_level))
        vlm_behavior = str(vlm_result.get("behavior", "unknown"))
        has_danger_person = any(d.get("class_name") == "person" and d.get("zone") == "danger" for d in detections)
        has_danger_or_cross = any(d.get("zone") == "danger" or d.get("cross_fence") for d in detections)
        yolo_has_target = bool(detections)
        vlm_valid = bool(vlm_result.get("is_valid_alarm", False))

        final_level = rule_level
        if RISK_ORDER.get(vlm_level, 0) > RISK_ORDER.get(final_level, 0):
            final_level = vlm_level
        if has_danger_or_cross and RISK_ORDER.get(final_level, 0) < RISK_ORDER["medium"]:
            final_level = "medium"
        if has_danger_person or vlm_behavior in {"crossing", "climbing", "intruding"}:
            final_level = "high"

        conflict = (not yolo_has_target and vlm_valid) or (yolo_has_target and not vlm_valid and rule_level in {"medium", "high"})
        vlm_real_failed = bool(vlm_result.get("real_call_failed"))
        if not yolo_has_target and not vlm_valid:
            final_level = "normal"
            is_alarm = False
        else:
            is_alarm = final_level in {"medium", "high"} or vlm_valid

        if final_level == "high":
            title = "铁路周界确认入侵"
            action = "立即处置"
        elif final_level == "medium":
            title = "铁路周界疑似入侵"
            action = "现场核查"
        elif final_level == "low":
            title = "铁路周界低风险关注"
            action = "人工关注"
        else:
            title = "铁路周界误报"
            action = "无需处理"

        if not is_alarm:
            reason = vlm_result.get("false_alarm_reason") or rule_result.get("reason") or "未发现有效入侵目标。"
        elif final_level == "high":
            reason = vlm_result.get("reason") or "目标进入危险区、跨越护网或存在入侵行为。"
        else:
            reason = rule_result.get("reason") or vlm_result.get("reason") or "目标靠近或进入铁路周界警戒区域。"
        if conflict:
            reason = f"{reason}（YOLO 与 VLM 结果存在差异，建议人工复核）"

        if vlm_real_failed:
            reason = f"{reason}（大模型复核未成功，建议人工确认）"

        return {
            "is_alarm": bool(is_alarm),
            "risk_level": final_level,
            "alarm_title": title,
            "alarm_reason": str(reason)[:140],
            "recommended_action": action,
            "needs_review": bool(conflict or vlm_real_failed),
        }

    @classmethod
    def _merge_previous_and_current(cls, previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        prev_final = previous.get("final_result") or {}
        curr_final = current.get("final_result") or {}
        prev_level = str(prev_final.get("risk_level", "normal"))
        curr_level = str(curr_final.get("risk_level", "normal"))
        if RISK_ORDER.get(prev_level, 0) > RISK_ORDER.get(curr_level, 0):
            merged = dict(prev_final)
            merged["needs_review"] = bool(prev_final.get("needs_review") or curr_final.get("needs_review"))
            merged["alarm_reason"] = f"报警图片复核：{merged.get('alarm_reason', '')}"[:140]
            return merged
        return curr_final

    @staticmethod
    def save_visualization(
        image_path: str,
        analysis_result: dict[str, Any],
        security_config: dict[str, Any],
        output_path: str,
    ) -> str | None:
        """Save a demo visualization with regions, boxes, and readable labels.

        OpenCV Hershey fonts cannot render Chinese reliably and often produce
        question marks. Geometry is drawn by OpenCV; text is drawn by PIL with a
        CJK font when available. If no CJK font is found, labels fall back to
        English to avoid mojibake.
        """
        try:
            import cv2
            import numpy as np

            image = cv2.imread(image_path)
            if image is None:
                return None
            regions = security_config.get("regions") or {}

            def draw_poly(points: list[list[int]], color: tuple[int, int, int], closed: bool = True) -> None:
                if not points:
                    return
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(image, [pts], closed, color, 2)

            draw_poly(regions.get("warning_zone") or [], (0, 255, 255), True)
            draw_poly(regions.get("danger_zone") or [], (0, 0, 255), True)
            fence = regions.get("fence_line") or []
            if len(fence) >= 2:
                pts = np.array(fence, dtype=np.int32)
                cv2.polylines(image, [pts], False, (255, 0, 0), 3)

            for det in analysis_result.get("detections", []):
                x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
                color = (0, 0, 255) if det.get("zone") == "danger" else (0, 255, 255) if det.get("zone") == "warning" else (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            try:
                from PIL import Image, ImageDraw, ImageFont

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(pil_img)

                def load_font(size: int) -> tuple[Any, bool]:
                    candidates = [
                        "C:/Windows/Fonts/msyh.ttc",
                        "C:/Windows/Fonts/simhei.ttf",
                        "C:/Windows/Fonts/simsun.ttc",
                        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                        "/System/Library/Fonts/PingFang.ttc",
                    ]
                    for font_path in candidates:
                        if Path(font_path).exists():
                            return ImageFont.truetype(font_path, size=size), True
                    return ImageFont.load_default(), False

                font_big, has_cjk = load_font(32)
                font_mid, _ = load_font(22)
                final = analysis_result.get("final_result", {})
                risk = str(final.get("risk_level", "unknown")).upper()
                title_cn = str(final.get("alarm_title", "铁路周界复核"))
                action_cn = str(final.get("recommended_action", ""))
                title = f"{risk} {title_cn}" if has_cjk else f"{risk} railway review"
                draw.text((20, 20), title, fill=(255, 0, 0), font=font_big)
                if action_cn:
                    draw.text((20, 58), action_cn if has_cjk else str(final.get("recommended_action", "")), fill=(255, 0, 0), font=font_mid)
                vlm = analysis_result.get("vlm_result") or analysis_result.get("llm_review") or {}
                vlm_mode = str(vlm.get("mode", "unknown"))
                vlm_reason = str(vlm.get("reason") or vlm.get("false_alarm_reason") or "")[:60]
                if vlm_reason:
                    llm_text = f"???({vlm_mode}): {vlm_reason}" if has_cjk else f"LLM({vlm_mode}): {vlm_reason}"
                    draw.text((20, 88), llm_text, fill=(255, 0, 0), font=font_mid)

                if regions.get("warning_zone"):
                    draw.text(tuple(int(v) for v in regions["warning_zone"][0]), "警戒区 warning" if has_cjk else "warning", fill=(255, 255, 0), font=font_mid)
                if regions.get("danger_zone"):
                    draw.text(tuple(int(v) for v in regions["danger_zone"][0]), "危险区 danger" if has_cjk else "danger", fill=(255, 0, 0), font=font_mid)
                if len(fence) >= 2:
                    draw.text(tuple(int(v) for v in fence[0]), "护网 fence" if has_cjk else "fence", fill=(0, 0, 255), font=font_mid)

                for det in analysis_result.get("detections", []):
                    x1, y1, _, _ = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
                    label = f"{det.get('class_name')} {float(det.get('confidence', 0.0)):.2f} {det.get('zone')}"
                    draw.text((x1, max(0, y1 - 26)), label, fill=(0, 255, 0), font=font_mid)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                final = analysis_result.get("final_result", {})
                risk = str(final.get("risk_level", "unknown")).upper()
                cv2.putText(image, f"{risk} railway review", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if regions.get("warning_zone"):
                    cv2.putText(image, "warning", tuple(regions["warning_zone"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if regions.get("danger_zone"):
                    cv2.putText(image, "danger", tuple(regions["danger_zone"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if len(fence) >= 2:
                    cv2.putText(image, "fence", tuple(fence[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(output_path, image):
                return output_path
        except Exception as exc:  # pragma: no cover - visualization is best effort
            logger.warning("Failed to save visualization for %s: %s", image_path, exc, exc_info=True)
        return None
