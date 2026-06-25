"""Local Gemma VLM provider for OpenAI-compatible LAN deployments."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pydantic import ValidationError

from src.evidence.evidence_schema import EventEvidence
from src.evidence.serializers import save_json
from src.vlm.json_validator import parse_vlm_review
from src.vlm.prompts import OUTPUT_SCHEMA, build_review_prompt, summarize_evidence
from src.vlm.review_provider import VLMReviewProvider
from src.vlm.vlm_schema import VLMReview

logger = logging.getLogger(__name__)

GEMMA_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.HTTPError,
    requests.exceptions.RequestException,
    json.JSONDecodeError,
    ValueError,
    ValidationError,
)


class LocalGemmaProvider(VLMReviewProvider):
    """Review provider for a local Gemma VLM served as /v1/chat/completions."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8082/v1/chat/completions",
        model: str = "gemma-4-26B",
        timeout_sec: float = 600.0,
        connect_timeout_seconds: float = 10.0,
        max_retries: int = 0,
        retry_backoff_seconds: float = 2.0,
        fallback_on_error: bool = True,
        artifact_dir: str | None = None,
        bearer_token: str = "sk-no-key-required",
        max_images: int = 1,
        evidence_mode: str = "minimal",
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_sec = float(timeout_sec)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.read_timeout_seconds = float(timeout_sec)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.fallback_on_error = bool(fallback_on_error)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.bearer_token = bearer_token
        self.max_images = max(0, int(max_images))
        self.evidence_mode = evidence_mode

    def review(self, evidence: EventEvidence) -> VLMReview:
        """Call local Gemma VLM and return a validated review."""
        vlm_input = self._vlm_input_metadata(evidence)
        payload = self._payload(evidence, vlm_input=vlm_input)
        self._write_request_summary(payload)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    "Local Gemma review attempt %s/%s: model=%s endpoint=%s connect_timeout=%.1fs read_timeout=%.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    self.model,
                    self.endpoint,
                    self.connect_timeout_seconds,
                    self.read_timeout_seconds,
                )
                response = requests.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.bearer_token}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=(self.connect_timeout_seconds, self.read_timeout_seconds),
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                self._write_raw_response(data)
                content = self._response_content(data)
                review, error = parse_vlm_review(content)
                if review is None:
                    raise ValueError(error["message"] if error else "invalid local Gemma JSON")
                review.metadata.update(self._base_metadata(success=True, retry_count=attempt, fallback=False))
                review.metadata["vlm_input"] = vlm_input
                self._annotate_sam_consistency(evidence, review)
                logger.info(
                    "Local Gemma review attempt %s/%s: success level=%s confidence=%.3f normalized=%s conflict_with_sam=%s",
                    attempt + 1,
                    self.max_retries + 1,
                    review.alarm_level_suggestion,
                    review.confidence,
                    review.metadata.get("normalized", False),
                    review.metadata.get("conflict_with_sam", False),
                )
                return review
            except GEMMA_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning(
                    "Local Gemma review attempt %s/%s failed: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    type(exc).__name__,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._write_error(exc, retry_count=attempt)
                if self.fallback_on_error:
                    fallback = self._fallback_review(evidence, type(exc).__name__, str(exc), retry_count=attempt)
                    fallback.metadata["vlm_input"] = vlm_input
                    self._annotate_sam_consistency(evidence, fallback)
                    return fallback
                raise

        assert last_exc is not None
        fallback = self._fallback_review(evidence, type(last_exc).__name__, str(last_exc), retry_count=self.max_retries)
        fallback.metadata["vlm_input"] = vlm_input
        self._annotate_sam_consistency(evidence, fallback)
        return fallback

    def _payload(self, evidence: EventEvidence, vlm_input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build OpenAI-compatible multimodal payload with compact evidence."""
        vlm_input = vlm_input or self._vlm_input_metadata(evidence)
        content: list[dict[str, Any]] = []
        for image_url in self._image_paths_to_data_urls(vlm_input["image_paths"]):
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": vlm_input["prompt_text"]})
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "thinking": {"type": "disabled"},
            "reasoning": {"exclude": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }

    def _review_text(self, evidence: EventEvidence) -> str:
        """Return a local-Gemma prompt for the selected evidence mode."""
        if self.evidence_mode == "qwen":
            return build_review_prompt(evidence)

        compact_evidence = self._minimal_evidence(evidence) if self.evidence_mode == "minimal" else self._compact_evidence(evidence)
        return (
            "Return only JSON. No analysis. No markdown. No code fence.\n"
            "Required keys: is_anomaly,event_type,alarm_level_suggestion,confidence,evidence_time,evidence_tracks,"
            "matched_rules,reason,possible_false_alarm,recommended_action.\n"
            "Allowed alarm_level_suggestion: none, low, medium, high.\n"
            "If no anomaly, use event_type=\"none\", alarm_level_suggestion=\"none\", evidence_time=[], evidence_tracks=[], matched_rules=[].\n"
            "If metadata.sam_decision.is_suspicious is true, keep the event for safety.\n"
            f"Example shape: {OUTPUT_SCHEMA}\n"
            "Evidence JSON:\n"
            f"{json.dumps(compact_evidence, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a JSON generator for railway alarm review. "
            "Output exactly one compact JSON object and nothing else."
        )

    def _vlm_input_metadata(self, evidence: EventEvidence) -> dict[str, Any]:
        """Return the non-secret evidence packet sent to the local VLM."""
        prompt_text = self._review_text(evidence)
        image_paths = self._sent_image_paths(evidence)
        return {
            "provider": "local_gemma",
            "evidence_mode": self.evidence_mode,
            "image_count": len(image_paths),
            "image_paths": image_paths,
            "prompt_text": prompt_text,
            "prompt_text_chars": len(prompt_text),
            "evidence_summary": summarize_evidence(evidence) if self.evidence_mode == "qwen" else self._minimal_evidence(evidence),
        }

    @staticmethod
    def _compact_evidence(evidence: EventEvidence) -> dict[str, Any]:
        """Compress event evidence for local VLMs with small context windows."""
        objects = []
        for track in evidence.objects[:30]:
            first_bbox = track.bboxes[0] if track.bboxes else None
            last_bbox = track.bboxes[-1] if track.bboxes else None
            first_point = track.trajectory[0] if track.trajectory else None
            last_point = track.trajectory[-1] if track.trajectory else None
            objects.append(
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "confidence": round(track.confidence, 3),
                    "bbox_count": len(track.bboxes),
                    "first_bbox": LocalGemmaProvider._compact_bbox(first_bbox.bbox) if first_bbox else None,
                    "last_bbox": LocalGemmaProvider._compact_bbox(last_bbox.bbox) if last_bbox else None,
                    "start_xy": LocalGemmaProvider._compact_xy(first_point.x, first_point.y) if first_point else None,
                    "end_xy": LocalGemmaProvider._compact_xy(last_point.x, last_point.y) if last_point else None,
                    "entered_rois": track.entered_rois,
                    "dwell_time": round(track.dwell_time, 2),
                    "direction": track.direction,
                }
            )
        keyframes = []
        for keyframe in evidence.keyframes[:5]:
            keyframes.append(
                {
                    "timestamp": round(keyframe.timestamp, 3),
                    "frame_path": keyframe.frame_path,
                    "annotated_frame_path": keyframe.annotated_frame_path,
                    "reason": keyframe.reason,
                    "boxes": [
                        {
                            "track_id": box.get("track_id"),
                            "label": box.get("label"),
                            "confidence": round(float(box.get("confidence", 0.0)), 3),
                            "bbox": LocalGemmaProvider._compact_bbox(box.get("bbox") or []),
                        }
                        for box in keyframe.boxes[:30]
                    ],
                }
            )
        windows = [
            {
                "window_id": window.window_id,
                "start": round(window.start, 3),
                "end": round(window.end, 3),
                "object_count": window.object_count,
                "active_tracks": window.active_tracks[:30],
                "triggered_rules": window.triggered_rules,
                "visual_summary": window.visual_summary,
            }
            for window in evidence.windows[:8]
        ]
        metadata = evidence.metadata
        sam_tracking = metadata.get("sam_tracking") if isinstance(metadata.get("sam_tracking"), dict) else {}
        sam_summary = sam_tracking.get("summary") if isinstance(sam_tracking.get("summary"), dict) else None
        compact_metadata = {
            "input_type": metadata.get("input_type"),
            "rule_source": metadata.get("rule_source"),
            "tracker": metadata.get("tracker"),
            "sam_tracking_summary": sam_summary,
            "sam_decision": LocalGemmaProvider._sam_decision(evidence),
            "visualization": metadata.get("visualization"),
        }
        return {
            "event_id": evidence.event_id,
            "camera_id": evidence.camera_id,
            "source": evidence.video_path,
            "time_range": evidence.time_range,
            "fps": evidence.fps,
            "counts": {
                "objects": len(evidence.objects),
                "rules": len(evidence.roi_rules),
                "triggered_rules": sum(1 for rule in evidence.roi_rules if rule.triggered),
                "keyframes": len(evidence.keyframes),
                "windows": len(evidence.windows),
            },
            "roi_rules": [rule.model_dump(mode="json") for rule in evidence.roi_rules],
            "objects": objects,
            "keyframes": keyframes,
            "windows": windows,
            "metadata": compact_metadata,
        }

    @staticmethod
    def _minimal_evidence(evidence: EventEvidence) -> dict[str, Any]:
        """Return the smallest useful evidence packet for small-context local VLMs."""
        objects = []
        for track in evidence.objects[:12]:
            bbox = track.bboxes[-1].bbox if track.bboxes else []
            objects.append(
                {
                    "id": track.track_id,
                    "label": track.label,
                    "conf": round(track.confidence, 3),
                    "bbox": LocalGemmaProvider._compact_bbox(bbox),
                }
            )
        triggered_rules = [rule.rule_id for rule in evidence.roi_rules if rule.triggered]
        object_count: dict[str, int] = {}
        for track in evidence.objects:
            object_count[track.label] = object_count.get(track.label, 0) + 1
        visualization = evidence.metadata.get("visualization")
        annotated_image = visualization.get("annotated_image") if isinstance(visualization, dict) else None
        return {
            "event_id": evidence.event_id,
            "camera_id": evidence.camera_id,
            "input_type": evidence.metadata.get("input_type"),
            "source": Path(evidence.video_path).name,
            "object_count": object_count,
            "objects": objects,
            "triggered_rules": triggered_rules,
            "sam_decision": LocalGemmaProvider._sam_decision(evidence),
            "annotated_image": annotated_image,
        }

    @staticmethod
    def _sam_decision(evidence: EventEvidence) -> dict[str, Any]:
        """Summarize the authoritative SAMTracking rule decision."""
        sam_meta = evidence.metadata.get("sam_tracking") if isinstance(evidence.metadata.get("sam_tracking"), dict) else {}
        summary = sam_meta.get("summary") if isinstance(sam_meta.get("summary"), dict) else {}
        rules = [rule for rule in evidence.roi_rules if rule.rule_type == "mask_iou_intrusion"]
        triggered = any(rule.triggered for rule in rules)
        severity = "none"
        evidence_tracks: list[int] = []
        matched_rules: list[str] = []
        for rule in rules:
            matched_rules.append(rule.rule_id)
            evidence_tracks.extend(rule.evidence_tracks)
            severity = LocalGemmaProvider._max_level(severity, rule.severity_hint)
        suspicious_frames = int(summary.get("suspicious_frame_count") or 0)
        alarm_frames = int(summary.get("alarm_frame_count") or 0)
        intrusion_events = int(summary.get("intrusion_event_count") or 0)
        is_suspicious = triggered or suspicious_frames > 0 or alarm_frames > 0 or intrusion_events > 0
        if severity == "none" and is_suspicious:
            severity = "high" if alarm_frames > 0 or intrusion_events > 0 else "medium"
        return {
            "rule_source": evidence.metadata.get("rule_source"),
            "track_mask_seen": bool(summary.get("track_mask_seen")),
            "detections_seen": bool(summary.get("detections_seen")),
            "is_suspicious": is_suspicious,
            "risk_level": severity,
            "matched_rules": matched_rules,
            "evidence_tracks": sorted(set(evidence_tracks)),
            "max_iou": round(float(summary.get("max_iou") or 0.0), 4),
            "max_object_overlap": round(float(summary.get("max_object_overlap") or 0.0), 4),
            "iou_threshold": summary.get("iou_threshold"),
            "object_overlap_threshold": summary.get("object_overlap_threshold"),
            "suspicious_frame_count": suspicious_frames,
            "alarm_frame_count": alarm_frames,
            "intrusion_event_count": intrusion_events,
        }

    @staticmethod
    def _annotate_sam_consistency(evidence: EventEvidence, review: VLMReview) -> None:
        """Mark review metadata when VLM and SAMTracking disagree."""
        sam_decision = LocalGemmaProvider._sam_decision(evidence)
        sam_suspicious = bool(sam_decision.get("is_suspicious"))
        vlm_suspicious = bool(review.is_anomaly) and review.alarm_level_suggestion != "none"
        conflict = sam_suspicious != vlm_suspicious
        review.metadata["sam_decision"] = sam_decision
        review.metadata["conflict_with_sam"] = conflict
        review.metadata["needs_review"] = conflict
        if conflict:
            review.metadata["conflict_reason"] = (
                "SAMTracking reported suspicious evidence but VLM did not."
                if sam_suspicious
                else "VLM reported anomaly but SAMTracking rule did not trigger."
            )

    @staticmethod
    def _max_level(left: str, right: str) -> str:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        return left if order.get(left, 0) >= order.get(right, 0) else right

    @staticmethod
    def _compact_bbox(bbox: list[Any]) -> list[float]:
        return [round(float(value), 1) for value in bbox[:4]]

    @staticmethod
    def _compact_xy(x: float, y: float) -> list[float]:
        return [round(float(x), 1), round(float(y), 1)]

    @staticmethod
    def _image_paths_to_data_urls(image_paths: list[str]) -> list[str]:
        urls: list[str] = []
        for candidate in image_paths:
            path = Path(candidate)
            mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            urls.append(f"data:{mime};base64,{data}")
        return urls

    def _sent_image_paths(self, evidence: EventEvidence) -> list[str]:
        """Return local image paths included in the VLM payload."""
        paths: list[str] = []
        seen: set[str] = set()
        for candidate in self._image_candidates(evidence):
            if not candidate or candidate in seen:
                continue
            path = Path(candidate)
            if not path.exists() or not path.is_file():
                continue
            seen.add(candidate)
            paths.append(candidate)
            if len(paths) >= self.max_images:
                return paths
        return paths

    @staticmethod
    def _image_candidates(evidence: EventEvidence) -> list[str]:
        """Return visualization images first, then raw keyframes."""
        candidates: list[str] = []
        visualization = evidence.metadata.get("visualization")
        if isinstance(visualization, dict):
            candidates.extend(
                [
                    str(visualization.get("annotated_image") or ""),
                    str(visualization.get("overview_image") or ""),
                ]
            )
        for keyframe in evidence.keyframes:
            candidates.extend([keyframe.annotated_frame_path or "", keyframe.frame_path])
        return candidates

    @staticmethod
    def _response_content(data: dict[str, Any]) -> str:
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
        text = str(content or "").strip()
        if text:
            return text
        return LocalGemmaProvider._extract_json_candidate(str(message.get("reasoning_content", "")))

    @staticmethod
    def _extract_json_candidate(text: str) -> str:
        """Extract the last complete JSON object from verbose local reasoning."""
        if not text:
            return ""
        start = text.rfind("{")
        while start >= 0:
            candidate = text[start:].strip()
            end = candidate.rfind("}")
            if end >= 0:
                candidate = candidate[: end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass
            start = text.rfind("{", 0, start)
        return ""

    def _base_metadata(self, success: bool, retry_count: int, fallback: bool) -> dict[str, Any]:
        return {
            "provider": "local_gemma",
            "success": success,
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout_seconds": self.timeout_sec,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "retry_count": retry_count,
            "max_retries": self.max_retries,
            "fallback": fallback,
            "evidence_mode": self.evidence_mode,
        }

    def _fallback_review(self, evidence: EventEvidence, error_type: str, error_message: str, retry_count: int) -> VLMReview:
        triggered = [rule for rule in evidence.roi_rules if rule.triggered]
        return VLMReview(
            is_anomaly=False,
            event_type="none",
            alarm_level_suggestion="none",
            confidence=0.0,
            evidence_time=[],
            evidence_tracks=[],
            matched_rules=[rule.rule_id for rule in triggered],
            reason=f"Local Gemma review failed: {error_type}. Fallback review generated.",
            possible_false_alarm=True,
            recommended_action="manual review recommended",
            metadata={
                **self._base_metadata(success=False, retry_count=retry_count, fallback=True),
                "error_type": error_type,
                "error_message": self._sanitize_error(error_message),
            },
        )

    def _write_request_summary(self, payload: dict[str, Any]) -> None:
        if not self.artifact_dir:
            return
        messages = payload.get("messages", [])
        image_count = 0
        text_chars = 0
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                image_count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image_url")
                text_chars += sum(len(str(item.get("text", ""))) for item in content if isinstance(item, dict) and item.get("type") == "text")
            elif isinstance(content, str):
                text_chars += len(content)
        save_json(
            {
                "provider": "local_gemma",
                "timestamp": self._now(),
                "endpoint": self.endpoint,
                "model": self.model,
                "image_count": image_count,
                "text_chars": text_chars,
                "evidence_mode": self.evidence_mode,
            },
            str(self.artifact_dir / "gemma_request_summary.json"),
        )

    def _write_raw_response(self, data: dict[str, Any]) -> None:
        if not self.artifact_dir:
            return
        logger.info("Local Gemma raw response artifact: %s", self.artifact_dir / "gemma_raw_response.json")
        save_json(
            {"provider": "local_gemma", "success": True, "timestamp": self._now(), "response": data},
            str(self.artifact_dir / "gemma_raw_response.json"),
        )

    def _write_error(self, exc: Exception, retry_count: int) -> None:
        if not self.artifact_dir:
            return
        logger.info("Local Gemma error artifact: %s", self.artifact_dir / "gemma_error.json")
        save_json(
            {
                **self._base_metadata(success=False, retry_count=retry_count, fallback=True),
                "error_type": type(exc).__name__,
                "error_message": self._sanitize_error(str(exc)),
                "timestamp": self._now(),
            },
            str(self.artifact_dir / "gemma_error.json"),
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _sanitize_error(message: str) -> str:
        return message.replace("Bearer ", "Bearer [REDACTED] ")[:500]
