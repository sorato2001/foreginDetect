"""Command-line pipeline for STEAD event analysis."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from src.alarm.alarm_engine import AlarmEngine
from src.config.settings import ensure_output_dir
from src.evidence.evidence_builder import build_object_tracks, empty_evidence
from src.evidence.serializers import save_model
from src.evidence.window_builder import WindowBuilder
from src.perception.detector import Detection
from src.perception.simple_iou_tracker import SimpleIOUTracker
from src.perception.yolo_detector import YoloDetector
from src.rules.rule_engine import RuleEngine
from src.vlm.mock_provider import MockVLMProvider
from src.vlm.qwen_provider import QwenProvider


def _fake_tracks() -> dict[int, list[Detection]]:
    """Return deterministic tracks for smoke tests and offline demos."""
    return {
        1: [
            Detection("person", 0.85, [120, 120, 170, 220], 0, 0.0),
            Detection("person", 0.88, [180, 150, 230, 250], 1, 2.0),
            Detection("person", 0.90, [260, 180, 310, 280], 2, 4.0),
        ]
    }


def _detect_video(video_path: str, mock_detections: bool = False) -> tuple[dict[int, list[Detection]], float, float]:
    """Detect and track objects in a video with graceful fallback."""
    if mock_detections:
        return _fake_tracks(), 25.0, 6.0
    try:
        import cv2
    except Exception:
        return {}, 0.0, 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}, 0.0, 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detector = YoloDetector()
    tracker = SimpleIOUTracker()
    sample_every = max(1, int(fps))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            timestamp = frame_index / fps if fps > 0 else 0.0
            tracker.update(detector.detect_frame(frame, frame_index, timestamp))
        frame_index += 1
    cap.release()
    duration = total / fps if fps > 0 and total else frame_index / fps if fps > 0 else 0.0
    return tracker.tracks, fps, duration


def run_pipeline(
    video_path: str,
    camera_id: str,
    rules_path: str,
    output_dir: str,
    vlm_provider: str = "mock",
    event_id: str | None = None,
    mock_detections: bool = False,
) -> dict:
    """Run STEAD event analysis and write JSON artifacts."""
    event_id = event_id or f"event_{uuid.uuid4().hex[:8]}"
    out = ensure_output_dir(output_dir)
    tracks, fps, duration = _detect_video(video_path, mock_detections=mock_detections)
    evidence = empty_evidence(event_id, camera_id, video_path, duration=duration, fps=fps or None)
    evidence.objects = build_object_tracks(tracks)
    rule_engine = RuleEngine.from_yaml(rules_path)
    evidence.roi_rules = rule_engine.evaluate(evidence)
    evidence.windows = WindowBuilder().build(evidence)
    evidence.metadata.update({"mock_detections": mock_detections, "stead_version": "stead_v1"})

    provider = QwenProvider() if vlm_provider == "qwen" else MockVLMProvider()
    review = provider.review(evidence)
    alarm = AlarmEngine().fuse(evidence, review)

    save_model(evidence, str(out / "event_evidence.json"))
    save_model(review, str(out / "vlm_review.json"))
    save_model(alarm, str(out / "alarm_result.json"))
    return {
        "event_id": event_id,
        "output_dir": str(out),
        "event_evidence": str(out / "event_evidence.json"),
        "vlm_review": str(out / "vlm_review.json"),
        "alarm_result": str(out / "alarm_result.json"),
        "final_level": alarm.final_level,
        "is_alarm": alarm.is_alarm,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze one surveillance event with STEAD.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--rules", default="configs/rules.example.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vlm-provider", choices=["mock", "qwen"], default="mock")
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--mock-detections", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        video_path=args.video,
        camera_id=args.camera_id,
        rules_path=args.rules,
        output_dir=args.output,
        vlm_provider=args.vlm_provider,
        event_id=args.event_id,
        mock_detections=args.mock_detections,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

