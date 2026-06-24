import json
from pathlib import Path

from src.evidence.evidence_builder import empty_evidence
from src.evidence.evidence_schema import KeyframeInfo, ObjectTrack, ROIRuleTrigger, TrajectoryPoint, WindowDescription
from src.vlm.local_gemma_provider import LocalGemmaProvider


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "is_anomaly": True,
                                "event_type": "restricted_area_intrusion",
                                "alarm_level_suggestion": "high",
                                "confidence": 0.9,
                                "evidence_time": [[0.0, 1.0]],
                                "evidence_tracks": [1],
                                "matched_rules": ["sam_mask_iou_intrusion"],
                                "reason": "目标进入轨道分割区域。",
                                "possible_false_alarm": False,
                                "recommended_action": "notify security staff and save the event clip",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


def test_local_gemma_provider_parses_openai_compatible_response(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr("src.vlm.local_gemma_provider.requests.post", fake_post)
    tmp_path = Path("outputs/test_artifacts/local_gemma_provider")
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"not-a-real-jpeg-but-good-enough-for-base64")
    evidence = empty_evidence("event_gemma", "cam01", "video.mp4")
    evidence.keyframes = [KeyframeInfo(timestamp=0.0, frame_path=str(image_path), reason="input_image")]
    evidence.roi_rules = [
        ROIRuleTrigger(
            rule_id="sam_mask_iou_intrusion",
            rule_type="mask_iou_intrusion",
            triggered=True,
            trigger_time=[0.0, 1.0],
            evidence_tracks=[1],
            severity_hint="high",
        )
    ]

    provider = LocalGemmaProvider(endpoint="http://local/v1/chat/completions", artifact_dir=str(tmp_path))
    review = provider.review(evidence)

    assert review.is_anomaly is True
    assert review.alarm_level_suggestion == "high"
    assert review.metadata["provider"] == "local_gemma"
    assert review.metadata["vlm_input"]["provider"] == "local_gemma"
    assert review.metadata["vlm_input"]["image_paths"] == [str(image_path)]
    assert "prompt_text" in review.metadata["vlm_input"]
    assert calls[0]["url"] == "http://local/v1/chat/completions"
    assert calls[0]["json"]["messages"][1]["content"][0]["type"] == "image_url"
    prompt_text = calls[0]["json"]["messages"][1]["content"][-1]["text"]
    assert "Evidence JSON" in prompt_text
    assert '"sam_decision"' in prompt_text
    assert "full_event_evidence_json" not in prompt_text
    assert Path(tmp_path / "gemma_raw_response.json").exists()


def test_local_gemma_prompt_includes_sam_decision():
    evidence = empty_evidence("event_sam", "cam01", "video.mp4")
    evidence.metadata["rule_source"] = "sam_tracking_mask_iou"
    evidence.metadata["sam_tracking"] = {
        "summary": {
            "track_mask_seen": True,
            "detections_seen": True,
            "max_iou": 0.2,
            "max_object_overlap": 0.6,
            "iou_threshold": 0.1,
            "object_overlap_threshold": 0.15,
            "suspicious_frame_count": 1,
            "alarm_frame_count": 1,
            "intrusion_event_count": 1,
        }
    }
    evidence.roi_rules = [
        ROIRuleTrigger(
            rule_id="sam_mask_iou_intrusion",
            rule_type="mask_iou_intrusion",
            triggered=True,
            evidence_tracks=[1],
            severity_hint="high",
        )
    ]

    payload = LocalGemmaProvider(max_images=0)._payload(evidence)
    prompt_text = payload["messages"][1]["content"][-1]["text"]

    assert '"sam_decision"' in prompt_text
    assert '"is_suspicious":true' in prompt_text
    assert '"risk_level":"high"' in prompt_text
    assert '"keyframes"' not in prompt_text
    assert '"windows"' not in prompt_text


def test_local_gemma_qwen_mode_uses_structured_video_prompt():
    evidence = empty_evidence("event_sam_video", "cam01", "video.mp4")
    evidence.metadata["rule_source"] = "sam_tracking_mask_iou"
    evidence.metadata["sam_tracking"] = {
        "summary": {
            "track_mask_seen": True,
            "detections_seen": True,
            "max_iou": 0.2,
            "max_object_overlap": 0.6,
            "suspicious_frame_count": 3,
            "alarm_frame_count": 2,
            "intrusion_event_count": 1,
        }
    }
    evidence.objects = [
        ObjectTrack(
            track_id=1,
            label="person",
            confidence=0.92,
            trajectory=[TrajectoryPoint(timestamp=0.0, x=10.0, y=20.0), TrajectoryPoint(timestamp=1.0, x=30.0, y=40.0)],
        )
    ]
    evidence.roi_rules = [
        ROIRuleTrigger(
            rule_id="sam_mask_iou_intrusion",
            rule_type="mask_iou_intrusion",
            triggered=True,
            evidence_tracks=[1],
            severity_hint="high",
        )
    ]
    evidence.windows = [
        WindowDescription(
            window_id="w0",
            start=0.0,
            end=1.0,
            object_count={"person": 1},
            active_tracks=[1],
            triggered_rules=["sam_mask_iou_intrusion"],
            visual_summary="person overlaps segmented railway track",
        )
    ]

    payload = LocalGemmaProvider(max_images=0, evidence_mode="qwen")._payload(evidence)
    prompt_text = payload["messages"][1]["content"][-1]["text"]

    assert "Track Stream" in prompt_text
    assert "SAMTracking Stream" in prompt_text
    assert "Window Stream" in prompt_text
    assert "sam_tracking" in prompt_text
    assert "sam_mask_iou_intrusion" in prompt_text
    assert "event_evidence_summary" in prompt_text


def test_local_gemma_review_metadata_contains_qwen_mode_evidence(monkeypatch, tmp_path):
    def fake_post(url, headers, json, timeout):
        return _Response()

    monkeypatch.setattr("src.vlm.local_gemma_provider.requests.post", fake_post)
    evidence = empty_evidence("event_sam_video", "cam01", "video.mp4")
    evidence.metadata["rule_source"] = "sam_tracking_mask_iou"
    evidence.metadata["sam_tracking"] = {
        "summary": {
            "track_mask_seen": True,
            "detections_seen": True,
            "max_object_overlap": 0.6,
            "alarm_frame_count": 2,
        }
    }
    evidence.roi_rules = [
        ROIRuleTrigger(
            rule_id="sam_mask_iou_intrusion",
            rule_type="mask_iou_intrusion",
            triggered=True,
            evidence_tracks=[1],
            severity_hint="high",
        )
    ]

    review = LocalGemmaProvider(
        endpoint="http://local/v1/chat/completions",
        artifact_dir=str(tmp_path),
        max_images=0,
        evidence_mode="qwen",
    ).review(evidence)

    vlm_input = review.metadata["vlm_input"]
    assert vlm_input["evidence_mode"] == "qwen"
    assert "SAMTracking Stream" in vlm_input["prompt_text"]
    assert vlm_input["evidence_summary"]["metadata"]["sam_tracking"]["summary"]["alarm_frame_count"] == 2


def test_local_gemma_extracts_json_from_reasoning_content():
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": 'analysis first {"is_anomaly":false,"event_type":"none","alarm_level_suggestion":"none","confidence":0.9,"evidence_time":[],"evidence_tracks":[],"matched_rules":[],"reason":"no intrusion","possible_false_alarm":false,"recommended_action":"none"}',
                }
            }
        ]
    }

    text = LocalGemmaProvider._response_content(data)

    assert text.startswith('{"is_anomaly"')
    assert "no intrusion" in text


def test_local_gemma_ignores_incomplete_reasoning_json():
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": 'analysis ```json\n{"is_anomaly": false, "event_type"',
                }
            }
        ]
    }

    assert LocalGemmaProvider._response_content(data) == ""
