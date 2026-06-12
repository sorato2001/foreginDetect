from src.evidence.evidence_builder import empty_evidence
from src.vlm.prompts import summarize_evidence


def test_sam_tracking_summary_is_visible_to_vlm_prompt():
    evidence = empty_evidence("event_sam", "cam01", "video.mp4")
    evidence.metadata["sam_tracking"] = {
        "summary": {
            "rule_type": "mask_iou_intrusion",
            "track_mask_seen": True,
            "max_iou": 0.42,
            "suspicious_frame_count": 3,
            "alarm_frame_count": 1,
        }
    }

    summary = summarize_evidence(evidence)

    assert summary["metadata"]["sam_tracking"]["summary"]["max_iou"] == 0.42
    assert summary["metadata"]["sam_tracking"]["summary"]["track_mask_seen"] is True
