from src.vlm.json_validator import parse_vlm_review


def test_valid_json():
    text = '{"is_anomaly": true, "event_type": "intrusion", "alarm_level_suggestion": "medium", "confidence": 0.8, "evidence_time": [[1,2]], "evidence_tracks": [1], "matched_rules": ["r1"], "reason": "ok", "possible_false_alarm": false, "recommended_action": "check"}'
    review, error = parse_vlm_review(text)
    assert error is None
    assert review is not None
    assert review.alarm_level_suggestion == "medium"


def test_invalid_json_returns_error():
    review, error = parse_vlm_review("not json")
    assert review is None
    assert error is not None


def test_missing_field_returns_error():
    review, error = parse_vlm_review('{"is_anomaly": true}')
    assert review is None
    assert error is not None

