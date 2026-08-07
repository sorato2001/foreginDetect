import json
import sys
from datetime import datetime
from pathlib import Path

from src.pipeline import analyze_event


def test_default_output_dir_uses_timestamp_and_avoids_collision(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    now = datetime(2026, 8, 7, 14, 45, 21)

    assert analyze_event._default_output_dir(now) == str(Path("outputs") / "20260807144521")

    (tmp_path / "outputs" / "20260807144521").mkdir(parents=True)
    assert analyze_event._default_output_dir(now) == str(Path("outputs") / "20260807144521_01")


def test_cli_default_output_writes_all_arguments_and_effective_settings(monkeypatch, tmp_path):
    output_dir = tmp_path / "outputs" / "20260807144521"
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return {"output_dir": kwargs["output_dir"], "is_alarm": False}

    monkeypatch.setattr(analyze_event, "_default_output_dir", lambda: str(output_dir))
    monkeypatch.setattr(analyze_event, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_event.py",
            "--video",
            "missing.mp4",
            "--camera-id",
            "cam01",
            "--vlm-provider",
            "mock",
        ],
    )

    assert analyze_event.main() == 0
    assert captured["output_dir"] == str(output_dir)

    config_path = output_dir / analyze_event.RUN_CONFIG_FILENAME
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["output_was_explicit"] is False
    assert payload["arguments_include_defaults"] is True
    assert payload["arguments"]["output"] == str(output_dir)
    assert payload["arguments"]["sam_imgsz"] == 640
    assert payload["arguments"]["sam2_enabled"] == "true"
    assert payload["effective_settings"]["sam2_enabled"] is True
    assert payload["effective_settings"]["sam_track_labels"] == []
    assert "--output" not in payload["command"]
    assert f'--output {output_dir}' in payload["replay_command"]


def test_cli_run_config_is_copied_to_batch_item_outputs(tmp_path):
    root = tmp_path / "batch"
    item_a = root / "a"
    item_b = root / "b"
    result = {"results": [{"output_dir": str(item_a)}, {"output_dir": str(item_b)}]}
    config = {"arguments_include_defaults": True, "arguments": {"output": str(root)}}

    paths = analyze_event._save_cli_run_config(
        config,
        analyze_event._result_output_dirs(result, str(root)),
    )

    assert len(paths) == 3
    for output_dir in (root, item_a, item_b):
        assert (output_dir / analyze_event.RUN_CONFIG_FILENAME).exists()
