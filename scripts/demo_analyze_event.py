"""Analyze a finalized event video or a stored SQLite event.

Examples:
    python scripts/demo_analyze_event.py --event-id 1
    python scripts/demo_analyze_event.py --video data/video/cam_001/xxx.mp4 --camera-id cam_001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analysis.event_analyzer import EventAnalyzer
from app.config import get_config
from app.database import EventDatabase


def main() -> int:
    parser = argparse.ArgumentParser(description="Railway perimeter event video review demo")
    parser.add_argument("--config", default="configs/cameras.yaml", help="YAML config path")
    parser.add_argument("--event-id", type=int, default=None, help="Event ID from SQLite")
    parser.add_argument("--video", default=None, help="Video path if not using --event-id")
    parser.add_argument("--camera-id", default=None, help="Camera ID if not using --event-id")
    parser.add_argument("--alarm-image", default=None, help="Optional original FTP alarm image")
    parser.add_argument("--output-json", default=None, help="Optional path to save analysis JSON")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(PROJECT_ROOT / config_path)
    config = get_config(config_path)

    event = None
    if args.event_id is not None:
        event = EventDatabase(config.db_path).get_event(args.event_id)
        if not event:
            raise SystemExit(f"event_id not found: {args.event_id}")
        camera_id = event.camera_id
        video_path = event.video_path
        alarm_image = event.image_path
        event_time = event.event_time
        previous_result = event.analysis_result
    else:
        if not args.video or not args.camera_id:
            raise SystemExit("Either --event-id or both --video and --camera-id are required")
        camera_id = args.camera_id
        video_path = args.video
        alarm_image = args.alarm_image
        event_time = datetime.now()
        previous_result = None

    camera_config = config.cameras.get(camera_id)
    if not camera_config:
        raise SystemExit(f"camera_id not found in config: {camera_id}")

    output_dir = None
    if video_path:
        output_dir = str(Path(video_path).with_suffix("")) + "_frames"
    analyzer = EventAnalyzer()
    result = analyzer.analyze_event(
        video_path=video_path,
        camera_id=camera_id,
        event_time=event_time,
        camera_config=camera_config,
        alarm_image_path=alarm_image,
        output_dir=output_dir,
        previous_result=previous_result,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
