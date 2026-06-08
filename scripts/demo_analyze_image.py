"""Analyze one alarm image with the railway perimeter review pipeline.

Usage:
    python scripts/demo_analyze_image.py --image path/to/alarm.jpg --camera-id cam_001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analysis.event_analyzer import EventAnalyzer
from app.config import get_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Railway perimeter image alarm review demo")
    parser.add_argument("--image", required=True, help="Path to an FTP alarm image")
    parser.add_argument("--camera-id", required=True, help="Camera ID in YAML config")
    parser.add_argument("--config", default="configs/cameras.yaml", help="YAML config path")
    parser.add_argument("--output-json", default=None, help="Optional path to save analysis JSON")
    parser.add_argument("--visualize", default=None, help="Optional path to save visualization image")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(PROJECT_ROOT / config_path)
    config = get_config(config_path)
    camera_config = config.cameras.get(args.camera_id)
    if not camera_config:
        raise SystemExit(f"camera_id not found in config: {args.camera_id}")

    analyzer = EventAnalyzer()
    result = analyzer.analyze_image(
        image_path=args.image,
        camera_id=args.camera_id,
        event_time=datetime.now(),
        camera_config=camera_config,
        save_visualization_path=args.visualize,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
