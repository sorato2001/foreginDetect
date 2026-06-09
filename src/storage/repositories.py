"""Repository helpers for STEAD artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_artifact(path: str) -> dict[str, Any]:
    """Load an artifact JSON file."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def event_output_dir(base_dir: str, event_id: str) -> Path:
    """Return the expected output directory for an event."""
    return Path(base_dir) / event_id

