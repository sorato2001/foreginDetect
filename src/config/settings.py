"""Settings loader for STEAD config files and environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def ensure_output_dir(path: str) -> Path:
    """Create and return an output directory path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out

