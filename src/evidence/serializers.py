"""JSON serializers for STEAD schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def save_model(model: BaseModel, path: str) -> None:
    """Save a Pydantic model as UTF-8 JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(model.model_dump_json(indent=2))


def save_json(data: dict[str, Any], path: str) -> None:
    """Save a dict as UTF-8 JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)

