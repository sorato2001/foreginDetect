"""Artifact routes for STEAD event outputs."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


def _load(output_root: str, event_id: str, filename: str) -> dict:
    path = Path(output_root) / event_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found for {event_id}")
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


@router.get("/events")
def list_events(output_root: str = Query("outputs")) -> dict:
    """List event output folders that contain alarm_result.json."""
    root = Path(output_root)
    events = []
    if root.exists():
        for item in sorted(root.iterdir()):
            alarm_path = item / "alarm_result.json"
            if alarm_path.exists():
                with open(alarm_path, "r", encoding="utf-8") as file_obj:
                    alarm = json.load(file_obj)
                events.append({"event_id": item.name, "alarm": alarm})
    return {"total": len(events), "events": events}


@router.get("/events/{event_id}/evidence")
def get_evidence(event_id: str, output_root: str = Query("outputs")) -> dict:
    """Return event_evidence.json."""
    return _load(output_root, event_id, "event_evidence.json")


@router.get("/events/{event_id}/vlm-review")
def get_vlm_review(event_id: str, output_root: str = Query("outputs")) -> dict:
    """Return vlm_review.json."""
    return _load(output_root, event_id, "vlm_review.json")


@router.get("/events/{event_id}/alarm")
def get_alarm(event_id: str, output_root: str = Query("outputs")) -> dict:
    """Return alarm_result.json."""
    return _load(output_root, event_id, "alarm_result.json")

