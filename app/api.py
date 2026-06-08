"""FastAPI-based REST API for system monitoring and querying.

The API keeps the original monitoring endpoints and exposes railway-security
analysis summaries for frontend alarm review dashboards.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


def _final_result(analysis_result: dict[str, Any] | None) -> dict[str, Any]:
    """Extract final_result with safe defaults."""
    if not analysis_result:
        return {}
    return analysis_result.get("final_result") or {}


def _llm_review(analysis_result: dict[str, Any] | None) -> dict[str, Any]:
    """Extract large-model review result with backward-compatible keys."""
    if not analysis_result:
        return {}
    return analysis_result.get("llm_review") or analysis_result.get("vlm_result") or {}


def _event_summary(event) -> dict[str, Any]:
    """Return one event item including railway alarm summary fields."""
    final = _final_result(event.analysis_result)
    llm = _llm_review(event.analysis_result)
    return {
        "id": event.id,
        "camera_id": event.camera_id,
        "media_type": event.media_type,
        "event_type": event.event_type,
        "event_time": event.event_time.isoformat(),
        "event_start_time": event.event_start_time.isoformat(),
        "event_end_time": event.event_end_time.isoformat(),
        "image_path": event.image_path,
        "video_path": event.video_path,
        "annotated_image_path": event.annotated_image_path,
        "risk_level": final.get("risk_level"),
        "alarm_title": final.get("alarm_title"),
        "alarm_reason": final.get("alarm_reason"),
        "recommended_action": final.get("recommended_action"),
        "llm_mode": llm.get("mode"),
        "llm_reason": llm.get("reason"),
        "llm_confidence": llm.get("confidence"),
        "llm_is_valid_alarm": llm.get("is_valid_alarm"),
        "is_alarm": final.get("is_alarm"),
        "needs_review": final.get("needs_review"),
        "analysis_result": event.analysis_result,
        "status": event.status,
        "alarm_count": event.alarm_count,
        "created_at": event.created_at.isoformat(),
    }


def create_app(event_system) -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Railway Perimeter Event Recording API",
        description="API for monitoring multi-camera railway perimeter event review system",
        version="1.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.event_system = event_system

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/status")
    async def system_status() -> dict:
        """Get overall system status."""
        status = app.event_system.get_status()
        status["timestamp"] = datetime.now().isoformat()
        return status

    @app.get("/cameras")
    async def list_cameras() -> dict:
        """List all configured cameras."""
        cameras = []
        for camera_id, config in app.event_system.config.cameras.items():
            recorder = app.event_system.recorder_manager.get_recorder(camera_id)
            active_event = app.event_system.scheduler.get_active_event(camera_id)
            security_cfg = getattr(config, "railway_security", {}) or {}
            camera_info = {
                "camera_id": camera_id,
                "name": config.name,
                "ip": config.ip,
                "enabled": config.enabled,
                "analysis_mode": app.event_system.config.analysis_mode,
                "railway_security_enabled": bool(security_cfg.get("enabled", False)),
                "recorder_healthy": recorder.is_healthy() if recorder else False,
                "active_event": None,
            }
            if active_event:
                camera_info["active_event"] = {
                    "status": active_event.status,
                    "media_type": active_event.media_type,
                    "alarm_time": active_event.alarm_time.isoformat(),
                    "record_until": active_event.record_until.isoformat(),
                    "alarm_count": active_event.alarm_count,
                    "final_result": (active_event.analysis_result or {}).get("final_result"),
                }
            cameras.append(camera_info)
        return {"total": len(cameras), "cameras": cameras, "timestamp": datetime.now().isoformat()}

    @app.get("/cameras/{camera_id}")
    async def get_camera(camera_id: str) -> dict:
        """Get detailed information for a specific camera."""
        config = app.event_system.config.cameras.get(camera_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        recorder = app.event_system.recorder_manager.get_recorder(camera_id)
        active_event = app.event_system.scheduler.get_active_event(camera_id)
        camera_info = {
            "camera_id": camera_id,
            "name": config.name,
            "ip": config.ip,
            "rtsp_url": config.rtsp_url,
            "enabled": config.enabled,
            "analysis_mode": app.event_system.config.analysis_mode,
            "railway_security": getattr(config, "railway_security", {}) or {},
            "cache_seconds": config.cache_seconds,
            "pre_seconds": config.pre_seconds,
            "post_seconds": config.post_seconds,
            "recorder": {
                "healthy": recorder.is_healthy() if recorder else False,
                "running": recorder.running if recorder else False,
                "restart_count": recorder.restart_count if recorder else 0,
            },
            "active_event": None,
        }
        if active_event:
            camera_info["active_event"] = {
                "status": active_event.status,
                "media_type": active_event.media_type,
                "alarm_time": active_event.alarm_time.isoformat(),
                "event_start": active_event.event_start.isoformat(),
                "record_until": active_event.record_until.isoformat(),
                "alarm_count": active_event.alarm_count,
                "analysis_result": active_event.analysis_result,
            }
        return camera_info

    @app.get("/events")
    async def list_events(
        camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
        limit: int = Query(100, ge=1, le=1000, description="Number of events to return"),
    ) -> dict:
        """List recorded events including final railway review summary."""
        events = app.event_system.db.get_events_by_camera(camera_id, limit=limit) if camera_id else app.event_system.db.get_all_events(limit=limit)
        return {
            "total": len(events),
            "events": [_event_summary(event) for event in events],
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/events/summary/stats")
    async def event_summary_stats(
        camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
        limit: int = Query(1000, ge=1, le=10000, description="Number of events to scan"),
    ) -> dict:
        """Return risk-level counts and alarm/false-alarm statistics."""
        events = app.event_system.db.get_events_by_camera(camera_id, limit=limit) if camera_id else app.event_system.db.get_all_events(limit=limit)
        risk_counter: Counter[str] = Counter()
        valid_alarm_count = 0
        false_alarm_count = 0
        needs_review_count = 0
        for event in events:
            final = _final_result(event.analysis_result)
            risk = str(final.get("risk_level") or "unknown")
            risk_counter[risk] += 1
            if final.get("is_alarm"):
                valid_alarm_count += 1
            else:
                false_alarm_count += 1
            if final.get("needs_review"):
                needs_review_count += 1
        return {
            "total": len(events),
            "risk_levels": dict(risk_counter),
            "valid_alarm_count": valid_alarm_count,
            "false_alarm_count": false_alarm_count,
            "needs_review_count": needs_review_count,
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/events/active/all")
    async def get_active_events() -> dict:
        """Get all currently active recording events."""
        active = app.event_system.scheduler.get_all_active_events()
        return {
            "total": len(active),
            "events": [
                {
                    "camera_id": event.camera_id,
                    "status": event.status,
                    "media_type": event.media_type,
                    "alarm_time": event.alarm_time.isoformat(),
                    "event_start": event.event_start.isoformat(),
                    "record_until": event.record_until.isoformat(),
                    "alarm_count": event.alarm_count,
                    "final_result": (event.analysis_result or {}).get("final_result"),
                }
                for event in active.values()
            ],
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/events/{event_id}/frames")
    async def get_event_frames(event_id: int) -> dict:
        """Return key-frame paths saved by event-level review."""
        event = app.event_system.db.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        artifacts = (event.analysis_result or {}).get("artifacts") or {}
        frames = artifacts.get("key_frames") or []
        return {"event_id": event_id, "total": len(frames), "frames": frames}

    @app.get("/events/{event_id}")
    async def get_event(event_id: int) -> dict:
        """Get complete details for a specific event."""
        event = app.event_system.db.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        data = _event_summary(event)
        data["updated_at"] = event.updated_at.isoformat()
        return data

    @app.get("/health/recorders")
    async def recorder_health() -> dict:
        """Get health status of all recorders."""
        health = app.event_system.recorder_manager.get_health_status()
        health["timestamp"] = datetime.now().isoformat()
        return health

    return app
