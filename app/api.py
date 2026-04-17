"""
FastAPI-based REST API for system monitoring and querying.
Provides endpoints for status, event list, and camera status.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


def create_app(event_system) -> FastAPI:
    """
    Create FastAPI application instance.
    
    Args:
        event_system: EventSystem instance
        
    Returns:
        FastAPI app
    """
    app = FastAPI(
        title="Multi-Camera Event Recording API",
        description="API for monitoring and querying multi-camera event system",
        version="1.0.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store reference to system
    app.event_system = event_system
    
    # Health check
    @app.get("/health")
    async def health() -> dict:
        """
        Health check endpoint.
        
        Returns:
            Health status
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }
    
    # System status
    @app.get("/status")
    async def system_status() -> dict:
        """
        Get overall system status.
        
        Returns:
            System status including recorders, scheduler, and listener
        """
        status = app.event_system.get_status()
        status["timestamp"] = datetime.now().isoformat()
        return status
    
    # List all cameras
    @app.get("/cameras")
    async def list_cameras() -> dict:
        """
        List all configured cameras.
        
        Returns:
            Dictionary with camera configurations and current status
        """
        cameras = []
        
        for camera_id, config in app.event_system.config.cameras.items():
            recorder = app.event_system.recorder_manager.get_recorder(camera_id)
            active_event = app.event_system.scheduler.get_active_event(camera_id)
            
            camera_info = {
                "camera_id": camera_id,
                "name": config.name,
                "ip": config.ip,
                "enabled": config.enabled,
                "analysis_mode": app.event_system.config.analysis_mode,
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
                }
            
            cameras.append(camera_info)
        
        return {
            "total": len(cameras),
            "cameras": cameras,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Get camera details
    @app.get("/cameras/{camera_id}")
    async def get_camera(camera_id: str) -> dict:
        """
        Get detailed information for a specific camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Camera information and status
        """
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
            }
        
        return camera_info
    
    # List events
    @app.get("/events")
    async def list_events(
        camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
        limit: int = Query(100, ge=1, le=1000, description="Number of events to return"),
    ) -> dict:
        """
        List recorded events.
        
        Args:
            camera_id: Optional camera ID filter
            limit: Maximum number of events to return
            
        Returns:
            List of event records
        """
        if camera_id:
            events = app.event_system.db.get_events_by_camera(camera_id, limit=limit)
        else:
            events = app.event_system.db.get_all_events(limit=limit)
        
        return {
            "total": len(events),
            "events": [
                {
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
                    "analysis_result": event.analysis_result,
                    "status": event.status,
                    "alarm_count": event.alarm_count,
                    "created_at": event.created_at.isoformat(),
                }
                for event in events
            ],
            "timestamp": datetime.now().isoformat(),
        }
    
    # Get event details
    @app.get("/events/{event_id}")
    async def get_event(event_id: int) -> dict:
        """
        Get details for a specific event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Event details
        """
        event = app.event_system.db.get_event(event_id)
        
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        
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
            "analysis_result": event.analysis_result,
            "status": event.status,
            "alarm_count": event.alarm_count,
            "created_at": event.created_at.isoformat(),
            "updated_at": event.updated_at.isoformat(),
        }
    
    # Get active events
    @app.get("/events/active/all")
    async def get_active_events() -> dict:
        """
        Get all currently active recording events.
        
        Returns:
            List of active events
        """
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
                }
                for event in active.values()
            ],
            "timestamp": datetime.now().isoformat(),
        }
    
    # Recorder health
    @app.get("/health/recorders")
    async def recorder_health() -> dict:
        """
        Get health status of all recorders.
        
        Returns:
            Recorder health information
        """
        health = app.event_system.recorder_manager.get_health_status()
        health["timestamp"] = datetime.now().isoformat()
        return health
    
    return app
