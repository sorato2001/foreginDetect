"""
Data models for the multi-camera event recording system.
"""

from enum import Enum
from typing import Any, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event types supported by the system."""
    LINE_CROSSING = "line_crossing"
    INTRUSION = "intrusion"
    LOITERING = "loitering"


class EventStatus(str, Enum):
    """Status of a recording event."""
    IDLE = "idle"              # No active event
    RECORDING = "recording"    # Event is recording, waiting for post-seconds
    FINALIZING = "finalizing"  # Event is being finalized, generating video


class CameraConfig(BaseModel):
    """Configuration for a single camera."""
    camera_id: str = Field(..., description="Unique camera identifier")
    name: str = Field(..., description="Human-readable camera name")
    ip: str = Field(..., description="Camera IP address")
    rtsp_url: str = Field(..., description="RTSP stream URL")
    ftp_image_dir: str = Field(..., description="Local directory where FTP uploads images")
    event_type: EventType = Field(default=EventType.LINE_CROSSING, description="Type of alarm event")
    pre_seconds: int = Field(default=5, description="Seconds to record before alarm")
    post_seconds: int = Field(default=10, description="Seconds to record after alarm")
    cache_seconds: int = Field(default=120, description="Total seconds of cache to maintain")
    cooldown_seconds: int = Field(default=3, description="Minimum interval between repeated alarms")
    max_extend_seconds: int = Field(default=60, description="Max extension for continued recording")
    output_dir: str = Field(..., description="Directory to save event videos")
    enabled: bool = Field(default=True, description="Whether this camera is enabled")
    railway_security: dict[str, Any] = Field(default_factory=dict, description="Railway perimeter analysis settings")


class AlarmEvent(BaseModel):
    """Alarm event received from FTP image."""
    camera_id: str = Field(..., description="Camera identifier")
    event_time: datetime = Field(..., description="Time of alarm event")
    image_path: str = Field(..., description="Path to alarm image file")
    event_type: EventType = Field(default=EventType.LINE_CROSSING, description="Type of event")


class ActiveEvent(BaseModel):
    """Active recording event for a camera."""
    camera_id: str
    alarm_time: datetime           # Time of first alarm
    event_start: datetime          # Start time for recording
    record_until: datetime         # Time to stop recording
    media_type: Literal["video", "image"] = "video"
    event_type: EventType = EventType.LINE_CROSSING  # Type of event
    status: EventStatus = EventStatus.RECORDING
    alarm_count: int = 1           # Count of alarms in this event
    last_alarm_time: Optional[datetime] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    analysis_result: Optional[dict] = None  # YOLO and VLM analysis results


class EventRecord(BaseModel):
    """Database record for a finalized event."""
    id: Optional[int] = None
    camera_id: str
    media_type: Literal["video", "image"] = "video"
    event_type: str
    event_time: datetime
    event_start_time: datetime
    event_end_time: datetime
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    analysis_result: Optional[dict] = None
    status: str = "completed"
    alarm_count: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SystemConfig(BaseModel):
    """Overall system configuration."""
    cameras: dict[str, CameraConfig] = Field(default_factory=dict)
    analysis_mode: Literal["video", "image"] = "video"
    save_annotated_images: bool = False
    base_cache_dir: str = "./data/cache"
    base_video_dir: str = "./data/video"
    db_path: str = "./data/db/events.db"
    log_dir: str = "./data/logs"
    log_level: str = "INFO"
    ffmpeg_path: str = "ffmpeg"
    scheduler_interval_sec: float = 1.0  # Check for finalization events every N seconds
    file_cleanup_interval_sec: float = 30.0  # Run cache cleanup every N seconds
    enable_api: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


