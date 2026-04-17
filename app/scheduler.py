"""
Scheduler module for managing event lifecycle and state machine.
Coordinates alarm detection, recording, and finalization.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

from .models import AlarmEvent, ActiveEvent, CameraConfig, EventRecord, EventStatus
from .database import EventDatabase
from .event_finalizer import EventFinalizer


logger = logging.getLogger(__name__)


class EventScheduler:
    """
    Orchestrates event recording state machine.
    Manages active events, handles alarm triggers, and coordinates finalization.
    """
    
    def __init__(
        self,
        db: EventDatabase,
        finalizer: EventFinalizer,
        cache_dir: str,
        output_dir: str,
        analysis_mode: str = "video",
    ):
        """
        Initialize event scheduler.
        
        Args:
            db: EventDatabase instance
            finalizer: EventFinalizer instance
            cache_dir: Base cache directory
            output_dir: Base output video directory
        """
        self.db = db
        self.finalizer = finalizer
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.analysis_mode = analysis_mode
        
        # Active events per camera: camera_id -> ActiveEvent
        self.active_events: Dict[str, ActiveEvent] = {}
        self.lock = threading.RLock()
        
        # Callbacks
        self.on_event_created: Optional[Callable[[ActiveEvent], None]] = None
        self.on_event_extended: Optional[Callable[[ActiveEvent], None]] = None
        self.on_event_finalized: Optional[Callable[[EventRecord], None]] = None
        
        logger.info("Initialized EventScheduler")
    
    def register_callbacks(
        self,
        on_created: Optional[Callable] = None,
        on_extended: Optional[Callable] = None,
        on_finalized: Optional[Callable] = None,
    ) -> None:
        """
        Register callbacks for event lifecycle.
        
        Args:
            on_created: Called when new event is created
            on_extended: Called when event is extended
            on_finalized: Called when event is finalized
        """
        if on_created:
            self.on_event_created = on_created
        if on_extended:
            self.on_event_extended = on_extended
        if on_finalized:
            self.on_event_finalized = on_finalized
    
    def handle_alarm(self, alarm_event: AlarmEvent, camera_config: CameraConfig) -> None:
        """
        Process an alarm event according to state machine rules.
        
        State transitions:
        - IDLE: Create new event
        - RECORDING: Extend record_until if within limits
        
        Args:
            alarm_event: AlarmEvent to process
            camera_config: CameraConfig for the camera
        """
        camera_id = alarm_event.camera_id
        
        with self.lock:
            current_event = self.active_events.get(camera_id)
            
            if current_event is None or current_event.status == EventStatus.IDLE:
                # Create new event
                self._create_event(alarm_event, camera_config)
            
            elif current_event.status == EventStatus.RECORDING:
                # Extend existing event
                self._extend_event(current_event, alarm_event, camera_config)
    
    def _create_event(self, alarm_event: AlarmEvent, camera_config: CameraConfig) -> None:
        """
        Create a new recording event.
        
        Args:
            alarm_event: Triggering alarm event
            camera_config: Camera configuration
        """
        camera_id = alarm_event.camera_id
        event_start = alarm_event.event_time - timedelta(seconds=camera_config.pre_seconds)
        record_until = alarm_event.event_time + timedelta(seconds=camera_config.post_seconds)
        
        active_event = ActiveEvent(
            camera_id=camera_id,
            alarm_time=alarm_event.event_time,
            event_start=event_start,
            record_until=record_until,
            media_type=self.analysis_mode,
            event_type=alarm_event.event_type,
            status=EventStatus.RECORDING,
            alarm_count=1,
            last_alarm_time=alarm_event.event_time,
            image_path=alarm_event.image_path,
        )
        
        self.active_events[camera_id] = active_event
        
        logger.info(f"Created event for {camera_id}: "
                   f"start={event_start}, record_until={record_until}")
        
        if self.on_event_created:
            self.on_event_created(active_event)
    
    def _extend_event(
        self,
        active_event: ActiveEvent,
        alarm_event: AlarmEvent,
        camera_config: CameraConfig,
    ) -> None:
        """
        Extend an existing recording event.
        
        Args:
            active_event: Current active event
            alarm_event: Triggering alarm event
            camera_config: Camera configuration
        """
        camera_id = alarm_event.camera_id
        
        # Calculate new record_until
        new_record_until = alarm_event.event_time + timedelta(
            seconds=camera_config.post_seconds
        )
        
        # Check max extension limit
        max_allowed = active_event.alarm_time + timedelta(
            seconds=camera_config.max_extend_seconds
        )
        
        if new_record_until > max_allowed:
            new_record_until = max_allowed
            logger.info(f"Event extension limited by max_extend_seconds for {camera_id}")
        
        active_event.alarm_count += 1
        active_event.last_alarm_time = alarm_event.event_time
        active_event.image_path = alarm_event.image_path

        # Only update if extending
        if new_record_until > active_event.record_until:
            active_event.record_until = new_record_until
            
            logger.info(f"Extended event for {camera_id}: "
                       f"record_until={new_record_until} "
                       f"(alarm_count={active_event.alarm_count})")
            
            if self.on_event_extended:
                self.on_event_extended(active_event)
    
    def check_finalization(self, camera_configs: dict[str, CameraConfig]) -> None:
        """
        Check for events that should be finalized.
        
        Called periodically by main loop. Marks RECORDING events as FINALIZING
        and triggers async finalization.
        
        Args:
            camera_configs: Dictionary of camera configurations
        """
        now = datetime.now()
        cameras_to_finalize = []
        
        with self.lock:
            for camera_id, active_event in list(self.active_events.items()):
                
                # Skip if not recording or already finalizing
                if active_event.status != EventStatus.RECORDING:
                    continue
                
                # Check if recording window has ended
                if now < active_event.record_until:
                    continue
                
                # Time to finalize
                cameras_to_finalize.append((camera_id, active_event))
                active_event.status = EventStatus.FINALIZING
        
        # Trigger finalization for each event (outside lock to avoid deadlock)
        for camera_id, active_event in cameras_to_finalize:
            logger.info(f"Initiating finalization for {camera_id}")
            
            camera_cache_dir = f"{self.cache_dir}/{camera_id}"
            
            # Determine output directory
            camera_config = camera_configs.get(camera_id)
            output_base = self.output_dir
            if camera_config and camera_config.output_dir:
                output_base = camera_config.output_dir
            
            self.finalizer.finalize_event(
                active_event,
                camera_cache_dir,
                output_base,
                on_complete=self._on_finalization_complete,
            )
    
    def _on_finalization_complete(
        self,
        active_event: ActiveEvent,
        video_path: Optional[str],
        status: str,
    ) -> None:
        """
        Callback when event finalization completes.
        
        Args:
            active_event: The finalized event
            video_path: Path to generated video (or None if failed)
            status: Status string: 'success', 'no_segments', 'compose_failed', etc.
        """
        camera_id = active_event.camera_id
        
        with self.lock:
            # Create database record
            event_record = EventRecord(
                camera_id=camera_id,
                media_type=active_event.media_type,
                event_type=active_event.event_type.value,
                event_time=active_event.alarm_time,
                event_start_time=active_event.event_start,
                event_end_time=active_event.record_until,
                image_path=active_event.image_path,
                video_path=video_path,
                annotated_image_path=active_event.annotated_image_path,
                analysis_result=active_event.analysis_result,
                status=status,
                alarm_count=active_event.alarm_count,
            )
            
            try:
                event_id = self.db.add_event(event_record)
                logger.info(f"Event record created: id={event_id}, "
                           f"camera={camera_id}, status={status}")
            except Exception as e:
                logger.error(f"Failed to record event to database: {e}")
            
            # Reset to idle
            if camera_id in self.active_events:
                del self.active_events[camera_id]
            
            if self.on_event_finalized:
                self.on_event_finalized(event_record)
    
    def get_active_event(self, camera_id: str) -> Optional[ActiveEvent]:
        """Get current active event for a camera."""
        with self.lock:
            return self.active_events.get(camera_id)
    
    def get_all_active_events(self) -> dict[str, ActiveEvent]:
        """Get all active events."""
        with self.lock:
            return dict(self.active_events)
    
    def get_status(self) -> dict:
        """
        Get scheduler status.
        
        Returns:
            Dictionary with status info
        """
        with self.lock:
            status = {
                "active_events": len(self.active_events),
                "events": [
                    {
                        "camera_id": event.camera_id,
                        "status": event.status,
                        "media_type": event.media_type,
                        "alarm_time": event.alarm_time.isoformat(),
                        "record_until": event.record_until.isoformat(),
                        "alarm_count": event.alarm_count,
                    }
                    for event in self.active_events.values()
                ]
            }
        
        return status


class SchedulerLoop:
    """Background loop for event finalization checks."""
    
    def __init__(
        self,
        scheduler: EventScheduler,
        camera_configs: dict[str, CameraConfig],
        check_interval: float = 1.0,
    ):
        """
        Initialize scheduler loop.
        
        Args:
            scheduler: EventScheduler instance
            camera_configs: Camera configurations
            check_interval: How often to check for finalization (seconds)
        """
        self.scheduler = scheduler
        self.camera_configs = camera_configs
        self.check_interval = check_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        logger.info("Initialized SchedulerLoop")
    
    def start(self) -> bool:
        """
        Start background scheduler loop.
        
        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("SchedulerLoop already running")
            return True
        
        self.running = True
        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="scheduler_loop",
        )
        self.thread.start()
        
        logger.info(f"Started SchedulerLoop (check_interval={self.check_interval}s)")
        return True
    
    def stop(self) -> None:
        """Stop background scheduler loop."""
        if not self.running:
            return
        
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Stopped SchedulerLoop")
    
    def _loop(self) -> None:
        """Main loop for scheduler."""
        logger.info("SchedulerLoop started")
        
        while self.running:
            try:
                self.scheduler.check_finalization(self.camera_configs)
                time.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                time.sleep(1)
        
        logger.info("SchedulerLoop stopped")
