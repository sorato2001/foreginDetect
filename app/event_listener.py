"""
Event listener module for monitoring FTP alarm image uploads.
Detects new alarm images and generates alarm events.
"""

import os
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable, Set
from collections import defaultdict

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .models import AlarmEvent, EventType, CameraConfig
from .utils import get_file_event_time


logger = logging.getLogger(__name__)


class AlarmImageHandler(FileSystemEventHandler):
    """Handles filesystem events for alarm image uploads."""
    
    def __init__(
        self,
        camera_configs: dict[str, CameraConfig],
        on_alarm: Callable[[AlarmEvent], None],
        cooldown_seconds: float = 1.0,
    ):
        """
        Initialize alarm handler.
        
        Args:
            camera_configs: Dictionary of CameraConfig by camera_id
            on_alarm: Callback function called when alarm is detected
            cooldown_seconds: Minimum interval between alarms for same camera
        """
        self.camera_configs = camera_configs
        self.on_alarm = on_alarm
        self.cooldown_seconds = cooldown_seconds
        
        # Track last alarm time per camera for cooldown
        self.last_alarm_times: dict[str, datetime] = defaultdict(
            lambda: datetime.now() - timedelta(seconds=cooldown_seconds * 2)
        )
        
        # Build mapping from FTP dir to camera_id for quick lookup
        self.ftp_dir_to_camera: dict[str, str] = {}
        self._build_ftp_mapping()
        
        logger.info(f"Initialized AlarmImageHandler with {len(camera_configs)} cameras")
    
    def _build_ftp_mapping(self) -> None:
        """Build mapping from FTP directories to camera IDs."""
        for camera_id, config in self.camera_configs.items():
            ftp_dir = config.ftp_image_dir
            if ftp_dir:
                # Normalize path
                ftp_dir = os.path.abspath(ftp_dir)
                self.ftp_dir_to_camera[ftp_dir] = camera_id
                logger.debug(f"Mapped FTP dir {ftp_dir} -> camera {camera_id}")
    
    def on_created(self, event) -> None:
        """Handle file creation events."""
        if event.is_directory:
            logger.debug(f"Ignoring directory creation: {event.src_path}")
            return
        
        logger.debug(f"File created event: {event.src_path}")
        self._check_alarm_image(event.src_path)
    
    def on_modified(self, event) -> None:
        """Handle file modification events (fallback detection)."""
        if event.is_directory:
            return
        
        logger.debug(f"File modified event: {event.src_path}")
        self._check_alarm_image(event.src_path)
    
    def _check_alarm_image(self, filepath: str) -> None:
        """
        Check if a file is an alarm image for a known camera.
        
        Args:
            filepath: Path to the file
        """
        logger.debug(f"Checking alarm image: {filepath}")
        
        # Check file extension (common image formats)
        if not self._is_image_file(filepath):
            logger.debug(f"Not an image file (wrong extension): {filepath}")
            return
        
        logger.debug(f"File has valid image extension: {filepath}")
        
        # Find which camera this FTP directory belongs to
        camera_id = self._find_camera_by_ftp_path(filepath)
        if not camera_id:
            logger.debug(f"No camera mapped to this FTP path: {filepath}")
            logger.debug(f"Current FTP mappings: {self.ftp_dir_to_camera}")
            return
        
        logger.debug(f"Mapped to camera {camera_id}")
        
        # Check cooldown
        now = datetime.now()
        last_alarm = self.last_alarm_times[camera_id]
        time_since_last = (now - last_alarm).total_seconds()
        
        if time_since_last < self.camera_configs[camera_id].cooldown_seconds:
            logger.debug(f"Skipping alarm for {camera_id} (cooldown active): "
                        f"{time_since_last:.1f}s < {self.camera_configs[camera_id].cooldown_seconds}s")
            return
        
        self.last_alarm_times[camera_id] = now
        
        # Parse event time from filename or file mtime
        try:
            event_time = get_file_event_time(filepath)
        except Exception as e:
            logger.warning(f"Failed to parse event time for {filepath}: {e}")
            event_time = datetime.now()
        
        # Create alarm event
        alarm_event = AlarmEvent(
            camera_id=camera_id,
            event_time=event_time,
            image_path=filepath,
            event_type=self.camera_configs[camera_id].event_type,
        )
        
        logger.info(f"Detected alarm event: camera={camera_id}, "
                   f"time={event_time}, image={os.path.basename(filepath)}")
        
        # Trigger callback
        try:
            self.on_alarm(alarm_event)
        except Exception as e:
            logger.error(f"Error processing alarm event: {e}", exc_info=True)
    
    def _find_camera_by_ftp_path(self, filepath: str) -> Optional[str]:
        """
        Find camera_id for a file based on FTP directory mapping.
        
        Args:
            filepath: Path to the file
            
        Returns:
            camera_id if found, None otherwise
        """
        filepath_abs = os.path.abspath(filepath)
        logger.debug(f"Looking for camera for file (abs path): {filepath_abs}")
        
        # Check for exact or parent directory matches
        for ftp_dir, camera_id in self.ftp_dir_to_camera.items():
            logger.debug(f"  Checking if '{filepath_abs}' starts with '{ftp_dir}'")
            if filepath_abs.startswith(ftp_dir):
                logger.debug(f"  -> Match found! Camera: {camera_id}")
                return camera_id
        
        logger.debug(f"  -> No match found in mappings")
        return None
    
    @staticmethod
    def _is_image_file(filepath: str) -> bool:
        """Check if file is an image."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        ext = os.path.splitext(filepath)[1].lower()
        return ext in image_extensions


class EventListener:
    """Monitors FTP directories for alarm images and triggers alarm events."""
    
    def __init__(self, camera_configs: dict[str, CameraConfig]):
        """
        Initialize event listener.
        
        Args:
            camera_configs: Dictionary of CameraConfig by camera_id
        """
        self.camera_configs = camera_configs
        self.observer: Optional[Observer] = None
        self.handlers: Set[AlarmImageHandler] = set()
        self.on_alarm: Optional[Callable[[AlarmEvent], None]] = None
        
        logger.info("Initialized EventListener")
    
    def register_alarm_callback(self, callback: Callable[[AlarmEvent], None]) -> None:
        """
        Register callback for alarm events.
        
        Args:
            callback: Function to call when alarm is detected
        """
        self.on_alarm = callback
        logger.debug("Registered alarm callback")
    
    def start(self) -> bool:
        """
        Start monitoring FTP directories.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.observer is not None and self.observer.is_alive():
            logger.warning("EventListener already running")
            return True
        
        if not self.on_alarm:
            logger.error("No alarm callback registered")
            return False
        
        try:
            self.observer = Observer()
            
            # Get unique FTP directories to monitor
            ftp_dirs = set()
            for camera_id, config in self.camera_configs.items():
                if config.ftp_image_dir and config.enabled:
                    ftp_dir = os.path.abspath(config.ftp_image_dir)
                    ftp_dirs.add(ftp_dir)
            
            if not ftp_dirs:
                logger.warning("No FTP directories configured or all cameras disabled")
                return False
            
            # Create handler and schedule watches
            handler = AlarmImageHandler(
                self.camera_configs,
                self.on_alarm,
                cooldown_seconds=1.0
            )
            self.handlers.add(handler)
            
            for ftp_dir in ftp_dirs:
                # Ensure directory exists
                Path(ftp_dir).mkdir(parents=True, exist_ok=True)
                self.observer.schedule(handler, ftp_dir, recursive=False)
                logger.info(f"Watching FTP directory: {ftp_dir}")
            
            self.observer.start()
            logger.info(f"EventListener started, monitoring {len(ftp_dirs)} directories")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start EventListener: {e}", exc_info=True)
            return False
    
    def stop(self) -> None:
        """Stop monitoring FTP directories."""
        if self.observer is None:
            return
        
        try:
            logger.info("Stopping EventListener")
            self.observer.stop()
            self.observer.join(timeout=5)
            logger.info("EventListener stopped")
        except Exception as e:
            logger.error(f"Error stopping EventListener: {e}")
    
    def is_running(self) -> bool:
        """Check if listener is active."""
        return self.observer is not None and self.observer.is_alive()
