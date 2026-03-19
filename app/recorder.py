"""
Recorder module for continuous RTSP stream capture and caching.
Maintains circular buffer of recent segments for each camera.
"""

import os
import logging
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from .models import CameraConfig, EventStatus
from .utils import cleanup_cache_files


logger = logging.getLogger(__name__)


class CameraRecorder:
    """Handles RTSP streaming and segment caching for a single camera."""
    
    def __init__(self, camera_config: CameraConfig, cache_dir: str, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize recorder for a camera.
        
        Args:
            camera_config: CameraConfig object
            cache_dir: Directory to store .ts segment files
            ffmpeg_path: Path to ffmpeg executable
        """
        self.config = camera_config
        self.cache_dir = os.path.join(cache_dir, camera_config.camera_id)
        self.ffmpeg_path = ffmpeg_path
        
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self.start_time: Optional[datetime] = None
        self.restart_count = 0
        self.last_error: Optional[str] = None
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized recorder for camera {camera_config.camera_id} "
                   f"(cache: {self.cache_dir})")
    
    def start(self) -> bool:
        """
        Start RTSP streaming and recording.
        
        Returns:
            True if stream started successfully, False otherwise
        """
        if self.running:
            logger.warning(f"Recorder already running for {self.config.camera_id}")
            return True
        
        if not self.config.enabled:
            logger.info(f"Camera {self.config.camera_id} is disabled, skipping start")
            return False
        
        try:
            # Construct ffmpeg command
            cmd = self._build_ffmpeg_command()
            
            logger.info(f"Starting recording for {self.config.camera_id}: {' '.join(cmd)}")
            
            # Start ffmpeg process - capture stderr to see errors
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout for unified logging
                universal_newlines=True,
                bufsize=1  # Line buffering
            )
            
            self.running = True
            self.start_time = datetime.now()
            
            # Start thread to monitor process
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                daemon=True,
                name=f"recorder_monitor_{self.config.camera_id}"
            )
            monitor_thread.start()
            
            logger.info(f"Recording started for camera {self.config.camera_id} "
                       f"(PID: {self.process.pid})")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to start recording for {self.config.camera_id}: {e}")
            return False
    
    def stop(self) -> None:
        """Stop RTSP streaming and recording."""
        if not self.running or self.process is None:
            return
        
        try:
            logger.info(f"Stopping recording for {self.config.camera_id}")
            self.process.terminate()
            
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing ffmpeg process for {self.config.camera_id}")
                self.process.kill()
                self.process.wait(timeout=5)
            
            self.running = False
            logger.info(f"Recording stopped for {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"Error stopping recorder for {self.config.camera_id}: {e}")
    
    def cleanup_old_segments(self) -> int:
        """
        Remove cache files older than cache_seconds.
        
        Returns:
            Number of files deleted
        """
        return cleanup_cache_files(self.cache_dir, self.config.cache_seconds)
    
    def _build_ffmpeg_command(self) -> list:
        """
        Build ffmpeg command for RTSP capture.
        
        Uses segment muxer to create 1-second .ts files.
        
        Returns:
            List of command arguments
        """
        # Output pattern: /path/to/cache/%s.ts (number them sequentially 0001, 0002, etc)
        # Normalize to forward slashes for cross-platform FFmpeg compatibility
        output_pattern = os.path.join(self.cache_dir, "%04d.ts").replace("\\", "/")
        
        cmd = [
            self.ffmpeg_path,
            "-rtsp_transport", "tcp",
            "-i", self.config.rtsp_url,
            "-c", "copy",                    # No re-encoding
            "-f", "segment",
            "-segment_time", "1",            # 1 second per segment
            "-segment_format", "mpegts",
            "-y",                             # Overwrite output files
            output_pattern
        ]
        
        return cmd
    
    def _monitor_process(self) -> None:
        """Monitor ffmpeg process and restart if it exits unexpectedly."""
        if self.process is None:
            return
        
        try:
            # Read FFmpeg output (stdout + stderr combined)
            if self.process.stdout:
                for line in self.process.stdout:
                    if line.strip():
                        logger.debug(f"[{self.config.camera_id}] {line.rstrip()}")
            
            # Wait for process to finish
            returncode = self.process.wait()
            
            if returncode != 0 and returncode != -15:  # -15 is SIGTERM
                logger.error(f"Recorder process exited with code {returncode} "
                           f"for {self.config.camera_id}")
                self.last_error = f"Process exited with code {returncode}"
            
            self.running = False
            
            if self.restart_count < 5:  # Limit restart attempts
                logger.warning(f"Restarting recorder for {self.config.camera_id} "
                             f"(attempt {self.restart_count + 1})")
                self.restart_count += 1
                time.sleep(2)  # Wait before restart
                self.start()
            else:
                logger.error(f"Too many restart attempts for {self.config.camera_id}, giving up")
        
        except Exception as e:
            logger.error(f"Error monitoring recorder process for {self.config.camera_id}: {e}")
            self.running = False
    
    def is_healthy(self) -> bool:
        """
        Check if recorder is healthy.
        
        Returns:
            True if running and process is alive
        """
        if not self.running or self.process is None:
            return False
        
        return self.process.poll() is None


class RecorderManager:
    """Manages multiple CameraRecorder instances."""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize recorder manager.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path
        self.recorders: Dict[str, CameraRecorder] = {}
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_running = False
        
        logger.info("Initialized RecorderManager")
    
    def add_camera(self, camera_config: CameraConfig, cache_dir: str) -> CameraRecorder:
        """
        Add a camera to management.
        
        Args:
            camera_config: CameraConfig object
            cache_dir: Base cache directory
            
        Returns:
            CameraRecorder instance
        """
        if camera_config.camera_id in self.recorders:
            logger.warning(f"Camera {camera_config.camera_id} already exists")
            return self.recorders[camera_config.camera_id]
        
        recorder = CameraRecorder(camera_config, cache_dir, self.ffmpeg_path)
        self.recorders[camera_config.camera_id] = recorder
        
        return recorder
    
    def start_all(self) -> int:
        """
        Start recording for all cameras.
        
        Returns:
            Number of successfully started recorders
        """
        started_count = 0
        
        for camera_id, recorder in self.recorders.items():
            if recorder.start():
                started_count += 1
        
        # Start cleanup thread
        if not self.cleanup_running:
            self.cleanup_running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="recorder_cleanup"
            )
            self.cleanup_thread.start()
        
        logger.info(f"Started {started_count}/{len(self.recorders)} recorders")
        return started_count
    
    def stop_all(self) -> None:
        """Stop all recorders."""
        self.cleanup_running = False
        
        for camera_id, recorder in self.recorders.items():
            recorder.stop()
        
        logger.info(f"Stopped all {len(self.recorders)} recorders")
    
    def get_recorder(self, camera_id: str) -> Optional[CameraRecorder]:
        """Get recorder for a camera."""
        return self.recorders.get(camera_id)
    
    def get_health_status(self) -> dict:
        """
        Get health status of all recorders.
        
        Returns:
            Dictionary with health info
        """
        status = {
            "total": len(self.recorders),
            "healthy": 0,
            "unhealthy": [],
        }
        
        for camera_id, recorder in self.recorders.items():
            if recorder.is_healthy():
                status["healthy"] += 1
            else:
                status["unhealthy"].append({
                    "camera_id": camera_id,
                    "running": recorder.running,
                    "last_error": recorder.last_error,
                    "restart_count": recorder.restart_count,
                })
        
        return status
    
    def _cleanup_loop(self) -> None:
        """Background loop for cleaning up old cache files."""
        logger.info("Starting cleanup loop")
        
        while self.cleanup_running:
            try:
                for camera_id, recorder in self.recorders.items():
                    recorder.cleanup_old_segments()
                
                # Run cleanup every 30 seconds
                time.sleep(30)
            
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(5)
        
        logger.info("Cleanup loop stopped")
