"""
Main application entry point.
Orchestrates all system components: recorders, listeners, scheduler, and API.
"""

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .config import get_config
from .database import EventDatabase
from .recorder import RecorderManager
from .event_listener import EventListener
from .scheduler import EventScheduler, SchedulerLoop
from .event_finalizer import EventFinalizer


# Configure logging
def setup_logging(log_dir: str, log_level: str = "INFO") -> None:
    """
    Configure Python logging.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / "event_system.log"
    
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured (level={log_level}, file={log_file})")


class EventSystem:
    """Main application system."""
    
    def __init__(
        self,
        config_path: str,
        log_level: Optional[str] = None,
        enable_api: Optional[bool] = None,
        analysis_mode: Optional[str] = None,
    ):
        """Initialize event system.
        
        Args:
            config_path: Path to YAML configuration file
            log_level: Optional log level override
            enable_api: Optional flag to enable/disable API
        """
        # Load configuration
        self.config = get_config(config_path)

        if log_level:
            self.config.log_level = log_level
        if enable_api is not None:
            self.config.enable_api = enable_api
        if analysis_mode:
            self.config.analysis_mode = analysis_mode

        # Setup logging
        setup_logging(self.config.log_dir, self.config.log_level)
        self.logger = logging.getLogger(__name__)

        self.logger.info("=" * 60)
        self.logger.info("Multi-Camera Event Recording System Starting")
        self.logger.info("=" * 60)
        
        # Initialize components
        self.db = EventDatabase(self.config.db_path)
        self.recorder_manager = RecorderManager(self.config.ffmpeg_path)
        self.event_listener = EventListener(self.config.cameras)
        self.event_finalizer = EventFinalizer(
            self.config.ffmpeg_path,
            analysis_mode=self.config.analysis_mode,
            save_annotated_images=self.config.save_annotated_images,
        )
        self.scheduler = EventScheduler(
            self.db,
            self.event_finalizer,
            self.config.base_cache_dir,
            self.config.base_video_dir,
            analysis_mode=self.config.analysis_mode,
        )
        self.scheduler_loop = SchedulerLoop(
            self.scheduler,
            self.config.cameras,
            self.config.scheduler_interval_sec,
        )
        
        # API server (optional)
        self.api_server = None
    
    def start(self) -> None:
        """Start all system components."""
        try:
            if self.config.analysis_mode == "video":
                self._start_recorders()
            else:
                self.logger.info("Image analysis mode enabled, skipping RTSP recorders")
            
            # Start event listener
            self._start_event_listener()
            
            # Start scheduler loop
            self._start_scheduler_loop()
            
            # Start API server (optional)
            if self.config.enable_api:
                self._start_api_server()
            
            self.logger.info("=" * 60)
            self.logger.info(f"System started successfully!")
            self.logger.info(f"Cameras: {len(self.config.cameras)}")
            self.logger.info(f"Analysis mode: {self.config.analysis_mode}")
            self.logger.info(f"Cache dir: {self.config.base_cache_dir}")
            self.logger.info(f"Video output dir: {self.config.base_video_dir}")
            self.logger.info(f"Database: {self.config.db_path}")
            if self.config.enable_api:
                self.logger.info(f"API: http://{self.config.api_host}:{self.config.api_port}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}", exc_info=True)
            self.stop()
            raise
    
    def stop(self) -> None:
        """Stop all system components."""
        self.logger.info("=" * 60)
        self.logger.info("Shutting down system...")
        self.logger.info("=" * 60)
        
        try:
            if self.api_server:
                self.logger.info("Stopping API server...")
                # API shutdown is handled by uvicorn
            
            self.logger.info("Stopping scheduler loop...")
            self.scheduler_loop.stop()
            
            self.logger.info("Waiting for event finalization...")
            self.event_finalizer.wait_all(timeout=60)
            
            self.logger.info("Stopping event listener...")
            self.event_listener.stop()
            
            self.logger.info("Stopping recorders...")
            self.recorder_manager.stop_all()
            
            self.logger.info("=" * 60)
            self.logger.info("System shutdown complete")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def _start_recorders(self) -> None:
        """Start all camera recorders."""
        self.logger.info(f"Starting recorders for {len(self.config.cameras)} cameras...")
        
        # Add all cameras to recorder manager
        for camera_id, camera_config in self.config.cameras.items():
            self.recorder_manager.add_camera(camera_config, self.config.base_cache_dir)
        
        # Start all recorders
        started = self.recorder_manager.start_all()
        self.logger.info(f"Started {started} recorders")
    
    def _start_event_listener(self) -> None:
        """Start FTP event listener."""
        self.logger.info("Starting event listener...")
        
        # Set alarm callback
        self.event_listener.register_alarm_callback(self._on_alarm_event)
        
        if not self.event_listener.start():
            raise RuntimeError("Failed to start event listener")
        
        self.logger.info("Event listener started")
    
    def _start_scheduler_loop(self) -> None:
        """Start scheduler background loop."""
        self.logger.info("Starting scheduler loop...")
        
        # Register callbacks
        self.scheduler.register_callbacks(
            on_created=self._on_event_created,
            on_extended=self._on_event_extended,
            on_finalized=self._on_event_finalized,
        )
        
        if not self.scheduler_loop.start():
            raise RuntimeError("Failed to start scheduler loop")
        
        self.logger.info("Scheduler loop started")
    
    def _start_api_server(self) -> None:
        """Start FastAPI server (if enabled)."""
        self.logger.info(f"Starting API server on {self.config.api_host}:{self.config.api_port}...")
        
        try:
            import uvicorn
            from .api import create_app
            
            app = create_app(self)
            
            # Run in background
            import threading
            config = uvicorn.Config(
                app,
                host=self.config.api_host,
                port=self.config.api_port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            
            self.api_server = server
            self.logger.info("API server started")
            
        except ImportError:
            self.logger.warning("FastAPI/uvicorn not available, skipping API server")
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
    
    def _on_alarm_event(self, alarm_event) -> None:
        """Handle alarm event from listener."""
        camera_id = alarm_event.camera_id
        camera_config = self.config.cameras.get(camera_id)
        
        if not camera_config:
            self.logger.warning(f"Received alarm for unknown camera: {camera_id}")
            return
        
        self.logger.info(f"Alarm event: camera={camera_id}, time={alarm_event.event_time}")
        
        # Process through scheduler
        self.scheduler.handle_alarm(alarm_event, camera_config)
    
    def _on_event_created(self, active_event) -> None:
        """Callback when new event is created."""
        self.logger.info(f"Event created: {active_event.camera_id} "
                        f"({active_event.event_start} -> {active_event.record_until})")
    
    def _on_event_extended(self, active_event) -> None:
        """Callback when event is extended."""
        self.logger.info(f"Event extended: {active_event.camera_id} "
                        f"record_until={active_event.record_until}")
    
    def _on_event_finalized(self, event_record) -> None:
        """Callback when event is finalized."""
        self.logger.info(f"Event finalized: {event_record.camera_id} "
                        f"media={event_record.media_type} "
                        f"video={event_record.video_path} status={event_record.status}")
        if event_record.annotated_image_path:
            self.logger.info(f"Annotated image: {event_record.annotated_image_path}")
        if event_record.analysis_result:
            analysis_status = event_record.analysis_result.get("status")
            self.logger.info(f"Analysis result status: {analysis_status}")
            vlm_analysis = event_record.analysis_result.get("vlm_analysis") or {}
            vlm_answer = vlm_analysis.get("answer")
            if vlm_answer:
                self.logger.info(f"VLM result: {vlm_answer[:300]}")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def get_status(self) -> dict:
        """Get system status."""
        return {
            "analysis_mode": self.config.analysis_mode,
            "recorder_health": self.recorder_manager.get_health_status(),
            "scheduler": self.scheduler.get_status(),
            "event_listener_running": self.event_listener.is_running(),
        }


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point.

    Args:
        argv: Optional list of command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Camera Event Recording System"
    )
    parser.add_argument(
        "-c", "--config",
        default="configs/cameras.yaml",
        help="Path to configuration file (default: configs/cameras.yaml)",
    )
    parser.add_argument(
        "-l", "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level override (optional)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable REST API server",
    )
    parser.add_argument(
        "--analysis-mode",
        default=None,
        choices=["video", "image"],
        help="Select analysis mode (overrides YAML config)",
    )

    args = parser.parse_args(argv)

    system = EventSystem(
        config_path=args.config,
        log_level=args.log_level,
        enable_api=not args.no_api,
        analysis_mode=args.analysis_mode,
    )

    stop_event = threading.Event()

    def _signal_handler(signum, frame) -> None:
        system.logger.info(f"Received signal {signum}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        system.start()

        while not stop_event.is_set():
            time.sleep(0.5)

    except Exception as e:
        system.logger.error(f"Fatal error: {e}", exc_info=True)
        system.stop()
        return 1

    finally:
        system.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
