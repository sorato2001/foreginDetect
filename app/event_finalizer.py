"""
Event finalizer module for composing recorded video segments or analyzing FTP images.
"""

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from .analysis.event_analyzer import EventAnalyzer
from .models import ActiveEvent, CameraConfig
from .utils import find_ts_files_in_range, generate_concat_file, get_video_output_path


logger = logging.getLogger(__name__)


class EventFinalizer:
    """Handles video composition or image analysis for finalized events."""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        analysis_mode: str = "video",
        save_annotated_images: bool = False,
        camera_configs: Optional[dict[str, CameraConfig]] = None,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.analysis_mode = analysis_mode
        self.save_annotated_images = save_annotated_images
        self.thread_pool: dict[str, threading.Thread] = {}
        self.camera_configs = camera_configs or {}
        self.event_analyzer = EventAnalyzer()
        logger.info(
            "Initialized EventFinalizer (analysis_mode=%s, save_annotated_images=%s)",
            analysis_mode,
            save_annotated_images,
        )

    @staticmethod
    def _normalize_base_output_dir(output_dir: str, camera_id: str) -> str:
        """Return a base output directory that does not already end with camera_id."""
        path = Path(output_dir)
        if path.name == camera_id:
            return str(path.parent)
        return str(path)

    @staticmethod
    def _resolve_event_status(analysis_result: Optional[dict], default_status: str = "success") -> str:
        """Map analyzer status to the event record status."""
        if not analysis_result:
            return default_status
        analysis_status = analysis_result.get("status")
        if analysis_status in {"success", "partial_success", "error"}:
            return analysis_status
        final_result = analysis_result.get("final_result") or {}
        if final_result:
            return "alarm" if final_result.get("is_alarm") else "reviewed_normal"
        return default_status

    def finalize_event(
        self,
        active_event: ActiveEvent,
        cache_dir: str,
        output_dir: str,
        on_complete: Optional[Callable[[ActiveEvent, Optional[str], str], None]] = None,
    ) -> None:
        """Finalize an event asynchronously based on its media type."""
        thread_key = f"finalize_{active_event.camera_id}_{active_event.alarm_time.timestamp()}"
        existing_thread = self.thread_pool.get(thread_key)
        if existing_thread and existing_thread.is_alive():
            logger.warning("Finalization already running for %s", thread_key)
            return

        if active_event.media_type == "image":
            target = self._finalize_image_sync
            args = (active_event, output_dir, on_complete)
        else:
            target = self._finalize_video_sync
            args = (active_event, cache_dir, output_dir, on_complete)

        thread = threading.Thread(
            target=target,
            args=args,
            daemon=True,
            name=thread_key,
        )
        thread.start()
        self.thread_pool[thread_key] = thread
        logger.info(
            "Started async finalization for %s (media_type=%s)",
            active_event.camera_id,
            active_event.media_type,
        )

    def _finalize_video_sync(
        self,
        active_event: ActiveEvent,
        cache_dir: str,
        output_dir: str,
        on_complete: Optional[Callable[[ActiveEvent, Optional[str], str], None]] = None,
    ) -> None:
        """Finalize a video event by composing TS files and analyzing the result."""
        try:
            output_dir = self._normalize_base_output_dir(output_dir, active_event.camera_id)
            logger.info(
                "Finalizing video event for %s: %s -> %s",
                active_event.camera_id,
                active_event.event_start,
                active_event.record_until,
            )

            ts_files = find_ts_files_in_range(
                cache_dir,
                active_event.event_start,
                active_event.record_until,
            )
            if not ts_files:
                logger.warning(
                    "No ts files found for event %s in range [%s, %s]",
                    active_event.camera_id,
                    active_event.event_start,
                    active_event.record_until,
                )
                if on_complete:
                    on_complete(active_event, None, "no_segments")
                return

            ts_file_paths = [filepath for filepath, _ in ts_files]
            concat_file = os.path.join(
                Path(output_dir).parent,
                f"concat_{active_event.camera_id}_{int(active_event.alarm_time.timestamp())}.txt",
            )

            try:
                generate_concat_file(ts_file_paths, concat_file)
            except Exception as exc:
                logger.error("Failed to generate concat file: %s", exc, exc_info=True)
                if on_complete:
                    on_complete(active_event, None, "concat_failed")
                return

            output_video = get_video_output_path(
                output_dir,
                active_event.camera_id,
                active_event.alarm_time,
            )

            try:
                video_path = self._compose_video(concat_file, output_video)
            finally:
                try:
                    os.remove(concat_file)
                except OSError:
                    pass

            active_event.video_path = video_path

            # Optional event-level railway review on sampled key frames. This runs in
            # the finalization worker thread, never in the RTSP recording loop.
            camera_config = self.camera_configs.get(active_event.camera_id)
            try:
                frame_output_dir = str(Path(output_video).with_suffix("")) + "_frames"
                active_event.analysis_result = self.event_analyzer.analyze_event(
                    video_path=video_path,
                    camera_id=active_event.camera_id,
                    event_time=active_event.alarm_time,
                    camera_config=camera_config,
                    alarm_image_path=active_event.image_path,
                    output_dir=frame_output_dir,
                    previous_result=active_event.analysis_result,
                )
            except Exception as exc:
                logger.error("Railway event analysis failed for %s: %s", active_event.camera_id, exc, exc_info=True)
                active_event.analysis_result = active_event.analysis_result or {
                    "status": "error",
                    "message": str(exc),
                }

            final_status = self._resolve_event_status(active_event.analysis_result)
            if active_event.analysis_result is not None:
                active_event.analysis_result["status"] = final_status
            logger.info(
                "Video railway analysis finished for %s with status=%s",
                active_event.camera_id,
                final_status,
            )

            if on_complete:
                on_complete(active_event, video_path, final_status)
        except subprocess.CalledProcessError:
            logger.error("Failed to compose video for %s", active_event.camera_id, exc_info=True)
            if on_complete:
                on_complete(active_event, None, "compose_failed")
        except Exception as exc:
            logger.error(
                "Unexpected error in video finalization for %s: %s",
                active_event.camera_id,
                exc,
                exc_info=True,
            )
            active_event.analysis_result = {"status": "error", "message": str(exc)}
            if on_complete:
                on_complete(active_event, active_event.video_path, "error")

    def _finalize_image_sync(
        self,
        active_event: ActiveEvent,
        output_dir: str,
        on_complete: Optional[Callable[[ActiveEvent, Optional[str], str], None]] = None,
    ) -> None:
        """Finalize an image event by analyzing the last FTP image in the event window."""
        try:
            output_dir = self._normalize_base_output_dir(output_dir, active_event.camera_id)
            logger.info(
                "Finalizing image event for %s using image %s",
                active_event.camera_id,
                active_event.image_path,
            )
            if not active_event.image_path:
                raise RuntimeError("No FTP image available for image event finalization")

            event_output_dir = str(
                Path(output_dir)
                / active_event.camera_id
                / active_event.alarm_time.strftime("%Y-%m-%d")
            )
            camera_config = self.camera_configs.get(active_event.camera_id)
            if not active_event.analysis_result:
                active_event.analysis_result = self.event_analyzer.analyze_image(
                    image_path=active_event.image_path,
                    camera_id=active_event.camera_id,
                    event_time=active_event.alarm_time,
                    camera_config=camera_config,
                )
            if self.save_annotated_images:
                vis_path = str(Path(event_output_dir) / f"{Path(active_event.image_path).stem}_railway.jpg")
                security_config = getattr(camera_config, "railway_security", {}) if camera_config else {}
                saved = self.event_analyzer.save_visualization(
                    active_event.image_path,
                    active_event.analysis_result,
                    security_config,
                    vis_path,
                )
                active_event.annotated_image_path = saved
                if saved:
                    active_event.analysis_result.setdefault("artifacts", {})["visualization_path"] = saved
            final_status = self._resolve_event_status(active_event.analysis_result)
            if active_event.analysis_result is not None:
                active_event.analysis_result["status"] = final_status
            logger.info(
                "Image analysis finished for %s with status=%s, annotated_image=%s",
                active_event.camera_id,
                final_status,
                active_event.annotated_image_path,
            )

            if on_complete:
                on_complete(active_event, None, final_status)
        except Exception as exc:
            logger.error(
                "Unexpected error in image finalization for %s: %s",
                active_event.camera_id,
                exc,
                exc_info=True,
            )
            active_event.analysis_result = {"status": "error", "message": str(exc)}
            if on_complete:
                on_complete(active_event, None, "error")

    def _compose_video(self, concat_file: str, output_path: str) -> str:
        """Use ffmpeg to compose a final MP4 from TS segments."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-y",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"FFmpeg did not create output file: {output_path}")
        return output_path

    def wait_all(self, timeout: float = 600) -> None:
        """Wait for all finalization threads to complete."""
        logger.info("Waiting for %s finalization threads...", len(self.thread_pool))
        for thread_key, thread in self.thread_pool.items():
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(
                        "Finalization thread %s did not complete within timeout",
                        thread_key,
                    )
        logger.info("All finalization threads completed")
