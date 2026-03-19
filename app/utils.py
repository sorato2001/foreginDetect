"""
Utility functions for the event recording system.
"""

import re
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List


logger = logging.getLogger(__name__)


def parse_event_time_from_filename(filename: str) -> Optional[datetime]:
    """
    Try to parse event time from filename.
    
    This function attempts to extract a timestamp from the filename.
    Common formats:
    - Unix timestamp: 1710717130
    - ISO format: 20240318_143530
    - Datetime format: 2024-03-18-14-35-30
    
    Args:
        filename: Filename to parse
        
    Returns:
        datetime object if parsing succeeds, None otherwise
    """
    basename = Path(filename).stem  # Remove extension
    
    # Try Unix timestamp (10 digits)
    match = re.search(r'\b(\d{10})\b', basename)
    if match:
        try:
            timestamp = int(match.group(1))
            # Sanity check: should be in reasonable range (2020-2050)
            if 1577836800 <= timestamp <= 2524608000:
                return datetime.fromtimestamp(timestamp)
        except (ValueError, OSError):
            pass
    
    # Try ISO format YYYYMMDD_HHMMSS
    match = re.search(r'(\d{8})_(\d{6})', basename)
    if match:
        try:
            date_str = match.group(1)
            time_str = match.group(2)
            datetime_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    
    # Try format YYYY-MM-DD-HH-MM-SS
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', basename)
    if match:
        try:
            datetime_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)}"
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    
    return None


def get_file_event_time(filepath: str) -> datetime:
    """
    Get event time from file, preferring parsed filename over modification time.
    
    Args:
        filepath: Path to the file
        
    Returns:
        datetime object from filename parsing or file modification time
    """
    # Try to parse from filename
    event_time = parse_event_time_from_filename(filepath)
    if event_time:
        logger.debug(f"Parsed event time from filename: {filepath} -> {event_time}")
        return event_time
    
    # Fall back to file modification time
    try:
        mtime = os.path.getmtime(filepath)
        event_time = datetime.fromtimestamp(mtime)
        logger.debug(f"Using file mtime as event time: {filepath} -> {event_time}")
        return event_time
    except OSError as e:
        logger.warning(f"Failed to get file mtime: {filepath}: {e}")
        return datetime.now()


def find_ts_files_in_range(
    cache_dir: str,
    start_time: datetime,
    end_time: datetime
) -> List[tuple[str, float]]:
    """
    Find all .ts files in cache directory within time range.
    
    Uses file modification time for matching since files are numbered sequentially.
    
    Args:
        cache_dir: Cache directory containing .ts files
        start_time: Start of time range
        end_time: End of time range
        
    Returns:
        List of (filepath, mtime_timestamp) tuples, sorted by file modification time
    """
    ts_files = []
    
    if not os.path.isdir(cache_dir):
        logger.warning(f"Cache directory not found: {cache_dir}")
        return ts_files
    
    start_ts = start_time.timestamp()
    end_ts = end_time.timestamp()
    
    try:
        for filename in os.listdir(cache_dir):
            if not filename.endswith('.ts'):
                continue
            
            filepath = os.path.join(cache_dir, filename)
            
            try:
                # Use file modification time for matching
                file_mtime = os.path.getmtime(filepath)
                
                # Check if within range (with 2 second tolerance)
                if start_ts - 2 <= file_mtime <= end_ts + 2:
                    ts_files.append((filepath, file_mtime))
            except (ValueError, OSError):
                # Skip files that can't be accessed
                continue
    
    except OSError as e:
        logger.error(f"Error reading cache directory: {cache_dir}: {e}")
        return ts_files
    
    # Sort by modification time
    ts_files.sort(key=lambda x: x[1])
    
    logger.debug(f"Found {len(ts_files)} ts files in range "
                f"[{start_time}, {end_time}]")
    
    return ts_files


def generate_concat_file(
    ts_files: List[str],
    concat_file_path: str
) -> None:
    """
    Generate FFmpeg concat demuxer file.
    
    Args:
        ts_files: List of .ts file paths
        concat_file_path: Path to write concat file
        
    Raises:
        IOError: If writing concat file fails
    """
    if not ts_files:
        raise ValueError("No ts files provided for concat")
    
    content = ""
    for ts_file in ts_files:
        # Convert to absolute path and normalize to forward slashes
        # FFmpeg concat demuxer requires absolute paths
        abs_path = os.path.abspath(ts_file)
        normalized_path = abs_path.replace("\\", "/")
        safe_path = normalized_path.replace("'", "'\\''")
        content += f"file '{safe_path}'\n"
        logger.debug(f"Added to concat: {safe_path}")
    
    try:
        with open(concat_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.debug(f"Generated concat file: {concat_file_path} with {len(ts_files)} files")
        logger.info(f"Concat file content (first 3 lines):\n" + "\n".join(content.split("\n")[:3]))
    except IOError as e:
        logger.error(f"Failed to write concat file: {concat_file_path}: {e}")
        raise


def cleanup_cache_files(cache_dir: str, keep_seconds: int) -> int:
    """
    Clean up old .ts files from cache directory.
    Keep only files newer than keep_seconds.
    
    Args:
        cache_dir: Cache directory to clean
        keep_seconds: Keep files from the last N seconds
        
    Returns:
        Number of files deleted
    """
    if not os.path.isdir(cache_dir):
        return 0
    
    now_ts = datetime.now().timestamp()
    cutoff_ts = now_ts - keep_seconds
    deleted_count = 0
    
    try:
        for filename in os.listdir(cache_dir):
            if not filename.endswith('.ts'):
                continue
            
            filepath = os.path.join(cache_dir, filename)
            
            try:
                # Use file modification time for age determination
                file_mtime = os.path.getmtime(filepath)
                
                if file_mtime < cutoff_ts:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.debug(f"Deleted cache file: {filepath}")
                    except OSError as e:
                        logger.warning(f"Failed to delete cache file: {filepath}: {e}")
            except OSError:
                # Skip files that can't be accessed
                continue
    
    except OSError as e:
        logger.error(f"Error reading cache directory: {cache_dir}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} cache files from {cache_dir}")
    
    return deleted_count


def get_video_output_path(
    base_video_dir: str,
    camera_id: str,
    event_time: datetime
) -> str:
    """
    Generate standard output path for event video.
    
    Format: {base_video_dir}/{camera_id}/{YYYY-MM-DD}/event_{camera_id}_{YYYYMMDD_HHMMSS}.mp4
    
    Args:
        base_video_dir: Base video directory
        camera_id: Camera identifier
        event_time: Time of the event
        
    Returns:
        Full path for output video file
    """
    date_str = event_time.strftime("%Y-%m-%d")
    time_str = event_time.strftime("%Y%m%d_%H%M%S")
    filename = f"event_{camera_id}_{time_str}.mp4"
    
    output_path = os.path.join(base_video_dir, camera_id, date_str, filename)
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    return output_path


def is_valid_rtsp_url(url: str) -> bool:
    """
    Check if URL looks like a valid RTSP URL.
    
    Args:
        url: URL string to check
        
    Returns:
        True if URL appears valid
    """
    return url.lower().startswith(('rtsp://', 'rtsps://', 'rtsp+uni://'))


def is_file_ready(filepath: str, check_interval: float = 0.1, timeout: float = 2.0) -> bool:
    """
    Check if file is ready for reading (not still being written).
    
    This is a simple heuristic - checks if file size is stable.
    
    Args:
        filepath: Path to check
        check_interval: Time to wait between size checks (seconds)
        timeout: Maximum time to wait (seconds)
        
    Returns:
        True if file appears ready
    """
    import time
    
    if not os.path.exists(filepath):
        return False
    
    try:
        size1 = os.path.getsize(filepath)
        time.sleep(check_interval)
        size2 = os.path.getsize(filepath)
        
        # If size is stable, file is ready
        return size1 == size2
    except OSError:
        return False
