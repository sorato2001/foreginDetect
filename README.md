# Multi-Camera Event Recording System

A Python-based system for continuous RTSP stream capture and alarm-triggered event video recording from multiple Hikvision network cameras. When cameras send alarm notifications via FTP, this system automatically saves video clips containing the pre-alarm and post-alarm footage.

## Features

- **Multi-Camera Support**: Handle dozens of simultaneous camera streams with independent recording control
- **Circular Buffer Caching**: Maintains a rolling buffer of recent video segments for each camera
- **Smart Event Recording**: Automatically captures video when alarms are triggered via FTP image uploads
- **Event Extension**: Automatically extends recording if alarms occur during the post-alarm window
- **FFmpeg Integration**: Uses ffmpeg for efficient streaming (copy mode, no re-encoding)
- **SQLite Logging**: Persistent database of all recorded events with metadata
- **REST API**: Optional FastAPI endpoints for system monitoring and event querying
- **Robust Error Handling**: Automatic recovery from streaming failures with restart logic
- **Detailed Logging**: Comprehensive logs for debugging and monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Camera Event System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │   Recorder 1     │  │   Recorder 2     │  │   Recorder N     │
│  │  (ffmpeg RTSP)   │  │  (ffmpeg RTSP)   │  │  (ffmpeg RTSP)   │
│  │  Circ Buffer     │  │  Circ Buffer     │  │  Circ Buffer     │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
│           │ streams              │ streams            │ streams
│           │ segment files        │ segment files      │ segment files
│           ▼                       ▼                    ▼
│  ┌─────────────────────────────────────────────────────────────┐
│  │           Circular Cache (data/cache/)                      │
│  │    {camera_id}/{unix_timestamp}.ts (1-second segments)      │
│  └────────────────┬────────────────────────────────────────────┘
│                   │
│  ┌────────────────┴─────────────┐
│  │                              │
│  ▼                              ▼
│  ┌──────────────┐  ┌────────────────────────┐
│  │   FTP Image  │  │  Event Scheduler       │
│  │   Listener   │  │  (State Machine)       │
│  └──────┬───────┘  │  IDLE/RECORDING/       │
│         │          │  FINALIZING            │
│         │ alarm    │                        │
│         └─────────►├────────────────────────┤
│                    │  - Create events       │
│                    │  - Extend recording    │
│                    │  - Trigger finalize    │
│                    └────────┬───────────────┘
│                             │
│                             ▼
│                    ┌────────────────────────┐
│                    │  Event Finalizer       │
│                    │  (FFmpeg Concat)       │
│                    └────────┬───────────────┘
│                             │
│         ┌───────────────────┴──────────┐
│         │                              │
│         ▼                              ▼
│  ┌──────────────┐  ┌──────────────────────┐
│  │ Event Video  │  │    Event Database    │
│  │ (data/video/)│  │    (SQLite)          │
│  └──────────────┘  └──────────────────────┘
│                              ▲
│                              │
│         ┌────────────────────┴───────────────┐
│         │                                    │
│         ▼                                    ▼
│  ┌──────────────┐  ┌──────────────────────┐
│  │ REST API     │  │  Logs                │
│  │ (FastAPI)    │  │  (data/logs/)        │
│  └──────────────┘  └──────────────────────┘
│                                              │
└──────────────────────────────────────────────┘
```

## System Requirements

- **Python**: 3.11 or newer
- **FFmpeg**: 4.2 or newer (compiled with libx264 for MP4 output)
- **OS**: Linux (Ubuntu 20.04+) preferred, other Unix-like systems, or WSL on Windows
- **Disk Space**: 
  - Cache: ~50-200 MB (depending on camera count and cache_seconds)
  - Videos: ~500 MB - several GB per day (depending on event frequency and video quality)

## Installation

### For Windows Users - Quick Start

See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed Windows-specific instructions.

**Quick summary:**
1. Install Python 3.11+ from https://www.python.org (check "Add to PATH")
2. Install FFmpeg from https://ffmpeg.org or via `choco install ffmpeg`
3. Double-click `scripts\run.bat`
4. Edit `configs\cameras.yaml` with your camera settings
5. Create FTP image directories

### For Linux (Ubuntu/Debian)

### 1. Install system dependencies

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3.11 python3.11-venv python3-pip

# Install FFmpeg
sudo apt install ffmpeg

# Verify installations
python3 --version
ffmpeg -version
```

### 2. Clone or extract the project

```bash
cd /path/to/multi_camera_event_system
```

### 3. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Create configuration file

**Windows:**
```cmd
copy configs\cameras.windows.yaml configs\cameras.yaml
```

**Linux/macOS:**
```bash
cp configs/cameras.example.yaml configs/cameras.yaml
```

### 2. Edit `configs/cameras.yaml`

Configure your cameras with RTSP URLs and parameters:

**Important for Windows path compatibility:**
- Use forward slashes `/` or double backslashes `\\` in YAML paths
- Single backslash `\` will cause parse errors
- Relative paths like `./data/ftp/cam_001` work on all platforms

```yaml
system:
  base_cache_dir: "./data/cache"
  base_video_dir: "./data/video"
  db_path: "./data/db/events.db"
  log_dir: "./data/logs"
  log_level: "INFO"

cameras:
  cam_001:
    name: "Parking Lot"
    ip: "192.168.1.100"
    rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream1"
    ftp_image_dir: "/data/ftp/cam_001"    # Where camera uploads alarm images
    event_type: "line_crossing"
    enabled: true
    pre_seconds: 5                         # Record 5 sec before alarm
    post_seconds: 10                       # Record 10 sec after alarm
    cache_seconds: 120                     # Keep 120 sec circular buffer
    cooldown_seconds: 3                    # Debounce repeated alarms
    max_extend_seconds: 60                 # Max event duration
    output_dir: "./data/video/cam_001"
```

**Windows (Command Prompt):**
```cmd
mkdir data\ftp\cam_001
mkdir data\ftp\cam_002
REM ... etc for each camera
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path "data\ftp\cam_001" -Force
New-Item -ItemType Directory -Path "data\ftp\cam_002" -Force
```

**Linux:**
```bash
mkdir -p /data/ftp/cam_001
mkdir -p /data/ftp/cam_002tsp://admin:password@ip:554/Streaming/Channels/101`
- `ftp_image_dir`: Local directory where camera uploads alarm images via FTP
- `pre_seconds`: How many seconds before alarm to include in video
- `post_seconds`: How many seconds after alarm to continue recording
- `cache_seconds`: Total rolling buffer (should be > pre_seconds + post_seconds)

### 3. Ensure FTP directories exist

```bash
mkdir -p /data/ftp/cam_001
mkdir -p /data/ftp/cam_002
# ... etc for each camera
chmod 777 /data/ftp/cam_*
```

Your Hikvision cameras should be configured (via their web UI or app) to upload alarm images to these directories via FTP.

## Running the System

### Windows

**Option 1: Double-click (easiest)**
```
Navigate to: scripts\run.bat
Double-click the file
```

**Option 2: Command Prompt**
```cmd
cd multi_camera_event_system
scripts\run.bat configs\cameras.yaml INFO
```

**Option 3: PowerShell**
```powershell
cd multi_camera_event_system
.\scripts\run.bat configs\cameras.yaml INFO
```

### Linux / macOS

**Option 1: Using the run script**
```bash
chmod +x scripts/run.sh
./scripts/run.sh configs/cameras.yaml INFO
```

**Option 2: Direct Python execution**
```bash
python3 -m app.main -c configs/cameras.yaml -l INFO
```

The system will:
1. Create necessary directories
**Windows (Command Prompt):**
```cmd
REM Simple way: Use Paint to create a test image
REM 1. Open Paint
REM 2. Create 1920x1080 image (or any size)
REM 3. Save as: data\ftp\cam_001\alarm_image.jpg

REM Or use PowerShell:
powershell -Command "
[System.Reflection.Assembly]::LoadWithPartialName('System.drawing') | Out-Null
$bmp = New-Object System.Drawing.Bitmap(1920, 1080)
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.Clear([System.Drawing.Color]::Blue)
$graphics.Dispose()
$bmp.Save('data\ftp\cam_001\alarm_image.jpg')
$bmp.Dispose()
"
```

**Windows (Python):**
```cmd
python -m pip install pillow
python -c "
from PIL import Image
import os
os.makedirs('data/ftp/cam_001', exist_ok=True)
img = Image.new('RGB', (1920, 1080), color='blue')
img.save('data/ftp/cam_001/alarm_image.jpg')
"
```

**Linux:**
```bash
python3 -c "
from PIL import Image
import os
os.makedirs('data/ftp/cam_001', exist_ok=True)
img = Image.new('RGB', (1920, 1080), color='blue')
img.save('data/ftp/cam_001/alarm_image.jpg')
"
```

The system should detect this image and create an event. Check logs in `data/logs/event_system.log`.

### Viewing generated videos

Once an event is finalized, the video will be saved to:
```
data\video\{camera_id}\{YYYY-MM-DD}\event_{camera_id}_{YYYYMMDD_HHMMSS}.mp4
```

Example:
```
data\video\cam_001\2024-03-18\event_cam_001_20240318_143530.mp4
```

Double-click to play with Windows Media Player, VLC, or any video player.

### Checking event database

**Windows:**
```cmd
REM Option 1: Install sqlite3 (if not already installed)
choco install sqlite

REM Then run:
sqlite3 data\db\events.db "SELECT * FROM events;"
sqlite3 data\db\events.db "SELECT * FROM events WHERE camera_id='cam_001';"
```

**Linux:**### Viewing generated videos

Once an event is finalized, the video will be saved to:
```
data/video/{camera_id}/{YYYY-MM-DD}/event_{camera_id}_{YYYYMMDD_HHMMSS}.mp4
```

Example:
```
data/video/cam_001/2024-03-18/event_cam_001_20240318_143530.mp4
```

### Checking event database

```bash
# List all events
sqlite3 data/db/events.db "SELECT * FROM events;"

# Events for specific camera
sqlite3 data/db/events.db "SELECT * FROM events WHERE camera_id='cam_001';"
```
not found on Windows

**Issue**: "ffmpeg is not installed or not in PATH"

**Solutions**:
1. Install FFmpeg:
   - Download from https://ffmpeg.org/download.html
   - Extract to `C:\Program Files\ffmpeg`
   - Add to PATH (see WINDOWS_SETUP.md for detailed steps)
   
2. Verify installation:
   ```cmd
   ffmpeg -version
   ```

3. You can also specify full path in config:
   ```yaml
   ffmpeg_path: "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"
   ```

### FFmpeg process not streaming

**Issue**: Logs show "Failed to start recording for camera_xxx"

**Solutions**:
- Verify RTSP URL is correct:
  - **Windows**: Install VLC from https://www.videolan.org/
    Then: File > Open Network Stream > paste RTSP URL
  - **Linux**: `vlc rtsp://admin:password@192.168.1.100:554/stream1`
- Check network connectivity to camera (ping the IP)

### System Status

```bash
curl http://localhost:8000/status
```cmd
  REM Windows
  dir data\ftp\cam_001\
  echo "test" > data\ftp\cam_001\test.txt
  ```
  ```bash
  # Linux
  ls -la /data/ftp/cam_001/
  touch /data/ftp/cam_001/test.txt
  ```

- Check image filename contains parseable timestamp or check file modification time
- Review logs for FTP listener errors:
  ```cmd
  REM Windows - use more or type command
  more data\logs\event_system.log | findstr "Detected alarm"
  ```
  ```bash
  # Linux
  grep "Detected alarm" data/logs/event_system.log
  ```

- Verify camera is actually uploading images:
  ```cmd
  REM Windows
  dir data\ftp\cam_001\
  ```
  ```bash
  # Linux
curl http://localhost:8000/cameras/cam_001
```

### List events

```bash
curl http://localhost:8000/events
curl http://localhost:8000/events?camera_id=cam_001&limit=50
```

### Get event details

```bash
curl http://localhost:8000/events/1
```

### Get active events

```bash
curl http://localhost:8000/events/active/all
```

### Recorder health


- **Windows**: Stop the system and check for stale processes:
  ```cmd
  tasklist | findstr python
  taskkill /F /IM python.exe
  ```

- **Linux**: Stop and check stale processes:
  ```bash
  pkill -f "python3 -m app.main"
  ```

- Delete lock files:
  ```cmd
  REM Windows
  del data\db\events.db-wal
  del data\db\events.db-shm
  ```
  ```bash
  # Linux
### FFmpeg process not streaming

**Issue**: Logs show "Failed to start recording for camera_xxx"

**Solutions**:
- Verify RTSP URL is correct by testing with VLC:
  ```bash
  vlc rtsp://admin:password@192.168.1.100:554/stream1
  ```
- Check network connectivity to camera
- Verify camera credentials
- Try adding connection parameters:
  ```yaml
  rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream1?timeout=10"
  ```

### Alarms not detected

**Issue**: Camera uploads images but no events are created

**Solutions**:
- Verify FTP directory exists and is writeable:
  ```bash
  ls -la /data/ftp/cam_001/
  touch /data/ftp/cam_001/test.txt
  ```
- Check image filename contains parseable timestamp or check file modification time
- Review logs for FTP listener errors:
  ```bash
  grep "Detected alarm" data/logs/event_system.log
  ```
- Verify camera is actually uploading images:
  ```bash
  ls -lah /data/ftp/cam_001/
  ```

### Video composition fails

**Issue**: Logs show "Failed to compose video" or "No ts files found"

**Solutions**:
- Verify ffmpeg is in PATH:
  ```bash
  which ffmpeg
  ffmpeg -version
  ```
- Check cache directory has segment files:
  ```bash
  ls -la data/cache/cam_001/
  ```
- Verify cache_seconds is large enough to cover pre + post seconds:
  ```
  cache_seconds > pre_seconds + post_seconds
  ```
- Check available disk space:
  ```bash
  df -h data/cache/
  ```

### High CPU or memory usage

**Issue**: System consumes too much CPU/memory

**Solutions**:
- Reduce camera count
- Decrease stream quality (configure camera to lower bitrate)
- Use copy mode in ffmpeg (default - verify with `grep -i copy`)
- Increase `cache_seconds` check interval

### Database locked errors

**Issue**: "database is locked" errors in logs

**Solutions**:
- Ensure only one instance is running
- Stop the system and check for stale processes:
  ```bash
  pkill -f "python3 -m app.main"
  ```
- Delete lock file if it exists:
  ```bash
  rm -f data/db/events.db-wal data/db/events.db-shm
  ```

## File Structure

```
multi_camera_event_system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── config.py               # Configuration loading
│   ├── models.py               # Data models
│   ├── database.py             # SQLite interface
│   ├── recorder.py             # RTSP recording workers
│   ├── event_listener.py       # FTP monitoring
│   ├── scheduler.py            # Event state machine
│   ├── event_finalizer.py      # Video composition
│   ├── utils.py                # Utility functions
│   └── api.py                  # REST API (optional)
├── configs/
│   └── cameras.example.yaml    # Configuration template
├── scripts/
│   └── run.sh                  # Startup script
├── data/
│   ├── cache/                  # Rolling segment buffers
│   ├── video/                  # Final event videos
│   ├── ftp/                    # FTP upload directory
│   ├── db/                     # SQLite database
│   └── logs/                   # Application logs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Performance & Scaling

### Circular Buffer Management

- Each camera maintains a rolling buffer of recent segments
- Cleanup runs every 30 seconds, removing segments older than `cache_seconds`
- Typical disk usage: 100-200 MB per camera per 2 minutes of cache

### Database Indexes

- Indexed on `camera_id` and `event_time` for fast queries
- SQLite can handle thousands of events without issues

### Concurrent Events

- System supports overlapping events from different cameras
- Same camera can extend recording if alarms occur during post-seconds window
- Thread-safe for multi-camera scenarios

### Recommendations

| Scale | Cameras | Notes |
|-------|---------|-------|
| Small | 1-5 | Single machine sufficient |
| Medium | 5-20 | Monitor CPU/memory, optimize bitrates |
| Large | 20-50+ | Consider multiple instances per server section |

## Extending the System

### Adding new event types

Edit `models.py`:
```python
class EventType(str, Enum):
    LINE_CROSSING = "line_crossing"
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    YOU_CUSTOM_TYPE = "you_custom_type"    # Add here
```

### Connecting to external systems

Modify `scheduler.py` callbacks:
```python
def on_finalized(event_record):
    # Send to Kafka, webhook, etc.
    send_to_kafka(event_record)
```

### Using PostgreSQL instead of SQLite

Replace `database.py` implementation with a PostgreSQL adapter.

### Integrating with Redis queue

Add event publishing to queue for distributed processing:
```python
redis_client.publish("events", event_record.json())
```

### Web UI dashboard

Implement a web frontend consuming the REST API.

## Limitations & Known Issues

1. **RTSP connection stability**: Network interruptions may cause stream loss; system auto-restarts but may miss events
2. **Segment timing accuracy**: 1-second segments provide ~1 second granularity
3. **Clock sync**: Requires system clock accuracy within a few seconds for filename parsing
4. **Single-server deployment**: Currently designed for single machine; scaling requires refactoring
5. **No authentication**: REST API has no authentication layer (use firewall/reverse proxy)

## Future Enhancements

- [ ] Redis-based cluster support for horizontal scaling
- [ ] PostgreSQL/TimescaleDB for larger deployments
- [ ] Advanced video indexing and search
- [ ] Real-time stream monitoring UI
- [ ] Event notification webhooks
- [ ] Machine learning integration for false positive filtering
- [ ] Multi-server coordination
- [ ] H.265/HEVC compression support
- [ ] Direct camera SD card download (for backup)

## License

This project is provided as-is for event recording purposes.

## Support

For issues or questions:
1. Check logs in `data/logs/event_system.log`
2. Review configuration in `configs/cameras.yaml`
3. Verify FFmpeg installation and RTSP/FTP connectivity
4. Check available disk space and system resources

---

**Version**: 1.0.0  
**Last Updated**: March 2024
