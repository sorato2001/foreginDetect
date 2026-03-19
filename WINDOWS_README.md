# 🪟 Multi-Camera Event Recording System - Windows Complete Guide

Welcome! This guide covers everything you need to run the multi-camera event recording system on Windows.

## 📚 Documentation Files

### Start Here
1. **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** ⭐
   - Step-by-step installation instructions
   - Prerequisites (Python, FFmpeg)
   - Camera configuration
   - Running the system
   - Troubleshooting

2. **[WINDOWS_FILES_INDEX.md](WINDOWS_FILES_INDEX.md)** ⭐
   - Explains all Windows-specific files
   - Quick reference for configuration
   - Common issues and solutions
   - File checklist

### Complete Documentation
3. **[README.md](README.md)**
   - Full system documentation
   - Architecture overview
   - API endpoints
   - Performance recommendations
   - Advanced usage

### Configuration Templates
4. **[configs/cameras.windows.yaml](configs/cameras.windows.yaml)**
   - Example configuration for Windows
   - Windows path examples
   - Detailed comments

5. **[.env-example](.env-example)**
   - Environment variable examples
   - Optional overrides

## 🚀 Quick Start (5 Minutes)

### Prerequisites

Before you start, install:

1. **Python 3.11+**
   - Download: https://www.python.org/downloads/
   - **IMPORTANT**: Check "Add Python to PATH" during installation
   - Verify: Open cmd and type `python --version`

2. **FFmpeg** (choose one)
   - **Easy**: `choco install ffmpeg` (if you have Chocolatey)
   - **Manual**: Download from https://ffmpeg.org/download.html
   - Verify: Open cmd and type `ffmpeg -version`

### Setup (One Time Only)

```cmd
REM Navigate to project directory
cd multi_camera_event_system

REM Run setup wizard (creates config, directories, and validates installation)
scripts\setup-windows.bat
```

This will:
- Check Python and FFmpeg
- Copy configuration template
- Create necessary directories
- Let you edit config in Notepad

### Configure Cameras

Edit `configs\cameras.yaml`:
- Add your camera RTSP URLs
- Configure FTP directories for alarm images
- Set pre/post recording times

Example:
```yaml
cameras:
  cam_001:
    name: "Front Gate"
    ip: "192.168.1.100"
    rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream1"
    ftp_image_dir: "./data/ftp/cam_001"
    pre_seconds: 5
    post_seconds: 10
    cache_seconds: 120
    enabled: true
```

### Run the System

Choose one method:

**Method 1: Desktop Shortcut (Recommended)**
```cmd
REM Create desktop shortcut (run once)
scripts\create-shortcut.bat

REM Then just double-click "Start-MultiCamera" on your desktop daily!
```

**Method 2: Double-Click**
```
scripts\run.bat
```

**Method 3: Command Prompt**
```cmd
scripts\run.bat configs\cameras.yaml INFO
```

**Method 4: PowerShell**
```powershell
.\scripts\run.ps1
```

---

## 🎬 Running Your First Event (Test)

### Create a test alarm image:

**Option A: Simple (Notepad)**
```cmd
REM Create FTP directory
mkdir data\ftp\cam_001

REM Create a simple test image file in the directory
copy nul data\ftp\cam_001\test_image.jpg
```

**Option B: Visual (Using Paint)**
1. Open Paint (type "paint" in Windows search)
2. Create an image (any size, just make sure it's RGB/JPEG)
3. File > Save As
4. Save as: `data/ftp/cam_001/alarm_image.jpg`
5. Type: JPEG

### Watch the system detect it:

1. The system should automatically detect the new image
2. Check `data\logs\event_system.log` to see if it detected the alarm:

```cmd
REM Windows: View recent log entries
more +10 data\logs\event_system.log
```

3. After event completes, video appears in:
```
data\video\cam_001\2024-03-18\event_cam_001_20240318_143530.mp4
```

---

## 📁 File Structure

```
multi_camera_event_system/
├── 🖥️  scripts/
│   ├── run.bat                 ← Double-click to run system
│   ├── run.ps1                 ← PowerShell version
│   ├── create-shortcut.bat     ← Creates desktop shortcut
│   └── setup-windows.bat       ← First-time setup wizard
│
├── 📝 configs/
│   ├── cameras.windows.yaml    ← Example config (Windows paths)
│   ├── cameras.example.yaml    ← Example config (Linux paths)
│   └── cameras.yaml            ← YOUR config (created from template)
│
├── 🐍 app/
│   ├── main.py                 ← Application entry point
│   ├── config.py               ← Config loading
│   ├── models.py               ← Data models
│   ├── database.py             ← SQLite operations
│   ├── recorder.py             ← RTSP streaming
│   ├── event_listener.py       ← FTP monitoring
│   ├── scheduler.py            ← Event scheduling
│   ├── event_finalizer.py      ← Video composition
│   ├── utils.py                ← Utility functions
│   └── api.py                  ← REST API (optional)
│
├── 📊 data/
│   ├── cache/                  ← Circular buffer segments
│   ├── video/                  ← Final event videos
│   ├── ftp/                    ← FTP alarm images
│   ├── db/                     ← SQLite database
│   └── logs/                   ← System logs
│
├── 📖 Windows Documentation
│   ├── WINDOWS_SETUP.md        ← Setup instructions
│   ├── WINDOWS_FILES_INDEX.md  ← File descriptions
│   └── THIS FILE               ← Quick reference
│
├── 📋 requirements.txt          ← Python dependencies
├── 📖 README.md                 ← Full documentation
├── .env-example                 ← Environment variables
└── .gitignore
```

---

## ⚙️ Configuration Guide

### Essential Settings

| Setting | Example | Notes |
|---------|---------|-------|
| `rtsp_url` | `rtsp://admin:pass@192.168.1.100:554/stream1` | Check camera manual for correct URL |
| `ftp_image_dir` | `./data/ftp/cam_001` | Where camera uploads alarm images |
| `pre_seconds` | `5` | Seconds BEFORE alarm to record |
| `post_seconds` | `10` | Seconds AFTER alarm to record |
| `cache_seconds` | `120` | Total rolling buffer size |
| `cooldown_seconds` | `3` | Debounce: min interval between alarms |

### Important Notes

1. **Paths in YAML**: Use `/` or `\\` not `\`
   - ✅ `C:/data/ftp/cam_001`
   - ✅ `C:\\data\\ftp\\cam_001`
   - ❌ `C:\data\ftp\cam_001` (won't work!)

2. **cache_seconds** must be >= pre_seconds + post_seconds
   - Example: pre=5, post=10, so cache must be ≥15 seconds

3. **FTP directories** must exist and camera must have write permission
   - Create manually or camera creates them

---

## 🔍 Monitoring the System

### Real-Time Logs

**Option 1: Notepad**
```cmd
REM Keep opening and refreshing file explorer
start explorer data\logs\
```

**Option 2: Command Prompt (tail-like)**
```cmd
REM View last 50 lines
powershell -Command "Get-Content data\logs\event_system.log -Tail 50 -Wait"
```

**Option 3: Windows Terminal** (Better)
```cmd
REM Install Windows Terminal from Microsoft Store
wt powershell -Command "Get-Content data\logs\event_system.log -Tail 50 -Wait"
```

### View Recorded Videos

```cmd
REM Open video folder
start explorer data\video\

REM Or navigate manually:
REM data\video\cam_001\2024-03-18\event_cam_001_20240318_143530.mp4
```

### Check Event Database

```cmd
REM Install sqlite3 if needed
choco install sqlite

REM Query events
sqlite3 data\db\events.db "SELECT camera_id, event_time, status FROM events ORDER BY event_time DESC LIMIT 10;"
```

---

## 🛠️ Common Tasks

### Stop the System

**If running in Command Prompt:**
- Press `Ctrl+C` in the cmd window

**If running in background:**
```cmd
REM Find python processes
tasklist | findstr python

REM Kill it
taskkill /IM python.exe /F
```

### View Current Events

```cmd
REM Show active events (while system is running)
powershell -Command "curl http://localhost:8000/events/active/all 2>$null | ConvertFrom-Json | ConvertTo-Json"
```

### Check Camera Status

```cmd
REM Show all cameras and their status
curl http://localhost:8000/cameras
```

### View All Recorded Events

```cmd
REM Show all recorded events
curl http://localhost:8000/events?limit=50
```

---

## ❌ Troubleshooting

### Problem: "Python not found"

**Solution:**
1. Download Python from https://www.python.org/downloads/
2. Run installer
3. **CRITICAL**: Check "Add Python to PATH"
4. Restart command prompt
5. Verify: `python --version`

### Problem: "ffmpeg not found"

**Solution:**
1. Install ffmpeg:
   - **Easy**: `choco install ffmpeg`
   - **Manual**: Download and add to PATH (see WINDOWS_SETUP.md)
2. Restart command prompt
3. Verify: `ffmpeg -version`

### Problem: "Config file not found"

**Solution:**
```cmd
REM Make sure configs\cameras.yaml exists
cd scripts
setup-windows.bat
```

### Problem: Alarms not detected

**Solution:**
1. Check FTP directory exists: `dir data\ftp\cam_001\`
2. Manually place an image there: `copy test_image.jpg data\ftp\cam_001\`
3. Check logs for errors: Look in `data\logs\event_system.log`
4. Check camera is uploading to correct location

### Problem: Videos not being created

**Solution:**
1. Verify ffmpeg works: `ffmpeg -version`
2. Check disk space: Click C: drive in File Explorer
3. Check logs for ffmpeg errors
4. Try with longer pre/post seconds

### Problem: "Port 8000 already in use"

**Solution:**
```yaml
# In cameras.yaml, change api_port:
api_port: 8001  # Instead of 8000
```

---

## 📞 Getting Help

1. **First**: Check [WINDOWS_SETUP.md](WINDOWS_SETUP.md) troubleshooting section
2. **Second**: Check `data\logs\event_system.log` for error messages
3. **Third**: Review [README.md](README.md) full documentation
4. **Last**: Check [WINDOWS_FILES_INDEX.md](WINDOWS_FILES_INDEX.md) quick reference

---

## ✨ Tips & Tricks

### Tip 1: Run Without Console Window (Hidden)

Create a shortcut in a text editor:
```
@echo off
start /b scripts\run.bat
```
Save as `run-hidden.bat`, then use `create-shortcut.bat` to create desktop shortcut.

### Tip 2: Log to File Rotation

Add to `cameras.yaml`:
```yaml
system:
  log_level: "INFO"
```

Configure Windows Task Scheduler to compress old logs daily.

### Tip 3: Auto-Start on Boot

Use Windows Task Scheduler:
1. Open Task Scheduler
2. Create Basic Task
3. Name: "MultiCamera Event System"
4. Trigger: At startup
5. Action: Start program `cmd.exe`
6. Arguments: `/c "C:\path\to\run.bat"`

### Tip 4: Monitor in Real-Time

Use Windows Terminal with auto-refresh:
```powershell
while ($true) { 
    Clear-Host
    Get-Content data\logs\event_system.log -Tail 30
    Start-Sleep 2
}
```

### Tip 5: Backup Videos Daily

Use Windows Task Scheduler to copy `data\video` to external drive daily.

---

## 🎯 System Architecture (Windows-Optimized)

```
┌─────────────────────────┐
│   Your Cameras (Hikvision etc.)
│   - Stream RTSP video
│   - Upload alarm images via FTP
└────────┬────────────────┘
         │
         ├──RTSP────────► FFmpeg (ffmpeg.exe)
         │                streams & buffers:
         │                data\cache\cam_001\*.ts
         │
         └──FTP────────► Explorer watches
                         data\ftp\cam_001\
                         Detects new images
                            │
                            ▼
                    Event Scheduler
                    - Creates event
                    - Records until post_seconds
                    - Extends if new alarms
                            │
                            ▼
                    Video Finalizer
                    - Finds *.ts segments
                    - Runs ffmpeg.exe concat
                    - Creates MP4
                            │
                            ▼
                    Output: data\video\
                    example:
                    cam_001\2024-03-18\
                    event_cam_001_20240318_143530.mp4
                            │
                            ▼
                    SQLite Database
                    data\db\events.db
```

---

## 📊 Performance on Windows

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Windows | 7 SP1 | 10 / 11 |
| CPU | 2 cores | 4 cores |
| RAM | 2 GB | 4+ GB |
| Storage | 500 MB (app) | 2+ GB (cache + videos) |
| Python | 3.11 | 3.11+ |
| FFmpeg | Compiled | Latest |

### Performance Tips

1. **CPU**: Use copy mode (default) - no re-encoding
2. **Network**: Use wired connection for cameras
3. **Disk**: Use SSD for cache/db, HDD OK for videos
4. **Memory**: Monitor Task Manager > Performance

---

## 🎓 Learning More

### Understand RTSP
- Docs: https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol
- Hikvision cameras: Check manual for URL format
- Common: `rtsp://admin:password@ip:554/stream1`

### Understand FFmpeg
- Docs: https://ffmpeg.org/documentation.html
- Tutorial: https://trac.ffmpeg.org/wiki/File:FFmpeg%20guide.pdf
- Testing: `ffmpeg -h full`

### Understand YAML
- Tutorial: https://yaml.org/start.html
- Online validator: https://www.yamllint.com

### Python Virtual Environments
- Docs: https://docs.python.org/3/tutorial/venv.html
- Why: Isolates project dependencies

---

## 📝 Checklists

### Before First Run
- [ ] Python 3.11+ installed
- [ ] FFmpeg installed and in PATH
- [ ] `configs\cameras.yaml` exists and populated
- [ ] FTP directories created: `data\ftp\cam_001`, etc.
- [ ] Camera RTSP URLs verified (test with VLC)
- [ ] Camera configured to upload images to FTP

### Maintenance
- [ ] Check `data\logs\event_system.log` weekly
- [ ] Backup important videos to external drive
- [ ] Monitor disk space in `data\` folder
- [ ] Review SQLite database occasionally for old events
- [ ] Test system with manual alarm image monthly

### If Moving System
- [ ] Backup all of `data\` folder
- [ ] Backup `configs\cameras.yaml`
- [ ] Update camera IPs if network changed
- [ ] Update FTP paths if storage changed

---

## 🎉 Success!

You should now have a working multi-camera event recording system on Windows!

**Next steps:**
1. Run `scripts\setup-windows.bat`
2. Edit `configs\cameras.yaml`
3. Double-click `scripts\run.bat` or use the desktop shortcut
4. Check logs as events are recorded
5. Find videos in `data\video\`

**Questions?** Refer to [WINDOWS_SETUP.md](WINDOWS_SETUP.md) or [README.md](README.md).

---

**Version**: 1.0.0 - Windows Edition  
**Last Updated**: March 2024  
**Compatibility**: Windows 7+, Windows 10, Windows 11
