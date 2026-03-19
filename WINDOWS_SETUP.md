"""
Quick setup guide for Windows users.

This file explains how to get the multi-camera event recording system
running on Windows in just a few steps.
"""

# ============================================================
# WINDOWS QUICK START
# ============================================================

# Step 1: Install Prerequisites
# ============================================================

# 1a. Python 3.11+ (required)
# - Download from: https://www.python.org/downloads/
# - IMPORTANT: Check "Add Python to PATH" during installation
# - Verify: Open Command Prompt and type: python --version
#   Should show: Python 3.11.x or newer

# 1b. FFmpeg (required)
# Option A: Chocolatey (recommended if you have it)
#   choco install ffmpeg
#
# Option B: Manual download
#   1. Go to https://ffmpeg.org/download.html
#   2. Click "Windows builds by BtbN"
#   3. Download "ffmpeg-master-latest-win64-gpl.zip"
#   4. Extract to: C:\Program Files\ffmpeg
#   5. Add to PATH:
#      - Press Win+X, search "Environment Variables"
#      - Click "Edit the system environment variables"
#      - Click "Environment Variables" button
#      - Under "System variables", click "Path"
#      - Click "New" and add: C:\Program Files\ffmpeg\bin
#      - Click OK, OK, OK
#   6. Restart Command Prompt
#
# Verify: Open Command Prompt and type: ffmpeg -version


# Step 2: Configure Cameras
# ============================================================

# 1. Open configs\cameras.windows.yaml in a text editor
#    (Right-click > Open with > Notepad)

# 2. For each camera, fill in:
#    - ip: Camera IP address (e.g., 192.168.1.100)
#    - rtsp_url: RTSP stream URL from camera
#      Example for Hikvision: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
#    - ftp_image_dir: Where camera will upload alarm images
#      Example: C:/Users/YourName/Documents/hikvision_ftp/cam_001
#      Or relative: ./data/ftp/cam_001

# 3. Create the FTP directories you specified:
#    - Open File Explorer
#    - Create folders matching your ftp_image_dir paths
#    - Make sure the camera can write to these directories

# 4. Configure cameras via their web UI:
#    - Open browser, go to http://<camera_ip>
#    - Login with admin/password
#    - Find "Event" or "Alarm" settings
#    - Enable "Line Crossing Detection" (or your event type)
#    - Find "Network" > "FTP" settings
#    - Set FTP Server: localhost (or your machine IP if on another machine)
#    - Set FTP Port: 21
#    - Set Upload Path: /cam_001 (matches your ftp_image_dir)
#    - Test and save


# Step 3: Run the System
# ============================================================

# Method 1: Double-click run.bat (easiest)
#   1. Navigate to: scripts\ folder
#   2. Double-click: run.bat
#   3. A command window will open and start the system

# Method 2: Create desktop shortcut (most convenient)
#   1. Navigate to: scripts\ folder
#   2. Double-click: create-shortcut.bat
#   3. A shortcut will be created on your desktop
#   4. Double-click the shortcut anytime to start

# Method 3: Command Prompt
#   1. Open Command Prompt
#   2. Navigate to the project directory:
#      cd C:\path\to\multi_camera_event_system
#   3. Run one of:
#      scripts\run.bat configs\cameras.yaml INFO
#      scripts\run.bat

# Method 4: PowerShell (modern approach)
#   1. Open PowerShell
#   2. Navigate to the project directory:
#      cd C:\path\to\multi_camera_event_system
#   3. Allow script execution (first time only):
#      Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser
#   4. Run:
#      .\scripts\run.ps1 -ConfigFile "configs\cameras.yaml" -LogLevel "INFO"
#      # or just:
#      .\scripts\run.ps1


# Step 4: Monitor Activity
# ============================================================

# Watch the log file:
#   - Open file: data\logs\event_system.log
#   - Watch for lines like:
#     "Detected alarm event: camera=cam_001"
#     "Event created: cam_001"
#     "Successfully finalized event"


# Step 5: View Generated Videos
# ============================================================

# Videos are saved to:
#   data\video\{camera_id}\{YYYY-MM-DD}\event_{camera_id}_{YYYYMMDD_HHMMSS}.mp4
#
# Example:
#   data\video\cam_001\2024-03-18\event_cam_001_20240318_143530.mp4
#
# You can double-click to play with Windows Media Player or VLC


# Step 6: Check Recorded Events
# ============================================================

# View event database (requires sqlite3):
#   1. Install sqlite3 for Windows:
#      choco install sqlite (if using Chocolatey)
#      Or download from: https://www.sqlite.org/download.html
#
#   2. View events:
#      sqlite3 data\db\events.db
#      SELECT * FROM events;
#      SELECT * FROM events WHERE camera_id='cam_001';
#      .quit


# Troubleshooting
# ============================================================

# Q: "ffmpeg is not installed or not in PATH"
# A: 1. Check ffmpeg is installed: ffmpeg -version
#    2. If not found, install it (see Step 1b)
#    3. Restart Command Prompt or reboot after adding to PATH

# Q: "Python is not installed or not in PATH"
# A: 1. Check Python is installed: python --version
#    2. If not found, install from python.org
#    3. Make sure "Add Python to PATH" was checked
#    4. Restart Command Prompt after installation

# Q: "Config file not found"
# A: 1. Make sure you have configs\cameras.yaml (not .example.yaml)
#    2. Use the provided cameras.windows.yaml as a template
#    3. Copy: copy configs\cameras.windows.yaml configs\cameras.yaml
#    4. Edit configs\cameras.yaml with your settings

# Q: "Alarms not detected"
# A: 1. Check FTP directory exists and is readable
#    2. Manually place an image file in the FTP directory
#    3. Check logs for errors: look in data\logs\event_system.log
#    4. Verify camera is uploading images to correct directory

# Q: "Videos not being created"
# A: 1. Check that ffmpeg is working: ffmpeg -version
#    2. Verify enough disk space: Check C: drive
#    3. Check logs for ffmpeg errors
#    4. Try with a longer pre_seconds and post_seconds

# Q: "Port 8000 already in use" (if API enabled)
# A: 1. Change api_port in cameras.yaml to unused port (e.g., 8001)
#    2. Or stop other services using port 8000

# Q: "Permission denied" for FTP directory
# A: 1. Check FTP directory is readable/writable
#    2. Right-click directory > Properties > Security
#    3. Make sure your user has Full Control
#    4. Try using absolute path instead of relative path


# Advanced: Running without console window (optional)
# ============================================================

# To run minimized without console window:
#   1. Create a shortcut to run.bat
#   2. Right-click shortcut > Properties
#   3. Under "Target", change to:
#      C:\windows\system32\cmd.exe /C "scripts\run.bat"
#   4. Change "Run" dropdown to: Minimized
#   5. Click OK


# Advanced: Running as Windows Service (optional)
# ============================================================

# Using NSSM (Non-Sucking Service Manager):
#   1. Download NSSM: https://nssm.cc/download
#   2. Extract nssm.exe
#   3. Open Command Prompt as Administrator
#   4. Run:
#      nssm install MultiCameraEvent "C:\path\to\scripts\run.bat"
#      nssm start MultiCameraEvent
#   5. Service will run at startup and restart if it crashes


# API Usage on Windows
# ============================================================

# If enable_api: true in your config:
#
# From Command Prompt or PowerShell:
#   curl http://localhost:8000/health
#   curl http://localhost:8000/status
#   curl http://localhost:8000/cameras
#   curl http://localhost:8000/events
#
# Or from browser:
#   http://localhost:8000/health
#   http://localhost:8000/cameras
#   http://localhost:8000/events


# ============================================================
# That's it! Your system should now be running.
# Check data\logs\event_system.log for details.
# ============================================================
