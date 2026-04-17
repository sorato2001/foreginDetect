@echo off
REM Multi-Camera Event Recording System - Windows Setup Wizard
REM 
REM This script performs one-time setup tasks:
REM - Copy example configs
REM - Create FTP directories
REM - Do basic validation
REM
REM Run this ONCE before running the system

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0..
cd /d "%SCRIPT_DIR%"

cls
echo.
echo =============================================
echo Multi-Camera Event Recording System
echo Windows Setup Wizard
echo =============================================
echo.

REM Check Python
echo Checking Python installation...
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where python3 >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Python not found!
        echo.
        echo Please install Python 3.11+ from:
        echo   https://www.python.org/downloads/
        echo.
        echo IMPORTANT: Check "Add Python to PATH" during installation
        echo.
        pause
        exit /b 1
    )
)
echo [OK] Python found
echo.

REM Check FFmpeg
echo Checking FFmpeg installation...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] FFmpeg not found in PATH
    echo.
    echo FFmpeg is required. Install via:
    echo   Option 1: choco install ffmpeg (if you have Chocolatey)
    echo   Option 2: Download from https://ffmpeg.org/download.html
    echo.
    echo You can proceed without FFmpeg, but the system won't work.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i "!CONTINUE!"=="n" (
        exit /b 1
    )
) else (
    echo [OK] FFmpeg found
)
echo.

REM Copy config template
echo Setting up configuration...
if exist "configs\cameras.yaml" (
    echo [OK] configs\cameras.yaml already exists
    set /p OVERWRITE="Overwrite with template? (y/n): "
    if /i "!OVERWRITE!"=="y" (
        copy /Y "configs\cameras.windows.yaml" "configs\cameras.yaml" >nul
        echo [DONE] Copied template
    )
) else (
    copy /Y "configs\cameras.windows.yaml" "configs\cameras.yaml" >nul
    echo [DONE] Created configs\cameras.yaml from template
)
echo.

REM Create FTP directories
echo Creating FTP directories...
if not exist "data\ftp\cam_001" (
    mkdir "data\ftp\cam_001"
    echo [DONE] Created data\ftp\cam_001
)
if not exist "data\ftp\cam_002" (
    mkdir "data\ftp\cam_002"
    echo [DONE] Created data\ftp\cam_002
)
if not exist "data\ftp\cam_003" (
    mkdir "data\ftp\cam_003"
    echo [DONE] Created data\ftp\cam_003
)
echo.

REM Create other directories
echo Creating working directories...
mkdir "data\cache" 2>nul
mkdir "data\video" 2>nul
mkdir "data\db" 2>nul
mkdir "data\logs" 2>nul
echo [DONE] All directories created
echo.

REM Summary
echo =============================================
echo Setup Complete!
echo =============================================
echo.
echo Next steps:
echo   1. Edit configs\cameras.yaml with your camera settings
echo   2. Run: scripts\run.bat
echo      Optional: scripts\run.bat configs\cameras.yaml INFO image
echo.
echo For detailed help:
echo   - See: WINDOWS_SETUP.md
echo   - See: WINDOWS_FILES_INDEX.md
echo   - See: README.md
echo.

set /p OPEN_CONFIG="Open configs\cameras.yaml now? (y/n): "
if /i "!OPEN_CONFIG!"=="y" (
    start notepad "configs\cameras.yaml"
)

echo.
echo Press Enter to close...
pause >nul
