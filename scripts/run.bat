@echo off
REM Multi-Camera Event Recording System - Windows Startup Script
REM 
REM This script sets up the environment and starts the system on Windows.
REM Usage: run.bat [config_file] [log_level] [analysis_mode]

setlocal enabledelayedexpansion

REM Get script directory
set SCRIPT_DIR=%~dp0..
set CONFIG_FILE=%1
set LOG_LEVEL=%2
set ANALYSIS_MODE=%3

if "%CONFIG_FILE%"=="" (
    set CONFIG_FILE=%SCRIPT_DIR%\configs\cameras.yaml
)

if "%LOG_LEVEL%"=="" (
    set LOG_LEVEL=INFO
)

if /I "%ANALYSIS_MODE%"=="DEFAULT" (
    set ANALYSIS_MODE=
)

echo.
echo =============================================
echo Multi-Camera Event Recording System
echo =============================================
echo.
echo Script directory: %SCRIPT_DIR%
echo Config file: %CONFIG_FILE%
echo Log level: %LOG_LEVEL%
if "%ANALYSIS_MODE%"=="" (
    echo Analysis mode: use YAML/default
) else (
    echo Analysis mode: %ANALYSIS_MODE%
)
echo.

REM Check if config file exists
if not exist "%CONFIG_FILE%" (
    echo ERROR: Config file not found: %CONFIG_FILE%
    echo.
    echo Please create a config file first:
    echo   copy configs\cameras.example.yaml configs\cameras.yaml
    echo   [Edit configs\cameras.yaml with your camera settings]
    echo.
    pause
    exit /b 1
)

REM Check if ffmpeg is available
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ffmpeg is not installed or not in PATH
    echo.
    echo Please install ffmpeg:
    echo   Download from: https://ffmpeg.org/download.html
    echo   Or use: choco install ffmpeg
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('ffmpeg -version 2^>nul ^| findstr /R "ffmpeg version"') do set FFMPEG_VERSION=%%i
echo [OK] ffmpeg found: %FFMPEG_VERSION%
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where python3 >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Python is not installed or not in PATH
        echo.
        echo Please install Python 3.11+:
        echo   Download from: https://www.python.org/downloads/
        echo   Make sure to check "Add Python to PATH" during installation
        echo.
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

echo [OK] Python found: %PYTHON_CMD%
echo.

REM Create virtual environment if needed
if not exist "%SCRIPT_DIR%\venv" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv "%SCRIPT_DIR%\venv"
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call "%SCRIPT_DIR%\venv\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/upgrade requirements
echo Installing Python dependencies...
pip install -q -r "%SCRIPT_DIR%\requirements.txt"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some dependencies may have failed to install
    echo Continuing anyway...
)

echo.
echo =============================================
echo Starting system...
echo =============================================
echo.

REM Run the application
cd /d "%SCRIPT_DIR%"
set RUN_CMD=python -m app.main -c "%CONFIG_FILE%" -l "%LOG_LEVEL%"
if not "%ANALYSIS_MODE%"=="" (
    set RUN_CMD=!RUN_CMD! --analysis-mode "%ANALYSIS_MODE%"
)
call !RUN_CMD!

pause
