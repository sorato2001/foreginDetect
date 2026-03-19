# Multi-Camera Event Recording System - Windows PowerShell Startup Script
# 
# This script sets up the environment and starts the system on Windows.
# Usage: .\run.ps1 [-ConfigFile "configs/cameras.yaml"] [-LogLevel "INFO"]
#
# NOTE: You may need to allow script execution first:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1

param(
    [string]$ConfigFile = "configs\cameras.yaml",
    [string]$LogLevel = "INFO"
)

$ScriptDir = Split-Path -Parent (Get-Item -Path $PSCommandPath).FullName
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Multi-Camera Event Recording System" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Script directory: $ScriptDir"
Write-Host "Project root: $ProjectRoot"
Write-Host "Config file: $ConfigFile"
Write-Host "Log level: $LogLevel"
Write-Host ""

# Resolve paths
if (![System.IO.Path]::IsPathRooted($ConfigFile)) {
    $ConfigFile = Join-Path $ProjectRoot $ConfigFile
}

# Check if config file exists
if (!(Test-Path $ConfigFile)) {
    Write-Error "Config file not found: $ConfigFile" -ErrorAction Stop
    exit 1
}

Write-Host "[OK] Config file found" -ForegroundColor Green

# Check if ffmpeg is available
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpeg) {
    Write-Error @"
ffmpeg is not installed or not in PATH

Please install ffmpeg:
  - Download from: https://ffmpeg.org/download.html
  - Or use: choco install ffmpeg
"@ -ErrorAction Stop
    exit 1
}

Write-Host "[OK] ffmpeg found: $($ffmpeg.Source)" -ForegroundColor Green

# Check if Python is available
$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $python) {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
}

if ($null -eq $python) {
    Write-Error @"
Python is not installed or not in PATH

Please install Python 3.11+:
  - Download from: https://www.python.org/downloads/
  - Make sure to check 'Add Python to PATH' during installation
"@ -ErrorAction Stop
    exit 1
}

Write-Host "[OK] Python found: $($python.Source)" -ForegroundColor Green

# Determine python command
$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }

# Create virtual environment if needed
$venvPath = Join-Path $ProjectRoot "venv"
if (!(Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & $pythonCmd -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment" -ErrorAction Stop
        exit 1
    }
}

# Activate virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (!(Test-Path $activateScript)) {
    Write-Error "Virtual environment activation script not found" -ErrorAction Stop
    exit 1
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& $activateScript

# Install/upgrade requirements
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
$requirementsFile = Join-Path $ProjectRoot "requirements.txt"

pip install -q -r $requirementsFile
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Some dependencies may have failed to install"
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Starting system..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Run the application
Push-Location $ProjectRoot
& python -m app.main -c $ConfigFile -l $LogLevel
Pop-Location

Write-Host ""
Write-Host "System stopped. Press Enter to close." -ForegroundColor Yellow
Read-Host
