#!/bin/bash

# Multi-Camera Event Recording System - Startup Script
# 
# This script sets up the environment and starts the system.
# Usage: ./run.sh [config_file] [log_level]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-$SCRIPT_DIR/configs/cameras.yaml}"
LOG_LEVEL="${2:-INFO}"

echo "=============================================="
echo "Multi-Camera Event Recording System"
echo "=============================================="
echo "Script dir: $SCRIPT_DIR"
echo "Config: $CONFIG_FILE"
echo "Log level: $LOG_LEVEL"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Please create a config file first:"
    echo "  cp configs/cameras.example.yaml configs/cameras.yaml"
    echo "  # Edit configs/cameras.yaml with your camera settings"
    exit 1
fi

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg is not installed or not in PATH"
    echo ""
    echo "Please install ffmpeg:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
    exit 1
fi

echo "✓ ffmpeg found: $(ffmpeg -version | head -n1)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade requirements
echo "Installing Python dependencies..."
pip install -q -r requirements.txt

echo ""
echo "=============================================="
echo "Starting system..."
echo "=============================================="
echo ""

# Run the application
cd "$SCRIPT_DIR"
python3 -m app.main -c "$CONFIG_FILE" -l "$LOG_LEVEL"
