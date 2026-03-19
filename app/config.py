"""
Configuration loading and validation module.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import yaml
from .models import SystemConfig, CameraConfig


logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_system_config(config_path: str) -> SystemConfig:
    """
    Load and validate system configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated SystemConfig object
        
    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If configuration is invalid
    """
    config_dict = load_yaml_config(config_path)
    
    # Extract cameras config
    cameras_config = config_dict.get('cameras', {})
    validated_cameras = {}
    
    for camera_id, camera_dict in cameras_config.items():
        if not isinstance(camera_dict, dict):
            logger.warning(f"Skipping invalid camera config: {camera_id}")
            continue
        
        camera_dict['camera_id'] = camera_id
        try:
            validated_cameras[camera_id] = CameraConfig(**camera_dict)
            logger.info(f"Loaded camera config: {camera_id}")
        except Exception as e:
            logger.error(f"Failed to load camera {camera_id}: {e}")
            raise ValueError(f"Invalid camera config for {camera_id}: {e}")
    
    # Extract system settings
    system_dict = config_dict.get('system', {})
    system_dict['cameras'] = validated_cameras
    
    try:
        system_config = SystemConfig(**system_dict)
        logger.info("System configuration validated successfully")
        return system_config
    except Exception as e:
        logger.error(f"Failed to validate system configuration: {e}")
        raise ValueError(f"Invalid system configuration: {e}")


def ensure_directories(config: SystemConfig) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: SystemConfig object
    """
    directories = [
        config.base_cache_dir,
        config.base_video_dir,
        config.log_dir,
        os.path.dirname(config.db_path),
    ]
    
    # Add camera-specific cache directories
    for camera_id in config.cameras:
        cache_path = os.path.join(config.base_cache_dir, camera_id)
        directories.append(cache_path)
        video_path = os.path.join(config.base_video_dir, camera_id)
        directories.append(video_path)
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def get_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    Get system configuration with directory validation.
    
    Args:
        config_path: Path to YAML config file. If None, tries default locations.
        
    Returns:
        SystemConfig object
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            "configs/cameras.yaml",
            "./cameras.yaml",
            "/etc/multi_camera_event/cameras.yaml",
        ]
        config_path = None
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(
                f"No configuration file found. Tried: {default_paths}. "
                "Please provide a valid config path."
            )
    
    config = load_system_config(config_path)
    ensure_directories(config)
    
    logger.info(f"Configuration loaded and validated. "
                f"Total cameras: {len(config.cameras)}")
    
    return config
