"""Utilities package for NKU Software Engineering project."""
from .camera_manager import CameraManager, get_camera_manager
from .tools import speecher_player # Import speecher_player instance

__all__ = [
    'speecher_player', # Export instance
    'get_camera_manager', 'CameraManager'
]
