# detection/__init__.py
"""Detection utilities for Football Pass Counter."""

from .ball_detection import fallback_detect_ball
from .yolo_processor import YOLOProcessor

__all__ = ['fallback_detect_ball', 'YOLOProcessor']
