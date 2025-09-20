# detection/ball_detection.py
"""
Fallback ball detection using color-based computer vision techniques.
Used when YOLO fails to detect the ball or as a backup method.
"""

import math
import cv2
import numpy as np
from typing import Optional, Tuple, List

from config.settings import (
    BALL_HSV_LOWER,
    BALL_HSV_UPPER,
    BALL_MIN_AREA,
    BALL_MAX_AREA,
    BALL_SEARCH_RADIUS
)


def fallback_detect_ball(frame: np.ndarray, 
                        last_center: Optional[Tuple[int, int]] = None,
                        search_radius: int = BALL_SEARCH_RADIUS) -> Optional[Tuple[int, int]]:
    """
    Detect ball using color-based computer vision as fallback when YOLO fails.
    
    This function uses HSV color space filtering to find white/light colored objects
    that match the expected characteristics of a football (soccer ball).
    
    Args:
        frame: Input video frame (BGR format)
        last_center: Last known ball position for focused search
        search_radius: Radius around last_center to search within
        
    Returns:
        Ball center coordinates (x, y) or None if not found
    """
    h, w = frame.shape[:2]
    
    # Define region of interest based on last known position
    if last_center is not None:
        x, y = last_center
        x1 = max(0, x - search_radius)
        y1 = max(0, y - search_radius)
        x2 = min(w, x + search_radius)
        y2 = min(h, y + search_radius)
        roi = frame[y1:y2, x1:x2]
        offset = (x1, y1)
    else:
        roi = frame
        offset = (0, 0)
    
    if roi is None or roi.size == 0:
        return None
    
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Create mask for white/light colored objects (typical ball colors)
    lower_bound = np.array(BALL_HSV_LOWER)
    upper_bound = np.array(BALL_HSV_UPPER)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Evaluate each contour as a potential ball candidate
    candidates = []
    for contour in contours:
        candidate = _evaluate_ball_candidate(contour, offset)
        if candidate is not None:
            candidates.append(candidate)
    
    if not candidates:
        return None
    
    # Select best candidate based on proximity or circularity
    return _select_best_candidate(candidates, last_center)


def _evaluate_ball_candidate(contour: np.ndarray, offset: Tuple[int, int]) -> Optional[Tuple[int, int, float, float]]:
    """
    Evaluate a contour as a potential ball candidate.
    
    Args:
        contour: OpenCV contour
        offset: Offset to apply to coordinates (for ROI)
        
    Returns:
        Tuple of (cx, cy, area, circularity) or None if not valid
    """
    area = cv2.contourArea(contour)
    
    # Filter by area constraints
    if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
        return None
    
    # Calculate circularity (measure of how round the object is)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    
    circularity = 4 * math.pi * (area / (perimeter * perimeter + 1e-9))
    
    # Calculate centroid
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    
    cx = int(moments["m10"] / moments["m00"]) + offset[0]
    cy = int(moments["m01"] / moments["m00"]) + offset[1]
    
    return (cx, cy, area, circularity)


def _select_best_candidate(candidates: List[Tuple[int, int, float, float]], 
                          last_center: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Select the best ball candidate from a list of candidates.
    
    Args:
        candidates: List of (cx, cy, area, circularity) tuples
        last_center: Last known ball position for proximity-based selection
        
    Returns:
        Best candidate center coordinates (x, y)
    """
    if last_center is not None:
        # Prefer candidates closest to last known position
        lx, ly = last_center
        candidates.sort(key=lambda c: math.hypot(c[0] - lx, c[1] - ly))
        return (candidates[0][0], candidates[0][1])
    else:
        # Prefer candidates with best circularity and larger area
        candidates.sort(key=lambda c: (-c[3], c[2]))  # Sort by circularity desc, area desc
        return (candidates[0][0], candidates[0][1])


def enhance_ball_detection(frame: np.ndarray) -> np.ndarray:
    """
    Enhance frame for better ball detection by applying preprocessing.
    
    Args:
        frame: Input frame
        
    Returns:
        Enhanced frame
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    
    # Enhance contrast
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced
