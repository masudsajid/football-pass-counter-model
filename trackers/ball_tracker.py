# trackers/ball_tracker.py
"""
Ball tracking module with smoothing and velocity prediction.
Handles ball position tracking with temporal smoothing and motion prediction.
"""

from typing import Optional, Tuple
from config.settings import SMOOTH_ALPHA


class BallTracker:
    """
    Tracks ball position with smoothing and velocity-based prediction.
    
    Features:
    - Exponential smoothing for position stability
    - Velocity estimation for motion prediction
    - Missing frame tolerance with prediction capability
    - Temporal consistency tracking
    """
    
    def __init__(self, alpha: float = SMOOTH_ALPHA):
        """
        Initialize the ball tracker.
        
        Args:
            alpha: Smoothing factor for exponential smoothing (0-1)
        """
        self.cx: Optional[int] = None
        self.cy: Optional[int] = None
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.last_frame_seen: int = -9999
        self.missing: int = 0
        self.alpha: float = alpha
        self.velocity_alpha: float = 0.3  # Separate alpha for velocity smoothing
    
    def update_with_detection(self, center: Optional[Tuple[int, int]], frame_idx: int) -> Optional[Tuple[int, int]]:
        """
        Update tracker with a new ball detection.
        
        Args:
            center: Ball center coordinates (x, y) or None if not detected
            frame_idx: Current frame index
            
        Returns:
            Smoothed ball position or None if no detection
        """
        if center is None:
            self.missing += 1
            return None
        
        x, y = center
        
        if self.cx is None:
            # First detection - initialize position
            self.cx, self.cy = x, y
            self.vx, self.vy = 0.0, 0.0
        else:
            # Update velocity with smoothing
            self.vx = (1 - self.velocity_alpha) * self.vx + self.velocity_alpha * (x - self.cx)
            self.vy = (1 - self.velocity_alpha) * self.vy + self.velocity_alpha * (y - self.cy)
            
            # Update position with exponential smoothing
            self.cx = int(self.alpha * x + (1 - self.alpha) * self.cx)
            self.cy = int(self.alpha * y + (1 - self.alpha) * self.cy)
        
        self.last_frame_seen = frame_idx
        self.missing = 0
        
        return (self.cx, self.cy)
    
    def predict(self) -> Optional[Tuple[int, int]]:
        """
        Predict ball position based on current position and velocity.
        
        Returns:
            Predicted ball position or None if no previous position
        """
        if self.cx is None:
            return None
        
        px = int(self.cx + self.vx)
        py = int(self.cy + self.vy)
        
        return (px, py)
    
    def get_current_position(self) -> Optional[Tuple[int, int]]:
        """Get the current tracked position."""
        if self.cx is None or self.cy is None:
            return None
        return (self.cx, self.cy)
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity components."""
        return (self.vx, self.vy)
    
    def get_missing_count(self) -> int:
        """Get number of consecutive missing frames."""
        return self.missing
    
    def is_tracking(self) -> bool:
        """Check if the tracker has a valid position."""
        return self.cx is not None and self.cy is not None
    
    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.cx = None
        self.cy = None
        self.vx = 0.0
        self.vy = 0.0
        self.last_frame_seen = -9999
        self.missing = 0
