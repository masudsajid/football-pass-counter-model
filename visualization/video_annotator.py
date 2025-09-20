# visualization/video_annotator.py
"""
Video annotation utilities for drawing bounding boxes, labels, and overlays.
Handles all visual elements added to video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from config.settings import (
    PLAYER_COLOR,
    BALL_COLOR,
    TEXT_COLOR
)


class VideoAnnotator:
    """
    Handles video frame annotations including bounding boxes, labels, and overlays.
    
    Features:
    - Player bounding box and ID annotations
    - Ball position marking
    - Pass counter and possession overlays
    - Customizable colors and fonts
    """
    
    def __init__(self):
        """Initialize video annotator with default settings."""
        self.player_color = PLAYER_COLOR
        self.ball_color = BALL_COLOR
        self.text_color = TEXT_COLOR
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
    
    def annotate_players(self, 
                        frame: np.ndarray, 
                        tracks: List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int]]]) -> np.ndarray:
        """
        Annotate player bounding boxes and IDs on the frame.
        
        Args:
            frame: Video frame to annotate
            tracks: List of (track_id, bbox, centroid) tuples
            
        Returns:
            Annotated frame
        """
        for tid, bbox, centroid in tracks:
            x1, y1, x2, y2 = bbox
            cx, cy = centroid
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.player_color, self.thickness)
            
            # Draw player ID label
            label = f"P{tid}"
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
            
            # Position label above bounding box
            label_x = x1
            label_y = y1 - 8 if y1 - 8 > label_size[1] else y1 + label_size[1] + 8
            
            # Draw label background for better visibility
            cv2.rectangle(frame, 
                         (label_x - 2, label_y - label_size[1] - 2),
                         (label_x + label_size[0] + 2, label_y + 2),
                         self.player_color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (label_x, label_y), 
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
            
            # Draw centroid point
            cv2.circle(frame, (cx, cy), 3, self.player_color, -1)
        
        return frame
    
    def annotate_ball(self, frame: np.ndarray, ball_center: Optional[Tuple[int, int]]) -> np.ndarray:
        """
        Annotate ball position on the frame.
        
        Args:
            frame: Video frame to annotate
            ball_center: Ball center coordinates or None
            
        Returns:
            Annotated frame
        """
        if ball_center is None:
            return frame
        
        bx, by = int(ball_center[0]), int(ball_center[1])
        
        # Draw ball circle
        cv2.circle(frame, (bx, by), 6, self.ball_color, -1)
        cv2.circle(frame, (bx, by), 8, (255, 255, 255), 1)  # White border
        
        # Draw ball label
        label = "Ball"
        label_pos = (bx + 10, by - 5)
        
        # Add text background for visibility
        label_size = cv2.getTextSize(label, self.font, 0.5, 1)[0]
        cv2.rectangle(frame,
                     (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                     (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                     self.ball_color, -1)
        
        cv2.putText(frame, label, label_pos, self.font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def annotate_possession(self, 
                           frame: np.ndarray, 
                           current_possessor: Optional[int],
                           frame_height: int) -> np.ndarray:
        """
        Annotate current ball possession on the frame.
        
        Args:
            frame: Video frame to annotate
            current_possessor: Current possessor player ID or None
            frame_height: Frame height for positioning
            
        Returns:
            Annotated frame
        """
        if current_possessor is None:
            return frame
        
        # Position text at bottom of frame
        text = f"Possessor: P{current_possessor}"
        text_pos = (30, frame_height - 40)
        
        # Add background for better visibility
        text_size = cv2.getTextSize(text, self.font, 0.9, 2)[0]
        cv2.rectangle(frame,
                     (text_pos[0] - 10, text_pos[1] - text_size[1] - 10),
                     (text_pos[0] + text_size[0] + 10, text_pos[1] + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text, text_pos, self.font, 0.9, self.text_color, 2)
        
        return frame
    
    def annotate_pass_counter(self, frame: np.ndarray, pass_count: int) -> np.ndarray:
        """
        Annotate pass counter on the frame.
        
        Args:
            frame: Video frame to annotate
            pass_count: Current pass count
            
        Returns:
            Annotated frame
        """
        text = f"Passes: {pass_count}"
        text_pos = (30, 40)
        
        # Add background for better visibility
        text_size = cv2.getTextSize(text, self.font, 1.2, 3)[0]
        cv2.rectangle(frame,
                     (text_pos[0] - 10, text_pos[1] - text_size[1] - 10),
                     (text_pos[0] + text_size[0] + 10, text_pos[1] + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text, text_pos, self.font, 1.2, self.text_color, 3)
        
        return frame
    
    def annotate_frame_info(self, 
                           frame: np.ndarray, 
                           frame_idx: int, 
                           fps: float,
                           frame_width: int) -> np.ndarray:
        """
        Annotate frame information (frame number, timestamp).
        
        Args:
            frame: Video frame to annotate
            frame_idx: Current frame index
            fps: Video frames per second
            frame_width: Frame width for positioning
            
        Returns:
            Annotated frame
        """
        timestamp = frame_idx / fps if fps > 0 else 0
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        text = f"Frame: {frame_idx} | Time: {minutes:02d}:{seconds:02d}"
        text_pos = (frame_width - 300, 30)
        
        # Add background
        text_size = cv2.getTextSize(text, self.font, 0.5, 1)[0]
        cv2.rectangle(frame,
                     (text_pos[0] - 5, text_pos[1] - text_size[1] - 5),
                     (text_pos[0] + text_size[0] + 5, text_pos[1] + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text, text_pos, self.font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def annotate_debug_info(self, 
                           frame: np.ndarray, 
                           debug_info: dict,
                           frame_height: int) -> np.ndarray:
        """
        Annotate debug information on the frame.
        
        Args:
            frame: Video frame to annotate
            debug_info: Dictionary containing debug information
            frame_height: Frame height for positioning
            
        Returns:
            Annotated frame
        """
        y_offset = frame_height - 120
        line_height = 20
        
        for key, value in debug_info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), 
                       self.font, 0.4, (255, 255, 0), 1)
            y_offset += line_height
        
        return frame
    
    def set_colors(self, player_color: Tuple[int, int, int], 
                  ball_color: Tuple[int, int, int], 
                  text_color: Tuple[int, int, int]) -> None:
        """
        Set custom colors for annotations.
        
        Args:
            player_color: BGR color for player annotations
            ball_color: BGR color for ball annotations
            text_color: BGR color for text annotations
        """
        self.player_color = player_color
        self.ball_color = ball_color
        self.text_color = text_color
