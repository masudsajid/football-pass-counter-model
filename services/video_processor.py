# services/video_processor.py
"""
Main video processing service that orchestrates the entire analysis pipeline.
Coordinates detection, tracking, pass counting, and visualization.
"""

import time
import cv2
from pathlib import Path
from typing import Tuple, Dict, List
from collections import defaultdict

from config.settings import DEFAULT_FPS, VIDEO_CODEC
from detection import YOLOProcessor, fallback_detect_ball
from trackers import PlayerTracker, BallTracker
from services.pass_detector import PassDetector
from visualization import VideoAnnotator, ChartGenerator


class VideoProcessor:
    """
    Main service for processing football videos with pass counting.
    
    Orchestrates the entire pipeline:
    1. Video loading and setup
    2. Object detection (YOLO + fallback)
    3. Player and ball tracking
    4. Pass detection and counting
    5. Video annotation and output
    6. Statistics and visualization generation
    """
    
    def __init__(self, model_path: str):
        """
        Initialize video processor.
        
        Args:
            model_path: Path to YOLO model file
        """
        self.model_path = model_path
        
        # Initialize components
        self.yolo_processor = YOLOProcessor(model_path)
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.pass_detector = PassDetector()
        self.video_annotator = VideoAnnotator()
        self.chart_generator = ChartGenerator()
        
        # State tracking
        self.frame_history = []
        self.pass_history = []
        self.player_positions = defaultdict(list)
        
        # Video properties
        self.fps = DEFAULT_FPS
        self.frame_width = 0
        self.frame_height = 0
    
    def process_video(self, 
                     input_path: str, 
                     output_video_path: str, 
                     chart_path: str) -> Tuple[int, float, Dict, int, int, str, str]:
        """
        Process a football video with pass counting and tracking.
        
        Args:
            input_path: Path to input video file
            output_video_path: Path for output annotated video
            chart_path: Path for output pass chart
            
        Returns:
            Tuple of (pass_count, elapsed_time, player_positions, width, height, video_path, chart_path)
        """
        start_time = time.time()
        
        # Setup video capture and writer
        cap, out = self._setup_video_io(input_path, output_video_path)
        
        try:
            # Process video frame by frame
            self._process_frames(cap, out)
            
            # Generate final chart
            self.chart_generator.generate_pass_chart(
                self.frame_history, 
                self.pass_history, 
                chart_path
            )
            
        finally:
            cap.release()
            out.release()
        
        elapsed_time = time.time() - start_time
        pass_count = self.pass_detector.get_pass_count()
        
        return (
            pass_count, 
            elapsed_time, 
            dict(self.player_positions), 
            self.frame_width, 
            self.frame_height, 
            output_video_path, 
            chart_path
        )
    
    def _setup_video_io(self, input_path: str, output_path: str) -> Tuple[cv2.VideoCapture, cv2.VideoWriter]:
        """Setup video input and output."""
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        if not out.isOpened():
            raise ValueError(f"Could not open output video file: {output_path}")
        
        return cap, out
    
    def _process_frames(self, cap: cv2.VideoCapture, out: cv2.VideoWriter) -> None:
        """Process video frames one by one."""
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Detect objects using YOLO
            player_detections, ball_candidates = self.yolo_processor.detect_objects(frame)
            
            # Update player tracker
            tracks = self.player_tracker.update(player_detections)
            
            # Handle ball detection and tracking
            ball_center = self._process_ball_detection(frame, ball_candidates, frame_idx)
            
            # Detect passes
            current_possessor, pass_occurred = self.pass_detector.detect_possession(
                ball_center, tracks, frame_idx
            )
            
            # Record player positions for heatmaps
            self._record_player_positions(tracks)
            
            # Annotate frame
            frame = self._annotate_frame(frame, tracks, ball_center, current_possessor, frame_idx)
            
            # Write annotated frame
            out.write(frame)
            
            # Record history for charts
            self.frame_history.append(frame_idx)
            self.pass_history.append(self.pass_detector.get_pass_count())
    
    def _process_ball_detection(self, 
                               frame, 
                               ball_candidates: List, 
                               frame_idx: int) -> Tuple[int, int]:
        """Process ball detection with fallback methods."""
        ball_center = None
        
        if ball_candidates:
            # Use highest confidence YOLO detection
            ball_candidates.sort(key=lambda x: -x[2])
            bc = ball_candidates[0]
            ball_center = (bc[0], bc[1])
            self.ball_tracker.update_with_detection(ball_center, frame_idx)
        else:
            # Try fallback detection
            last_pred = self.ball_tracker.predict()
            fallback_center = fallback_detect_ball(frame, last_center=last_pred)
            
            if fallback_center is None and self.ball_tracker.cx is None:
                # Try full-frame fallback detection
                fallback_center = fallback_detect_ball(frame, last_center=None)
            
            if fallback_center is not None:
                ball_center = fallback_center
                self.ball_tracker.update_with_detection(ball_center, frame_idx)
            else:
                # Use prediction if available
                pred = self.ball_tracker.predict()
                if pred is not None:
                    ball_center = pred
        
        return ball_center
    
    def _record_player_positions(self, tracks: List) -> None:
        """Record player positions for heatmap generation."""
        for tid, bbox, centroid in tracks:
            cx, cy = centroid
            if 0 <= cx < self.frame_width and 0 <= cy < self.frame_height:
                self.player_positions[tid].append((cx, cy))
    
    def _annotate_frame(self, 
                       frame, 
                       tracks: List, 
                       ball_center, 
                       current_possessor, 
                       frame_idx: int):
        """Annotate frame with all visual elements."""
        # Annotate players
        frame = self.video_annotator.annotate_players(frame, tracks)
        
        # Annotate ball
        frame = self.video_annotator.annotate_ball(frame, ball_center)
        
        # Annotate possession
        frame = self.video_annotator.annotate_possession(
            frame, current_possessor, self.frame_height
        )
        
        # Annotate pass counter
        frame = self.video_annotator.annotate_pass_counter(
            frame, self.pass_detector.get_pass_count()
        )
        
        # Annotate frame info
        frame = self.video_annotator.annotate_frame_info(
            frame, frame_idx, self.fps, self.frame_width
        )
        
        return frame
    
    def get_processing_statistics(self) -> dict:
        """Get comprehensive processing statistics."""
        pass_stats = self.pass_detector.get_statistics()
        
        return {
            'video_info': {
                'fps': self.fps,
                'width': self.frame_width,
                'height': self.frame_height,
                'total_frames': len(self.frame_history)
            },
            'detection_info': self.yolo_processor.get_model_info(),
            'tracking_info': {
                'active_players': self.player_tracker.get_track_count(),
                'ball_tracking': self.ball_tracker.is_tracking()
            },
            'pass_statistics': pass_stats,
            'player_positions': {
                pid: len(positions) for pid, positions in self.player_positions.items()
            }
        }
    
    def reset(self) -> None:
        """Reset processor to initial state."""
        self.player_tracker.reset()
        self.ball_tracker.reset()
        self.pass_detector.reset()
        
        self.frame_history.clear()
        self.pass_history.clear()
        self.player_positions.clear()
        
        self.fps = DEFAULT_FPS
        self.frame_width = 0
        self.frame_height = 0
