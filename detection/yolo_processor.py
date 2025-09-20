# detection/yolo_processor.py
"""
YOLO model wrapper for object detection in football videos.
Handles model loading, inference, and result processing.
"""

from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO

from config.settings import (
    YOLO_CONF,
    PLAYER_CONF_THR,
    BALL_CONF_THR,
    YOLO_IMAGE_SIZE
)


class YOLOProcessor:
    """
    Wrapper class for YOLO model to detect players and balls in football videos.
    
    Features:
    - Model loading and initialization
    - Confidence-based filtering for different object types
    - Batch processing capability
    - Result formatting for downstream processing
    """
    
    def __init__(self, model_path: str, conf_threshold: float = YOLO_CONF):
        """
        Initialize YOLO processor.
        
        Args:
            model_path: Path to the YOLO model file (.pt)
            conf_threshold: Global confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model: Optional[YOLO] = None
        self.player_conf_thr = PLAYER_CONF_THR
        self.ball_conf_thr = BALL_CONF_THR
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model from the specified path."""
        try:
            self.model = YOLO(self.model_path)
            print(f"Successfully loaded YOLO model from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {e}")
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, float, Tuple[int, int, int, int]]]]:
        """
        Detect players and balls in a single frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (player_detections, ball_candidates) where:
            - player_detections: List of (x1, y1, x2, y2) bounding boxes
            - ball_candidates: List of (cx, cy, confidence, bbox) tuples
        """
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, imgsz=YOLO_IMAGE_SIZE)[0]
        
        player_detections = []
        ball_candidates = []
        
        # Process each detection
        for box in results.boxes:
            detection_info = self._extract_detection_info(box)
            if detection_info is None:
                continue
            
            cls_id, label, confidence, bbox = detection_info
            
            # Classify detection type and apply appropriate confidence thresholds
            if self._is_player_detection(label, confidence):
                player_detections.append(bbox)
            elif self._is_ball_detection(label, confidence):
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                ball_candidates.append((cx, cy, confidence, bbox))
        
        return player_detections, ball_candidates
    
    def _extract_detection_info(self, box) -> Optional[Tuple[int, str, float, Tuple[int, int, int, int]]]:
        """
        Extract information from a YOLO detection box.
        
        Args:
            box: YOLO detection box object
            
        Returns:
            Tuple of (class_id, label, confidence, bbox) or None if invalid
        """
        try:
            # Extract class information
            cls_id = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
            label = self.model.names[cls_id] if cls_id in self.model.names else str(cls_id)
            
            # Extract confidence
            confidence = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(box.conf)
            
            # Extract bounding box coordinates
            xy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else box.xyxy[0].numpy()
            x1, y1, x2, y2 = [int(v) for v in xy]
            
            return cls_id, label, confidence, (x1, y1, x2, y2)
        
        except Exception as e:
            print(f"Error extracting detection info: {e}")
            return None
    
    def _is_player_detection(self, label: str, confidence: float) -> bool:
        """Check if detection is a valid player."""
        player_labels = {"player", "person"}
        return (label.lower() in player_labels and confidence >= self.player_conf_thr)
    
    def _is_ball_detection(self, label: str, confidence: float) -> bool:
        """Check if detection is a valid ball."""
        ball_labels = {"ball", "sports ball", "soccerball"}
        return (label.lower() in ball_labels and confidence >= self.ball_conf_thr)
    
    def set_confidence_thresholds(self, player_conf: float, ball_conf: float) -> None:
        """
        Update confidence thresholds for different object types.
        
        Args:
            player_conf: Confidence threshold for player detections
            ball_conf: Confidence threshold for ball detections
        """
        self.player_conf_thr = player_conf
        self.ball_conf_thr = ball_conf
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "class_names": list(self.model.names.values()) if self.model.names else [],
            "confidence_threshold": self.conf_threshold,
            "player_confidence": self.player_conf_thr,
            "ball_confidence": self.ball_conf_thr
        }
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, float, Tuple[int, int, int, int]]]]]:
        """
        Detect objects in multiple frames (batch processing).
        
        Args:
            frames: List of video frames
            
        Returns:
            List of detection results for each frame
        """
        results = []
        for frame in frames:
            frame_results = self.detect_objects(frame)
            results.append(frame_results)
        return results
