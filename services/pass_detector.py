# services/pass_detector.py
"""
Pass detection service using ball possession analysis.
Implements logic for detecting when passes occur between players.
"""

from typing import List, Tuple, Optional

from config.settings import MIN_HOLD_FRAMES, BBOX_MARGIN


class PassDetector:
    """
    Detects passes between players based on ball possession changes.
    
    Features:
    - Ball-player containment detection
    - Temporal smoothing to reduce false positives
    - Possession change tracking with hold frames
    - Pass counting with validation
    """
    
    def __init__(self, min_hold_frames: int = MIN_HOLD_FRAMES, bbox_margin: int = BBOX_MARGIN):
        """
        Initialize pass detector.
        
        Args:
            min_hold_frames: Minimum frames to confirm possession change
            bbox_margin: Margin for ball-player containment check
        """
        self.min_hold_frames = min_hold_frames
        self.bbox_margin = bbox_margin
        
        # State tracking
        self.pass_counter = 0
        self.current_possessor: Optional[int] = None
        self.candidate_possessor: Optional[int] = None
        self.candidate_hold = 0
        
        # History tracking
        self.possession_history = []
        self.pass_events = []
    
    def detect_possession(self, 
                         ball_center: Optional[Tuple[int, int]], 
                         tracks: List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int]]],
                         frame_idx: int) -> Tuple[Optional[int], bool]:
        """
        Detect current ball possessor and check for pass events.
        
        Args:
            ball_center: Ball center coordinates (x, y) or None
            tracks: List of (track_id, bbox, centroid) tuples for players
            frame_idx: Current frame index
            
        Returns:
            Tuple of (current_possessor_id, pass_occurred)
        """
        if ball_center is None:
            return self._handle_no_ball_detection(frame_idx)
        
        # Find player containing the ball
        new_possessor = self._find_ball_possessor(ball_center, tracks)
        
        # Update possession state and detect passes
        pass_occurred = self._update_possession_state(new_possessor, frame_idx)
        
        return self.current_possessor, pass_occurred
    
    def _find_ball_possessor(self, 
                            ball_center: Tuple[int, int], 
                            tracks: List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int]]]) -> Optional[int]:
        """
        Find which player currently possesses the ball based on containment.
        
        Args:
            ball_center: Ball center coordinates
            tracks: Player tracking information
            
        Returns:
            Player ID who possesses the ball, or None
        """
        bx, by = int(ball_center[0]), int(ball_center[1])
        
        for tid, bbox, centroid in tracks:
            x1, y1, x2, y2 = bbox
            
            # Check if ball is within expanded bounding box
            if ((x1 - self.bbox_margin) <= bx <= (x2 + self.bbox_margin) and
                (y1 - self.bbox_margin) <= by <= (y2 + self.bbox_margin)):
                return tid
        
        return None
    
    def _handle_no_ball_detection(self, frame_idx: int) -> Tuple[Optional[int], bool]:
        """Handle case when ball is not detected."""
        self.candidate_possessor = None
        self.candidate_hold = 0
        
        # Record in history
        self.possession_history.append({
            'frame': frame_idx,
            'possessor': self.current_possessor,
            'ball_detected': False
        })
        
        return self.current_possessor, False
    
    def _update_possession_state(self, new_possessor: Optional[int], frame_idx: int) -> bool:
        """
        Update possession state with temporal smoothing and detect passes.
        
        Args:
            new_possessor: Newly detected possessor ID
            frame_idx: Current frame index
            
        Returns:
            True if a pass occurred, False otherwise
        """
        pass_occurred = False
        
        if new_possessor is None:
            # No clear possessor
            self.candidate_possessor = None
            self.candidate_hold = 0
        else:
            if self.current_possessor is None:
                # No current possessor, establish new one
                pass_occurred = self._establish_new_possession(new_possessor, frame_idx)
            else:
                # Current possessor exists, check for changes
                pass_occurred = self._check_possession_change(new_possessor, frame_idx)
        
        # Record in history
        self.possession_history.append({
            'frame': frame_idx,
            'possessor': self.current_possessor,
            'candidate': self.candidate_possessor,
            'candidate_hold': self.candidate_hold,
            'ball_detected': True
        })
        
        return pass_occurred
    
    def _establish_new_possession(self, new_possessor: int, frame_idx: int) -> bool:
        """Establish possession when no current possessor exists."""
        if self.candidate_possessor == new_possessor:
            self.candidate_hold += 1
        else:
            self.candidate_possessor = new_possessor
            self.candidate_hold = 1
        
        if self.candidate_hold >= self.min_hold_frames:
            self.current_possessor = self.candidate_possessor
            self.candidate_possessor = None
            self.candidate_hold = 0
            
            # Record possession establishment (not counted as a pass)
            self.pass_events.append({
                'frame': frame_idx,
                'from_player': None,
                'to_player': self.current_possessor,
                'type': 'establishment'
            })
        
        return False
    
    def _check_possession_change(self, new_possessor: int, frame_idx: int) -> bool:
        """Check for possession changes and detect passes."""
        if new_possessor == self.current_possessor:
            # Same possessor, reset candidate
            self.candidate_possessor = None
            self.candidate_hold = 0
            return False
        
        # Different possessor detected
        if self.candidate_possessor == new_possessor:
            self.candidate_hold += 1
        else:
            self.candidate_possessor = new_possessor
            self.candidate_hold = 1
        
        if self.candidate_hold >= self.min_hold_frames:
            # Possession change confirmed
            if new_possessor != self.current_possessor:
                # Valid pass detected
                self.pass_counter += 1
                
                # Record pass event
                self.pass_events.append({
                    'frame': frame_idx,
                    'from_player': self.current_possessor,
                    'to_player': new_possessor,
                    'type': 'pass',
                    'pass_number': self.pass_counter
                })
                
                self.current_possessor = new_possessor
                self.candidate_possessor = None
                self.candidate_hold = 0
                
                return True
        
        return False
    
    def get_pass_count(self) -> int:
        """Get current pass count."""
        return self.pass_counter
    
    def get_current_possessor(self) -> Optional[int]:
        """Get current ball possessor."""
        return self.current_possessor
    
    def get_pass_events(self) -> List[dict]:
        """Get list of all pass events."""
        return self.pass_events.copy()
    
    def get_possession_history(self) -> List[dict]:
        """Get full possession history."""
        return self.possession_history.copy()
    
    def reset(self) -> None:
        """Reset detector to initial state."""
        self.pass_counter = 0
        self.current_possessor = None
        self.candidate_possessor = None
        self.candidate_hold = 0
        self.possession_history.clear()
        self.pass_events.clear()
    
    def get_statistics(self) -> dict:
        """Get pass detection statistics."""
        if not self.pass_events:
            return {
                'total_passes': 0,
                'unique_players': 0,
                'average_possession_duration': 0
            }
        
        # Count unique players involved in passes
        unique_players = set()
        for event in self.pass_events:
            if event['from_player'] is not None:
                unique_players.add(event['from_player'])
            if event['to_player'] is not None:
                unique_players.add(event['to_player'])
        
        # Calculate average possession duration (frames between passes)
        pass_frames = [event['frame'] for event in self.pass_events if event['type'] == 'pass']
        if len(pass_frames) > 1:
            durations = [pass_frames[i+1] - pass_frames[i] for i in range(len(pass_frames)-1)]
            avg_duration = sum(durations) / len(durations)
        else:
            avg_duration = 0
        
        return {
            'total_passes': self.pass_counter,
            'unique_players': len(unique_players),
            'average_possession_duration': avg_duration,
            'pass_events': len(self.pass_events)
        }
