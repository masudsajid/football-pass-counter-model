# trackers/player_tracker.py
"""
Player tracking module using spatial matching and missing frame tolerance.
Handles multi-object tracking for football players with re-identification.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any

from config.settings import (
    PLAYER_MISSING_TOLERANCE,
    MAX_PLAYER_MATCH_DIST,
    MAX_PLAYERS
)


class PlayerTracker:
    """
    Tracks multiple players across video frames using spatial matching.
    
    Features:
    - Spatial distance-based matching between detections and existing tracks
    - Missing frame tolerance to handle temporary occlusions
    - Maximum player limit to prevent over-tracking
    - Greedy assignment algorithm for optimal matching
    """
    
    def __init__(self, 
                 max_missing: int = PLAYER_MISSING_TOLERANCE,
                 max_dist: int = MAX_PLAYER_MATCH_DIST,
                 max_players: int = MAX_PLAYERS):
        """
        Initialize the player tracker.
        
        Args:
            max_missing: Maximum frames a track can be missing before deletion
            max_dist: Maximum distance for matching detections to tracks
            max_players: Maximum number of players to track simultaneously
        """
        self.tracks: Dict[int, Dict[str, Any]] = {}  # tid -> track info
        self.next_id = 1
        self.max_missing = max_missing
        self.max_dist = max_dist
        self.max_players = max_players
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int]]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of bounding boxes (x1, y1, x2, y2)
            
        Returns:
            List of (track_id, bbox, centroid) tuples
        """
        assigned = {}
        new_centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in detections]
        new_bboxes = detections.copy()
        
        track_ids = list(self.tracks.keys())
        
        # Perform spatial matching if we have both tracks and detections
        if track_ids and new_centroids:
            assigned = self._perform_spatial_matching(track_ids, new_centroids)
        
        # Update assigned tracks and track used detections
        updated_ids, used_detections = self._update_assigned_tracks(assigned, new_bboxes, new_centroids)
        
        # Handle missing tracks
        self._handle_missing_tracks(updated_ids)
        
        # Create new tracks for unmatched detections
        self._create_new_tracks(new_bboxes, new_centroids, used_detections)
        
        # Return current tracks
        return self._get_current_tracks()
    
    def _perform_spatial_matching(self, track_ids: List[int], new_centroids: List[Tuple[int, int]]) -> Dict[int, int]:
        """Perform spatial matching between tracks and detections using greedy assignment."""
        assigned = {}
        
        # Build distance matrix: tracks x detections
        dists = []
        for tid in track_ids:
            tx, ty = self.tracks[tid]['centroid']
            row = [math.hypot(tx - nx, ty - ny) for (nx, ny) in new_centroids]
            dists.append(row)
        dists = np.array(dists)
        
        # Greedy assignment (smallest distance first)
        while True:
            if dists.size == 0:
                break
                
            i, j = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
            min_val = dists[i, j]
            
            if min_val > self.max_dist:
                break
                
            tid = track_ids[i]
            if tid in assigned or j in assigned.values():
                dists[i, j] = np.inf
                if np.isinf(dists).all():
                    break
                continue
                
            assigned[tid] = j
            dists[i, :] = np.inf
            dists[:, j] = np.inf
            
            if np.isinf(dists).all():
                break
        
        return assigned
    
    def _update_assigned_tracks(self, assigned: Dict[int, int], 
                              new_bboxes: List[Tuple[int, int, int, int]], 
                              new_centroids: List[Tuple[int, int]]) -> Tuple[set, set]:
        """Update tracks that were successfully matched to detections."""
        updated_ids = set()
        used_detections = set()
        
        for tid, j in assigned.items():
            bbox = new_bboxes[j]
            cx, cy = new_centroids[j]
            
            self.tracks[tid]['bbox'] = bbox
            self.tracks[tid]['centroid'] = (cx, cy)
            self.tracks[tid]['missing'] = 0
            
            updated_ids.add(tid)
            used_detections.add(j)
        
        return updated_ids, used_detections
    
    def _handle_missing_tracks(self, updated_ids: set) -> None:
        """Handle tracks that weren't updated (increment missing count, delete if necessary)."""
        for tid in list(self.tracks.keys()):
            if tid not in updated_ids:
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > self.max_missing:
                    del self.tracks[tid]
    
    def _create_new_tracks(self, new_bboxes: List[Tuple[int, int, int, int]], 
                          new_centroids: List[Tuple[int, int]], 
                          used_detections: set) -> None:
        """Create new tracks for unmatched detections (respecting max_players limit)."""
        for j, bbox in enumerate(new_bboxes):
            if j in used_detections:
                continue
                
            if self.next_id <= self.max_players:
                cx, cy = new_centroids[j]
                tid = self.next_id
                self.next_id += 1
                
                self.tracks[tid] = {
                    'bbox': bbox,
                    'centroid': (cx, cy),
                    'missing': 0
                }
    
    def _get_current_tracks(self) -> List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int]]]:
        """Get current tracks in the expected format."""
        return [(tid, info['bbox'], info['centroid']) for tid, info in self.tracks.items()]
    
    def get_track_count(self) -> int:
        """Get the current number of active tracks."""
        return len(self.tracks)
    
    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.tracks.clear()
        self.next_id = 1
