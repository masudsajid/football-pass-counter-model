# visualization/heatmap_generator.py
"""
Heatmap generation utilities for player movement visualization.
Creates density maps and calculates distance metrics.
"""

import math
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from config.settings import (
    HEATMAP_BINS,
    HEATMAP_DPI,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M
)


class HeatmapGenerator:
    """
    Generates heatmaps and movement analysis for football players.
    
    Features:
    - 2D density heatmap generation
    - Distance calculation in real-world coordinates
    - Multiple visualization styles
    - Configurable binning and resolution
    """
    
    def __init__(self, bins: int = HEATMAP_BINS, dpi: int = HEATMAP_DPI):
        """
        Initialize heatmap generator.
        
        Args:
            bins: Number of bins for histogram (resolution)
            dpi: DPI for saved images
        """
        self.bins = bins
        self.dpi = dpi
    
    def generate_player_heatmap(self, 
                               points: List[Tuple[int, int]], 
                               frame_width: int, 
                               frame_height: int, 
                               output_path: str,
                               player_id: Optional[int] = None) -> Optional[Tuple[str, float]]:
        """
        Generate heatmap for a player's movement and calculate distance covered.
        
        Args:
            points: List of (x, y) coordinates in pixel space
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
            output_path: Path to save the heatmap image
            player_id: Player ID for title (optional)
            
        Returns:
            Tuple of (saved_path, distance_in_meters) or None if no points
        """
        if not points:
            return None
        
        # Calculate distance covered in real-world coordinates
        distance_m = self._calculate_distance_covered(points, frame_width, frame_height)
        
        # Extract coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        # Create 2D histogram (heatmap)
        heatmap, xedges, yedges = np.histogram2d(
            xs, ys, 
            bins=self.bins, 
            range=[[0, frame_width], [0, frame_height]]
        )
        heatmap = heatmap.T  # Transpose for correct orientation
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.imshow(
            heatmap, 
            origin='lower', 
            cmap='magma', 
            interpolation='bilinear',
            extent=[0, frame_width, 0, frame_height], 
            aspect='auto'
        )
        
        plt.colorbar(label='Presence Density')
        
        # Set title based on player ID
        title = f'Player Movement Heatmap (Distance: {distance_m:.2f}m)'
        if player_id is not None:
            title = f'Player P{player_id} Movement Heatmap (Distance: {distance_m:.2f}m)'
        
        plt.title(title)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path, distance_m
    
    def _calculate_distance_covered(self, points: List[Tuple[int, int]], 
                                  frame_width: int, frame_height: int) -> float:
        """
        Calculate total distance covered by converting pixel coordinates to meters.
        
        Assumes standard football pitch dimensions (120m x 80m).
        
        Args:
            points: List of (x, y) pixel coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Distance covered in meters
        """
        if len(points) < 2:
            return 0.0
        
        total_distance = 0.0
        
        for i in range(1, len(points)):
            # Calculate pixel displacement
            dx_px = points[i][0] - points[i-1][0]
            dy_px = points[i][1] - points[i-1][1]
            
            # Convert to real-world coordinates (meters)
            dx_m = (dx_px / frame_width) * PITCH_LENGTH_M
            dy_m = (dy_px / frame_height) * PITCH_WIDTH_M
            
            # Add Euclidean distance
            distance_segment = math.hypot(dx_m, dy_m)
            total_distance += distance_segment
        
        return total_distance
    
    def generate_team_heatmap(self, 
                             player_positions: dict, 
                             frame_width: int, 
                             frame_height: int, 
                             output_path: str) -> Optional[str]:
        """
        Generate combined heatmap for all players.
        
        Args:
            player_positions: Dict of player_id -> list of (x, y) positions
            frame_width: Video frame width
            frame_height: Video frame height
            output_path: Path to save combined heatmap
            
        Returns:
            Path to saved heatmap or None if no data
        """
        if not player_positions:
            return None
        
        # Combine all positions
        all_points = []
        for positions in player_positions.values():
            all_points.extend(positions)
        
        if not all_points:
            return None
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        # Create combined heatmap
        heatmap, xedges, yedges = np.histogram2d(
            xs, ys, 
            bins=self.bins, 
            range=[[0, frame_width], [0, frame_height]]
        )
        heatmap = heatmap.T
        
        plt.figure(figsize=(12, 8))
        plt.imshow(
            heatmap, 
            origin='lower', 
            cmap='plasma', 
            interpolation='bilinear',
            extent=[0, frame_width, 0, frame_height], 
            aspect='auto'
        )
        
        plt.colorbar(label='Team Activity Density')
        plt.title(f'Team Movement Heatmap ({len(player_positions)} players)')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_trajectory_plot(self, 
                               points: List[Tuple[int, int]], 
                               output_path: str,
                               player_id: Optional[int] = None) -> Optional[str]:
        """
        Generate trajectory plot showing player movement path.
        
        Args:
            points: List of (x, y) coordinates
            output_path: Path to save trajectory plot
            player_id: Player ID for title
            
        Returns:
            Path to saved plot or None if insufficient data
        """
        if len(points) < 2:
            return None
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        plt.figure(figsize=(10, 6))
        
        # Plot trajectory line
        plt.plot(xs, ys, 'b-', alpha=0.7, linewidth=2, label='Movement Path')
        
        # Mark start and end points
        plt.scatter(xs[0], ys[0], c='green', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(xs[-1], ys[-1], c='red', s=100, marker='s', label='End', zorder=5)
        
        # Add direction arrows (sample every N points for clarity)
        step = max(1, len(points) // 20)
        for i in range(0, len(points) - step, step):
            dx = xs[i + step] - xs[i]
            dy = ys[i + step] - ys[i]
            plt.arrow(xs[i], ys[i], dx * 0.3, dy * 0.3, 
                     head_width=10, head_length=15, fc='blue', alpha=0.5)
        
        title = 'Player Movement Trajectory'
        if player_id is not None:
            title = f'Player P{player_id} Movement Trajectory'
        
        plt.title(title)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
