# visualization/chart_generator.py
"""
Chart generation utilities for pass counting and game statistics.
Creates various types of charts and statistical visualizations.
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from config.settings import CHART_DPI


class ChartGenerator:
    """
    Generates charts and statistical visualizations for football analysis.
    
    Features:
    - Pass counting charts (cumulative and per-frame)
    - Statistical summaries
    - Time-series analysis
    - Customizable styling
    """
    
    def __init__(self, dpi: int = CHART_DPI):
        """
        Initialize chart generator.
        
        Args:
            dpi: DPI for saved charts
        """
        self.dpi = dpi
    
    def generate_pass_chart(self, 
                           frame_history: List[int], 
                           pass_history: List[int], 
                           output_path: str,
                           chart_type: str = 'cumulative') -> str:
        """
        Generate pass counting chart.
        
        Args:
            frame_history: List of frame indices
            pass_history: List of cumulative pass counts
            output_path: Path to save the chart
            chart_type: Type of chart ('cumulative', 'step', 'rate')
            
        Returns:
            Path to saved chart
        """
        plt.figure(figsize=(12, 6))
        
        if chart_type == 'cumulative':
            self._plot_cumulative_passes(frame_history, pass_history)
        elif chart_type == 'step':
            self._plot_step_passes(frame_history, pass_history)
        elif chart_type == 'rate':
            self._plot_pass_rate(frame_history, pass_history)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_cumulative_passes(self, frame_history: List[int], pass_history: List[int]) -> None:
        """Plot cumulative passes over time."""
        plt.plot(frame_history, pass_history, linewidth=2, color='#2E86AB')
        plt.fill_between(frame_history, pass_history, alpha=0.3, color='#2E86AB')
        
        plt.xlabel('Frame Number')
        plt.ylabel('Cumulative Passes')
        plt.title('Cumulative Passes Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add final count annotation
        if pass_history:
            final_count = pass_history[-1]
            plt.annotate(f'Total: {final_count} passes', 
                        xy=(frame_history[-1], final_count),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=12, fontweight='bold')
    
    def _plot_step_passes(self, frame_history: List[int], pass_history: List[int]) -> None:
        """Plot step-wise pass progression."""
        plt.step(frame_history, pass_history, where='post', linewidth=2, color='#A23B72')
        
        plt.xlabel('Frame Number')
        plt.ylabel('Cumulative Passes')
        plt.title('Pass Count Progression (Step Chart)')
        plt.grid(True, alpha=0.3)
    
    def _plot_pass_rate(self, frame_history: List[int], pass_history: List[int]) -> None:
        """Plot pass rate over time (passes per time window)."""
        if len(frame_history) < 2:
            plt.text(0.5, 0.5, 'Insufficient data for rate calculation', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            return
        
        # Calculate pass rate (passes per 100 frames)
        window_size = 100
        rates = []
        rate_frames = []
        
        for i in range(window_size, len(pass_history)):
            start_idx = max(0, i - window_size)
            passes_in_window = pass_history[i] - pass_history[start_idx]
            rates.append(passes_in_window)
            rate_frames.append(frame_history[i])
        
        if rates:
            plt.plot(rate_frames, rates, linewidth=2, color='#F18F01', marker='o', markersize=3)
            plt.xlabel('Frame Number')
            plt.ylabel(f'Passes per {window_size} frames')
            plt.title(f'Pass Rate Over Time (Window: {window_size} frames)')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient data for rate calculation', 
                    transform=plt.gca().transAxes, ha='center', va='center')
    
    def generate_statistics_summary(self, 
                                  pass_history: List[int], 
                                  frame_history: List[int],
                                  fps: float,
                                  output_path: str) -> str:
        """
        Generate statistical summary chart.
        
        Args:
            pass_history: List of pass counts
            frame_history: List of frame indices
            fps: Video frames per second
            output_path: Path to save summary
            
        Returns:
            Path to saved summary
        """
        if not pass_history or not frame_history:
            return output_path
        
        # Calculate statistics
        total_passes = pass_history[-1] if pass_history else 0
        total_frames = len(frame_history)
        duration_seconds = total_frames / fps if fps > 0 else 0
        duration_minutes = duration_seconds / 60
        
        # Calculate pass rate statistics
        pass_rate_per_minute = total_passes / duration_minutes if duration_minutes > 0 else 0
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Basic statistics
        stats_text = f"""
        Total Passes: {total_passes}
        Duration: {duration_minutes:.1f} min
        Pass Rate: {pass_rate_per_minute:.1f} passes/min
        Total Frames: {total_frames}
        """
        ax1.text(0.1, 0.5, stats_text, transform=ax1.transAxes, 
                fontsize=14, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax1.set_title('Game Statistics')
        ax1.axis('off')
        
        # 2. Pass progression
        ax2.plot(frame_history, pass_history, linewidth=2)
        ax2.set_title('Pass Progression')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Cumulative Passes')
        ax2.grid(True, alpha=0.3)
        
        # 3. Pass intervals (time between passes)
        if len(pass_history) > 1:
            pass_intervals = []
            for i in range(1, len(pass_history)):
                if pass_history[i] > pass_history[i-1]:
                    # Find previous pass
                    for j in range(i-1, -1, -1):
                        if j == 0 or pass_history[j] > pass_history[j-1]:
                            interval = (frame_history[i] - frame_history[j]) / fps
                            pass_intervals.append(interval)
                            break
            
            if pass_intervals:
                ax3.hist(pass_intervals, bins=min(10, len(pass_intervals)), 
                        alpha=0.7, color='green', edgecolor='black')
                ax3.set_title('Pass Interval Distribution')
                ax3.set_xlabel('Seconds between passes')
                ax3.set_ylabel('Frequency')
            else:
                ax3.text(0.5, 0.5, 'No pass intervals to display', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Pass Intervals')
        else:
            ax3.text(0.5, 0.5, 'Insufficient data', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Pass Intervals')
        
        # 4. Activity timeline
        if len(frame_history) > 10:
            # Create activity heatmap (passes per time segment)
            segment_size = len(frame_history) // 10
            segments = []
            segment_labels = []
            
            for i in range(0, len(frame_history), segment_size):
                end_idx = min(i + segment_size, len(frame_history) - 1)
                passes_in_segment = pass_history[end_idx] - (pass_history[i] if i > 0 else 0)
                segments.append(passes_in_segment)
                start_time = frame_history[i] / fps / 60  # minutes
                segment_labels.append(f'{start_time:.1f}m')
            
            ax4.bar(range(len(segments)), segments, color='orange', alpha=0.7)
            ax4.set_title('Activity Timeline')
            ax4.set_xlabel('Time Segment')
            ax4.set_ylabel('Passes in Segment')
            ax4.set_xticks(range(len(segment_labels)))
            ax4.set_xticklabels(segment_labels, rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for timeline', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Activity Timeline')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
