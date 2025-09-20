# config/settings.py
"""
Configuration settings for the Football Pass Counter application.
Contains all constants and configurable parameters.
"""

# YOLO Detection Parameters
YOLO_CONF = 0.02
PLAYER_CONF_THR = 0.30
BALL_CONF_THR = 0.10

# Player Tracking Parameters
MAX_PLAYER_MATCH_DIST = 90         # px: distance threshold for matching detections to existing tracks
PLAYER_MISSING_TOLERANCE = 30      # frames tolerated missing before deleting track
MAX_PLAYERS = 22                   # maximum number of players to track

# Ball Tracking Parameters
BALL_SEARCH_RADIUS = 120           # fallback detection area radius
SMOOTH_ALPHA = 0.6                 # ball smoothing alpha

# Pass Detection Parameters
MIN_HOLD_FRAMES = 2                # frames threshold before confirming possessor change
BBOX_MARGIN = 6                    # margin for ball-player containment check

# Video Processing Parameters
DEFAULT_FPS = 25.0
VIDEO_CODEC = 'mp4v'
YOLO_IMAGE_SIZE = 640

# Visualization Parameters
HEATMAP_BINS = 60
CHART_DPI = 150
HEATMAP_DPI = 150

# Pitch Dimensions (in meters)
PITCH_LENGTH_M = 120
PITCH_WIDTH_M = 80

# Colors (BGR format for OpenCV)
PLAYER_COLOR = (200, 100, 0)
BALL_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 200, 200)

# Ball Detection Color Ranges (HSV)
BALL_HSV_LOWER = (0, 0, 180)
BALL_HSV_UPPER = (180, 60, 255)
BALL_MIN_AREA = 8
BALL_MAX_AREA = 5000

# UI Configuration
PAGE_TITLE = "Football Pass Counter"
DEFAULT_MODEL_PATH = "best.pt"
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov"]
