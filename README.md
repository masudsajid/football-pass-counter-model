# ⚽ Football Pass Counter

A computer vision application that automatically counts football passes in videos using YOLO object detection and tracking algorithms. Built with Streamlit for an easy-to-use web interface.

## Features

- **Automatic Pass Counting**: Detects and tracks players and the ball to count passes
- **Real-time Visualization**: Shows annotated video with player tracking and ball detection
- **Interactive Web Interface**: Easy-to-use Streamlit web app
- **Export Results**: Download annotated videos and pass count charts
- **Player Tracking**: Maintains consistent player IDs throughout the video
- **Ball Detection**: Uses both YOLO detection and fallback computer vision methods

## Requirements

- **Python**: 3.8 - 3.11 (3.12 not supported due to PyTorch compatibility)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM recommended
- **GPU**: Optional but recommended for faster processing

## Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd football_pass_counter
```

Or download and extract the project files to a folder.

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web interface)
- Ultralytics YOLO (object detection)
- OpenCV (computer vision)
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- Matplotlib (plotting)

### 4. Download the YOLO Model

Make sure you have the `best.pt` model file in the project directory. This should be a trained YOLO model that can detect:
- **Players** (class: "player")
- **Ball** (class: "ball")

If you don't have a model file, you'll need to train one or obtain a pre-trained model for football detection.

## Usage

### 1. Start the Application

```bash
streamlit run app.py
```

The application will start and automatically open in your web browser at `http://localhost:8501`.

### 2. Upload a Video

1. Click "Browse files" to upload a football video
2. Supported formats: MP4, AVI, MOV
3. The video will be saved temporarily for processing

### 3. Configure Model Path

- By default, the model path is set to `best.pt`
- If your model file has a different name, update the path in the text input field

### 4. Process the Video

1. Click "Start processing" to begin analysis
2. A 3D animation will show while processing (this may take several minutes)
3. The application will:
   - Detect players and track them across frames
   - Detect and track the ball
   - Count passes between players
   - Generate an annotated video
   - Create a pass count chart

### 5. View Results

After processing completes, you'll see:
- **Total pass count** in the success message
- **Annotated video** with player tracking and ball detection
- **Pass count chart** showing passes over time
- **Download buttons** for the annotated video and chart

## How It Works

### Player Detection & Tracking
- Uses YOLO to detect players in each frame
- Implements a tracking algorithm to maintain consistent player IDs
- Handles players temporarily leaving the frame

### Ball Detection
- Primary: YOLO detection for ball recognition
- Fallback: Computer vision techniques using color detection and shape analysis
- Predicts ball position when detection fails

### Pass Counting Logic
- Tracks which player has possession of the ball
- Counts a pass when possession changes between different players
- Requires minimum hold time to avoid false positives

## Configuration Parameters

The application uses several configurable parameters:

```python
YOLO_CONF = 0.02              # YOLO confidence threshold
PLAYER_CONF_THR = 0.30        # Player detection confidence
BALL_CONF_THR = 0.10          # Ball detection confidence
MAX_PLAYER_MATCH_DIST = 90    # Max distance for player tracking
PLAYER_MISSING_TOLERANCE = 30 # Frames before removing lost player
MIN_HOLD_FRAMES = 2           # Minimum frames to confirm possession
BALL_SEARCH_RADIUS = 120      # Search radius for fallback ball detection
SMOOTH_ALPHA = 0.6            # Ball position smoothing factor
```

## Troubleshooting

### Common Issues

**"Model path not found" error:**
- Ensure `best.pt` exists in the project directory
- Check the model path input field

**Slow processing:**
- Use a GPU-enabled PyTorch installation
- Reduce video resolution or length
- Close other applications to free up memory

**Poor detection accuracy:**
- Ensure your YOLO model is trained on similar football footage
- Adjust confidence thresholds in the code
- Use higher quality input videos

**Memory errors:**
- Process shorter video segments
- Reduce video resolution
- Increase system RAM

### Performance Tips

1. **Use GPU**: Install CUDA-enabled PyTorch for faster processing
2. **Video Quality**: Higher resolution videos provide better detection
3. **Lighting**: Well-lit videos with clear player/ball contrast work best
4. **Camera Angle**: Overhead or sideline views typically work better than end-zone views

## File Structure

```
football_pass_counter/
├── app.py              # Main Streamlit application
├── best.pt             # YOLO model file (you need to provide this)
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── venv/              # Virtual environment (created during setup)
```

## Dependencies

- **streamlit**: Web application framework
- **ultralytics**: YOLO object detection
- **opencv-python-headless**: Computer vision operations
- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## License

This project is open source. Please check the license file for details.

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify your YOLO model is compatible
4. Check that your Python version is supported (3.8-3.11)
