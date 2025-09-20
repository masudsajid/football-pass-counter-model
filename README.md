# ⚽ Football Pass Counter - Modular Architecture

A modular, scalable football pass counting application using YOLO object detection, multi-object tracking, and advanced computer vision techniques.

## 🏗️ Architecture Overview

This project has been refactored into a clean, modular architecture following Django-style best practices:

```
football-pass-counter-model/
├── config/                 # Configuration and settings
│   ├── __init__.py
│   └── settings.py        # All constants and parameters
├── detection/              # Object detection modules
│   ├── __init__.py
│   ├── ball_detection.py  # Fallback ball detection using CV
│   └── yolo_processor.py  # YOLO model wrapper
├── trackers/              # Object tracking modules
│   ├── __init__.py
│   ├── player_tracker.py  # Multi-player tracking
│   └── ball_tracker.py    # Ball tracking with smoothing
├── services/              # Business logic services
│   ├── __init__.py
│   ├── pass_detector.py   # Pass detection logic
│   └── video_processor.py # Main video processing service
├── visualization/         # Visualization and charts
│   ├── __init__.py
│   ├── heatmap_generator.py  # Player heatmaps
│   ├── chart_generator.py    # Statistical charts
│   └── video_annotator.py    # Video frame annotations
├── ui/                    # User interface components
│   ├── __init__.py
│   ├── streamlit_app.py   # Main Streamlit application
│   └── components.py      # Reusable UI components
├── main.py               # New modular entry point
├── app.py               # Legacy monolithic version
└── requirements.txt     # Dependencies
```

## 🚀 Quick Start

### New Modular Version (Recommended)
```bash
# Run the new modular application
streamlit run main.py
# or
python main.py
```

### Legacy Version (Backward Compatibility)
```bash
# Run the original monolithic version
streamlit run app.py
```

## 🔧 Features

### Detection & Tracking
- **YOLO-based Detection**: Advanced object detection for players and ball
- **Multi-Object Tracking**: Spatial matching with re-identification
- **Fallback Detection**: Computer vision-based ball detection when YOLO fails
- **Temporal Smoothing**: Ball position smoothing and velocity prediction

### Analysis & Visualization
- **Pass Counting**: Automatic pass detection with validation
- **Player Heatmaps**: Movement density visualization with distance calculation
- **Statistical Charts**: Pass progression, intervals, and activity timelines
- **Real-time Annotation**: Live video annotation with bounding boxes and labels

### User Interface
- **Modern Web UI**: Clean, responsive Streamlit interface
- **Progress Tracking**: Real-time processing indicators with 3D animations
- **Download Support**: Export videos, charts, and heatmaps
- **Session Management**: Persistent state across interactions

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd football-pass-counter-model

# Install dependencies
pip install -r requirements.txt

# Ensure you have your YOLO model file (best.pt)
# Place it in the project root or specify the path in the UI
```

## 🛠️ Configuration

Edit `config/settings.py` to customize detection and tracking parameters:

```python
# YOLO Detection Parameters
YOLO_CONF = 0.02
PLAYER_CONF_THR = 0.30
BALL_CONF_THR = 0.10

# Tracking Parameters
MAX_PLAYER_MATCH_DIST = 90
PLAYER_MISSING_TOLERANCE = 30
MIN_HOLD_FRAMES = 2

# Visualization Parameters
HEATMAP_BINS = 60
CHART_DPI = 150
```

## 📊 Usage

1. **Upload Video**: Select a football video file (MP4, AVI, MOV)
2. **Set Model Path**: Specify path to your YOLO model file
3. **Start Processing**: Click process and wait for analysis
4. **View Results**: Explore annotated video, pass charts, and player heatmaps
5. **Download**: Export results for further analysis

## 🏛️ Module Details

### Configuration (`config/`)
- Centralized settings and constants
- Easy parameter tuning
- Environment-specific configurations

### Detection (`detection/`)
- **YOLOProcessor**: Model loading, inference, and result processing
- **ball_detection**: Fallback detection using HSV color filtering and contour analysis

### Trackers (`trackers/`)
- **PlayerTracker**: Spatial matching with greedy assignment algorithm
- **BallTracker**: Temporal smoothing with velocity-based prediction

### Services (`services/`)
- **VideoProcessor**: Main orchestration service
- **PassDetector**: Ball possession analysis and pass counting logic

### Visualization (`visualization/`)
- **HeatmapGenerator**: 2D density maps with distance calculation
- **ChartGenerator**: Statistical visualizations and summaries
- **VideoAnnotator**: Frame annotation utilities

### UI (`ui/`)
- **StreamlitApp**: Main application interface
- **UIComponents**: Reusable UI elements and styling

## 🔄 Migration from Legacy

The legacy monolithic version (`app.py`) is preserved for backward compatibility. The new modular structure offers:

- **Better Organization**: Clear separation of concerns
- **Easier Maintenance**: Modular components for testing and updates
- **Enhanced Performance**: Optimized processing pipeline
- **Improved UI**: Better styling and user experience
- **Robust Error Handling**: Comprehensive validation and error management

## 🧪 Development

### Adding New Features
1. Create new modules in appropriate directories
2. Update `__init__.py` files for imports
3. Add configuration parameters to `config/settings.py`
4. Update UI components as needed

### Testing
```bash
# Test individual modules
python -m pytest tests/

# Run specific component tests
python -c "from detection import YOLOProcessor; print('Detection module OK')"
```

## 📈 Performance Considerations

- **Memory Usage**: Large videos may require significant RAM
- **Processing Time**: Depends on video length and resolution
- **Model Size**: YOLO model affects loading time and inference speed
- **Disk Space**: Temporary files created during processing

## 🤝 Contributing

1. Follow the modular architecture patterns
2. Add appropriate documentation and type hints
3. Update configuration files for new parameters
4. Test both legacy and modular versions
5. Maintain backward compatibility

## 📄 License

This project follows the same license as the original implementation.

## 🆘 Troubleshooting

### Common Issues
- **Model Not Found**: Ensure YOLO model path is correct
- **Processing Fails**: Check video format and file integrity
- **Memory Errors**: Reduce video resolution or process shorter clips
- **Import Errors**: Verify all dependencies are installed

### Getting Help
- Check the legacy version (`app.py`) for reference implementation
- Review configuration settings in `config/settings.py`
- Examine error logs in the Streamlit interface