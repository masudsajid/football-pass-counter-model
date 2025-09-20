# ui/streamlit_app.py
"""
Streamlit user interface for the Football Pass Counter application.
Provides a web-based interface for video upload, processing, and results visualization.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

from config.settings import PAGE_TITLE, DEFAULT_MODEL_PATH, SUPPORTED_VIDEO_FORMATS
from services import VideoProcessor
from visualization import HeatmapGenerator
from ui.components import UIComponents


class StreamlitApp:
    """
    Main Streamlit application class for Football Pass Counter.
    
    Features:
    - Video upload and model configuration
    - Real-time processing with progress indication
    - Results visualization (video, charts, heatmaps)
    - Download functionality for outputs
    - Session state management
    """
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.ui_components = UIComponents()
        self.heatmap_generator = HeatmapGenerator()
        self._setup_page_config()
    
    def _setup_page_config(self) -> None:
        """Setup Streamlit page configuration."""
        st.set_page_config(page_title=PAGE_TITLE, layout="wide")
        st.title("⚽ Football Pass Counter — Advanced Analysis")
    
    def run(self) -> None:
        """Run the main Streamlit application."""
        # Render header and input section
        self._render_input_section()
        
        # Handle video processing
        if st.session_state.get("uploaded_file") is not None:
            self._handle_video_processing()
        else:
            self._render_welcome_message()
        
        # Render footer
        self._render_footer()
    
    def _render_input_section(self) -> None:
        """Render the input section for model path and video upload."""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            model_path = st.text_input(
                "Model path (best.pt)", 
                value=DEFAULT_MODEL_PATH,
                help="Path to your trained YOLO model file"
            )
            st.session_state["model_path"] = model_path
        
        with col2:
            uploaded_file = st.file_uploader(
                "Upload a football video", 
                type=SUPPORTED_VIDEO_FORMATS,
                help="Supported formats: " + ", ".join(SUPPORTED_VIDEO_FORMATS)
            )
            st.session_state["uploaded_file"] = uploaded_file
        
        # Display model validation
        if model_path:
            if Path(model_path).exists():
                st.success(f"✅ Model found: {model_path}")
            else:
                st.error(f"❌ Model not found: {model_path}")
    
    def _handle_video_processing(self) -> None:
        """Handle the main video processing workflow."""
        uploaded_file = st.session_state["uploaded_file"]
        model_path = st.session_state["model_path"]
        
        # Save uploaded file
        temp_dir = self._setup_temp_directory(uploaded_file)
        input_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"📁 Video saved: {uploaded_file.name}")
        
        # Processing section
        col1, col2 = st.columns([1, 3])
        
        with col1:
            start_processing = st.button(
                "🚀 Start Processing", 
                type="primary",
                disabled=not Path(model_path).exists()
            )
        
        with col2:
            if not Path(model_path).exists():
                st.error("Please provide a valid model path before processing.")
        
        # Handle processing
        if start_processing:
            self._process_video(input_path, temp_dir, model_path)
        
        # Display results if available
        if st.session_state.get("processing_complete"):
            self._render_results()
    
    def _setup_temp_directory(self, uploaded_file) -> str:
        """Setup temporary directory for processing."""
        if "temp_dir" not in st.session_state:
            st.session_state["temp_dir"] = tempfile.mkdtemp()
        return st.session_state["temp_dir"]
    
    def _process_video(self, input_path: str, temp_dir: str, model_path: str) -> None:
        """Process the video with progress indication."""
        output_video_path = os.path.join(temp_dir, "annotated_output.mp4")
        chart_path = os.path.join(temp_dir, "pass_chart.png")
        
        # Show processing animation
        placeholder = st.empty()
        with placeholder.container():
            st.components.v1.html(self.ui_components.get_processing_animation(), height=300)
        
        try:
            with st.spinner("🔄 Processing video with YOLO model — this may take a while..."):
                # Initialize video processor
                processor = VideoProcessor(model_path)
                
                # Process video
                result = processor.process_video(input_path, output_video_path, chart_path)
                passes, elapsed, player_positions, frame_w, frame_h, out_vid, out_chart = result
                
                # Store results in session state
                self._store_processing_results(
                    passes, elapsed, player_positions, frame_w, frame_h,
                    temp_dir, out_vid, out_chart, processor.get_processing_statistics()
                )
            
            placeholder.empty()
            st.success(f"✅ Processing completed in {elapsed:.1f}s — Total passes: {passes}")
            
        except Exception as e:
            placeholder.empty()
            st.error(f"❌ Processing failed: {str(e)}")
            st.exception(e)
    
    def _store_processing_results(self, passes, elapsed, player_positions, frame_w, frame_h,
                                temp_dir, out_vid, out_chart, stats):
        """Store processing results in session state."""
        st.session_state.update({
            "processing_complete": True,
            "passes": passes,
            "elapsed_time": elapsed,
            "player_positions": player_positions,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "temp_dir": temp_dir,
            "output_video": out_vid,
            "chart_path": out_chart,
            "processing_stats": stats
        })
    
    def _render_results(self) -> None:
        """Render processing results and visualizations."""
        st.subheader("📊 Processing Results")
        
        # Results summary
        self._render_results_summary()
        
        # Video and chart outputs
        self._render_video_and_chart()
        
        # Player analysis section
        self._render_player_analysis()
        
        # Download section
        self._render_downloads()
    
    def _render_results_summary(self) -> None:
        """Render summary statistics."""
        stats = st.session_state.get("processing_stats", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Passes", st.session_state.get("passes", 0))
        
        with col2:
            st.metric("Processing Time", f"{st.session_state.get('elapsed_time', 0):.1f}s")
        
        with col3:
            video_info = stats.get("video_info", {})
            st.metric("Total Frames", video_info.get("total_frames", 0))
        
        with col4:
            tracking_info = stats.get("tracking_info", {})
            st.metric("Active Players", tracking_info.get("active_players", 0))
    
    def _render_video_and_chart(self) -> None:
        """Render annotated video and pass chart."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎥 Annotated Video")
            if st.session_state.get("output_video"):
                st.video(st.session_state["output_video"])
            else:
                st.info("No video output available")
        
        with col2:
            st.subheader("📈 Pass Analysis Chart")
            if st.session_state.get("chart_path"):
                st.image(st.session_state["chart_path"], use_column_width=True)
            else:
                st.info("No chart available")
    
    def _render_player_analysis(self) -> None:
        """Render player heatmap analysis."""
        st.subheader("🗺️ Player Movement Analysis")
        
        positions = st.session_state.get("player_positions", {})
        if not positions:
            st.info("No player position data available")
            return
        
        # Player selection
        player_ids = sorted(positions.keys())
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_pid = st.selectbox(
                "Select Player", 
                player_ids, 
                format_func=lambda x: f"Player {x}",
                help="Choose a player to view their movement heatmap"
            )
            
            if st.button("Generate Heatmap"):
                self._generate_player_heatmap(selected_pid, positions)
        
        with col2:
            # Display heatmap if available
            heatmap_key = f"heatmap_P{selected_pid}"
            if heatmap_key in st.session_state:
                heatmap_path, distance = st.session_state[heatmap_key]
                st.image(heatmap_path, caption=f"Player {selected_pid} — Distance: {distance:.2f}m")
    
    def _generate_player_heatmap(self, player_id, positions):
        """Generate heatmap for selected player."""
        temp_dir = st.session_state.get("temp_dir")
        if not temp_dir:
            st.error("Temporary directory not available")
            return
        
        heatmap_path = os.path.join(temp_dir, f"heatmap_P{player_id}.png")
        
        try:
            result = self.heatmap_generator.generate_player_heatmap(
                positions.get(player_id, []),
                st.session_state.get("frame_w", 0),
                st.session_state.get("frame_h", 0),
                heatmap_path,
                player_id
            )
            
            if result:
                heatmap_path_saved, distance = result
                st.session_state[f"heatmap_P{player_id}"] = (heatmap_path_saved, distance)
                st.success(f"Heatmap generated for Player {player_id}")
            else:
                st.warning(f"No position data available for Player {player_id}")
                
        except Exception as e:
            st.error(f"Failed to generate heatmap: {str(e)}")
    
    def _render_downloads(self) -> None:
        """Render download section."""
        st.subheader("💾 Downloads")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.get("output_video"):
                with open(st.session_state["output_video"], "rb") as f:
                    st.download_button(
                        "📹 Download Annotated Video",
                        data=f,
                        file_name="annotated_football_video.mp4",
                        mime="video/mp4"
                    )
        
        with col2:
            if st.session_state.get("chart_path"):
                with open(st.session_state["chart_path"], "rb") as f:
                    st.download_button(
                        "📊 Download Pass Chart",
                        data=f,
                        file_name="pass_analysis_chart.png",
                        mime="image/png"
                    )
        
        with col3:
            # Download heatmaps
            positions = st.session_state.get("player_positions", {})
            if positions:
                player_ids = sorted(positions.keys())
                selected_for_download = st.selectbox(
                    "Select Player Heatmap", 
                    player_ids,
                    format_func=lambda x: f"Player {x} Heatmap",
                    key="download_select"
                )
                
                heatmap_key = f"heatmap_P{selected_for_download}"
                if heatmap_key in st.session_state:
                    heatmap_path, _ = st.session_state[heatmap_key]
                    with open(heatmap_path, "rb") as f:
                        st.download_button(
                            f"🗺️ Download P{selected_for_download} Heatmap",
                            data=f,
                            file_name=f"player_{selected_for_download}_heatmap.png",
                            mime="image/png"
                        )
    
    def _render_welcome_message(self) -> None:
        """Render welcome message when no video is uploaded."""
        st.info("👆 Upload a football video above to get started with pass analysis!")
        
        # Feature overview
        st.subheader("🔧 Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Detection & Tracking:**
            - YOLO-based player and ball detection
            - Multi-object tracking with re-identification
            - Fallback ball detection using computer vision
            """)
        
        with col2:
            st.markdown("""
            **Analysis & Visualization:**
            - Automatic pass counting and validation
            - Player movement heatmaps
            - Statistical analysis and charts
            """)
    
    def _render_footer(self) -> None:
        """Render application footer."""
        st.markdown("---")
        st.caption(
            "⚽ Football Pass Counter — Built with YOLO, OpenCV, and Streamlit. "
            "Ensure all required packages are installed."
        )
