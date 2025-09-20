# ui/components.py
"""
UI components and utilities for the Streamlit application.
Contains reusable UI elements and styling components.
"""

import streamlit as st


class UIComponents:
    """
    Collection of reusable UI components for the Streamlit app.
    
    Features:
    - Processing animations and loading indicators
    - Styled information boxes
    - Custom CSS styling
    - Interactive elements
    """
    
    def __init__(self):
        """Initialize UI components."""
        self._apply_custom_css()
    
    def _apply_custom_css(self) -> None:
        """Apply custom CSS styling to the Streamlit app."""
        st.markdown("""
        <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        
        .processing-container {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 1rem;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def get_processing_animation(self) -> str:
        """
        Get HTML for 3D processing animation.
        
        Returns:
            HTML string with Three.js animation
        """
        return '''
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            html, body {
              margin: 0;
              height: 100%;
              overflow: hidden;
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            #info {
              position: absolute;
              top: 20px;
              left: 50%;
              transform: translateX(-50%);
              color: white;
              font-size: 18px;
              font-weight: bold;
              text-shadow: 0 2px 4px rgba(0,0,0,0.5);
              z-index: 100;
            }
            #progress {
              position: absolute;
              bottom: 30px;
              left: 50%;
              transform: translateX(-50%);
              color: white;
              font-size: 14px;
              z-index: 100;
            }
          </style>
        </head>
        <body>
        <div id="info">🔄 Processing Video...</div>
        <div id="progress">Analyzing player movements and detecting passes</div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r148/three.min.js"></script>
        <script>
          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
          const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
          renderer.setSize(window.innerWidth, window.innerHeight);
          renderer.setClearColor(0x000000, 0);
          document.body.appendChild(renderer.domElement);

          // Create multiple rotating objects
          const objects = [];
          
          // Main cube
          const geometry1 = new THREE.BoxGeometry(1, 1, 1);
          const material1 = new THREE.MeshNormalMaterial();
          const cube = new THREE.Mesh(geometry1, material1);
          objects.push({mesh: cube, speed: {x: 0.01, y: 0.02, z: 0.005}});
          scene.add(cube);
          
          // Smaller spheres orbiting
          for(let i = 0; i < 3; i++) {
            const geometry = new THREE.SphereGeometry(0.2, 16, 16);
            const material = new THREE.MeshBasicMaterial({
              color: new THREE.Color().setHSL(i/3, 0.8, 0.6),
              transparent: true,
              opacity: 0.8
            });
            const sphere = new THREE.Mesh(geometry, material);
            objects.push({
              mesh: sphere, 
              speed: {x: 0.02 + i*0.01, y: 0.015 + i*0.005, z: 0.01},
              orbit: {radius: 2 + i*0.5, angle: i*Math.PI*2/3}
            });
            scene.add(sphere);
          }

          camera.position.z = 5;
          
          let time = 0;
          function animate() {
            requestAnimationFrame(animate);
            time += 0.016;
            
            objects.forEach((obj, index) => {
              if(obj.orbit) {
                // Orbital motion
                obj.orbit.angle += obj.speed.x;
                obj.mesh.position.x = Math.cos(obj.orbit.angle) * obj.orbit.radius;
                obj.mesh.position.y = Math.sin(obj.orbit.angle) * obj.orbit.radius * 0.5;
                obj.mesh.position.z = Math.sin(obj.orbit.angle * 2) * 0.5;
              }
              
              // Rotation
              obj.mesh.rotation.x += obj.speed.x;
              obj.mesh.rotation.y += obj.speed.y;
              obj.mesh.rotation.z += obj.speed.z;
            });
            
            // Camera gentle movement
            camera.position.x = Math.sin(time * 0.2) * 0.5;
            camera.position.y = Math.cos(time * 0.15) * 0.3;
            camera.lookAt(0, 0, 0);
            
            renderer.render(scene, camera);
          }
          animate();
          
          window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
          });
        </script>
        </body>
        </html>
        '''
    
    def render_metric_card(self, title: str, value: str, delta: str = None) -> None:
        """
        Render a styled metric card.
        
        Args:
            title: Metric title
            value: Metric value
            delta: Optional delta/change value
        """
        delta_html = f"<div style='color: #28a745; font-size: 0.8rem;'>{delta}</div>" if delta else ""
        
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 0.9rem; color: #666;">{title}</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin: 0.25rem 0;">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def render_success_message(self, message: str) -> None:
        """Render a styled success message."""
        st.markdown(f"""
        <div class="success-box">
            ✅ {message}
        </div>
        """, unsafe_allow_html=True)
    
    def render_info_message(self, message: str) -> None:
        """Render a styled info message."""
        st.markdown(f"""
        <div class="info-box">
            ℹ️ {message}
        </div>
        """, unsafe_allow_html=True)
    
    def render_feature_grid(self, features: list) -> None:
        """
        Render a grid of feature cards.
        
        Args:
            features: List of dicts with 'title', 'description', 'icon'
        """
        cols = st.columns(len(features))
        
        for i, feature in enumerate(features):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; border: 1px solid #ddd; border-radius: 0.5rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{feature.get('icon', '🔧')}</div>
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">{feature['title']}</div>
                    <div style="font-size: 0.9rem; color: #666;">{feature['description']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def render_progress_indicator(self, progress: float, message: str = "") -> None:
        """
        Render a custom progress indicator.
        
        Args:
            progress: Progress value between 0 and 1
            message: Optional progress message
        """
        progress_percent = int(progress * 100)
        
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>{message}</span>
                <span>{progress_percent}%</span>
            </div>
            <div style="background-color: #e9ecef; border-radius: 0.25rem; height: 0.5rem;">
                <div style="background-color: #007bff; width: {progress_percent}%; height: 100%; border-radius: 0.25rem; transition: width 0.3s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_stats_summary(self, stats: dict) -> None:
        """
        Render a comprehensive stats summary.
        
        Args:
            stats: Dictionary containing various statistics
        """
        st.markdown("### 📊 Processing Summary")
        
        # Create columns for different stat categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Video Info**")
            video_info = stats.get('video_info', {})
            for key, value in video_info.items():
                st.write(f"• {key.replace('_', ' ').title()}: {value}")
        
        with col2:
            st.markdown("**Detection Stats**")
            detection_info = stats.get('detection_info', {})
            for key, value in detection_info.items():
                if isinstance(value, list):
                    st.write(f"• {key.replace('_', ' ').title()}: {len(value)} items")
                else:
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
        
        with col3:
            st.markdown("**Pass Analysis**")
            pass_stats = stats.get('pass_statistics', {})
            for key, value in pass_stats.items():
                st.write(f"• {key.replace('_', ' ').title()}: {value}")
    
    def render_help_section(self) -> None:
        """Render help and usage instructions."""
        with st.expander("ℹ️ Help & Usage Instructions"):
            st.markdown("""
            ### How to Use Football Pass Counter
            
            1. **Upload Video**: Choose a football video file (MP4, AVI, or MOV)
            2. **Set Model Path**: Ensure your YOLO model file path is correct
            3. **Start Processing**: Click the processing button and wait for analysis
            4. **View Results**: Explore the annotated video, charts, and heatmaps
            5. **Download**: Save your results for further analysis
            
            ### Tips for Best Results
            - Use videos with clear view of the field
            - Ensure good lighting and minimal camera movement
            - Higher resolution videos generally produce better results
            - Make sure your YOLO model is trained on football/soccer data
            
            ### Troubleshooting
            - If processing fails, check that your model file exists
            - Large video files may take significant time to process
            - Ensure you have sufficient disk space for temporary files
            """)
    
    def get_loading_placeholder(self, message: str = "Loading...") -> str:
        """Get a simple loading placeholder HTML."""
        return f"""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 1.2rem; margin-bottom: 1rem;">{message}</div>
            <div style="display: inline-block; width: 40px; height: 40px; border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
