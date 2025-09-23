import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
from ultralytics import YOLO
import tempfile
import os
import shutil

# ---------------- VIDEO PROCESSING ----------------
def process_video(input_path, model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Save processed video
    out_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    player_tracks = {}   # { "Player 1": [(x,y), ...], ... }
    player_ids = {}      # Map YOLO IDs → Player names
    passes_per_frame = []
    total_passes = 0
    prev_ball_owner = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        results = model.track(frame, persist=True)
        frame_passes = 0

        if results[0].boxes.id is not None:
            for box, track_id, cls in zip(results[0].boxes.xyxy,
                                          results[0].boxes.id.cpu().numpy(),
                                          results[0].boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if int(cls) == 2:  # player
                    if track_id not in player_ids:
                        player_ids[track_id] = f"Player {len(player_ids)+1}"
                    pid = player_ids[track_id]

                    if pid not in player_tracks:
                        player_tracks[pid] = []
                    player_tracks[pid].append((cx, cy))

                    # Draw player box + ID
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, pid, (int(x1), int(y1)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                elif int(cls) == 0:  # ball
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

                    # Find nearest player
                    min_dist, owner = 1e9, None
                    for p_id, pts in player_tracks.items():
                        if pts:
                            px, py = pts[-1]
                            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                            if dist < min_dist:
                                min_dist, owner = dist, p_id

                    # Pass detection
                    if owner is not None and owner != prev_ball_owner:
                        if prev_ball_owner is not None:
                            total_passes += 1
                            frame_passes += 1
                        prev_ball_owner = owner

        passes_per_frame.append(frame_passes)

        # Overlay total passes on frame
        cv2.putText(frame, f"Passes: {total_passes}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        out.write(frame)

    cap.release()
    out.release()
    return out_path, player_tracks, passes_per_frame, total_passes, frame_w, frame_h, fps


# ---------------- PASS PER FRAME CHART ----------------
def plot_pass_chart(passes_per_frame):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(passes_per_frame)), passes_per_frame, color="blue")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Passes in Frame")
    ax.set_title("Passes per Frame")
    st.pyplot(fig)


# ---------------- HEATMAP ON PITCH ----------------
def plot_player_heatmap_on_pitch(player_tracks, player="All Players",
                                 frame_w=1920, frame_h=1080, fps=30):
    coords = []
    if player == "All Players":
        for pts in player_tracks.values():
            coords.extend(pts)
    else:
        coords = player_tracks.get(player, [])

    if not coords:
        st.warning(f"No data available for {player}")
        return

    xs, ys = zip(*coords)
    xs = [x / frame_w * 120 for x in xs]              # normalize to pitch length (meters)
    ys = [(1 - (y / frame_h)) * 80 for y in ys]       # invert Y to match pitch coords (meters)

    fig, ax = plt.subplots(figsize=(13.5, 8))
    fig.set_facecolor("#22312b")
    ax.patch.set_facecolor("#22312b")

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    pitch.draw(ax=ax)

    sns.kdeplot(x=xs, y=ys, fill=True, levels=100, cmap="magma",
                alpha=0.7, thresh=0.05, ax=ax)

    ax.set_title(f"{player} Positional Heatmap", color="white", fontsize=16)

    # Metrics: distance covered and top speed for an individual player (overlay on heatmap)
    if player != "All Players":
        total_distance_m = 0.0
        top_speed_mps = 0.0

        # compute using consecutive points in pitch meters
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            segment_m = np.sqrt(dx * dx + dy * dy)
            total_distance_m += segment_m
            if fps and fps > 0:
                speed_mps = segment_m * fps
                if speed_mps > top_speed_mps:
                    top_speed_mps = speed_mps

        stats_text = f"Distance: {total_distance_m:.1f} m\nTop speed: {top_speed_mps:.2f} m/s"
        ax.text(2, 76, stats_text, color="white", fontsize=12,
                bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.4'))

    st.pyplot(fig)


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="⚽ Football Pass Counter", layout="wide")
st.title("⚽ Football Pass Counter — Streamlit")

uploaded_video = st.file_uploader("Upload match video", type=["mp4", "avi", "mov"])
model_path = st.text_input("Enter YOLO model path", "best.pt")

if uploaded_video is not None and model_path:
    temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    if st.button("Start Processing"):
        (out_path, player_tracks, passes_per_frame,
         total_passes, frame_w, frame_h, fps) = process_video(temp_video_path, model_path)

        st.session_state["out_path"] = out_path
        st.session_state["player_tracks"] = player_tracks
        st.session_state["passes_per_frame"] = passes_per_frame
        st.session_state["total_passes"] = total_passes
        st.session_state["frame_w"] = frame_w
        st.session_state["frame_h"] = frame_h
        st.session_state["fps"] = fps

        # Save a persistent copy on disk
        try:
            outputs_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            base_name, _ = os.path.splitext(uploaded_video.name)
            persistent_path = os.path.join(outputs_dir, f"{base_name}_annotated.mp4")
            shutil.copyfile(out_path, persistent_path)
            st.session_state["persistent_video_path"] = persistent_path
        except Exception as e:
            st.warning(f"Could not save a persistent copy of the video: {e}")

# ---------------- RESULTS SECTION ----------------
if "player_tracks" in st.session_state:
    st.subheader("Processed Video")
    # Use raw bytes for more reliable playback across environments
    try:
        with open(st.session_state["out_path"], "rb") as vf:
            st.video(vf.read())
    except Exception:
        st.video(st.session_state["out_path"])  # fallback

    # Show save location and download option
    if "persistent_video_path" in st.session_state and os.path.exists(st.session_state["persistent_video_path"]):
        st.success(f"Annotated video saved at: {st.session_state['persistent_video_path']}")
        with open(st.session_state["persistent_video_path"], "rb") as f:
            st.download_button(label="Download annotated video",
                               data=f,
                               file_name=os.path.basename(st.session_state["persistent_video_path"]),
                               mime="video/mp4")

    st.subheader("Analytics")
    st.write(f"✅ Total passes in video: {st.session_state['total_passes']}")

    plot_pass_chart(st.session_state["passes_per_frame"])

    players = ["All Players"] + list(st.session_state["player_tracks"].keys())
    selected_player = st.selectbox("Select Player", players)
    plot_player_heatmap_on_pitch(st.session_state["player_tracks"],
                                 player=selected_player,
                                 frame_w=st.session_state["frame_w"],
                                 frame_h=st.session_state["frame_h"],
                                 fps=st.session_state.get("fps", 30))
