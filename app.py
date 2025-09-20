import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
from ultralytics import YOLO
import tempfile
import os

# ---------------- VIDEO PROCESSING ----------------
def process_video(input_path, model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
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

                if int(cls) == 2:  # 👈 assume class 2 = player
                    if track_id not in player_ids:
                        player_ids[track_id] = f"Player {len(player_ids)+1}"
                    pid = player_ids[track_id]

                    if pid not in player_tracks:
                        player_tracks[pid] = []
                    player_tracks[pid].append((cx, cy))

                elif int(cls) == 0:  # 👈 assume class 0 = ball
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

    cap.release()
    return player_tracks, passes_per_frame, total_passes


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
                                 frame_w=1920, frame_h=1080):
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
    xs = [x / frame_w * 120 for x in xs]  # normalize to pitch length
    ys = [y / frame_h * 80 for y in ys]   # normalize to pitch width

    fig, ax = plt.subplots(figsize=(13.5, 8))
    fig.set_facecolor("#22312b")
    ax.patch.set_facecolor("#22312b")

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    pitch.draw(ax=ax)

    sns.kdeplot(x=xs, y=ys, fill=True, levels=100, cmap="magma",
                alpha=0.7, thresh=0.05, ax=ax)

    ax.set_title(f"{player} Positional Heatmap", color="white", fontsize=16)
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
        player_tracks, passes_per_frame, total_passes = process_video(temp_video_path, model_path)

        st.session_state["player_tracks"] = player_tracks
        st.session_state["passes_per_frame"] = passes_per_frame
        st.session_state["total_passes"] = total_passes

# ---------------- RESULTS SECTION ----------------
if "player_tracks" in st.session_state:
    st.subheader("Processed Video")
    st.video(temp_video_path)

    st.subheader("Analytics")
    st.write(f"✅ Total passes in video: {st.session_state['total_passes']}")

    plot_pass_chart(st.session_state["passes_per_frame"])

    players = ["All Players"] + list(st.session_state["player_tracks"].keys())
    selected_player = st.selectbox("Select Player", players)
    plot_player_heatmap_on_pitch(st.session_state["player_tracks"], player=selected_player)
