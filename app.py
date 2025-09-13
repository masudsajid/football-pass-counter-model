import streamlit as st
import tempfile
import os
from pathlib import Path
import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -----------------------
# UI / Paths
# -----------------------
st.set_page_config(page_title="Football Pass Counter", layout="wide")
st.title("⚽ Football Pass Counter — Streamlit UI")

MODEL_PATH_INPUT = st.text_input("best.pt", value="best.pt")

uploaded_file = st.file_uploader("Upload a football video (mp4, avi)", type=["mp4", "avi", "mov"]) 

# 3D animation HTML (three.js rotating cube)
THREE_JS_HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>html,body{margin:0;height:100%;overflow:hidden;background:#0b1020}#info{position:absolute;top:8px;left:8px;color:#fff;font-family:Arial}</style>
</head>
<body>
<div id="info">Processing... sit tight — counting passes</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r148/three.min.js"></script>
<script>
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const geometry = new THREE.BoxGeometry(1,1,1);
  const material = new THREE.MeshNormalMaterial();
  const cube = new THREE.Mesh(geometry, material);
  scene.add(cube);

  camera.position.z = 3;

  function animate() {
    requestAnimationFrame(animate);
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.02;
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', ()=>{
    camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight);
  });
</script>
</body>
</html>
'''

# -----------------------
# Core processing code (adapted from your script)
# -----------------------

# Params (kept same as your script)
YOLO_CONF = 0.02
PLAYER_CONF_THR = 0.30
BALL_CONF_THR = 0.10
MAX_PLAYER_MATCH_DIST = 90
PLAYER_MISSING_TOLERANCE = 30
MIN_HOLD_FRAMES = 2
BALL_SEARCH_RADIUS = 120
SMOOTH_ALPHA = 0.6


def fallback_detect_ball(frame, last_center=None, search_radius=BALL_SEARCH_RADIUS):
    h, w = frame.shape[:2]
    if last_center is not None:
        x, y = last_center
        x1 = max(0, x - search_radius); y1 = max(0, y - search_radius)
        x2 = min(w, x + search_radius); y2 = min(h, y + search_radius)
        roi = frame[y1:y2, x1:x2]
        offset = (x1, y1)
    else:
        roi = frame
        offset = (0, 0)

    if roi is None or roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 8 or area > 5000:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"]) + offset[0]
        cy = int(M["m01"] / M["m00"]) + offset[1]
        candidates.append((cx, cy, area, circularity))

    if not candidates:
        return None

    if last_center is not None:
        lx, ly = last_center
        candidates.sort(key=lambda c: math.hypot(c[0]-lx, c[1]-ly))
        return (candidates[0][0], candidates[0][1])
    else:
        candidates.sort(key=lambda c: (-c[3], c[2]))
        return (candidates[0][0], candidates[0][1])


class PlayerTracker:
    def __init__(self, max_missing=PLAYER_MISSING_TOLERANCE, max_dist=MAX_PLAYER_MATCH_DIST):
        self.tracks = {}
        self.next_id = 1
        self.max_missing = max_missing
        self.max_dist = max_dist

    def update(self, detections):
        assigned = {}
        new_centroids = [((x1+x2)//2, (y1+y2)//2) for (x1,y1,x2,y2) in detections]
        new_bboxes = detections.copy()

        track_ids = list(self.tracks.keys())
        if track_ids and new_centroids:
            dists = []
            for tid in track_ids:
                tx, ty = self.tracks[tid]['centroid']
                row = [math.hypot(tx-nx, ty-ny) for (nx,ny) in new_centroids]
                dists.append(row)
            dists = np.array(dists)
            while True:
                i, j = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
                minval = dists[i, j]
                if minval > self.max_dist:
                    break
                tid = track_ids[i]
                if tid in assigned or j in assigned.values():
                    dists[i, j] = np.inf
                    if np.isinf(dists).all(): break
                    continue
                assigned[tid] = j
                dists[i, :] = np.inf
                dists[:, j] = np.inf
                if np.isinf(dists).all(): break

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

        for tid in list(self.tracks.keys()):
            if tid not in updated_ids:
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > self.max_missing:
                    del self.tracks[tid]

        for j, bbox in enumerate(new_bboxes):
            if j in used_detections:
                continue
            cx, cy = new_centroids[j]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {'bbox': bbox, 'centroid': (cx, cy), 'missing': 0}

        out = []
        for tid, info in self.tracks.items():
            out.append((tid, info['bbox'], info['centroid']))
        return out


class BallTracker:
    def __init__(self, alpha=SMOOTH_ALPHA):
        self.cx = None
        self.cy = None
        self.vx = 0.0
        self.vy = 0.0
        self.last_frame_seen = -9999
        self.missing = 0
        self.alpha = alpha

    def update_with_detection(self, center, frame_idx):
        if center is None:
            self.missing += 1
            return None
        x, y = center
        if self.cx is None:
            self.cx, self.cy = x, y
            self.vx, self.vy = 0.0, 0.0
        else:
            self.vx = 0.7*self.vx + 0.3*(x - self.cx)
            self.vy = 0.7*self.vy + 0.3*(y - self.cy)
            self.cx = int(self.alpha*x + (1-self.alpha)*self.cx)
            self.cy = int(self.alpha*y + (1-self.alpha)*self.cy)
        self.last_frame_seen = frame_idx
        self.missing = 0
        return (self.cx, self.cy)

    def predict(self):
        if self.cx is None:
            return None
        px = int(self.cx + self.vx)
        py = int(self.cy + self.vy)
        return (px, py)


# The heavy-lifting processing function

def process_video(input_path: str, output_video_path: str, chart_path: str, model_path: str):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    pass_counter = 0
    current_possessor = None
    candidate_possessor = None
    candidate_hold = 0

    frame_idx = 0
    start_time = time.time()

    frame_history = []
    pass_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=YOLO_CONF, imgsz=640)[0]

        player_detections = []
        ball_candidates = []

        for box in results.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = float(box.conf)
            xy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xy]
            if label == "player" and conf >= PLAYER_CONF_THR:
                player_detections.append((x1, y1, x2, y2))
            elif label == "ball" and conf >= BALL_CONF_THR:
                bx = (x1+x2)//2; by = (y1+y2)//2
                ball_candidates.append((bx, by, conf, (x1,y1,x2,y2)))

        tracks = player_tracker.update(player_detections)

        ball_center = None
        if ball_candidates:
            ball_candidates.sort(key=lambda x: -x[2])
            bc = ball_candidates[0]
            ball_center = (bc[0], bc[1])
            ball_tracker.update_with_detection(ball_center, frame_idx)
        else:
            last_pred = ball_tracker.predict()
            fb = fallback_detect_ball(frame, last_center=last_pred)
            if fb is None and ball_tracker.cx is None:
                fb = fallback_detect_ball(frame, last_center=None)
            if fb is not None:
                ball_center = fb
                ball_tracker.update_with_detection(ball_center, frame_idx)
            else:
                pred = ball_tracker.predict()
                if pred is not None:
                    ball_center = pred

        # Draw players
        for tid, bbox, centroid in tracks:
            x1,y1,x2,y2 = bbox
            cx,cy = centroid
            cv2.rectangle(frame, (x1,y1),(x2,y2),(200,100,0),2)
            cv2.putText(frame, f"P{tid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,100,0), 2)
            cv2.circle(frame, (cx,cy), 3, (200,100,0), -1)

        # Draw ball
        if ball_center is not None:
            bx, by = int(ball_center[0]), int(ball_center[1])
            cv2.circle(frame, (bx,by), 6, (0,255,0), -1)
            cv2.putText(frame, "Ball", (bx+8, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Pass counting logic
        new_possessor = None
        if ball_center is not None:
            bx, by = int(ball_center[0]), int(ball_center[1])
            for tid, bbox, centroid in tracks:
                x1,y1,x2,y2 = bbox
                margin = 6
                if (x1-margin) <= bx <= (x2+margin) and (y1-margin) <= by <= (y2+margin):
                    new_possessor = tid
                    break

        if new_possessor is None:
            candidate_possessor = None
            candidate_hold = 0
        else:
            if current_possessor is None:
                if candidate_possessor == new_possessor:
                    candidate_hold += 1
                else:
                    candidate_possessor = new_possessor
                    candidate_hold = 1
                if candidate_hold >= MIN_HOLD_FRAMES:
                    current_possessor = candidate_possessor
                    candidate_possessor = None
                    candidate_hold = 0
            else:
                if new_possessor == current_possessor:
                    candidate_possessor = None
                    candidate_hold = 0
                else:
                    if candidate_possessor == new_possessor:
                        candidate_hold += 1
                    else:
                        candidate_possessor = new_possessor
                        candidate_hold = 1
                    if candidate_hold >= MIN_HOLD_FRAMES:
                        if new_possessor != current_possessor:
                            pass_counter += 1
                        current_possessor = new_possessor
                        candidate_possessor = None
                        candidate_hold = 0

        if current_possessor is not None:
            cv2.putText(frame, f"Possessor: P{current_possessor}", (30, H-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,200), 2)
        cv2.putText(frame, f"Passes: {pass_counter}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,200), 3)

        out.write(frame)

        frame_history.append(frame_idx)
        pass_history.append(pass_counter)

    cap.release()
    out.release()

    # Save chart
    plt.figure(figsize=(10,5))
    plt.plot(frame_history, pass_history, label="Passes")
    plt.xlabel("Frame")
    plt.ylabel("Pass Counter")
    plt.title("Passes per Frame")
    plt.legend()
    plt.grid(True)
    plt.savefig(chart_path)
    plt.close()

    elapsed = time.time() - start_time

    return pass_counter, elapsed


# -----------------------
# Streamlit flow
# -----------------------
if uploaded_file is not None:
    tdir = tempfile.mkdtemp()
    input_path = os.path.join(tdir, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Saved uploaded video to: {input_path}")

    output_video_path = os.path.join(tdir, "annotated_output.mp4")
    chart_path = os.path.join(tdir, "pass_chart.png")

    start_btn = st.button("Start processing")

    if start_btn:
        if not Path(MODEL_PATH_INPUT).exists():
            st.error(f"Model path not found: {MODEL_PATH_INPUT}")
        else:
            placeholder = st.empty()
            # show 3D animation while processing (client-side)
            with placeholder.container():
                st.components.v1.html(THREE_JS_HTML, height=400)

            # Run processing (blocking) — the HTML animation keeps running client-side
            with st.spinner("Processing video with YOLO model — this may take a while..."):
                passes, elapsed = process_video(input_path, output_video_path, chart_path, MODEL_PATH_INPUT)

            # remove animation
            placeholder.empty()

            st.success(f"Processing finished in {elapsed:.1f}s — total passes: {passes}")

            # Show annotated video
            st.subheader("Annotated video")
            st.video(output_video_path)

            # Show chart
            st.subheader("Passes per Frame")
            st.image(chart_path, use_column_width=True)

            # Downloads
            with open(output_video_path, "rb") as f:
                btn = st.download_button(label="Download annotated video", data=f, file_name="annotated_output.mp4", mime="video/mp4")
            with open(chart_path, "rb") as f:
                st.download_button(label="Download pass chart", data=f, file_name="pass_chart.png", mime="image/png")

            # Also show a small CSV table (frame,passes)
            import csv
            csv_path = os.path.join(tdir, "passes.csv")
            # regenerate simple csv from chart data by re-running a lighter pass over the video is expensive, so
            # instead we create a CSV with cumulative pass counts per frame based on the chart we saved earlier.
            # For clarity we will reconstruct from the saved chart image is not helpful; instead, run a tiny second pass
            # is inefficient. To keep this example simple, we will not produce the CSV here. If you want CSV, enable
            # a parameter to save frame_history inside process_video and return it.

            st.info("If you want a CSV of frame vs pass counts, rerun with CSV output enabled (I can add this flag).")

else:
    st.info("Upload a video above, then set your model path and press Start processing.")


# Footer
st.markdown("---")
st.caption("App built with ultralytics YOLO, OpenCV and Streamlit. Make sure requirements are installed in your venv before running: pip install -r requirements.txt")