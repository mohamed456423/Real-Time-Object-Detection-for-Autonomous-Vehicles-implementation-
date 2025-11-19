import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from datetime import datetime
import random
import pandas as pd

# =============================
# CONFIG
# =============================
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
FRAME_SKIP = 1  # Live camera should NOT skip frames

# =============================
# Helper: Generate YOLO-style colors
# =============================
def get_class_colors(names_dict):
    random.seed(42)
    colors = {}
    for cls_id, cls_name in names_dict.items():
        colors[cls_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    return colors


def draw_yolo_box(frame, box, cls_name, conf, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{cls_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="YOLOv8 Video Analysis", layout="wide")
st.title("🚗 YOLOv8 Real-Time Object Detection Dashboard")

mode = st.sidebar.radio(
    "Choose Mode:",
    ["📹 Upload Video", "📷 Live Camera Detection"]
)

model = YOLO("best.pt")  # Load model once
class_colors = get_class_colors(model.names)


# ================================================================
# 📷 UPDATED LIVE CAMERA LOGIC (Start → Run → Stop → Generate Report)
# ================================================================
if mode == "📷 Live Camera Detection":

    st.subheader("📡 Live Camera Object Detection")

    start_button = st.button("▶ Start Camera")
    stop_button = st.button("⛔ Stop Camera")

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    report_placeholder = st.empty()

    # Keep camera state in session state
    if "running" not in st.session_state:
        st.session_state.running = False

    if start_button:
        st.session_state.running = True
        report_placeholder.empty()  # Clear old report

    if stop_button:
        st.session_state.running = False

    # Run live camera LOOP
    if st.session_state.running:

        # Initialize camera stats
        class_counts = {name: 0 for name in model.names.values()}
        fps_list = []
        total_frames = 0
        low_light = False
        prev_time = time.time()

        cap = cv2.VideoCapture(0)
        status_placeholder.info("📡 **Camera Running…**")

        while st.session_state.running:

            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("❌ Cannot access webcam.")
                break

            total_frames += 1

            # Brightness check
            brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if brightness < 60:
                low_light = True

            # YOLO inference
            results = model(frame, conf=CONF_THRESHOLD)

            for b in results[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                cls_name = model.names[cls_id]

                class_counts[cls_name] += 1
                frame = draw_yolo_box(frame, (x1, y1, x2, y2), cls_name, conf, class_colors[cls_name])

            # FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            fps_list.append(fps)
            prev_time = current_time

            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Stop immediately if user clicks stop
            if stop_button:
                st.session_state.running = False
                break

        cap.release()

        # =================================================
        # After STOP → Generate full report
        # =================================================
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        status_placeholder.success("📄 Live Camera Session Ended — Report Ready")

        with report_placeholder.container():
            st.markdown("## 📊 Live Camera Detection Report")

            col1, col2 = st.columns(2)
            col1.metric("Total Frames", total_frames)
            col2.metric("Average FPS", f"{avg_fps:.2f}")

            st.markdown("### 🌍 Environment Conditions")
            st.write(f"- 🌑 Low Light: {'🟢 Yes' if low_light else '🔴 No'}")

            # Detection Statistics
            st.markdown("### 📦 Object Detection Stats")
            df_counts = pd.DataFrame({
                "Class": list(class_counts.keys()),
                "Detections": list(class_counts.values())
            })
            st.dataframe(df_counts)
            st.bar_chart(df_counts.set_index("Class"))

            st.markdown("### 🧠 Insights")

            if df_counts["Detections"].sum() == 0:
                st.info("No objects detected.")
            else:
                most = df_counts.loc[df_counts["Detections"].idxmax()]
                st.write(f"**Most Detected Object:** `{most['Class']}` ({most['Detections']} times)")

            if avg_fps < 12:
                st.warning("⚠ Low FPS — try using YOLOv8n.")
            else:
                st.success("🚀 Good real-time performance!")


# ================================================================
# 📹 VIDEO UPLOAD MODE (Your original version, unchanged)
# ================================================================
elif mode == "📹 Upload Video":
    st.write("Upload a video, run YOLO detection, and generate an interactive report.")

    video_file = st.file_uploader("🎥 Upload Video", type=["mp4", "avi", "mov"])
    stop_button = st.button("⛔ Stop Analysis")

    if video_file:

        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        st.video(video_path)

        # Initialize report
        class_counts = {name: 0 for name in model.names.values()}
        report = {
            "video": video_file.name,
            "date": str(datetime.now()),
            "total_frames": 0,
            "processed_frames": 0,
            "average_fps": 0,
            "environment_assessment": {"low_light": False},
        }

        cap = cv2.VideoCapture(video_path)
        fps_list = []
        prev_time = time.time()
        frame_counter = 0

        st.subheader("📡 Live Detection Feed")
        frame_placeholder = st.empty()

        while cap.isOpened():
            if stop_button:
                break

            ret, frame = cap.read()
            if not ret:
                break

            report["total_frames"] += 1
            frame_counter += 1

            if frame_counter % FRAME_SKIP != 0:
                continue

            report["processed_frames"] += 1

            brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if brightness < 60:
                report["environment_assessment"]["low_light"] = True

            results = model(frame)
            for b in results[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                cls_name = model.names[cls_id]
                class_counts[cls_name] += 1

                frame = draw_yolo_box(frame, (x1, y1, x2, y2), cls_name, conf, class_colors[cls_name])

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            fps_list.append(1 / (time.time() - prev_time))
            prev_time = time.time()

        cap.release()
        report["average_fps"] = sum(fps_list)/len(fps_list) if fps_list else 0

        st.success("📄 Video Analysis Complete!")

        st.markdown("## 📊 Final Interactive Report")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Frames", report["total_frames"])
        col2.metric("Processed Frames", report["processed_frames"])
        col3.metric("Average FPS", f"{report['average_fps']:.2f}")

        st.markdown("### 🌍 Environment Assessment")
        st.write(
            f"""
            - 🌑 **Low Light:** {"🟢 Yes" if report["environment_assessment"]["low_light"] else "🔴 No"}
            """
        )

        st.markdown("### 📦 Object Detection Statistics")
        df_counts = pd.DataFrame({
            "Class": list(class_counts.keys()),
            "Detections": list(class_counts.values())
        })

        st.dataframe(df_counts)
        st.bar_chart(df_counts.set_index("Class"))

        st.markdown("### 🧠 Insights")

        most_detected = df_counts.loc[df_counts["Detections"].idxmax()]
        st.write(f"**Most Detected Object:** `{most_detected['Class']}` ({most_detected['Detections']} times)")
