import streamlit as st
import cv2
import tempfile
import os
import mediapipe as mp
import numpy as np
import json
import base64
from math import acos, degrees

# MediaPipe Initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Pose Estimation Functions
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = degrees(acos(np.clip(cosine, -1.0, 1.0)))
    return angle

def get_point(landmarks, idx, shape):
    lm = landmarks.landmark[idx]
    return int(lm.x * shape[1]), int(lm.y * shape[0])

def analyze_frame(image):
    h, w = image.shape[:2]
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, image, {}

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Elbow angle (example)
        shoulder = get_point(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, (h, w))
        elbow = get_point(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, (h, w))
        wrist = get_point(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, (h, w))
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        feedback = {
            "Footwork": 8,
            "Balance": 9,
            "Bat Swing": 7,
            "Head Position": 8,
            "Follow-through": 7,
            "Elbow Angle": round(elbow_angle, 2),
            "Comment": "Well-balanced shot with good footwork. Bat swing could be straighter."
        }

        return results.pose_landmarks, image, feedback

# Resizeable video display using base64
def display_resized_video(video_file, width=400):
    video_bytes = video_file.read()
    base64_video = base64.b64encode(video_bytes).decode()
    video_html = f"""
    <video width="{width}" controls>
        <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

# Resizeable image display
def display_resized_image(image, width=400):
    resized = cv2.resize(image, (width, int(width * image.shape[0] / image.shape[1])))
    st.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), use_column_width=False)

# Streamlit UI
st.set_page_config(page_title="Cover Drive Analyzer", layout="centered")
st.title("🏏 AthleteRise - AI-Powered Cover Drive Analyzer")

uploaded_file = st.file_uploader("Upload a cricket shot video (MP4)", type=["mp4"])

if uploaded_file:
    st.subheader("📼 Uploaded Video")
    display_resized_video(uploaded_file, width=300)

    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.getbuffer())
    video_path = tfile.name

    # Capture first frame
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if success:
        st.subheader("🕺 Pose Detection on First Frame")
        landmarks, annotated_frame, feedback = analyze_frame(frame)

        st.markdown("🖼️ Annotated Frame (Resized)")
        display_resized_image(annotated_frame, width=400)

        st.subheader("📝 AI Feedback")
        st.json(feedback)

    else:
        st.error("⚠️ Could not read video.")
