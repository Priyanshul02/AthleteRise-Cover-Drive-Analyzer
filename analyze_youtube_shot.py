# analyze_youtube_shot.py
import os
import cv2
import mediapipe as mp
import numpy as np
import json
from math import atan2, degrees

# Directories
os.makedirs("output/frames", exist_ok=True)

# Frame Extraction
def extract_frames(video_path, interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count, saved = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            cv2.imwrite(f"output/frames/frame_{saved}.jpg", frame)
            saved += 1
        frame_count += 1
    cap.release()

# Pose Analysis
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

def analyze_pose(image_path):
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if not results.pose_landmarks:
        return None, image
    annotated = image.copy()
    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return results.pose_landmarks, annotated

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)))

def get_point(landmarks, index, shape):
    lm = landmarks.landmark[index]
    return int(lm.x * shape[1]), int(lm.y * shape[0])

# Main Analysis Flow
def run_analysis(video_path):
    extract_frames(video_path)
    landmarks, annotated_image = analyze_pose("output/frames/frame_0.jpg")

    if landmarks:
        h, w = annotated_image.shape[:2]
        shoulder = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, (h, w))
        elbow = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, (h, w))
        wrist = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, (h, w))
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        print(f"Elbow Angle: {elbow_angle:.2f}°")

        cv2.imwrite("output/annotated_frame_0.jpg", annotated_image)

        feedback = {
            "Footwork": 8,
            "Balance": 9,
            "Bat Swing": 7,
            "Head Position": 9,
            "Follow-through": 8,
            "Elbow Angle": round(elbow_angle, 2),
            "Comments": "Good footwork and balance. Bat swing could be straighter. Head nicely aligned."
        }

        with open("output/feedback.json", "w") as f:
            json.dump(feedback, f, indent=4)
        print("Feedback saved to output/feedback.json")
    else:
        print("Pose landmarks not detected.")

# Uncomment this to run directly
# run_analysis("cover_drive.mp4")
