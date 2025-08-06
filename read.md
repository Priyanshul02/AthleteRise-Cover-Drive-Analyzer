# 🏏 AthleteRise – Cover Drive Analyzer

A Streamlit-based AI tool for analyzing the biomechanics of a cricket **cover drive** using pose estimation. Upload a video, and the app automatically extracts the first frame, detects the pose using MediaPipe, calculates elbow angle and other metrics, and provides visual + textual feedback.

---

## 📦 Features

- 🔍 Pose estimation with **MediaPipe**
- 🎯 Biomechanical metric: Elbow angle (example)
- 📝 Auto-scored feedback on:
  - Footwork
  - Balance
  - Bat Swing
  - Head Position
  - Follow-through
- 📼 Streamlit UI with **resized video & image**
- 📁 JSON feedback report generation

---

## 🚀 How to Run

### 1. 🛠️ Install Dependencies

```bash
pip install streamlit opencv-python mediapipe numpy
