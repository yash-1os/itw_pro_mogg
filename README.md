[project_readme (1).md](https://github.com/user-attachments/files/34903134/project_readme.1.md)
# DeepFace Emotion & Face-Mogging System

A dual-interface AI vision application that performs real-time facial expression analysis and competitive "Face-Mogging" rankings. It features both a local Flask web dashboard for webcam/image analysis and a transparent, click-through Tkinter desktop overlay for live meeting platforms (Zoom, Discord, Google Meet).

## Features

### 1. Dual Operating Modes
* **Standard Emotion Mode:** Tracks a single subject's dominant emotion and provides real-time confidence scores for 7 core emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
* **Face-Mogging Mode:** Detects multiple faces in a frame and mathematically ranks them. 
  * **#1 Rank (The Mogger):** Crowned with a gold tracking reticle and their Mog Score.
  * **#2+ Ranks (Mogged):** Automatically tracked with a tilted, black censor bar drawn directly across their eyes, overlaid with bold red "MOGGED" text and their losing score.

### 2. Dual Interfaces
* **Web Dashboard (`app.py`):** A Flask-powered UI with vanilla HTML/JS. Supports drag-and-drop image analysis and live MJPEG webcam streaming with real-time JSON telemetry polling.
* **Meeting Overlay (`meeting_overlay.py`):** A standalone desktop HUD. Drag a selection box over any video call to capture the screen region. The UI floats transparently over your applications, allowing you to click through it while tracking participants.

### 3. Post-Session Analytics
When a live tracking session is stopped, the application aggregates the frame-by-frame emotion data, saves it to a CSV (`telemetry_history.csv` or `meeting_log.csv`), and generates a dark-themed Matplotlib timeline graph of the psychological behavioral trends.

## How the Model Works

The application uses a hybrid computer vision pipeline to ensure high performance and prevent stream freezing:

1. **Face Detection (OpenCV):** Instead of relying on heavy AI detectors, the system converts the video frame to grayscale, equalizes the lighting contrast, and uses lightweight OpenCV Haar Cascades (`haarcascade_frontalface_default.xml` and `alt2`) to rapidly draw bounding boxes around faces in real-time.
2. **Eye Tracking:** A secondary cascade (`haarcascade_eye.xml`) scans only the top 60% of the detected face to pinpoint the left and right pupils. This prevents false positives (like nostrils) and is used to calculate the exact trigonometric angle of a person's head tilt.
3. **Emotion Extraction (DeepFace):** The system pads the detected face box by 15%, crops it out of the main frame, and feeds *only* that cropped image into the DeepFace AI model. DeepFace analyzes the facial features and returns a confidence percentage array for 7 core emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

## The Mog Score Algorithm
The 0-100 Mog Score is calculated using three weighted metrics:
1. **Symmetry (45%):** The face crop is normalized, split, and the left side is correlated against a flipped right side. Severe head tilts (calculated via pupil distance and angle) incur a penalty.
2. **Composure / Stoic Aura (40%):** A custom behavioral formula that rewards `neutral` and `happy` expressions while heavily penalizing erratic or volatile emotions (`fear`, `sad`, `surprise`, `disgust`).
3. **Presence (15%):** The bounding box pixel area relative to the largest face in the current frame, rewarding physical proximity to the camera.

## How to Run

1. **Quick Start (Windows):** 
   Simply double-click the `Start_App.bat` file in the main folder to launch the configuration menu.
2. **Manual Launch:** 
   Open a terminal in the project folder and run:
   ```bash
   python launcher.py
   ```
From the launcher, you can toggle the Face-Mogging mode and choose to open either the Web Dashboard or the Live Meeting Overlay.

## Repository Management
This project includes a custom Bash script (`doitblud.sh`) to automate GitHub uploads. 
To sync your latest changes to GitHub from a Linux terminal (or WSL/Ubuntu):
```bash
./doitblud.sh .
```
