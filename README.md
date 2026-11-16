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

## The Mog Score Algorithm
The 0-100 Mog Score is calculated using three weighted metrics:
1. **Symmetry (45%):** The face crop is normalized, split, and the left side is correlated against a flipped right side. Severe head tilts (calculated via pupil distance and angle) incur a penalty.
2. **Composure / Stoic Aura (40%):** A custom behavioral formula that rewards `neutral` and `happy` expressions while heavily penalizing erratic or volatile emotions (`fear`, `sad`, `surprise`, `disgust`).
3. **Presence (15%):** The bounding box pixel area relative to the largest face in the current frame, rewarding physical proximity to the camera.

## Technology Stack
* **Computer Vision:** OpenCV (`cv2`) with local Haar Cascades (`frontalface_default`, `frontalface_alt2`, `eye`) for highly robust, offline bounding-box detection.
* **AI Engine:** DeepFace (used strictly for emotion extraction on pre-cropped regions to bypass internal detection crashes).
* **Backend:** Python, Flask, NumPy.
* **Desktop App:** Tkinter, `mss` (fast screen capture), Win32 `ctypes` (for transparent window layering and anti-capture masking).

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd itw_pro_mogg
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install flask deepface opencv-python numpy matplotlib mss
   ```

3. **Run the Application:**
   * **Windows:** Simply double-click `Start_App.bat` to launch the mode selector.
   * **Manual Launch:** Run `python launcher.py` to open the main configuration menu. From there, you can select your operating mode and launch either the Web Dashboard or the Live Meeting Overlay.

## Repository Management
This project includes a custom Bash script (`doitblud.sh`) to automate GitHub uploads. 
To sync your latest changes to GitHub from a Linux terminal (or WSL/Ubuntu):
```bash
./doitblud.sh .
```
