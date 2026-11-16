import sys
import os

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open("overlay_error.log", "w", encoding="utf-8")

# Quiet TensorFlow and keep DeepFace working with Keras 3 installs (only when tf-keras is installed).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
try:
    import tf_keras  # noqa: F401
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
except ImportError:
    pass

import mss
import math
import ctypes
import traceback
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
import threading
import time
import csv
import json
import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CHROMA_KEY = "#010203"  # Transparent mask color for Windows overlay
GOLD = "#ffd700"

# Composure weights: calm reads as composed, erratic emotions are punished
COMPOSURE_W = {"neutral": 1.0, "happy": 0.85, "surprise": -0.2,
               "angry": -0.5, "disgust": -0.6, "sad": -0.7, "fear": -0.9}

MAX_FACES = 6          # faces tracked / ranked at once
EMO_REFRESH = 0.8      # seconds before a face's emotions are re-analysed
LOSS_GRACE = 0.9       # keep drawing a face this long after the detector drops it (stops flicker)

is_running = False
latest_frame = None
face_locked = False
current_frame_count = 0
live_telemetry = {"dominant": "neutral", "emotions": {k: 0.0 for k in EMOTION_KEYS}, "faces": []}
session_history = []
lock = threading.Lock()


# ----------------------------------------------------------------------------
# Error log (the app runs without a console, so failures must land in a file)
# ----------------------------------------------------------------------------
def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


_last_err = {"msg": "", "t": 0.0}


def log_error(context):
    msg = traceback.format_exc()
    now = time.time()
    if msg == _last_err["msg"] and now - _last_err["t"] < 30:
        return
    _last_err["msg"], _last_err["t"] = msg, now
    try:
        with open(os.path.join(_app_dir(), "overlay_error.log"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {context}\n{msg}\n")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Face-Mogging mode flag: shared with the launcher and the web dashboard through mode_config.json
# ----------------------------------------------------------------------------
def _mode_file():
    return os.path.join(_app_dir(), "mode_config.json")


def load_mogging_mode():
    try:
        with open(_mode_file(), "r") as f:
            return bool(json.load(f).get("mogging_mode", True))
    except Exception:
        return True          # no file yet: keep the overlay's original behaviour (mogging on)


def save_mogging_mode(value):
    try:
        with open(_mode_file(), "w") as f:
            json.dump({"mogging_mode": bool(value)}, f)
    except Exception:
        log_error("save_mogging_mode")


def make_window_clickthrough_and_capture_safe(tk_win):
    """Makes the overlay click-through to apps underneath and invisible to mss screen capture."""
    try:
        tk_win.update_idletasks()
        user32 = ctypes.windll.user32
        hwnd = tk_win.winfo_id()
        parent = user32.GetParent(hwnd)
        target_hwnd = parent if parent else hwnd

        # WS_EX_LAYERED (0x80000) | WS_EX_TRANSPARENT (0x20) -> clicks pass straight through
        GWL_EXSTYLE = -20
        style = user32.GetWindowLongW(target_hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(target_hwnd, GWL_EXSTYLE, style | 0x00080000 | 0x00000020)

        # WDA_EXCLUDEFROMCAPTURE (0x11) -> mss sees clean faces underneath without flickering
        user32.SetWindowDisplayAffinity(target_hwnd, 0x00000011)
        if target_hwnd != hwnd:
            user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Detection: offline Haar cascades (frontal_default, falling back to alt2) + eye cascade for tilt
# ----------------------------------------------------------------------------
def _cascade_dirs():
    dirs = []
    try:
        dirs.append(cv2.data.haarcascades)
    except Exception:
        pass
    dirs.append(os.path.join(os.path.dirname(cv2.__file__), "data"))
    dirs += [_app_dir(), os.path.join(_app_dir(), "haarcascades")]     # drop the XML files here if cv2 has none
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        dirs += [os.path.join(base, "cv2", "data"), base]
    return dirs


def load_cascade(name):
    for d in _cascade_dirs():
        path = os.path.join(d, name)
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                return c
    return None


_cascades = {}


def get_cascades():
    if not _cascades:
        _cascades["face"] = load_cascade("haarcascade_frontalface_default.xml")
        _cascades["face2"] = load_cascade("haarcascade_frontalface_alt2.xml")
        _cascades["eye"] = load_cascade("haarcascade_eye.xml")
    return _cascades


def _iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def detect_eyes(gray, x, y, w, h):
    """Pupil-ish centres from the eye cascade on the upper 60% of the face box: (lx, ly, rx, ry) or None."""
    cas = get_cascades().get("eye")
    if cas is None or w < 40:
        return None
    rh = max(1, int(h * 0.6))
    roi = gray[y:y + rh, x:x + w]
    if roi.size == 0:
        return None
    roi = cv2.equalizeHist(roi)
    mn = max(8, int(w * 0.12))
    mx = max(mn + 2, int(w * 0.45))
    found = cas.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(mn, mn), maxSize=(mx, mx))
    left = right = None
    for (ex, ey, ew, eh) in found:
        cx, cy = ex + ew / 2.0, ey + eh / 2.0
        if not (0.15 * h <= cy <= 0.55 * h):
            continue
        cand = (ew * eh, cx, cy)
        if cx < w * 0.5:
            if left is None or cand[0] > left[0]:
                left = cand
        else:
            if right is None or cand[0] > right[0]:
                right = cand
    if left is None or right is None:
        return None
    dx, dy = right[1] - left[1], right[2] - left[2]
    if dx < 0.2 * w or abs(dy) > 0.7 * dx:
        return None
    return (x + left[1], y + left[2], x + right[1], y + right[2])


def detect_faces_haar(gray):
    cas = get_cascades()
    H, W = gray.shape[:2]
    scale = 1.0
    work = gray
    if W > 1600:                                   # big captures: detect on a downscaled copy
        scale = 1600.0 / W
        work = cv2.resize(gray, (1600, max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
    work = cv2.equalizeHist(work)
    min_sz = max(32, int(min(work.shape[:2]) * 0.05))

    rects = []
    for key in ("face", "face2"):
        c = cas.get(key)
        if c is None:
            continue
        found = c.detectMultiScale(work, scaleFactor=1.1, minNeighbors=5, minSize=(min_sz, min_sz))
        if len(found):
            rects = [tuple(int(v / scale) for v in r) for r in found]
            break

    rects.sort(key=lambda r: -r[2] * r[3])
    kept = []
    for r in rects:
        if all(_iou(r, k) < 0.4 for k in kept):
            kept.append(r)

    out = []
    for (x, y, w, h) in kept[:MAX_FACES]:
        x, y = max(0, x), max(0, y)
        w, h = min(W - x, w), min(H - y, h)
        if w < 20 or h < 20:
            continue
        out.append({"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2.0, "cy": y + h / 2.0,
                    "eyes": detect_eyes(gray, x, y, w, h)})
    return out


# Detector choice: MediaPipe face detection (bundled model, offline, eye keypoints for tilt) is preferred.
# Haar is the fallback; note that OpenCV 5 wheels no longer ship the Haar XML files.
engine = {"detector": "starting...", "error": ""}
_mp = {"det": None}


def _get_mp_detector():
    if _mp["det"] is None:
        import mediapipe as mp
        _mp["det"] = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return _mp["det"]


def init_detector():
    """MediaPipe first (finds more faces and gives real eye keypoints); Haar cascades as the fallback."""
    try:
        _get_mp_detector()
        engine["detector"] = "MediaPipe"
        return
    except Exception:
        log_error("MediaPipe face detector unavailable, trying Haar")
    cas = get_cascades()
    if cas.get("face") is not None or cas.get("face2") is not None:
        engine["detector"] = "Haar"
        return
    raise RuntimeError("No face detector available. Run: pip install mediapipe==0.10.14 "
                       "(needs the 'solutions' API), or put the Haar cascade XML files next to the app.")


def detect_faces_mediapipe(frame):
    H, W = frame.shape[:2]
    res = _get_mp_detector().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out = []
    for det in (res.detections or []):
        rb = det.location_data.relative_bounding_box
        x, y = max(0, int(rb.xmin * W)), max(0, int(rb.ymin * H))
        w, h = min(W - x, int(rb.width * W)), min(H - y, int(rb.height * H))
        if w < 20 or h < 20:
            continue
        eyes = None
        kps = det.location_data.relative_keypoints
        if len(kps) >= 2:                                   # keypoints 0/1 are the two eyes
            ax, ay, bx, by = kps[0].x * W, kps[0].y * H, kps[1].x * W, kps[1].y * H
            if ax > bx:
                ax, ay, bx, by = bx, by, ax, ay
            eyes = (ax, ay, bx, by)
        out.append({"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2.0, "cy": y + h / 2.0, "eyes": eyes})
    out.sort(key=lambda d: -d["w"] * d["h"])
    return out[:MAX_FACES]


def detect_faces(frame, gray):
    if engine["detector"] == "MediaPipe":
        return detect_faces_mediapipe(frame)
    return detect_faces_haar(gray)


def eye_params(x, y, w, h, eyes):
    """Eye-line centre (as fractions of the box), bar width ratio and head tilt in degrees, or None."""
    if not eyes:
        return None
    lx, ly, rx, ry = eyes
    dist = math.hypot(rx - lx, ry - ly)
    if dist <= 6:
        return None
    ang = max(-35.0, min(35.0, math.degrees(math.atan2(ry - ly, rx - lx))))
    bar_w = max(int(w * 0.85), int(dist * 2.15))
    return ((lx + rx) / 2.0 - x) / w, ((ly + ry) / 2.0 - y) / h, bar_w / float(w), ang


# ----------------------------------------------------------------------------
# Tracking: keeps identity, smoothed geometry, emotions and scores between frames
# ----------------------------------------------------------------------------
class FaceTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now):
        used = set()
        for d in sorted(detections, key=lambda d: -d["w"] * d["h"]):
            best, best_d = None, None
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                dist = math.hypot(d["cx"] - t["cx"], d["cy"] - t["cy"])
                if dist < 0.75 * max(d["w"], t["w"]) and (best_d is None or dist < best_d):
                    best, best_d = tid, dist
            ep = eye_params(d["x"], d["y"], d["w"], d["h"], d["eyes"])
            if best is None:
                best = self.next_id
                self.next_id += 1
                ex, ey, bwr, ang = ep if ep else (0.5, 0.38, 0.88, 0.0)
                self.tracks[best] = {
                    "id": best, "cx": d["cx"], "cy": d["cy"], "w": float(d["w"]), "h": float(d["h"]),
                    "ex": ex, "ey": ey, "bwr": bwr, "ang": ang,
                    "emo": None, "emo_t": 0.0, "sym": None, "comp": 70.0, "mog": None,
                }
            else:
                t = self.tracks[best]
                a = 0.5
                t["cx"] = a * d["cx"] + (1 - a) * t["cx"]
                t["cy"] = a * d["cy"] + (1 - a) * t["cy"]
                t["w"] = a * d["w"] + (1 - a) * t["w"]
                t["h"] = a * d["h"] + (1 - a) * t["h"]
                if ep:                                  # no eyes this frame -> keep last good geometry
                    b = 0.4
                    for key, val in zip(("ex", "ey", "bwr", "ang"), ep):
                        t[key] = b * val + (1 - b) * t[key]
            t = self.tracks[best]
            t["seen"] = now
            t["det"] = (d["x"], d["y"], d["w"], d["h"])
            used.add(best)
        for tid in [k for k, t in self.tracks.items() if now - t["seen"] > LOSS_GRACE]:
            del self.tracks[tid]


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def symmetry_score(gray_crop, tilt_deg):
    """Left vs flipped-right correlation of a 64x64 crop, penalised for heavy head tilt (0-100)."""
    if gray_crop is None or gray_crop.size == 0:
        return 70.0
    norm = cv2.resize(gray_crop, (64, 64)).astype(np.float32)
    left = norm[:, :32].ravel()
    right = cv2.flip(norm[:, 32:], 1).ravel()
    if left.std() < 1e-3 or right.std() < 1e-3:
        corr = 0.0
    else:
        corr = float(np.corrcoef(left, right)[0, 1])
    sym = 50.0 + 50.0 * min(1.0, max(0.0, corr))        # 50-100: an off-centre crop is never scored zero
    tilt = abs(tilt_deg)
    if tilt > 8:
        sym *= max(0.5, 1.0 - (tilt - 8) / 60.0)
    return sym


def composure_score(emo):
    if not emo:
        return 70.0
    raw = sum(COMPOSURE_W[k] * emo.get(k, 0.0) / 100.0 for k in EMOTION_KEYS)
    return min(1.0, max(0.0, raw)) * 100.0


def dominant_of(emo):
    return max(emo, key=emo.get) if emo else "neutral"


# ----------------------------------------------------------------------------
# DeepFace: emotions only, on padded face crops (never on the whole screen)
# ----------------------------------------------------------------------------
_fast_fail = 0


def analyze_emotions(crop):
    global _fast_fail
    try:
        res = None
        if _fast_fail < 3:
            try:
                res = DeepFace.analyze(crop, actions=["emotion"], enforce_detection=False,
                                       detector_backend="skip", silent=True)
                _fast_fail = 0
            except Exception:
                _fast_fail += 1
        if res is None:
            res = DeepFace.analyze(crop, actions=["emotion"], enforce_detection=False)
        if isinstance(res, list):
            res = res[0]
        e = res["emotion"]
        return {k: float(e.get(k, 0.0)) for k in EMOTION_KEYS}
    except Exception:
        log_error("DeepFace.analyze")
        return None


def analyze_frame(frame, tracker, now):
    """Detect -> track -> emotions -> score -> rank. Returns ranked face dicts (best first)."""
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.update(detect_faces(frame, gray), now)

    live = [t for t in tracker.tracks.values() if now - t["seen"] <= LOSS_GRACE]
    if not live:
        return []

    # refresh emotions for the stalest faces that are visible right now (padded 15% crop)
    visible = sorted((t for t in live if t["seen"] == now), key=lambda t: t["emo_t"])
    for t in visible[:3]:
        if t["emo"] is not None and time.time() - t["emo_t"] < EMO_REFRESH:
            break
        x, y, w, h = t["det"]
        px, py = int(0.15 * w), int(0.15 * h)
        crop = frame[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]
        if crop.size == 0:
            continue
        emo = analyze_emotions(np.ascontiguousarray(crop))
        if emo:
            prev = t["emo"]
            t["emo"] = emo if prev is None else {k: 0.5 * prev[k] + 0.5 * emo[k] for k in EMOTION_KEYS}
            t["emo_t"] = time.time()

    max_area = max(t["w"] * t["h"] for t in live)
    for t in live:
        if t["seen"] == now:
            x, y, w, h = t["det"]
            s = symmetry_score(gray[y:y + h, x:x + w], t["ang"])
            t["sym"] = s if t["sym"] is None else 0.5 * t["sym"] + 0.5 * s
        sym = t["sym"] if t["sym"] is not None else 70.0
        t["comp"] = composure_score(t["emo"])
        pres = 100.0 * (t["w"] * t["h"]) / max_area
        raw = 0.45 * sym + 0.40 * t["comp"] + 0.15 * pres
        t["mog"] = raw if t["mog"] is None else 0.65 * t["mog"] + 0.35 * raw   # smooth: no rank flicker

    live.sort(key=lambda t: -t["mog"])
    top = live[0]["mog"]
    ranked = []
    for i, t in enumerate(live):
        w, h = t["w"], t["h"]
        x, y = t["cx"] - w / 2.0, t["cy"] - h / 2.0
        eye_x, eye_y = x + t["ex"] * w, y + t["ey"] * h
        ranked.append({
            "id": t["id"],
            "rank": i + 1,
            "mog_score": round(t["mog"], 1),
            "symmetry": round(t["sym"] if t["sym"] is not None else 70.0, 1),
            "composure": round(t["comp"], 1),
            "dominant": dominant_of(t["emo"]),
            "emotions": dict(t["emo"]) if t["emo"] else {k: 0.0 for k in EMOTION_KEYS},
            "has_emo": t["emo"] is not None,
            "gap": round(t["mog"] - top, 1),
            "box": (int(x), int(y), int(w), int(h)),
            "eye_bar": (eye_x, eye_y, int(t["bwr"] * w), max(22, int(h * 0.22)), t["ang"]),
        })
    return ranked


class RegionSelector:
    def __init__(self):
        self.region = None
        self.start_x = 0
        self.start_y = 0
        self.rect = None

        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.30)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="black", cursor="cross")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.create_text(
            self.root.winfo_screenwidth() // 2, 50,
            text="DRAG A BOX AROUND SPEAKERS  |  ENTER for Full Screen  |  ESC to Cancel",
            fill="#42d99a", font=("Segoe UI", 14, "bold")
        )

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Return>", self.select_fullscreen)
        self.root.bind("<Escape>", self.cancel)

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#42d99a", width=3
        )

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        left = min(self.start_x, event.x)
        top = min(self.start_y, event.y)
        width = abs(event.x - self.start_x)
        height = abs(event.y - self.start_y)
        if width > 30 and height > 30:
            self.region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
            self.root.destroy()

    def select_fullscreen(self, event=None):
        self.region = {"left": 0, "top": 0, "width": self.root.winfo_screenwidth(), "height": self.root.winfo_screenheight()}
        self.root.destroy()

    def cancel(self, event=None):
        self.region = None
        self.root.destroy()

    def get_region(self):
        self.root.mainloop()
        return self.region


def screen_capture_worker(region):
    global latest_frame, current_frame_count, is_running
    with mss.mss() as sct:
        while is_running:
            try:
                sct_img = sct.grab(region)
                # BGRA -> contiguous BGR (OpenCV and DeepFace need a contiguous array)
                latest_frame = np.ascontiguousarray(np.array(sct_img)[:, :, :3])
                current_frame_count += 1
            except Exception:
                log_error("screen_capture_worker")
                time.sleep(0.5)
            time.sleep(0.03)


def ai_analysis_worker():
    global latest_frame, face_locked, live_telemetry, session_history, is_running
    tracker = FaceTracker()
    try:
        init_detector()
    except Exception as e:
        engine["error"] = str(e)
        log_error("init_detector")
        return
    analyze_emotions(np.zeros((96, 96, 3), np.uint8))       # warm-up: load the emotion model once

    while is_running:
        frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        frame_copy = frame.copy()
        fc = current_frame_count
        try:
            ranked = analyze_frame(frame_copy, tracker, time.time())
            with lock:
                if ranked:
                    top = ranked[0]
                    face_locked = True
                    live_telemetry["dominant"] = top["dominant"]
                    live_telemetry["emotions"] = top["emotions"]
                    live_telemetry["faces"] = ranked
                    if top["has_emo"]:
                        session_history.append({
                            "frame": fc,
                            "dominant": top["dominant"],
                            "emotions": top["emotions"]
                        })
                else:
                    face_locked = False
                    live_telemetry["faces"] = []
        except Exception:
            log_error("ai_analysis_worker")
            with lock:
                face_locked = False
                live_telemetry["faces"] = []
        time.sleep(0.03)


class FloatingHUD:
    def __init__(self, region):
        self.region = region
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.configure(bg="#10182a", highlightbackground="#f5b800", highlightthickness=1)
        self._fonts = {}

        win_w, win_h = 285, 392
        screen_w = self.root.winfo_screenwidth()
        self.root.geometry(f"{win_w}x{win_h}+{screen_w - win_w - 20}+20")

        self.root.bind("<ButtonPress-1>", self.start_move)
        self.root.bind("<B1-Motion>", self.do_move)

        # Transparent Click-Through Canvas directly over the captured video region
        self.face_overlay = tk.Toplevel(self.root)
        self.face_overlay.overrideredirect(True)
        self.face_overlay.attributes("-topmost", True)
        self.face_overlay.configure(bg=CHROMA_KEY)
        self.face_overlay.wm_attributes("-transparentcolor", CHROMA_KEY)
        self.face_overlay.geometry(f"{region['width']}x{region['height']}+{region['left']}+{region['top']}")

        self.face_canvas = tk.Canvas(
            self.face_overlay, width=region["width"], height=region["height"],
            bg=CHROMA_KEY, highlightthickness=0
        )
        self.face_canvas.pack(fill="both", expand=True)
        make_window_clickthrough_and_capture_safe(self.face_overlay)

        # Control HUD UI
        top_frame = tk.Frame(self.root, bg="#080d19", padx=10, pady=6)
        top_frame.pack(fill="x")

        self.status_label = tk.Label(
            top_frame, text="● SCANNING...", font=("Segoe UI", 8, "bold"),
            fg="#ff969f", bg="#080d19"
        )
        self.status_label.pack(side="left")

        stop_btn = tk.Button(
            top_frame, text="STOP & PLOT", font=("Segoe UI", 8, "bold"),
            fg="#080d19", bg="#42d99a", activebackground="#6ee7d1",
            relief="flat", cursor="hand2", padx=8, pady=2,
            command=self.stop_and_summarize
        )
        stop_btn.pack(side="right")

        self.mog_var = tk.BooleanVar(value=load_mogging_mode())
        tk.Checkbutton(
            self.root, text="Face-Mogging Mode", variable=self.mog_var, command=self._on_mog_toggle,
            font=("Segoe UI", 9, "bold"), fg="#ff4a4a", bg="#10182a", selectcolor="#080d19",
            activebackground="#10182a", activeforeground="#ff4a4a", cursor="hand2", anchor="w"
        ).pack(fill="x", padx=12, pady=(6, 0))

        self.dom_label = tk.Label(
            self.root, text="MOG SCORE: --", font=("Segoe UI", 14, "bold"),
            fg="#f5b800", bg="#10182a"
        )
        self.dom_label.pack(pady=(8, 2))

        self.conf_label = tk.Label(
            self.root, text="Loading Face-Mogging engine...", font=("Segoe UI", 8),
            fg="#8797ff", bg="#10182a"
        )
        self.conf_label.pack(pady=(0, 6))

        self.versus_label = tk.Label(
            self.root, text="Solo Mode (1 face)", font=("Segoe UI", 8, "bold"),
            fg="#eef3ff", bg="#182442", padx=8, pady=4
        )
        self.versus_label.pack(fill="x", padx=12, pady=(0, 8))

        self.bars = {}
        self.pct_labels = {}
        bars_container = tk.Frame(self.root, bg="#10182a", padx=12)
        bars_container.pack(fill="both", expand=True)

        for emo in EMOTION_KEYS:
            row = tk.Frame(bars_container, bg="#10182a")
            row.pack(fill="x", pady=2)

            tk.Label(
                row, text=emo.capitalize(), width=8, anchor="w",
                font=("Segoe UI", 8), fg="#dfe7fb", bg="#10182a"
            ).pack(side="left")

            track = tk.Frame(row, bg="#202b42", width=125, height=8)
            track.pack(side="left", padx=4)
            track.pack_propagate(False)

            fill = tk.Frame(track, bg="#42d99a", width=0, height=8)
            fill.pack(side="left")
            self.bars[emo] = fill

            pct = tk.Label(
                row, text="0.0%", width=5, anchor="e",
                font=("Segoe UI", 8), fg="#98a7c8", bg="#10182a"
            )
            pct.pack(side="right")
            self.pct_labels[emo] = pct

        tk.Label(
            self.root, text="Drag anywhere to move HUD",
            font=("Segoe UI", 7), fg="#657493", bg="#10182a"
        ).pack(pady=(4, 6))

    def _on_mog_toggle(self):
        save_mogging_mode(self.mog_var.get())

    def start_move(self, event):
        self._x = event.x
        self._y = event.y

    def do_move(self, event):
        x = self.root.winfo_x() + (event.x - self._x)
        y = self.root.winfo_y() + (event.y - self._y)
        self.root.geometry(f"+{x}+{y}")

    # ---- drawing helpers -------------------------------------------------
    def _font(self, family, px, weight="bold"):
        key = (family, px, weight)
        f = self._fonts.get(key)
        if f is None:
            f = tkfont.Font(root=self.root, family=family, size=-px, weight=weight)   # negative = pixels
            self._fonts[key] = f
        return f

    def _fit_px(self, text, max_w, max_h):
        """Largest Impact pixel size whose text fits inside max_w x max_h (auto-scaled MOGGED)."""
        px = max(8, int(max_h))
        w = self._font("Impact", px).measure(text)
        if w > max_w and w > 0:
            px = max(8, int(px * max_w / w))
        return px

    def _draw_label(self, text, x, y, w, color, lift=0):
        """'#rank SCORE: n' above the face; X/Y are clamped so it never leaves the captured region."""
        cw, ch = self.region["width"], self.region["height"]
        px = max(12, min(22, int(w * 0.12)))
        font = self._font("Segoe UI", px)
        tw = font.measure(text)
        tx = min(max(4, x), max(4, cw - tw - 4))
        ty = min(max(px + 6, y - 6 - lift), ch - 4)
        self.face_canvas.create_text(tx + 1, ty + 1, text=text, anchor="sw", fill="#000000", font=font)
        self.face_canvas.create_text(tx, ty, text=text, anchor="sw", fill=color, font=font)

    def _draw_reticle(self, x, y, w, h):
        """Gold tracking reticle for the #1 mogger: corner brackets plus side ticks. No censor bar."""
        c = self.face_canvas
        pad = int(0.06 * w)
        x1, y1, x2, y2 = x - pad, y - pad, x + w + pad, y + h + pad
        L = int(0.22 * min(w, h))
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            c.create_line(px + dx * L, py, px, py, px, py + dy * L, fill=GOLD, width=3, capstyle="round")
        mx, my, tick = (x1 + x2) // 2, (y1 + y2) // 2, max(6, L // 3)
        c.create_line(mx, y1, mx, y1 + tick, fill=GOLD, width=2)
        c.create_line(mx, y2, mx, y2 - tick, fill=GOLD, width=2)
        c.create_line(x1, my, x1 + tick, my, fill=GOLD, width=2)
        c.create_line(x2, my, x2 - tick, my, fill=GOLD, width=2)
        return pad

    def _draw_censor(self, eye_bar):
        """Black bar across the eyes rotated to the head tilt, with auto-scaled red MOGGED on top."""
        cx, cy, bw, bh, angle_deg = eye_bar
        hw, hh = bw / 2.0, bh / 2.0
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        corners = []
        for lx, ly in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            corners.extend([cx + lx * cos_t - ly * sin_t, cy + lx * sin_t + ly * cos_t])
        self.face_canvas.create_polygon(corners, fill="#08080a", outline="#1a1a1f", width=1)

        px = self._fit_px("MOGGED", bw * 0.88, bh * 0.78)
        font = self._font("Impact", px)
        try:
            self.face_canvas.create_text(cx, cy, text="MOGGED", fill="#ff2626", font=font, angle=-angle_deg)
        except tk.TclError:
            self.face_canvas.create_text(cx, cy, text="MOGGED", fill="#ff2626", font=font)

    def draw_overlay(self, faces):
        """Every face gets '#rank SCORE: n'. #1 gets the gold reticle; #2+ get the rotated MOGGED bar."""
        self.face_canvas.delete("all")
        mogging = self.mog_var.get()
        for f in faces:
            x, y, w, h = f["box"]
            if not mogging:      # plain mode: white box + the face's dominant emotion
                self.face_canvas.create_rectangle(x, y, x + w, y + h, outline="#ffffff", width=1)
                self._draw_label(f["dominant"].upper() if f["has_emo"] else "ANALYZING", x, y, w, "#42d99a")
                continue
            label = f"#{f['rank']} SCORE: {int(round(f['mog_score']))}"
            if f["rank"] == 1:
                pad = self._draw_reticle(x, y, w, h)
                self._draw_label(label, x, y, w, GOLD, lift=pad + 4)
            else:
                self._draw_censor(f["eye_bar"])
                self._draw_label(label, x, y, w, "#ffffff")

    def update_ui(self):
        if not is_running:
            return

        with lock:
            locked = face_locked
            faces = list(live_telemetry.get("faces", []))
            emotions = dict(live_telemetry["emotions"])

        self.draw_overlay(faces)
        self.root.lift()

        mogging = self.mog_var.get()
        primary = None
        if not mogging and faces:       # plain mode reports the biggest face, like the web dashboard
            primary = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            emotions = dict(primary["emotions"])

        if locked and faces:
            top = faces[0]
            self.status_label.config(text=f"● {len(faces)} FACE(S) LOCKED", fg="#42d99a")
            if not mogging:
                self.dom_label.config(text=primary["dominant"].upper() if primary["has_emo"] else "ANALYZING")
                self.conf_label.config(text=f"Largest face | {len(faces)} face(s) tracked", fg="#8797ff")
                self.versus_label.config(text="Mogging mode off", fg="#98a7c8")
            else:
                self.dom_label.config(text=f"#1 MOGGER: {top['mog_score']}")
                self.conf_label.config(text=f"SYM {top['symmetry']}% | AURA {top['composure']}% | {top['dominant'].upper()}")

                if len(faces) > 1:
                    runner = faces[1]
                    self.versus_label.config(
                        text=f"vs #2 MOGGED: {runner['mog_score']} ({runner['gap']} pts)",
                        fg="#ff969f"
                    )
                else:
                    self.versus_label.config(text="Solo Mode — Add 2nd face to compare", fg="#98a7c8")
        else:
            self.dom_label.config(text="MOG SCORE: --" if mogging else "EMOTION: --")
            self.status_label.config(text="● SCANNING...", fg="#ff969f")
            if engine["error"]:
                self.conf_label.config(text=engine["error"], fg="#ff969f", wraplength=260)
            else:
                self.conf_label.config(text=f"Align speaker faces in box ({engine['detector']})", fg="#8797ff")

        for emo in EMOTION_KEYS:
            score = emotions.get(emo, 0.0)
            bar_w = int((max(0.0, min(100.0, score)) / 100.0) * 125)
            self.bars[emo].config(width=bar_w)
            self.pct_labels[emo].config(text=f"{score:.1f}%")

        self.root.after(100, self.update_ui)

    def stop_and_summarize(self):
        global is_running
        is_running = False
        self.root.destroy()
        generate_meeting_report()

    def run(self):
        global is_running
        is_running = True
        threading.Thread(target=screen_capture_worker, args=(self.region,), daemon=True).start()
        threading.Thread(target=ai_analysis_worker, daemon=True).start()
        self.update_ui()
        self.root.mainloop()


def generate_meeting_report():
    with lock:
        history = list(session_history)

    if not history:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Meeting Analysis", "Session ended, but no faces were detected during the capture.")
        root.destroy()
        return

    counts = {}
    for entry in history:
        d = entry["dominant"]
        counts[d] = counts.get(d, 0) + 1
    overall_dominant = max(counts, key=counts.get)
    dominance_pct = (counts[overall_dominant] / len(history)) * 100

    with open("meeting_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Dominant_Emotion", "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
        for e in history:
            em = e["emotions"]
            writer.writerow([
                e["frame"], e["dominant"],
                round(em.get("angry", 0), 2), round(em.get("disgust", 0), 2),
                round(em.get("fear", 0), 2), round(em.get("happy", 0), 2),
                round(em.get("sad", 0), 2), round(em.get("surprise", 0), 2),
                round(em.get("neutral", 0), 2)
            ])

    frames = [e["frame"] for e in history]
    plt.figure(figsize=(10, 5.5), facecolor="#0b1323")
    ax = plt.gca()
    ax.set_facecolor("#0b1323")

    plt.plot(frames, [e["emotions"].get("angry", 0) for e in history], color="#ff4a4a", linewidth=2, label="Angry")
    plt.plot(frames, [e["emotions"].get("happy", 0) for e in history], color="#42d99a", linewidth=2, label="Happy")
    plt.plot(frames, [e["emotions"].get("sad", 0) for e in history], color="#5968ff", linewidth=2, label="Sad")
    plt.plot(frames, [e["emotions"].get("surprise", 0) for e in history], color="#f1c40f", linewidth=2, label="Surprise")
    plt.plot(frames, [e["emotions"].get("neutral", 0) for e in history], color="#98a7c8", linewidth=2, label="Neutral", alpha=0.6)

    plt.title(
        f"Meeting Behavioral Timeline  |  Most Frequent: {overall_dominant.upper()} ({dominance_pct:.1f}%)",
        color="white", fontsize=13, pad=15, fontweight="bold"
    )
    plt.xlabel("Captured Frame Number", color="#98a7c8", fontsize=11)
    plt.ylabel("Confidence Score (%)", color="#98a7c8", fontsize=11)

    ax.tick_params(colors="#98a7c8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#202c45")

    plt.legend(facecolor="#10182a", edgecolor="#202c45", labelcolor="white", loc="upper right")
    plt.grid(color="#202c45", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.savefig("meeting_emotion_plot.png", bbox_inches="tight", dpi=120)
    plt.show()


if __name__ == "__main__":
    try:
        selector = RegionSelector()
        selected_region = selector.get_region()
        if selected_region:
            hud = FloatingHUD(selected_region)
            hud.run()
    except Exception:
        err_root = tk.Tk()
        err_root.withdraw()
        messagebox.showerror("Meeting Overlay Error", traceback.format_exc())
        err_root.destroy()
