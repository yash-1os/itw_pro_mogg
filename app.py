import os, sys, math, json, time, threading, base64

# Quiet TensorFlow and make DeepFace work with Keras 3 installs (only if tf-keras is present).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
try:
    import tf_keras  # noqa: F401
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
except ImportError:
    pass

# NOTE: import order (DeepFace -> MediaPipe -> OpenCV) is kept from the original on purpose;
# changing it can trigger DLL / protobuf conflicts on Windows.
print("[1/5] Loading Flask...", flush=True)
from flask import Flask, render_template, request, jsonify, Response

print("[2/5] Loading DeepFace AI (slow on first boot)...", flush=True)
from deepface import DeepFace

print("[3/5] Loading MediaPipe...", flush=True)
import mediapipe as mp
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    sys.exit("This MediaPipe install has no 'solutions' API. Run:  pip install mediapipe==0.10.14")

print("[4/5] Loading OpenCV + NumPy...", flush=True)
import cv2
import numpy as np

print("[5/5] Initializing application...", flush=True)
app = Flask(__name__)
print("[debug] running file:", os.path.abspath(__file__), flush=True)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODE_FILE = os.path.join(BASE_DIR, "mode_config.json")   # absolute: works from any working directory

EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
MAX_FACES = 8
MAX_ANALYZED_FACES = 6          # DeepFace runs on at most this many faces (largest first)
ANALYZE_INTERVAL = 0.30         # seconds between DeepFace runs per face

# Composure weights: calm = good, erratic = bad
# A smile counts as much as a calm face; only negative / erratic expressions lose points.
COMPOSURE_W = {"neutral": 1.0, "happy": 1.0, "surprise": 0.1,
               "angry": -0.4, "disgust": -0.5, "sad": -0.6, "fear": -0.8}

# Final score = W_SYM * symmetry + W_COMP * composure.
# Face SIZE is deliberately NOT part of the score any more: in a selfie the person holding the
# camera always has the biggest face, which just measures distance to the lens, not looks.
W_SYM, W_COMP = 0.60, 0.40

GOLD = (0, 215, 255)   # BGR
RED = (0, 0, 255)

# ----------------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------------
lock = threading.Lock()                 # guards tracks, pending_jobs, live_telemetry
live_telemetry = {
    "dominant_emotion": "neutral",
    "emotions": {k: 0.0 for k in EMOTION_KEYS},
    "leaderboard": [],
    "faces": 0,
    "mogging_mode": False,
}
tracks = {}          # tid -> {cx, cy, w, seen, emo, mog, last_job}
pending_jobs = {}    # tid -> BGR crop waiting for DeepFace
next_tid = 1

frame_cond = threading.Condition()
latest_jpeg = None
frame_id = 0

viewers = 0
viewers_lock = threading.Lock()
pipe_lock = threading.Lock()
_pipe = {"thread": None, "stop": None}


@app.after_request
def disable_browser_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ----------------------------------------------------------------------------
# Mode flag (mode_config.json), re-read live so toggling needs no restart
# ----------------------------------------------------------------------------
_mode_cache = {"t": 0.0, "val": False}


def ensure_mode_file():
    if not os.path.exists(MODE_FILE):
        try:
            with open(MODE_FILE, "w") as f:
                json.dump({"mogging_mode": False}, f)
        except OSError:
            pass


def mogging_enabled():
    now = time.time()
    if now - _mode_cache["t"] < 0.5:
        return _mode_cache["val"]
    _mode_cache["t"] = now
    try:
        with open(MODE_FILE, "r") as f:
            _mode_cache["val"] = bool(json.load(f).get("mogging_mode", False))
    except Exception:
        pass    # keep the last known value
    return _mode_cache["val"]


# ----------------------------------------------------------------------------
# Face geometry (MediaPipe Face Mesh landmarks -> box, eye line, tilt)
# ----------------------------------------------------------------------------
def extract_face(lm, W, H, ox=0.0, oy=0.0, FW=None, FH=None):
    """W,H = size of the image the landmarks came from; ox,oy = where that image sits inside the
    full picture (for tiles); FW,FH = full picture size. Returned coordinates are full-picture pixels."""
    FW = FW or W
    FH = FH or H
    xs = np.array([p.x for p in lm.landmark], dtype=np.float32) * W + ox
    ys = np.array([p.y for p in lm.landmark], dtype=np.float32) * H + oy
    zs = np.array([p.z for p in lm.landmark], dtype=np.float32) * W      # z shares x's scale
    x1, x2 = max(0.0, float(xs.min())), min(FW - 1.0, float(xs.max()))
    y1, y2 = max(0.0, float(ys.min())), min(FH - 1.0, float(ys.max()))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    # Eye centres: mean of each eye's outer/inner corner. Image-left eye = A, image-right = B.
    ax, ay = float(xs[33] + xs[133]) / 2, float(ys[33] + ys[133]) / 2
    bx, by = float(xs[263] + xs[362]) / 2, float(ys[263] + ys[362]) / 2
    theta = math.atan2(by - ay, bx - ax)                      # head tilt (radians)
    eye_span = math.hypot(float(xs[263] - xs[33]), float(ys[263] - ys[33]))
    return {
        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
        "w": w, "h": h, "area": w * h, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
        "ecx": (ax + bx) / 2, "ecy": (ay + by) / 2, "theta": theta, "eye_span": eye_span,
        "pts": np.stack([xs, ys, zs], axis=1),
    }


def filter_faces(faces, min_rel_area=0.10, verbose=False):
    """Drop tiny / oddly shaped detections (hands, background blobs)."""
    if not faces:
        return faces
    biggest = max(f["area"] for f in faces)
    out = []
    for f in faces:
        why = None
        if f["w"] < 40 or f["h"] < 40:
            why = f"too small ({f['w']:.0f}x{f['h']:.0f}px)"
        elif not (0.75 <= f["h"] / f["w"] <= 2.4):
            why = f"odd shape (h/w={f['h'] / f['w']:.2f})"
        elif f["area"] < min_rel_area * biggest:
            why = f"much smaller than the biggest face ({f['area'] / biggest:.0%})"
        elif not (0.20 <= f["eye_span"] / f["w"] <= 1.0):
            why = f"odd eye spacing ({f['eye_span'] / f['w']:.2f})"
        if why:
            if verbose:
                print(f"[faces] dropped face at ({f['cx']:.0f},{f['cy']:.0f}): {why}", flush=True)
            continue
        out.append(f)
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
# Left/right mirror pairs of the MediaPipe face mesh (eyes, brows, mouth, nose, cheeks, jaw)
_SYM_PAIRS = [(33, 263), (133, 362), (159, 386), (145, 374), (61, 291), (70, 300), (105, 334),
              (93, 323), (172, 397), (136, 365), (129, 358), (98, 327), (234, 454)]


def landmark_symmetry(pts):
    """0-100 symmetry from the mesh itself. Head yaw and roll are rotated out first, so a face that
    is simply turned or tilted is not punished (the old pixel-mirror method did exactly that)."""
    P = np.asarray(pts, dtype=np.float64).copy()
    a, b = P[234], P[454]
    yaw = math.atan2(b[2] - a[2], b[0] - a[0])
    if abs(yaw) > math.radians(55):                     # near-profile: symmetry is not measurable
        return 75.0
    c, s = math.cos(yaw), math.sin(yaw)
    P[:, 0], P[:, 2] = P[:, 0] * c + P[:, 2] * s, -P[:, 0] * s + P[:, 2] * c
    a, b = P[234], P[454]
    roll = math.atan2(b[1] - a[1], b[0] - a[0])
    c, s = math.cos(roll), math.sin(roll)
    P[:, 0], P[:, 1] = P[:, 0] * c + P[:, 1] * s, -P[:, 0] * s + P[:, 1] * c
    width = abs(P[454, 0] - P[234, 0])
    if width < 1.0:
        return 70.0
    pairs = []
    for l, r in _SYM_PAIRS:
        xl, xr = (l, r) if P[l, 0] <= P[r, 0] else (r, l)
        pairs.append((P[xl], P[xr]))
    mid = float(np.median([(pl[0] + pr[0]) / 2.0 for pl, pr in pairs]))     # face midline
    errs = [(abs((mid - pl[0]) - (pr[0] - mid)) + abs(pl[1] - pr[1])) / width for pl, pr in pairs]
    asym = float(np.mean(errs))
    return 55.0 + 45.0 * math.exp(-asym / 0.07)


def symmetry_score(gray, f):
    if f.get("pts") is not None:
        return landmark_symmetry(f["pts"])
    crop = gray[f["y1"]:f["y2"], f["x1"]:f["x2"]]
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (64, 64)).astype(np.float32)
    left = crop[:, :32].ravel()
    right = cv2.flip(crop[:, 32:], 1).ravel()
    if left.std() < 1e-3 or right.std() < 1e-3:
        corr = 0.0
    else:
        corr = float(np.corrcoef(left, right)[0, 1])
    sym = 50.0 + 50.0 * min(1.0, max(0.0, corr))                # 50-100: an off-centre crop is never scored zero
    tilt = abs(math.degrees(f["theta"]))
    if tilt > 8:                                                 # heavy tilt penalty
        sym *= max(0.5, 1.0 - (tilt - 8) / 60.0)
    return sym


def composure_score(emo):
    if not emo:
        return 70.0
    raw = sum(COMPOSURE_W[k] * emo.get(k, 0.0) / 100.0 for k in EMOTION_KEYS)
    return min(1.0, max(0.0, raw)) * 100.0


def dominant(emo):
    return max(emo, key=emo.get) if emo else "neutral"


# ----------------------------------------------------------------------------
# Simple centroid tracker so emotions and score smoothing follow a face between frames
# ----------------------------------------------------------------------------
def assign_tracks(faces, now):
    global next_tid
    used = set()
    for f in sorted(faces, key=lambda f: -f["area"]):
        best, best_d = None, None
        for tid, t in tracks.items():
            if tid in used:
                continue
            d = math.hypot(f["cx"] - t["cx"], f["cy"] - t["cy"])
            if d < 0.75 * max(f["w"], t["w"]) and (best_d is None or d < best_d):
                best, best_d = tid, d
        if best is None:
            best = next_tid
            next_tid += 1
            tracks[best] = {"emo": None, "mog": None, "last_job": 0.0}
        tracks[best].update(cx=f["cx"], cy=f["cy"], w=f["w"], seen=now)
        used.add(best)
        f["tid"] = best
    for tid in [k for k, t in tracks.items() if now - t["seen"] > 1.0]:
        del tracks[tid]
        pending_jobs.pop(tid, None)


# ----------------------------------------------------------------------------
# DeepFace worker (emotions only, on padded face crops)
# ----------------------------------------------------------------------------
_fast_fail = 0
_last_err = ""


def analyze_emotions(crop):
    global _fast_fail, _last_err
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
    except Exception as ex:
        msg = repr(ex)
        if msg != _last_err:
            _last_err = msg
            print("[deepface] error:", msg, flush=True)
        return None


def ai_worker(stop):
    analyze_emotions(np.zeros((96, 96, 3), np.uint8))      # warm-up: load model before first real face
    while not stop.is_set():
        with lock:
            jobs = list(pending_jobs.items())
            pending_jobs.clear()
        if not jobs:
            time.sleep(0.03)
            continue
        for tid, crop in jobs:
            if stop.is_set():
                break
            emo = analyze_emotions(crop)
            if emo is None:
                continue
            with lock:
                t = tracks.get(tid)
                if t is None:
                    continue
                prev = t["emo"]
                t["emo"] = emo if prev is None else {k: 0.5 * prev[k] + 0.5 * emo[k] for k in EMOTION_KEYS}


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_text_outlined(frame, text, org, scale, color, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_label(frame, text, f, color, lift=0):
    """Text above the box; Y and X are clamped so it never leaves the frame."""
    H, W = frame.shape[:2]
    scale = max(0.5, min(1.0, f["w"] / 220.0))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(min(max(4, f["x1"]), max(4, W - tw - 4)))
    y = int(min(max(th + 6, f["y1"] - 10 - lift), H - 4))
    draw_text_outlined(frame, text, (x, y), scale, color, 2)


def draw_reticle(frame, f):
    """Gold tracking reticle for the #1 face: corner brackets + side ticks."""
    pad = int(0.06 * f["w"])
    x1, y1, x2, y2 = f["x1"] - pad, f["y1"] - pad, f["x2"] + pad, f["y2"] + pad
    L = int(0.22 * min(f["w"], f["h"]))
    t = 3
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * L, py), GOLD, t, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * L), GOLD, t, cv2.LINE_AA)
    mx, my, tick = (x1 + x2) // 2, (y1 + y2) // 2, max(6, L // 3)
    cv2.line(frame, (mx, y1), (mx, y1 + tick), GOLD, 2, cv2.LINE_AA)
    cv2.line(frame, (mx, y2), (mx, y2 - tick), GOLD, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, my), (x1 + tick, my), GOLD, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, my), (x2 - tick, my), GOLD, 2, cv2.LINE_AA)


def draw_rotated_text(frame, text, cx, cy, theta_rad, target_w, color):
    """cv2.putText cannot rotate, so render to a mask, rotate it, alpha-composite onto the frame."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (bw, _), _ = cv2.getTextSize(text, font, 1.0, 1)
    scale = max(0.25, target_w / max(bw, 1))
    thick = max(1, int(round(scale * 2.2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    w, h = tw + 2 * pad, th + 2 * pad
    diag = int(math.hypot(w, h)) + 4
    canvas = np.zeros((diag, diag), np.uint8)
    ox, oy = (diag - w) // 2, (diag - h) // 2
    cv2.putText(canvas, text, (ox + pad, oy + pad + th), font, scale, 255, thick, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), -math.degrees(theta_rad), 1.0)
    rot = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR)

    H, W = frame.shape[:2]
    x0, y0 = int(round(cx - diag / 2)), int(round(cy - diag / 2))
    fx1, fy1, fx2, fy2 = max(0, x0), max(0, y0), min(W, x0 + diag), min(H, y0 + diag)
    if fx2 <= fx1 or fy2 <= fy1:
        return
    a = rot[fy1 - y0:fy2 - y0, fx1 - x0:fx2 - x0].astype(np.float32)[..., None] / 255.0
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def draw_censor(frame, f):
    """Black bar across the eyes, rotated to the head tilt, with auto-scaled red MOGGED."""
    L = f["eye_span"] * 1.30
    Hb = max(14.0, L * 0.30)
    th = f["theta"]
    c, s = math.cos(th), math.sin(th)
    pts = [(f["ecx"] + dx * c - dy * s, f["ecy"] + dx * s + dy * c)
           for dx, dy in ((-L / 2, -Hb / 2), (L / 2, -Hb / 2), (L / 2, Hb / 2), (-L / 2, Hb / 2))]
    cv2.fillPoly(frame, [np.array(pts, np.int32)], (0, 0, 0), cv2.LINE_AA)
    draw_rotated_text(frame, "MOGGED", f["ecx"], f["ecy"], th, L * 0.86, RED)


def annotate(frame, faces, mogging):
    if mogging:
        ranked = sorted(faces, key=lambda f: -f["mog"])
        for rank, f in enumerate(ranked, 1):
            label = f"#{rank} SCORE: {int(round(f['mog']))}"
            if rank == 1:
                draw_reticle(frame, f)
                draw_label(frame, label, f, GOLD, lift=int(0.06 * f["w"]) + 4)   # clear of the reticle
            else:
                draw_censor(frame, f)
                draw_label(frame, label, f, (255, 255, 255))
    else:
        for f in faces:
            cv2.rectangle(frame, (f["x1"], f["y1"]), (f["x2"], f["y2"]), (255, 255, 255), 1)
            draw_label(frame, dominant(f["emo"]).upper() if f["emo"] else "ANALYZING", f, (154, 217, 66))
    return frame


# ----------------------------------------------------------------------------
# Per-frame pipeline
# ----------------------------------------------------------------------------
def process_frame(frame, fm):
    H, W = frame.shape[:2]
    small = frame if W <= 640 else cv2.resize(frame, (640, int(H * 640 / W)))
    res = fm.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))      # landmarks are normalised -> scale-free

    faces = []
    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks:
            f = extract_face(lm, W, H)
            if f:
                faces.append(f)
    faces = filter_faces(faces)

    mogging = mogging_enabled()
    now = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_area = max((f["area"] for f in faces), default=1.0)

    with lock:
        assign_tracks(faces, now)

        # queue DeepFace jobs (padded 15% crops), largest faces first
        for f in sorted(faces, key=lambda f: -f["area"])[:MAX_ANALYZED_FACES]:
            t = tracks[f["tid"]]
            if now - t["last_job"] >= ANALYZE_INTERVAL and f["tid"] not in pending_jobs:
                px, py = int(0.15 * f["w"]), int(0.15 * f["h"])
                crop = frame[max(0, f["y1"] - py):min(H, f["y2"] + py),
                             max(0, f["x1"] - px):min(W, f["x2"] + px)]
                if crop.size:
                    pending_jobs[f["tid"]] = crop.copy()
                    t["last_job"] = now

        for f in faces:
            t = tracks[f["tid"]]
            f["emo"] = t["emo"]
            sym = symmetry_score(gray, f)
            comp = composure_score(t["emo"])
            raw = W_SYM * sym + W_COMP * comp
            t["mog"] = raw if t["mog"] is None else 0.7 * t["mog"] + 0.3 * raw     # smooth to stop rank flicker
            f["mog"] = t["mog"]

        ranked = sorted(faces, key=lambda f: -f["mog"])
        live_telemetry["leaderboard"] = [
            {"rank": i, "id": f["tid"], "score": round(float(f["mog"]), 1), "emotion": dominant(f["emo"])}
            for i, f in enumerate(ranked, 1)
        ]
        live_telemetry["faces"] = len(faces)
        live_telemetry["mogging_mode"] = mogging
        if faces:
            primary = max(faces, key=lambda f: f["area"])
            if primary["emo"]:
                live_telemetry["emotions"] = {k: float(v) for k, v in primary["emo"].items()}
                live_telemetry["dominant_emotion"] = dominant(primary["emo"])

    return annotate(frame, faces, mogging)


# ----------------------------------------------------------------------------
# Camera + background pipeline (one capture thread shared by every browser tab)
# ----------------------------------------------------------------------------
def open_camera():
    idx = int(os.environ.get("CAMERA_INDEX", "0"))
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for b in backends:
        cap = cv2.VideoCapture(idx, b)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def placeholder_frame(msg):
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(img, msg, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def publish(img):
    global latest_jpeg, frame_id
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    with frame_cond:
        latest_jpeg = buf.tobytes()
        frame_id += 1
        frame_cond.notify_all()


def pipeline_loop(stop):
    print("[debug] pipeline thread started, opening camera...", flush=True)
    cap = open_camera()
    print("[debug] camera object:", "OPENED" if cap is not None else "FAILED", flush=True)
    if cap is None:
        print("[camera] Could not open the webcam. Close other apps using it (Zoom, Teams, "
              "Camera app) or set CAMERA_INDEX=1.", flush=True)
        while not stop.is_set():
            publish(placeholder_frame("Camera not available - close other apps using it"))
            time.sleep(0.5)
        return
    try:
        with mp_face_mesh.FaceMesh(max_num_faces=MAX_FACES, refine_landmarks=False,
                                   min_detection_confidence=0.6, min_tracking_confidence=0.5) as fm:
            fails = 0
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 50:
                        print("[camera] Lost the webcam stream.", flush=True)
                        publish(placeholder_frame("Camera stream lost - reload the page"))
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                if frame_id == 0:
                    print("[debug] got first camera frame, processing...", flush=True)
                try:
                    out = process_frame(frame, fm)
                except Exception as e:
                    print("[pipeline] frame error:", repr(e), flush=True)
                    out = frame
                publish(out)
                if frame_id == 1:
                    print("[debug] FIRST FRAME PUBLISHED", flush=True)
    except BaseException:
        import traceback
        print("[debug] PIPELINE THREAD CRASHED:", flush=True)
        traceback.print_exc()
    finally:
        cap.release()
        with lock:
            tracks.clear()
            pending_jobs.clear()


def start_pipeline():
    with pipe_lock:
        t, s = _pipe["thread"], _pipe["stop"]
        if t is not None and t.is_alive() and s is not None and not s.is_set():
            return
        if t is not None and t.is_alive():       # old one still shutting down: wait so the camera is free
            s.set()
            t.join(timeout=5)
        stop = threading.Event()
        th = threading.Thread(target=pipeline_loop, args=(stop,), daemon=True)
        _pipe["thread"], _pipe["stop"] = th, stop
        th.start()
        threading.Thread(target=ai_worker, args=(stop,), daemon=True).start()


def stop_pipeline():
    with pipe_lock:
        if _pipe["stop"] is not None:
            _pipe["stop"].set()


def pipeline_alive():
    t = _pipe["thread"]
    return t is not None and t.is_alive()


def gen_frames():
    global viewers
    print("[debug] /video_feed requested", flush=True)
    with viewers_lock:
        viewers += 1
    start_pipeline()
    with frame_cond:
        last = frame_id          # wait for the NEXT frame, never send latest_jpeg before one exists
    try:
        while True:
            with frame_cond:
                frame_cond.wait_for(lambda: frame_id != last, timeout=2.0)
                if frame_id == last:
                    if not pipeline_alive():
                        break
                    continue
                jpg, last = latest_jpeg, frame_id
            if jpg is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    except BaseException as e:
        import traceback
        print("[debug] gen_frames ended with:", repr(e), flush=True)
        traceback.print_exc()
        raise
    finally:
        print("[debug] gen_frames closed", flush=True)
        with viewers_lock:
            viewers -= 1
            if viewers <= 0:
                stop_pipeline()


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------------------
# Small-face detection for uploads: MediaPipe's detector sees a ~128px copy of whatever it is given,
# so faces further from the lens get missed. We also scan overlapping zoomed-in tiles.
# ----------------------------------------------------------------------------
def _overlap_of_smaller(a, b):
    ix = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    iy = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    if ix <= 0 or iy <= 0:
        return 0.0
    return (ix * iy) / max(1.0, min(a["area"], b["area"]))


def detect_faces_multi(img, fm):
    H, W = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    passes = [(0, 0, W, H, 0)]
    if max(W, H) >= 640:
        for level, (frac, n) in enumerate(((0.6, 2), (0.4, 3)), start=1):
            tw, th = int(W * frac), int(H * frac)
            for y0 in np.linspace(0, H - th, n).astype(int):
                for x0 in np.linspace(0, W - tw, n).astype(int):
                    passes.append((int(x0), int(y0), int(x0) + tw, int(y0) + th, level))

    cands = []
    for x0, y0, x1, y1, level in passes:
        res = fm.process(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
        if not res.multi_face_landmarks:
            continue
        for lm in res.multi_face_landmarks:
            f = extract_face(lm, x1 - x0, y1 - y0, x0, y0, W, H)
            if f is None:
                continue
            m = 4        # a face touching a tile edge that is not the picture edge is cut off: skip it
            if ((x0 > 0 and f["x1"] <= x0 + m) or (y0 > 0 and f["y1"] <= y0 + m) or
                    (x1 < W and f["x2"] >= x1 - m) or (y1 < H and f["y2"] >= y1 - m)):
                continue
            f["_level"] = level
            cands.append(f)

    cands.sort(key=lambda f: (-f["_level"], -f["area"]))      # prefer the sharper (zoomed) detection
    kept = []
    for f in cands:
        if all(_overlap_of_smaller(f, k) < 0.5 for k in kept):
            kept.append(f)
    return kept, len(cands)


# ----------------------------------------------------------------------------
# Uploaded-image analysis (one shot: no tracker, no shared live state)
# ----------------------------------------------------------------------------
analyze_lock = threading.Lock()


def analyze_image(img):
    """Detect faces, run emotions + mog scoring, draw the same overlay as the live stream."""
    big = max(img.shape[:2])
    if big > 1600:                                           # keep big photos fast
        k = 1600.0 / big
        img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=MAX_FACES,
                               refine_landmarks=False, min_detection_confidence=0.4) as fm:
        merged, n_cands = detect_faces_multi(img, fm)
    faces = filter_faces(merged, min_rel_area=0.03, verbose=True)
    print(f"[upload] {n_cands} raw detections -> {len(merged)} unique faces -> {len(faces)} kept", flush=True)
    if not faces:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ordered = sorted(faces, key=lambda f: -f["area"])
    for i, f in enumerate(ordered):
        f["emo"] = None
        if i < MAX_ANALYZED_FACES:
            px, py = int(0.15 * f["w"]), int(0.15 * f["h"])
            crop = img[max(0, f["y1"] - py):min(H, f["y2"] + py),
                       max(0, f["x1"] - px):min(W, f["x2"] + px)]
            if crop.size:
                f["emo"] = analyze_emotions(crop.copy())
        sym = symmetry_score(gray, f)
        comp = composure_score(f["emo"])
        f["mog"] = W_SYM * sym + W_COMP * comp
        print(f"[upload] face {i + 1}: symmetry={sym:.1f} composure={comp:.1f} -> score={f['mog']:.1f} "
              f"({dominant(f['emo']) if f['emo'] else 'no emotion'})", flush=True)

    primary = ordered[0]
    if primary["emo"] is None:
        raise RuntimeError("the emotion model returned nothing (see the console for [deepface] errors)")

    mogging = mogging_enabled()
    ranked = sorted(faces, key=lambda f: -f["mog"])
    out = annotate(img.copy(), faces, mogging)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("could not encode the result image")

    return {
        "emotions": {k: float(v) for k, v in primary["emo"].items()},
        "dominant_emotion": dominant(primary["emo"]),
        "leaderboard": [{"rank": i, "id": i, "score": round(float(f["mog"]), 1), "emotion": dominant(f["emo"])}
                        for i, f in enumerate(ranked, 1)],
        "faces": len(faces),
        "faces_found_before_filter": len(merged),
        "mogging_mode": mogging,
        "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"),
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image received."}), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read that image file."}), 400
    try:
        with analyze_lock:
            result = analyze_image(img)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    if result is None:
        return jsonify({"error": "No face detected in this image. Try a clearer, front-facing photo."}), 422
    return jsonify(result)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream_data")
def stream_data():
    with lock:
        return jsonify(live_telemetry)


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    stop_pipeline()
    return jsonify({"status": "stopped"})


@app.route("/mode", methods=["GET"])
def get_mode():
    return jsonify({"mogging_mode": mogging_enabled()})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("mogging_mode", False))
    try:
        with open(MODE_FILE, "w") as f:
            json.dump({"mogging_mode": enabled}, f)
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    _mode_cache.update(t=0.0, val=enabled)
    return jsonify({"mogging_mode": enabled})


if __name__ == "__main__":
    ensure_mode_file()
    host = os.environ.get("HOST", "127.0.0.1")      # 127.0.0.1 = this PC only (webcam stays private)
    port = int(os.environ.get("PORT", "5000"))
    print("\n=======================================================")
    print(f"READY!  Open  http://127.0.0.1:{port}  in your browser")
    print("=======================================================\n", flush=True)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
