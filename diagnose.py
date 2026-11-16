import sys, os, traceback
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tf_keras  # noqa
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
except ImportError:
    pass

def step(name, fn):
    print(f"\n--- {name} ---", flush=True)
    try:
        r = fn()
        print("PASS", r if r is not None else "")
        return True
    except BaseException:
        print("FAIL")
        traceback.print_exc()
        return False

print("Python:", sys.version)
print("Running from:", os.getcwd())
print("Script file :", os.path.abspath(__file__))

def v():
    import importlib.metadata as m
    out = []
    for p in ["flask", "deepface", "tensorflow", "tf-keras", "keras",
              "mediapipe", "opencv-python", "numpy", "protobuf", "mss", "matplotlib"]:
        try:
            out.append(f"{p}=={m.version(p)}")
        except Exception:
            out.append(f"{p}: NOT INSTALLED")
    return "\n  " + "\n  ".join(out)
step("package versions", v)

def cam():
    import cv2
    for name, b in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
        c = cv2.VideoCapture(0, b)
        ok = c.isOpened() and c.read()[0]
        c.release()
        print(" ", name, "->", ok)
step("camera index 0", cam)

def mesh():
    import mediapipe as mp
    import numpy as np
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=False,
            min_detection_confidence=0.6, min_tracking_confidence=0.5) as fm:
        fm.process(np.zeros((480, 640, 3), np.uint8))
step("MediaPipe FaceMesh", mesh)

def df():
    from deepface import DeepFace
    import numpy as np
    r = DeepFace.analyze(np.zeros((224, 224, 3), np.uint8), actions=["emotion"], enforce_detection=False)
    return "emotion model works"
step("DeepFace emotion model (may download weights, needs internet)", df)

def files():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ["app.py", "templates/index.html", "static/app.js", "static/style.css"]:
        print(" ", "OK     " if os.path.exists(os.path.join(here, p)) else "MISSING", p)
step("project files", files)

print("\nDone. Copy everything above and send it.")
