"""
Flask server for YOLO11 Wildlife + Fire/Smoke Anomaly Detection
Uses results.plot() for reliable bounding box visualization.
"""

import os
import cv2
import time
import threading
import numpy as np
import sys
from pathlib import Path
from flask import Flask, Response, render_template, request, jsonify
from ultralytics import YOLO
import werkzeug

# ─────────────────────────────── CONFIG ───────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "train" / "yolo11l_wildlife_fire" / "weights" / "best.pt"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.05  # VERY low for debugging

FIRE_CLASSES  = {"fire", "smoke"}
ANIMAL_CLASSES = {"buffalo", "elephant", "tiger", "wild_boar"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB


def resolve_model_path() -> Path:
    """Find best available detection weights."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    candidates = sorted((BASE_DIR / "runs" / "train").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    fallback = BASE_DIR / "best_yolo11l_wildlife_fire.pt"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"No model weights found. Tried: {MODEL_PATH}, runs/train/*/weights/best.pt, {fallback}")

# ─────────────────────────────── GLOBAL STATE ─────────────────────────────────
class DetectionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cap: cv2.VideoCapture | None = None
        self.source: str = "none"
        self.running: bool = False
        self.animal_detection_enabled: bool = True

        self.fire_detected: bool = False
        self.animal_detected: bool = False
        self.confidence: float = 0.0
        self.detections: list = []

        self._fps_t = time.time()
        self._fps_count = 0
        self.fps: float = 0.0

    def update_fps(self):
        self._fps_count += 1
        now = time.time()
        elapsed = now - self._fps_t
        if elapsed >= 1.0:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t = now

state = DetectionState()

# ─────────────────────────────── LOAD MODEL ───────────────────────────────────
print(f"[INFO] Loading model from: {MODEL_PATH}")
try:
    model_file = resolve_model_path()
    print(f"[INFO] Using weights: {model_file}")
    model = YOLO(str(model_file))
    print(f"[INFO] Model loaded ✓. Classes: {model.names}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    sys.exit(1)

# ─────────────────────────────── FRAME GENERATOR ─────────────────────────────

def process_frame(frame: np.ndarray) -> np.ndarray:
    try:
        if frame is None or frame.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            verbose=False,
        )[0]
        num_boxes = len(results.boxes)

        # Manual plotting for reliability in OpenCV stream
        plotted_frame = frame.copy()

        fire_det = False
        animal_det = False
        max_conf = 0.0
        frame_detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names.get(cls_id, f"cls{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            should_draw = False
            color = (0, 255, 0)

            if label in FIRE_CLASSES:
                fire_det = True
                should_draw = True
                color = (0, 0, 255)
                print(f"[INFO] FIRE! {label} ({conf:.2f})")
            if label in ANIMAL_CLASSES and state.animal_detection_enabled:
                animal_det = True
                should_draw = True
                color = (255, 180, 0)
                print(f"[INFO] ANIMAL! {label} ({conf:.2f})")

            if label not in FIRE_CLASSES and label not in ANIMAL_CLASSES:
                should_draw = True

            if should_draw:
                cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    plotted_frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            max_conf = max(max_conf, conf)
            frame_detections.append({
                "label": label, 
                "conf": round(conf * 100, 1), 
                "bbox": [x1, y1, x2, y2]
            })

        # Display metadata
        state.update_fps()
        info_text = f"FPS: {state.fps:.1f} | Detections: {num_boxes} | Conf: {CONFIDENCE_THRESHOLD}"
        cv2.putText(plotted_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if fire_det:
            cv2.rectangle(plotted_frame, (0, 0), (plotted_frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(plotted_frame, "!!! FIRE DETECTED !!!", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        with state.lock:
            state.fire_detected = fire_det
            state.animal_detected = animal_det
            state.confidence = max_conf * 100
            state.detections = frame_detections

        return plotted_frame
    except Exception as e:
        print(f"[ERROR] Process frame failed: {e}")
        return frame


def generate_frames():
    while True:
        with state.lock:
            cap, running = state.cap, state.running

        if not running or cap is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "INACTIVE", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,100,100), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            with state.lock: state.running = False
            continue

        frame = process_frame(frame)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# ─────────────────────────────── ROUTES ───────────────────────────────────────

@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed(): return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_camera", methods=["POST"])
def start_camera():
    idx = request.get_json(silent=True).get("camera_index", 0) if request.get_json(silent=True) else 0
    with state.lock:
        if state.cap: state.cap.release()
        state.cap = cv2.VideoCapture(int(idx))
        if not state.cap.isOpened():
            state.cap = None
            state.running = False
            return jsonify({"error": f"Cannot open camera index {idx}"}), 400
        state.running, state.source = True, "camera"
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    save_path = UPLOAD_FOLDER / werkzeug.utils.secure_filename(f.filename)
    f.save(str(save_path))
    with state.lock:
        if state.cap: state.cap.release()
        state.cap = cv2.VideoCapture(str(save_path))
        if not state.cap.isOpened():
            state.cap = None
            state.running = False
            return jsonify({"error": "Cannot open uploaded video (codec/format unsupported)"}), 400
        state.running, state.source = True, "video"
    return jsonify({"status": "ok"})

@app.route("/stop_camera", methods=["POST"])
@app.route("/stop_video", methods=["POST"])
def stop():
    with state.lock:
        state.running = False
        if state.cap: state.cap.release(); state.cap = None
        state.fire_detected = False
        state.animal_detected = False
        state.confidence = 0.0
        state.detections = []
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    with state.lock:
        return jsonify({
            "running": state.running,
            "source": state.source,
            "fire_detected": state.fire_detected,
            "animal_detected": state.animal_detected,
            "confidence": state.confidence,
            "detections": state.detections,
            "fps": round(state.fps, 1),
            "animal_detection_enabled": state.animal_detection_enabled,
        })

@app.route("/enable_animal_detection", methods=["POST"])
def enable():
    with state.lock: state.animal_detection_enabled = True
    return jsonify({"status": "ok", "message": "Animal detection enabled"})

@app.route("/disable_animal_detection", methods=["POST"])
def disable():
    with state.lock: state.animal_detection_enabled = False
    return jsonify({"status": "ok", "message": "Animal detection disabled"})

if __name__ == "__main__":
    print("Server starting...")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=5000, threaded=True)
