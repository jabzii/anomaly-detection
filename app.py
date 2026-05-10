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
import traceback
from pathlib import Path
from flask import Flask, Response, render_template, request, jsonify
from ultralytics import YOLO
import werkzeug

# ── Load .env file automatically (so credentials don't need manual export) ──────
def _load_dotenv(path: str = ".env"):
    """Minimal .env loader – works without python-dotenv installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        print(f"[ENV] Loaded {path} via python-dotenv")
        return
    except ImportError:
        pass
    env_path = Path(__file__).resolve().parent / path
    if not env_path.exists():
        print(f"[ENV] No .env file found at {env_path}")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
    print(f"[ENV] Loaded {env_path} via built-in parser")

_load_dotenv()

try:
    from twilio.rest import Client
except Exception:
    Client = None

# ─────────────────────────────── CONFIG ───────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "balanced_model" / "weights" / "best.pt"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.80 # VERY low for debugging

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
ALERT_TO_PHONE_NUMBER = os.getenv("ALERT_TO_PHONE_NUMBER", "")
SMS_COOLDOWN_SECONDS        = int(os.getenv("SMS_COOLDOWN_SECONDS", "60"))
ANIMAL_SMS_COOLDOWN_SECONDS = int(os.getenv("ANIMAL_SMS_COOLDOWN_SECONDS", "120"))
# Require this many consecutive frames before sending an SMS (avoids false positives)
CONFIRMATION_FRAMES         = int(os.getenv("CONFIRMATION_FRAMES", "3"))
# Collect all animal species appearing within this window into ONE SMS
ANIMAL_BATCH_WINDOW_SECONDS = int(os.getenv("ANIMAL_BATCH_WINDOW_SECONDS", "5"))

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
        self.sms_sending_enabled: bool = True

        self.fire_detected: bool = False
        self.animal_detected: bool = False
        self.confidence: float = 0.0
        self.detections: list = []

        self._fps_t = time.time()
        self._fps_count = 0
        self.fps: float = 0.0

        # ── Fire SMS state ──────────────────────────────────────────────────
        self.last_fire_alert_ts: float = 0.0
        self.fire_alert_sent_for_current_event: bool = False
        self.fire_confirm_count: int = 0          # consecutive frames with fire

        # ── Animal SMS state ────────────────────────────────────────────────
        self.animal_confirm_count: dict  = {}    # label -> consecutive frame count
        self.animal_alert_active: dict   = {}    # label -> bool (already in batch/sent)
        self.last_global_animal_ts: float = 0.0  # global cooldown after any animal SMS
        # Batch window: collect multiple species before firing one SMS
        self.animal_batch_pending: bool  = False  # window is open
        self.animal_batch_start_ts: float = 0.0   # when window opened
        self.animal_batch_labels: set    = set()  # species queued in this batch

    def update_fps(self):
        self._fps_count += 1
        now = time.time()
        elapsed = now - self._fps_t
        if elapsed >= 1.0:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t = now

state = DetectionState()


def twilio_ready() -> bool:
    return bool(
        Client is not None
        and TWILIO_ACCOUNT_SID
        and TWILIO_AUTH_TOKEN
        and TWILIO_PHONE_NUMBER
        and ALERT_TO_PHONE_NUMBER
    )


def print_sms_diagnostics():
    """Print a full readiness report at startup so misconfiguration is obvious."""
    print("\n" + "═" * 55)
    print("  SMS / TWILIO DIAGNOSTICS")
    print("═" * 55)
    print(f"  twilio package   : {'✅ installed' if Client is not None else '❌ NOT installed  →  pip install twilio'}")
    print(f"  ACCOUNT_SID      : {'✅ ' + TWILIO_ACCOUNT_SID[:8] + '...' if TWILIO_ACCOUNT_SID else '❌ MISSING'}")
    print(f"  AUTH_TOKEN       : {'✅ set' if TWILIO_AUTH_TOKEN else '❌ MISSING'}")
    print(f"  FROM number      : {'✅ ' + TWILIO_PHONE_NUMBER if TWILIO_PHONE_NUMBER else '❌ MISSING'}")
    print(f"  TO number        : {'✅ ' + ALERT_TO_PHONE_NUMBER if ALERT_TO_PHONE_NUMBER else '❌ MISSING'}")
    print(f"  Fire cooldown    : {SMS_COOLDOWN_SECONDS}s")
    print(f"  Animal cooldown  : {ANIMAL_SMS_COOLDOWN_SECONDS}s (global, after each batch)")
    print(f"  Animal batch win : {ANIMAL_BATCH_WINDOW_SECONDS}s (multi-species collected here)")
    print(f"  Confirm frames   : {CONFIRMATION_FRAMES} consecutive frames needed")
    print(f"  Overall ready    : {'✅ SMS ENABLED' if twilio_ready() else '❌ SMS DISABLED'}")
    print("═" * 55 + "\n")


def send_alert_sms(alert_type: str, labels: list[str], source: str):
    """
    Unified SMS sender for fire AND animal alerts.
    alert_type : "fire" | "animal" | "fire+animal"
    labels     : list of detected class names e.g. ["fire", "tiger"]
    source     : "camera" | "video" | "rtsp"

    Strategy to save credits:
    - One SMS per event (confirmation threshold already applied before calling this).
    - If fire+animal detected simultaneously, batched into a SINGLE message.
    - Runs in a background daemon thread — never blocks video pipeline.
    """
    if not twilio_ready():
        print("[WARN] SMS not sent: Twilio not configured or package missing.")
        return

    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Replace specific animal names with "animal" for the SMS
        display_labels = ["animal" if l in ANIMAL_CLASSES else l for l in labels]
        unique_labels = list(dict.fromkeys(display_labels))   # preserve order, deduplicate
        label_str = ", ".join(l.upper() for l in unique_labels)

        if alert_type == "fire":
            prefix = "🔥 FIRE/SMOKE ALERT"
        elif alert_type == "animal":
            prefix = "🐾 WILDLIFE ALERT"
        else:                          # fire+animal combined
            prefix = "🔥🐾 FIRE + WILDLIFE ALERT"

        msg = (
            f"{prefix}\n"
            f"Detected: {label_str}\n"
            f"Source: {source}\n"
            f"Time: {timestamp}"
        )

        if not state.sms_sending_enabled:
            print(f"[INFO] [{alert_type.upper()}] EVENT TRIGGERED but SMS SKIPPED (SMS is toggled OFF). Labels: {label_str}")
            return

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        sms = client.messages.create(
            body=msg,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_TO_PHONE_NUMBER,
        )
        print(f"[INFO] [{alert_type.upper()}] SMS sent. SID: {sms.sid} | Labels: {label_str}")
    except Exception as e:
        print(f"[ERROR] Failed to send SMS: {e}")
        traceback.print_exc()

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

print_sms_diagnostics()

# ─────────────────────────────── FRAME GENERATOR ─────────────────────────────

def process_frame(frame: np.ndarray) -> np.ndarray:
    try:
        if frame is None or frame.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        results = model.track(
    source=frame,
    persist=True, # Keeps track of objects across frames
    conf=CONFIDENCE_THRESHOLD,    # You can safely lower this to 50% when tracking is on
    imgsz=416,    # Make sure this matches your training imgsz
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
                color = (0, 0, 255)           # Red box for fire/smoke
                print(f"[INFO] FIRE! {label} ({conf:.2f})")

            elif label in ANIMAL_CLASSES and state.animal_detection_enabled:
                animal_det = True
                should_draw = True
                color = (255, 180, 0)         # Orange box for wildlife
                print(f"[INFO] ANIMAL! {label} ({conf:.2f})")

            # Only draw boxes for fire and animal — skip all other classes
            if should_draw:
                # Show "Animal" for wildlife, exact name for fire/smoke
                display_label = "Animal" if label in ANIMAL_CLASSES else label
                cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    plotted_frame,
                    f"{display_label} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Track all detections for status API (even if not drawn)
            if label in FIRE_CLASSES or (label in ANIMAL_CLASSES and state.animal_detection_enabled):
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

        # ── SMS Trigger Logic ─────────────────────────────────────────────────
        # Strategy: require CONFIRMATION_FRAMES consecutive frames before sending
        # any SMS. This eliminates false positives and wastes zero credits on
        # transient noise. Fire and each animal class are tracked independently.
        sms_payload = None   # (alert_type, labels) — set only when we should send

        # Collect detected labels this frame
        fire_labels_this_frame   = [d["label"] for d in frame_detections if d["label"] in FIRE_CLASSES]
        animal_labels_this_frame = [d["label"] for d in frame_detections
                                    if d["label"] in ANIMAL_CLASSES] if state.animal_detection_enabled else []

        with state.lock:
            state.fire_detected   = fire_det
            state.animal_detected = animal_det
            state.confidence      = max_conf * 100
            state.detections      = frame_detections
            now = time.time()
            current_source = state.source

            # ── Fire confirmation & cooldown ─────────────────────────────────
            if fire_det:
                state.fire_confirm_count += 1
            else:
                state.fire_confirm_count = 0
                state.fire_alert_sent_for_current_event = False   # reset for next event

            fire_cooldown_ok  = (now - state.last_fire_alert_ts) >= SMS_COOLDOWN_SECONDS
            fire_confirmed    = state.fire_confirm_count >= CONFIRMATION_FRAMES
            fire_ready_to_sms = (fire_det and fire_confirmed
                                 and not state.fire_alert_sent_for_current_event
                                 and fire_cooldown_ok)

            # ── Fire debug log (only when fire is actively detected) ──────────
            if fire_det:
                cooldown_remaining = max(0, SMS_COOLDOWN_SECONDS - (now - state.last_fire_alert_ts))
                print(
                    f"[SMS-DBG][FIRE] confirm={state.fire_confirm_count}/{CONFIRMATION_FRAMES} "
                    f"event_sent={state.fire_alert_sent_for_current_event} "
                    f"cooldown_ok={fire_cooldown_ok} (remaining={cooldown_remaining:.0f}s) "
                    f"→ {'WILL SEND' if fire_ready_to_sms else 'BLOCKED'}"
                )

            if fire_ready_to_sms:
                state.last_fire_alert_ts = now
                state.fire_alert_sent_for_current_event = True

            # ── Animal confirmation + batch-window collection ─────────────────
            # How it works:
            #   1. Each species needs CONFIRMATION_FRAMES consecutive detections.
            #   2. Once confirmed, it is added to animal_batch_labels and a
            #      ANIMAL_BATCH_WINDOW_SECONDS timer starts (if not already running).
            #   3. Any OTHER species confirmed during that window are added too.
            #   4. When the window expires → ONE SMS with ALL collected species.
            #   5. A global cooldown (ANIMAL_SMS_COOLDOWN_SECONDS) then applies
            #      before any new animal batch can be triggered.
            global_cooldown_ok = (now - state.last_global_animal_ts) >= ANIMAL_SMS_COOLDOWN_SECONDS

            for lbl in ANIMAL_CLASSES:
                # Update consecutive-frame counter
                if lbl in animal_labels_this_frame:
                    state.animal_confirm_count[lbl] = state.animal_confirm_count.get(lbl, 0) + 1
                else:
                    state.animal_confirm_count[lbl] = 0
                    state.animal_alert_active[lbl]  = False   # gone → allow re-entry next time

                confirmed    = state.animal_confirm_count.get(lbl, 0) >= CONFIRMATION_FRAMES
                already_queued = state.animal_alert_active.get(lbl, False)

                # Debug log when this species is visible
                if lbl in animal_labels_this_frame:
                    gc_remaining = max(0, ANIMAL_SMS_COOLDOWN_SECONDS - (now - state.last_global_animal_ts))
                    print(
                        f"[SMS-DBG][ANIMAL:{lbl}] "
                        f"confirm={state.animal_confirm_count.get(lbl,0)}/{CONFIRMATION_FRAMES} "
                        f"queued={already_queued} "
                        f"global_cooldown_ok={global_cooldown_ok} (remaining={gc_remaining:.0f}s) "
                        f"→ {'QUEUING' if (confirmed and not already_queued and global_cooldown_ok) else 'BLOCKED'}"
                    )

                # Queue species into the batch window if all conditions met
                if (lbl in animal_labels_this_frame
                        and confirmed
                        and not already_queued
                        and global_cooldown_ok):
                    state.animal_batch_labels.add(lbl)
                    state.animal_alert_active[lbl] = True   # don't double-queue
                    if not state.animal_batch_pending:
                        state.animal_batch_pending  = True
                        state.animal_batch_start_ts = now
                        print(
                            f"[SMS-DBG] Animal batch window OPENED "
                            f"(collecting for {ANIMAL_BATCH_WINDOW_SECONDS}s) "
                            f"first species: {lbl}"
                        )

            # ── Check if batch window has expired → emit ONE combined SMS ──────
            animal_ready_labels = []
            if state.animal_batch_pending:
                elapsed = now - state.animal_batch_start_ts
                print(
                    f"[SMS-DBG] Batch window: {elapsed:.1f}s / {ANIMAL_BATCH_WINDOW_SECONDS}s "
                    f"| queued={sorted(state.animal_batch_labels)}"
                )
                if elapsed >= ANIMAL_BATCH_WINDOW_SECONDS:
                    animal_ready_labels           = sorted(state.animal_batch_labels)
                    state.animal_batch_labels     = set()
                    state.animal_batch_pending    = False
                    state.last_global_animal_ts   = now
                    print(f"[SMS-DBG] Batch window CLOSED → sending 1 SMS for: {animal_ready_labels}")

            # ── Merge fire + animal into fewest possible SMS messages ──────────
            if fire_ready_to_sms and animal_ready_labels:
                sms_payload = ("fire+animal", fire_labels_this_frame + animal_ready_labels)
            elif fire_ready_to_sms:
                sms_payload = ("fire", fire_labels_this_frame)
            elif animal_ready_labels:
                sms_payload = ("animal", animal_ready_labels)

        if sms_payload:
            alert_type, labels = sms_payload
            print(f"[SMS] Dispatching {alert_type.upper()} alert → labels={labels} source={current_source}")
            threading.Thread(
                target=send_alert_sms,
                args=(alert_type, labels, current_source),
                daemon=True,
            ).start()

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
            "sms_sending_enabled": state.sms_sending_enabled,
        })

@app.route("/enable_animal_detection", methods=["POST"])
def enable():
    with state.lock: state.animal_detection_enabled = True
    return jsonify({"status": "ok", "message": "Animal detection enabled"})

@app.route("/disable_animal_detection", methods=["POST"])
def disable():
    with state.lock: state.animal_detection_enabled = False
    return jsonify({"status": "ok", "message": "Animal detection disabled"})

@app.route("/toggle_sms", methods=["POST"])
def toggle_sms():
    req = request.get_json(silent=True) or {}
    enable = req.get("enable", True)
    with state.lock: state.sms_sending_enabled = bool(enable)
    return jsonify({"status": "ok", "sms_sending_enabled": state.sms_sending_enabled})

if __name__ == "__main__":
    print("Server starting...")
    sys.stdout.flush()
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
