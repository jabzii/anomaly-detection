# 🔥🐾 Wildlife & Fire Anomaly Detection System

A real-time computer vision system that detects **fire, smoke, and dangerous wildlife** from live camera feeds or uploaded videos, and sends **SMS alerts via Twilio** when a threat is confirmed.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithm Deep Dive](#algorithm-deep-dive)
- [System Architecture](#system-architecture)
- [Project File Reference](#project-file-reference)
- [Detected Classes](#detected-classes)
- [SMS Alert System](#sms-alert-system)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Training Your Own Model](#training-your-own-model)
- [API Endpoints](#api-endpoints)
- [Hardware Requirements](#hardware-requirements)

---

## Overview

This system monitors video streams in real-time and raises alerts when it detects:

- 🔥 **Fire or smoke** — triggers an immediate SMS alert
- 🐾 **Dangerous wildlife** — buffalo, elephant, tiger, wild boar — triggers a batched wildlife SMS

It is built on top of **YOLO11** (You Only Look Once, version 11) — a state-of-the-art single-stage object detection neural network — served through a **Flask web server** with a live MJPEG video stream viewable in any browser.

---

## Algorithm Deep Dive

### 1. Object Detection — YOLO11

**YOLO (You Only Look Once)** is a real-time object detection algorithm. Unlike two-stage detectors (e.g. Faster R-CNN), YOLO processes the entire image in a single forward pass through a convolutional neural network, making it fast enough for live video.

#### How YOLO11 works:

```
Input Frame (416×416 px)
        ↓
  Backbone CNN         ← extracts spatial features at multiple scales
  (C2f + SPPF blocks)
        ↓
  Neck (FPN + PAN)     ← fuses features from different scales
        ↓
  Detection Head       ← predicts bounding boxes + class scores + confidence
        ↓
  NMS (Non-Max         ← removes duplicate overlapping boxes,
  Suppression)            keeps the one with highest confidence
        ↓
  Output: [x1,y1,x2,y2, class_id, confidence] per detected object
```

#### Key YOLO11 improvements over earlier versions:
- **C2f blocks** — improved gradient flow via cross-stage partial connections
- **Anchor-free detection** — predicts box center offsets directly, no predefined anchor boxes
- **Decoupled head** — separate branches for classification and box regression
- **Scalable** — nano (n), small (s), medium (m), large (l), extra-large (x) variants

This project uses `yolo11l.pt` (large) weights fine-tuned on 6 custom classes.

---

### 2. Object Tracking — ByteTrack

Once YOLO detects objects in a frame, **ByteTrack** assigns persistent IDs across frames.

```python
results = model.track(source=frame, persist=True, conf=0.65, imgsz=416)
```

- `persist=True` keeps tracker state between frames
- Each detected object gets a track ID so the system knows "this is still the same tiger"
- Reduces jitter and false triggers from single-frame detections

---

### 3. Confirmation Threshold (False Positive Filter)

A single-frame detection is **not enough to trigger an SMS**. The system requires **3 consecutive frames** with the same class before any alert fires:

```
Frame 1: fire detected  → counter = 1  (BLOCKED)
Frame 2: fire detected  → counter = 2  (BLOCKED)
Frame 3: fire detected  → counter = 3  (SEND SMS ✅)
Frame 4: fire gone      → counter reset to 0
```

This eliminates false positives caused by image compression artifacts, motion blur, or brief lighting changes.

---

### 4. Multi-Animal Batch Window

When multiple wildlife species appear simultaneously (e.g. tiger + elephant + buffalo), the system uses a **5-second collection window** to batch them into a **single SMS** instead of sending 3 separate messages:

```
Tiger confirmed   → batch window OPENS (5s timer)
Elephant confirmed (2s later) → added to batch
Buffalo confirmed (3s later)  → added to batch
Timer expires after 5s
→ ONE SMS: "TIGER, ELEPHANT, BUFFALO detected"
```

This prevents SMS credit waste when herds appear.

---

### 5. Detection Pipeline Per Frame

```
VideoCapture.read()
      ↓
 process_frame()
      ├── model.track()          ← YOLO11 inference + ByteTrack
      ├── classify detections    ← fire/smoke vs wildlife
      ├── draw bounding boxes    ← OpenCV rectangle + label text
      ├── update DetectionState  ← thread-safe shared state
      ├── confirmation counter   ← require N consecutive frames
      ├── batch window check     ← collect multi-species
      └── trigger SMS (daemon thread) ← non-blocking Twilio call
      ↓
 MJPEG encode → Flask stream → Browser
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    start.py (Dashboard)                  │
│  Kills stale port → launches app.py → rich terminal UI  │
└────────────────────┬────────────────────────────────────┘
                     │ subprocess (stdout pipe)
┌────────────────────▼────────────────────────────────────┐
│                      app.py (Flask Server)               │
│                                                          │
│  /video_feed  ──→  generate_frames()                    │
│                         ↓                               │
│                    process_frame()                       │
│                    ├── YOLO11 model.track()              │
│                    ├── Fire / Animal classification      │
│                    ├── SMS trigger logic                 │
│                    └── OpenCV frame annotation           │
│                                                          │
│  /status      ──→  DetectionState (thread-safe)         │
│  /start_camera──→  cv2.VideoCapture(index)              │
│  /upload      ──→  cv2.VideoCapture(file_path)          │
│  /stop_camera ──→  cap.release()                        │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   Twilio REST API   │  (SMS sent in daemon thread)
          └─────────────────────┘
```

---

## Project File Reference

### 🚀 Runtime Files (used to run the system)

| File | Purpose |
|---|---|
| `start.py` | **Main entry point.** Kills stale port, launches `app.py`, shows live terminal dashboard with all events, alerts, and SMS log |
| `app.py` | Flask web server. Runs YOLO11 inference on every video frame, manages detection state, handles SMS alerts |
| `.env` | Twilio credentials and system configuration. **Never commit this file to git.** |
| `templates/index.html` | Browser dashboard UI (video feed + status indicators) |
| `uploads/` | Temporary storage for uploaded video files |

### 🧠 Model Files (the trained neural network)

| File / Directory | Purpose |
|---|---|
| `runs/detect/balanced_model/weights/best.pt` | **Primary model weights** — used by `app.py` by default |
| `best_yolo11l_wildlife_fire.pt` | Convenience copy of the best model in the project root |
| `yolo11l.pt` | Base YOLO11 large pretrained weights (ImageNet) — used as starting point for training |
| `yolo11n.pt` | YOLO11 nano — used for fast training runs |
| `yolo11s.pt` | YOLO11 small — intermediate option |

### 🏋️ Training Files (used to train / retrain the model)

| File | Purpose |
|---|---|
| `train_yolo11.py` | Full training script. Configures YOLO11 hyperparameters, runs training, evaluates on test split, copies best weights |
| `master_dataset/` | The complete annotated dataset (images + YOLO-format labels). Contains `train/`, `val/`, `test/` splits and `data.yaml` |
| `runs/train/` | Training output: loss curves, confusion matrices, checkpoint weights per epoch |

### 🛠️ Dataset Preparation Scripts (developer / one-time use)

| File | Purpose |
|---|---|
| `balance_dataset.py` | Resamples the dataset so all 6 classes have equal representation. Prevents the model from being biased toward common classes |
| `organize_yolo_data.py` | Converts raw image collections into proper YOLO directory structure (`images/train`, `labels/train`, etc.) |
| `clean_dataset.py` | Removes corrupt images, mismatched label files, and empty annotations |
| `fix_labels.py` | Corrects class ID mismatches in label files (e.g. if class indices shifted after reorganisation) |
| `analyze.py` | Prints per-class image/label counts and flags imbalance issues |
| `formater/` | Additional label format conversion utilities |

### 📦 Archive / Reference

| File / Directory | Purpose |
|---|---|
| `archive(2)/` | Old scripts and experiments kept for reference |
| `datasets/` | Raw source datasets before processing |
| `Train_YOLO_Models.ipynb` | Jupyter notebook version of the training pipeline (for Google Colab use) |
| `server.log` | Last server run log output |

---

## Detected Classes

The model outputs 6 classes (IDs match the trained `best.pt`):

| ID | Class | Alert Type | SMS Triggered |
|---|---|---|---|
| 0 | `smoke` | 🔥 Fire/Smoke | ✅ Yes — immediately |
| 1 | `fire` | 🔥 Fire/Smoke | ✅ Yes — immediately |
| 2 | `buffalo` | 🐾 Wildlife | ✅ Yes — batched |
| 3 | `elephant` | 🐾 Wildlife | ✅ Yes — batched |
| 4 | `tiger` | 🐾 Wildlife | ✅ Yes — batched |
| 5 | `wild_boar` | 🐾 Wildlife | ✅ Yes — batched |

---

## SMS Alert System

Powered by **Twilio**. Alerts are sent in a background daemon thread so they never block video processing.

### Alert conditions

| Trigger | Condition | Cooldown |
|---|---|---|
| Fire/Smoke SMS | 3+ consecutive frames with fire/smoke, not already alerted for this event | 60 seconds |
| Wildlife SMS | 3+ consecutive frames of a species + 5s batch window expired | 120 seconds (global) |
| Combined SMS | Fire AND wildlife confirmed in same window | Both merged into 1 message |

### Example SMS messages

```
🔥 FIRE/SMOKE ALERT
Detected: FIRE
Source: camera
Time: 2026-05-09 12:20:11

🐾 WILDLIFE ALERT
Detected: TIGER, ELEPHANT
Source: camera
Time: 2026-05-09 12:22:05

🔥🐾 FIRE + WILDLIFE ALERT
Detected: SMOKE, WILD_BOAR
Source: video
Time: 2026-05-09 12:25:33
```

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU recommended (CUDA 11.8+) — CPU fallback available
- Twilio account (free trial works)

### Steps

```bash
# 1. Clone the project
git clone <repo-url>
cd main_project

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install ultralytics flask twilio python-dotenv opencv-python rich werkzeug

# 4. Set up credentials
cp .env.example .env   # then fill in your Twilio values
```

---

## Configuration

Edit `.env` in the project root:

```env
# Twilio credentials (get from twilio.com/console)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx        # Your Twilio number
ALERT_TO_PHONE_NUMBER=+91xxxxxxxxxx     # Number to receive alerts

# Tuning (optional — these are the defaults)
SMS_COOLDOWN_SECONDS=60          # Min gap between fire SMS alerts
ANIMAL_SMS_COOLDOWN_SECONDS=120  # Min gap between wildlife batches
CONFIRMATION_FRAMES=3            # Frames needed before alerting
ANIMAL_BATCH_WINDOW_SECONDS=5    # Window to collect multi-species
```

---

## Running the System

### Option A — Dashboard mode (recommended)

```bash
source .venv/bin/activate
python start.py
```

This opens a live terminal dashboard showing:
- System status (server, model, SMS readiness)
- All detection events (deduplicated, with hit count)
- SMS alerts sent log with Twilio SIDs
- Filtered system log (no polling spam)

### Option B — Direct server mode

```bash
source .venv/bin/activate
python app.py
```

Then open `http://localhost:9000` in your browser.

### Using the web interface

1. Open `http://localhost:9000`
2. Click **Start Camera** for live webcam feed, or **Upload Video** for a file
3. The system will automatically detect and alert
4. Click **Stop** to end the session

---

## Training Your Own Model

```bash
source .venv/bin/activate

# Fast training (20 epochs, nano model — for testing)
FAST_TRAIN=1 python train_yolo11.py

# Full training (1000 epochs, large model — for production)
FAST_TRAIN=0 python train_yolo11.py
```

The best weights are automatically copied to:
```
runs/train/yolo11l_wildlife_fire/weights/best.pt
best_yolo11l_wildlife_fire.pt   ← root copy for convenience
```

### Dataset structure expected

```
master_dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Training hyperparameters (key settings in `train_yolo11.py`)

| Parameter | Fast mode | Full mode | Purpose |
|---|---|---|---|
| `model` | yolo11n.pt | yolo11l.pt | Nano for speed, large for accuracy |
| `epochs` | 20 | 1000 | Training iterations |
| `imgsz` | 416 | 512 | Input resolution |
| `optimizer` | AdamW | AdamW | Faster convergence than SGD |
| `lr0` | 0.0001 | 0.0001 | Conservative LR for stability |
| `amp` | False | False | Disabled to prevent NaN losses on 6GB VRAM |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard UI |
| `GET` | `/video_feed` | Live MJPEG video stream |
| `GET` | `/status` | JSON — current detection state, FPS, detections list |
| `POST` | `/start_camera` | Start webcam. Body: `{"camera_index": 0}` |
| `POST` | `/upload` | Upload a video file (multipart/form-data) |
| `POST` | `/stop_camera` | Stop camera or video playback |
| `POST` | `/stop_video` | Alias for `/stop_camera` |
| `POST` | `/enable_animal_detection` | Re-enable wildlife detection |
| `POST` | `/disable_animal_detection` | Disable wildlife detection (fire still active) |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU fallback) | NVIDIA 6GB VRAM (RTX 3050+) |
| Storage | 2 GB | 20 GB (for full dataset) |
| OS | Ubuntu 20.04+ / Windows 10+ | Ubuntu 22.04 |

---

## Notes

- **Port conflict:** If you see `Address already in use`, run `fuser -k 9000/tcp` or use `python start.py` which handles this automatically.
- **`sudo` not needed:** Run as a normal user inside the `.venv`. Using `sudo` breaks the virtualenv path.
- **ANIMAL_CLASSES** must match the class names output by your trained model exactly (case-sensitive). Check with `model.names` at startup.

