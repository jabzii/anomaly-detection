"""
YOLO11 Full Training Script
Dataset  : master_dataset  (buffalo, elephant, tiger, wild_boar, fire, smoke)
Model    : yolo11l.pt  (large — best fit for 6 GB VRAM)
GPU      : NVIDIA RTX 3050 6GB Laptop GPU
"""

# ── CUDA memory optimisation (must be set before torch import) ────────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

# ─────────────────────────────── PATHS ────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATASET_YAML = BASE_DIR / "master_dataset" / "data.yaml"
RUNS_DIR     = BASE_DIR / "runs" / "train"

# Fast mode: export FAST_TRAIN=1 for quickest turnaround
FAST_TRAIN = os.getenv("FAST_TRAIN", "1") == "1"

# ─────────────────────────────── HYPERPARAMETERS ──────────────────────────────
CONFIG = {
    # ── Model ─────────────────────────────────────────────────────────────────
    "model"       : "yolo11n.pt" if FAST_TRAIN else "yolo11l.pt",  # nano is much faster
    "data"        : str(DATASET_YAML),  # dataset config

    # ── Core training ─────────────────────────────────────────────────────────
    "epochs"      : 20 if FAST_TRAIN else 1000,
    "imgsz"       : 416 if FAST_TRAIN else 512,
    "batch"       : 32 if FAST_TRAIN else 16,
    "workers"     : 8,                  # more parallel data loading
    "rect"        : True,               # rectangular batches → removes padding waste
    "cache"       : "disk",             # reuse scan cache from previous run

    # ── Optimiser ─────────────────────────────────────────────────────────────
    "optimizer"   : "AdamW",            # AdamW converges faster than SGD
    "lr0"         : 0.0001,             # initial learning rate (reduced for stability)
    "lrf"         : 0.01,               # final LR = lr0 * lrf
    "momentum"    : 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr" : 0.1,

    # ── Augmentation ──────────────────────────────────────────────────────────
    "hsv_h"       : 0.015,
    "hsv_s"       : 0.7,
    "hsv_v"       : 0.4,
    "degrees"     : 5.0,               # rotation
    "translate"   : 0.1,
    "scale"       : 0.5,
    "shear"       : 2.0,
    "perspective" : 0.0,
    "flipud"      : 0.0,
    "fliplr"      : 0.5,
    "mosaic"      : 0.5 if FAST_TRAIN else 1.0,
    "mixup"       : 0.0 if FAST_TRAIN else 0.1,
    "copy_paste"  : 0.0 if FAST_TRAIN else 0.1,

    # ── Loss weights ──────────────────────────────────────────────────────────
    "box"         : 7.5,
    "cls"         : 0.5,
    "dfl"         : 1.5,

    # ── Early stopping & checkpointing ────────────────────────────────────────
    "patience"    : 10 if FAST_TRAIN else 30,
    "save"        : True,
    "save_period" : -1 if FAST_TRAIN else 10,  # only save final checkpoints in fast mode

    # ── Evaluation & logging ──────────────────────────────────────────────────
    "val"         : True,               # always validate for monitoring
    "plots"       : True,               # always plot
    "verbose"     : True,
    "project"     : str(RUNS_DIR),
    "name"        : "yolo11l_wildlife_fire",
    "exist_ok"    : True,

    # ── Device ────────────────────────────────────────────────────────────────
    "device"      : 0,                 # GPU index (auto-fallback to CPU if unavailable)
    "amp"         : False,             # Disabled to avoid NaN issues on mixed precision
}

# ──────────────────────────────── HELPERS ─────────────────────────────────────

def verify_dataset():
    """Quick sanity-check before launching training."""
    yaml_path = Path(CONFIG["data"])
    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found at: {yaml_path}")
        sys.exit(1)

    base = yaml_path.parent
    for split in ("train", "val", "test"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if split in ("train", "val") and not img_dir.exists():
            print(f"[ERROR] Missing image dir: {img_dir}")
            sys.exit(1)
        if split == "test" and not img_dir.exists():
            print(f"  [test ]  images:      0   labels:      0   (test split missing; will skip test eval)")
            continue
        n_imgs = len(list(img_dir.glob("*.*")))
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  [{split:5}]  images: {n_imgs:6,d}   labels: {n_lbls:6,d}")

    print()


def print_banner():
    cuda_ok = torch.cuda.is_available()
    device_label = f"GPU {CONFIG['device']}" if cuda_ok else "CPU"

    print("=" * 65)
    print("  YOLO11 FULL TRAINING  —  Wildlife + Fire/Smoke Detection")
    print("=" * 65)
    print(f"  Model    : {CONFIG['model']}")
    print(f"  Mode     : {'FAST_TRAIN' if FAST_TRAIN else 'FULL_TRAIN'}")
    print(f"  Dataset  : {CONFIG['data']}")
    print(f"  Epochs   : {CONFIG['epochs']}")
    print(f"  Img size : {CONFIG['imgsz']}")
    print(f"  Batch    : {CONFIG['batch']}")
    print(f"  Rect     : {CONFIG['rect']}")
    print(f"  Optimizer: {CONFIG['optimizer']}")
    print(f"  AMP      : {CONFIG['amp']}")
    print(f"  Device   : {device_label}")
    if cuda_ok:
        print(f"  GPU Name : {torch.cuda.get_device_name(int(CONFIG['device']))}")
        print(f"  CUDA     : {torch.version.cuda}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(int(CONFIG['device'])).total_memory / 1e9:.1f} GB")
    else:
        print("  CUDA     : not available")
    print("=" * 65)
    print()


def prepare_runtime():
    """Make runtime config safe for the current machine."""
    if torch.cuda.is_available():
        return

    CONFIG["device"] = "cpu"
    CONFIG["amp"] = False
    print("[WARN] CUDA not available. Falling back to CPU (AMP disabled).\n")


# ──────────────────────────────── MAIN ────────────────────────────────────────

def main():
    prepare_runtime()
    print_banner()

    # Verify dataset integrity
    print("[1/3] Verifying dataset …")
    verify_dataset()

    # Load model
    print(f"[2/3] Loading model: {CONFIG['model']} …")
    model = YOLO(CONFIG["model"])

    # Launch training
    print("[3/3] Starting training …\n")
    results = model.train(**CONFIG)

    # ── Post-training summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    print(f"  Results  saved to : {save_dir}")
    print(f"  Best weights      : {best_pt} ({'found' if best_pt.exists() else 'missing'})")
    print(f"  Last weights      : {last_pt} ({'found' if last_pt.exists() else 'missing'})")

    # ── Run final validation on test split ────────────────────────────────────
    test_images_dir = DATASET_YAML.parent / "images" / "test"
    if FAST_TRAIN:
        print("\n[INFO] Skipping test evaluation in FAST_TRAIN mode.")
    elif best_pt.exists() and test_images_dir.exists():
        print("\n[Evaluating on test split …]")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(
            data=CONFIG["data"],
            split="test",
            imgsz=CONFIG["imgsz"],
            device=CONFIG["device"],
            plots=True,
        )

        print("\n── Test-split Metrics ──────────────────────────────────────")
        print(f"  mAP50      : {metrics.box.map50:.4f}")
        print(f"  mAP50-95   : {metrics.box.map:.4f}")
        print(f"  Precision  : {metrics.box.mp:.4f}")
        print(f"  Recall     : {metrics.box.mr:.4f}")
        print("=" * 65)
    else:
        print("\n[INFO] Skipping test evaluation (missing best.pt or test split).")

    # ── Convenience copy of best model to project root ─────────────────────────
    dest = BASE_DIR / "best_yolo11l_wildlife_fire.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, dest)
        print(f"\n  Best model copied to: {dest}")
    else:
        print("\n  Best model not copied (best.pt missing).")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
