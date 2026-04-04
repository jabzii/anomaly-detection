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

# ─────────────────────────────── HYPERPARAMETERS ──────────────────────────────
CONFIG = {
    # ── Model ─────────────────────────────────────────────────────────────────
    "model"       : "yolo11l.pt",       # YOLO11 large — best balance for 6 GB VRAM
    "data"        : str(DATASET_YAML),  # dataset config

    # ── Core training ─────────────────────────────────────────────────────────
    "epochs"      : 1,                # full training run
    "imgsz"       : 512,                # 512 → ~36% faster than 640, still high quality
    "batch"       : 16,                 # 6 GB free → 16 saturates GPU throughput
    "workers"     : 8,                  # more parallel data loading
    "rect"        : True,               # rectangular batches → removes padding waste
    "cache"       : "disk",             # reuse scan cache from previous run

    # ── Optimiser ─────────────────────────────────────────────────────────────
    "optimizer"   : "AdamW",            # AdamW converges faster than SGD
    "lr0"         : 0.001,              # initial learning rate
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
    "mosaic"      : 1.0,               # mosaic augmentation (4-image)
    "mixup"       : 0.1,               # MixUp augmentation
    "copy_paste"  : 0.1,               # copy-paste for small objects

    # ── Loss weights ──────────────────────────────────────────────────────────
    "box"         : 7.5,
    "cls"         : 0.5,
    "dfl"         : 1.5,

    # ── Early stopping & checkpointing ────────────────────────────────────────
    "patience"    : 30,                # stop if no improvement for 30 epochs
    "save"        : True,
    "save_period" : 10,                # save checkpoint every 10 epochs

    # ── Evaluation & logging ──────────────────────────────────────────────────
    "val"         : True,
    "plots"       : True,
    "verbose"     : True,
    "project"     : str(RUNS_DIR),
    "name"        : "yolo11l_wildlife_fire",
    "exist_ok"    : True,

    # ── Device ────────────────────────────────────────────────────────────────
    "device"      : 0,                 # GPU 0  (RTX 3050)
    "amp"         : True,              # Automatic Mixed Precision → faster + lower VRAM
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
        if not img_dir.exists():
            print(f"[ERROR] Missing image dir: {img_dir}")
            sys.exit(1)
        n_imgs = len(list(img_dir.glob("*.*")))
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  [{split:5}]  images: {n_imgs:6,d}   labels: {n_lbls:6,d}")

    print()


def print_banner():
    print("=" * 65)
    print("  YOLO11 FULL TRAINING  —  Wildlife + Fire/Smoke Detection")
    print("=" * 65)
    print(f"  Model    : {CONFIG['model']}")
    print(f"  Dataset  : {CONFIG['data']}")
    print(f"  Epochs   : {CONFIG['epochs']}")
    print(f"  Img size : {CONFIG['imgsz']}")
    print(f"  Batch    : {CONFIG['batch']}")
    print(f"  Rect     : {CONFIG['rect']}")
    print(f"  Optimizer: {CONFIG['optimizer']}")
    print(f"  AMP      : {CONFIG['amp']}")
    print(f"  Device   : GPU {CONFIG['device']}  ({torch.cuda.get_device_name(0)})")
    print(f"  CUDA     : {torch.version.cuda}")
    print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 65)
    print()


# ──────────────────────────────── MAIN ────────────────────────────────────────

def main():
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
    best_pt  = save_dir / "weights" / "best.pt"
    last_pt  = save_dir / "weights" / "last.pt"

    print(f"  Results  saved to : {save_dir}")
    print(f"  Best weights      : {best_pt}")
    print(f"  Last weights      : {last_pt}")

    # ── Run final validation on test split ────────────────────────────────────
    print("\n[Evaluating on test split …]")
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(
        data   = CONFIG["data"],
        split  = "test",
        imgsz  = CONFIG["imgsz"],
        device = CONFIG["device"],
        plots  = True,
    )

    print("\n── Test-split Metrics ──────────────────────────────────────")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print("=" * 65)

    # ── Convenience copy of best model to project root ─────────────────────────
    dest = BASE_DIR / "best_yolo11l_wildlife_fire.pt"
    shutil.copy2(best_pt, dest)
    print(f"\n  Best model copied to: {dest}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
