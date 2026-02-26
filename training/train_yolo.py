"""
train_yolo.py
=============
Module 2: YOLO TB Detection Model — Training Script.

This script trains a YOLOv8 model to detect Tuberculosis lesions in chest X-ray images.

What it does:
  1. Loads a pretrained YOLOv8 model (transfer learning from COCO weights)
  2. Fine-tunes it on your annotated TB chest X-ray dataset
  3. Saves the best-performing weights to models/tb_yolo.pt
  4. Logs training metrics: loss, mAP, precision, recall

Medical Context:
  Transfer learning lets us leverage general image features learned from millions
  of images (COCO dataset), then adapt the model to recognize specific TB lesion
  patterns in chest X-rays. This dramatically reduces the data required.

Training Commands:
  Basic training:
    python training/train_yolo.py

  With custom config:
    python training/train_yolo.py --epochs 50 --batch 8

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys
import argparse
import shutil
from datetime import datetime

# ── Add project root to Python path ───────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_yolo(epochs: int      = None,
               batch: int       = None,
               model_size: str  = None,
               resume: bool     = False) -> str:
    """
    Train a YOLOv8 model on the TB chest X-ray dataset.

    Args:
        epochs     (int):  Number of training epochs. Defaults to config.TRAIN_EPOCHS.
        batch      (int):  Batch size. Defaults to config.TRAIN_BATCH_SIZE.
        model_size (str):  YOLOv8 variant ('yolov8n', 'yolov8s', 'yolov8m', etc.).
        resume     (bool): If True, resume training from the last checkpoint.

    Returns:
        str: Path to the saved best model weights.

    Raises:
        FileNotFoundError: If dataset.yaml does not exist.
        RuntimeError: If training fails.
    """
    # ── Lazy import (allows the file to be imported without requiring ultralytics) ─
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    # ── Use defaults from config if not overridden ─────────────────────────────
    epochs     = epochs     or config.TRAIN_EPOCHS
    batch      = batch      or config.TRAIN_BATCH_SIZE
    model_size = model_size or config.YOLO_MODEL_SIZE

    # ── Verify dataset exists ──────────────────────────────────────────────────
    if not os.path.exists(config.DATASET_YAML):
        raise FileNotFoundError(
            f"dataset.yaml not found at {config.DATASET_YAML}. "
            "Run data/dataset_loader.py first to generate it."
        )

    print("=" * 60)
    print("  AI TB Detection — YOLO Training")
    print("=" * 60)
    print(f"  Model    : {model_size}")
    print(f"  Epochs   : {epochs}")
    print(f"  Batch    : {batch}")
    print(f"  Dataset  : {config.DATASET_YAML}")
    print(f"  Resume   : {resume}")
    print("=" * 60)

    # ── Initialize model ───────────────────────────────────────────────────────
    if resume and os.path.exists(config.MODEL_WEIGHTS):
        # Resume from existing TB weights (fine-tuning scenario)
        print(f"[INFO] Resuming from existing weights: {config.MODEL_WEIGHTS}")
        model = YOLO(config.MODEL_WEIGHTS)
    else:
        # Start from pretrained COCO weights — transfer learning
        pretrained_weights = f"{model_size}.pt"
        print(f"[INFO] Starting from pretrained weights: {pretrained_weights}")
        model = YOLO(pretrained_weights)

    # ── Run training ───────────────────────────────────────────────────────────
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"tb_training_{timestamp}"

    print(f"\n[INFO] Training started — run ID: {project_name}")

    results = model.train(
        data        = config.DATASET_YAML,
        epochs      = epochs,
        batch       = batch,
        imgsz       = config.IMAGE_SIZE,
        lr0         = config.TRAIN_LR,
        workers     = config.TRAIN_WORKERS,
        patience    = config.TRAIN_PATIENCE,
        project     = os.path.join(config.BASE_DIR, "runs", "train"),
        name        = project_name,
        seed        = config.RANDOM_SEED,
        exist_ok    = True,
        verbose     = True,
        # Data augmentation params (passed to Ultralytics augmentation pipeline)
        flipud      = 0.0,    # No vertical flip (X-rays always upright)
        fliplr      = 0.5,    # 50% horizontal flip
        degrees     = 10.0,   # Random rotation ±10°
        scale       = 0.3,    # Scale ±30%
        hsv_v       = 0.4,    # Value/brightness augmentation (useful for X-rays)
        hsv_s       = 0.0,    # No saturation change (X-rays are grayscale)
    )

    # ── Save best weights to models/ directory ─────────────────────────────────
    best_weights_src = os.path.join(
        config.BASE_DIR, "runs", "train", project_name, "weights", "best.pt"
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, config.MODEL_WEIGHTS)
        print(f"\n[SUCCESS] Best model weights saved to: {config.MODEL_WEIGHTS}")
    else:
        print(f"[WARNING] best.pt not found at expected location: {best_weights_src}")
        print("          Check the runs/train/ directory for trained weights.")

    return config.MODEL_WEIGHTS


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING VALIDATION (Quick sanity check after training)
# ─────────────────────────────────────────────────────────────────────────────

def validate_model(weights_path: str = None) -> dict:
    """
    Run YOLO validation on the validation set to get mAP, precision, recall.

    Args:
        weights_path (str): Path to model .pt file. Defaults to config.MODEL_WEIGHTS.

    Returns:
        dict: Validation metrics dict with keys: mAP50, mAP50-95, precision, recall.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    weights_path = weights_path or config.MODEL_WEIGHTS

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    print(f"[INFO] Validating model: {weights_path}")
    model   = YOLO(weights_path)
    metrics = model.val(data=config.DATASET_YAML,
                        imgsz=config.IMAGE_SIZE,
                        conf=config.CONFIDENCE_THRESHOLD)

    results = {
        "mAP50"     : float(metrics.box.map50),
        "mAP50-95"  : float(metrics.box.map),
        "precision" : float(metrics.box.p.mean()),
        "recall"    : float(metrics.box.r.mean()),
    }

    print("\n[VALIDATION RESULTS]")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command line arguments for training script."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for TB lesion detection on chest X-rays"
    )
    parser.add_argument("--epochs",  type=int, default=None,
                        help=f"Training epochs (default: {config.TRAIN_EPOCHS})")
    parser.add_argument("--batch",   type=int, default=None,
                        help=f"Batch size (default: {config.TRAIN_BATCH_SIZE})")
    parser.add_argument("--model",   type=str, default=None,
                        help=f"YOLO variant (default: {config.YOLO_MODEL_SIZE})")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume training from existing tb_yolo.pt weights")
    parser.add_argument("--validate",action="store_true",
                        help="Run validation only (no training)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.validate:
        # Just validate existing weights
        validate_model()
    else:
        # Run training
        saved_path = train_yolo(
            epochs     = args.epochs,
            batch      = args.batch,
            model_size = args.model,
            resume     = args.resume,
        )
        print(f"\n[DONE] Training complete. Model saved to: {saved_path}")

        # Auto-validate after training
        print("\n[INFO] Running post-training validation...")
        validate_model(saved_path)
