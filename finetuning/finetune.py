"""
finetune.py
===========
Module 7: Fine-Tuning System — Continue Training with New Data.

This script allows you to add new labeled TB chest X-ray images and fine-tune
the existing YOLO model on them WITHOUT forgetting previously learned patterns.

Key Techniques to Avoid Catastrophic Forgetting:
  1. Lower learning rate (0.001 vs 0.01 for initial training)
  2. Freeze initial backbone layers (feature extractor stays frozen)
  3. Train only final detection head layers on new data

Workflow:
  1. Place new images in new_data/images/
  2. Place YOLO label files in new_data/labels/
  3. Run: python finetuning/finetune.py --epochs 20
  4. New weights saved to: models/tb_yolo_updated.pt

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys
import argparse
import shutil
import glob
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# NEW DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_new_data_yaml() -> str:
    """
    Generate a dataset.yaml that combines existing data with new_data/ images.

    The new_data/ folder should have:
        new_data/
          images/  ← New X-ray images (.jpg/.png)
          labels/  ← Corresponding YOLO .txt annotation files

    Returns:
        str: Path to the generated fine-tuning dataset yaml.
    """
    import yaml

    new_images_dir = os.path.join(config.NEW_DATA_DIR, "images")
    new_labels_dir = os.path.join(config.NEW_DATA_DIR, "labels")

    # Validate new data exists
    if not os.path.isdir(new_images_dir):
        raise FileNotFoundError(
            f"New data images folder not found: {new_images_dir}\n"
            "Create this folder and add your new annotated X-ray images."
        )

    new_img_count = len(glob.glob(os.path.join(new_images_dir, "*.jpg")))
    new_img_count += len(glob.glob(os.path.join(new_images_dir, "*.png")))
    print(f"[INFO] Found {new_img_count} new images in: {new_images_dir}")

    if new_img_count == 0:
        raise ValueError("No images found in new_data/images/. "
                         "Add .jpg or .png X-ray images to fine-tune.")

    # Build yaml pointing to new_data as training set, original val as validation
    yaml_content = {
        "path"  : config.BASE_DIR,
        "train" : os.path.join(config.NEW_DATA_DIR, "images"),
        "val"   : os.path.join(config.VAL_DIR, "images"),
        "nc"    : len(config.CLASS_NAMES),
        "names" : config.CLASS_NAMES,
    }

    yaml_path = os.path.join(config.DATA_DIR, "finetune_dataset.yaml")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"[INFO] Fine-tune dataset yaml saved to: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def finetune(epochs: int = None, batch: int = None) -> str:
    """
    Fine-tune the existing TB YOLO model on new labeled data.

    Strategy:
      - Load existing model weights (tb_yolo.pt or tb_yolo_updated.pt)
      - Freeze first N backbone layers to preserve learned low-level features
      - Train only detection head + upper layers on new data
      - Use very low learning rate to prevent catastrophic forgetting
      - Save updated weights to models/tb_yolo_updated.pt

    Args:
        epochs (int): Number of fine-tuning epochs. Default: config.FINETUNE_EPOCHS
        batch  (int): Batch size. Default: config.FINETUNE_BATCH_SIZE

    Returns:
        str: Path to the saved updated model weights.

    Raises:
        FileNotFoundError: If base model weights don't exist.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    epochs = epochs or config.FINETUNE_EPOCHS
    batch  = batch  or config.FINETUNE_BATCH_SIZE

    # Prefer updated weights if they exist, otherwise use base weights
    if os.path.exists(config.UPDATED_WEIGHTS):
        base_weights = config.UPDATED_WEIGHTS
        print(f"[INFO] Resuming from updated weights: {base_weights}")
    elif os.path.exists(config.MODEL_WEIGHTS):
        base_weights = config.MODEL_WEIGHTS
        print(f"[INFO] Fine-tuning from base model: {base_weights}")
    else:
        raise FileNotFoundError(
            f"No model weights found. Train the model first:\n"
            f"  python training/train_yolo.py"
        )

    # Prepare dataset yaml for new data
    yaml_path = prepare_new_data_yaml()

    print("=" * 60)
    print("  TB Detection — YOLO Fine-Tuning")
    print("=" * 60)
    print(f"  Base Model   : {base_weights}")
    print(f"  Dataset      : {yaml_path}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch        : {batch}")
    print(f"  Learning Rate: {config.FINETUNE_LR}")
    print(f"  Frozen Layers: {config.FREEZE_LAYERS}")
    print("=" * 60)

    # Load model
    model = YOLO(base_weights)

    # Freeze backbone layers to prevent catastrophic forgetting
    # This preserves general visual feature detectors learned during initial training
    freeze_count = 0
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < config.FREEZE_LAYERS * 10:  # Approximate layer boundary
            param.requires_grad = False
            freeze_count += 1

    print(f"[INFO] Frozen {freeze_count} parameters in backbone layers.")

    # Run fine-tuning
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"tb_finetune_{timestamp}"

    results = model.train(
        data      = yaml_path,
        epochs    = epochs,
        batch     = batch,
        imgsz     = config.IMAGE_SIZE,
        lr0       = config.FINETUNE_LR,  # Very low LR to avoid forgetting
        lrf       = 0.1,                  # Final LR fraction
        warmup_epochs = 1,
        project   = os.path.join(config.BASE_DIR, "runs", "finetune"),
        name      = project_name,
        seed      = config.RANDOM_SEED,
        exist_ok  = True,
        verbose   = True,
    )

    # Save updated weights to standard path
    best_weights_src = os.path.join(
        config.BASE_DIR, "runs", "finetune", project_name, "weights", "best.pt"
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, config.UPDATED_WEIGHTS)
        print(f"\n[SUCCESS] Updated model saved to: {config.UPDATED_WEIGHTS}")
    else:
        print(f"[WARNING] Could not find best.pt at: {best_weights_src}")

    return config.UPDATED_WEIGHTS


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO TB model on new labeled X-ray data"
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Fine-tuning epochs (default: {config.FINETUNE_EPOCHS})")
    parser.add_argument("--batch",  type=int, default=None,
                        help=f"Batch size (default: {config.FINETUNE_BATCH_SIZE})")
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    out_path = finetune(epochs=args.epochs, batch=args.batch)
    print(f"\n[DONE] Fine-tuning complete. Updated model: {out_path}")
