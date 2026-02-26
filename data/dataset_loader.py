"""
dataset_loader.py
=================
Module 1: Dataset Handling for the AI TB Detection System.

This module is responsible for:
  - Downloading the Roboflow TB chest X-ray dataset
  - Verifying the YOLO-format folder structure
  - Generating the dataset.yaml configuration file required by Ultralytics YOLO
  - Providing a utility to inspect sample images with their annotations

Medical Context:
  The dataset contains chest X-ray images annotated with TB lesion bounding boxes.
  YOLO annotation format stores each box as:
      <class_id> <x_center> <y_center> <width> <height>   (all values normalized 0–1)

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys
import yaml
import random
import numpy as np
import cv2

# ── Add project root to path so we can import config ──────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# ROBOFLOW DATASET DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────

def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int):
    """
    Download the TB chest X-ray dataset from Roboflow in YOLOv8 format.

    Args:
        api_key   (str): Your Roboflow API key (get from app.roboflow.com → Settings)
        workspace (str): Your Roboflow workspace name (e.g. "myworkspace")
        project   (str): The project slug on Roboflow (e.g. "tuberculosis-detection")
        version   (int): Dataset version number to download

    Returns:
        str: Local path where the dataset was downloaded, or None on failure.

    Usage example:
        download_roboflow_dataset(
            api_key="your_key_here",
            workspace="your_workspace",
            project="tuberculosis-detection",
            version=1
        )
    """
    try:
        from roboflow import Roboflow  # Requires: pip install roboflow
    except ImportError:
        print("[ERROR] roboflow package not found. Run: pip install roboflow")
        return None

    print(f"[INFO] Connecting to Roboflow — project: {project} v{version}")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=config.DATA_DIR)
    print(f"[INFO] Dataset downloaded to: {config.DATA_DIR}")
    return config.DATA_DIR


# ─────────────────────────────────────────────────────────────────────────────
# DATASET YAML GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_yaml(train_path: str = None,
                           val_path: str   = None,
                           test_path: str  = None) -> str:
    """
    Generate the dataset.yaml file required by Ultralytics YOLO for training.

    The yaml file tells YOLO where to find image folders and what classes exist.

    Args:
        train_path (str): Path to training images directory  (default: config.TRAIN_DIR)
        val_path   (str): Path to validation images directory (default: config.VAL_DIR)
        test_path  (str): Path to test images directory       (default: config.TEST_DIR)

    Returns:
        str: Path to the generated dataset.yaml file.
    """
    train_path = train_path or config.TRAIN_DIR
    val_path   = val_path   or config.VAL_DIR
    test_path  = test_path  or config.TEST_DIR

    # Build the yaml structure Ultralytics expects
    yaml_content = {
        "path"  : config.DATA_DIR,   # Root dataset directory
        "train" : "train/images",    # Relative path from root
        "val"   : "val/images",
        "test"  : "test/images",
        "nc"    : len(config.CLASS_NAMES),   # Number of classes
        "names" : config.CLASS_NAMES,        # Class name list
    }

    # Write to disk
    yaml_path = config.DATASET_YAML
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"[INFO] dataset.yaml saved to: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# DATASET VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset_structure() -> bool:
    """
    Verify that the YOLO dataset folder structure is correct.

    Expected layout inside data/:
        train/images/   — Training X-ray images (.jpg/.png)
        train/labels/   — Corresponding YOLO annotation .txt files
        val/images/
        val/labels/
        test/images/
        test/labels/

    Returns:
        bool: True if structure is valid, False otherwise.
    """
    required_subdirs = [
        os.path.join(config.TRAIN_DIR, "images"),
        os.path.join(config.TRAIN_DIR, "labels"),
        os.path.join(config.VAL_DIR,   "images"),
        os.path.join(config.VAL_DIR,   "labels"),
    ]

    all_ok = True
    for d in required_subdirs:
        if os.path.isdir(d):
            count = len(os.listdir(d))
            print(f"  [OK]  {d}  ({count} files)")
        else:
            print(f"  [MISSING] {d}")
            all_ok = False

    if all_ok:
        print("[INFO] Dataset structure verified successfully.")
    else:
        print("[WARNING] Some directories are missing. "
              "Download the dataset first using download_roboflow_dataset().")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# DATASET STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_statistics() -> dict:
    """
    Count images and annotations across train/val/test splits.

    Returns:
        dict: Keys are split names, values are dicts with image/label counts.
    """
    stats = {}
    for split, split_dir in [("train", config.TRAIN_DIR),
                               ("val",   config.VAL_DIR),
                               ("test",  config.TEST_DIR)]:
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")

        img_count = len([f for f in os.listdir(img_dir)
                         if f.endswith((".jpg", ".jpeg", ".png"))]
                        ) if os.path.isdir(img_dir) else 0
        lbl_count = len([f for f in os.listdir(lbl_dir)
                         if f.endswith(".txt")]
                        ) if os.path.isdir(lbl_dir) else 0

        stats[split] = {"images": img_count, "labels": lbl_count}
        print(f"  [{split.upper():5s}] Images: {img_count:4d}  |  Labels: {lbl_count:4d}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE VISUALIZER (Augmentation Preview)
# ─────────────────────────────────────────────────────────────────────────────

def apply_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Apply basic data augmentation to a chest X-ray image.

    Augmentations used:
      - Horizontal flip  → Simulates mirror-image X-rays
      - Random rotation  → ±15° to handle tilted images
      - Contrast adjust  → Makes lesions more visible under different conditions

    Args:
        image (np.ndarray): Input BGR image loaded with OpenCV.

    Returns:
        np.ndarray: Augmented image.
    """
    # ── 1. Horizontal flip (50% probability) ──────────────────────────────────
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # ── 2. Random rotation ±15 degrees ────────────────────────────────────────
    angle = random.uniform(-15, 15)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           borderMode=cv2.BORDER_REFLECT_101)

    # ── 3. Contrast & brightness adjustment ───────────────────────────────────
    alpha = random.uniform(0.8, 1.2)   # Contrast multiplier (1.0 = unchanged)
    beta  = random.randint(-20, 20)    # Brightness offset
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return image


def visualize_sample(image_path: str, label_path: str = None, augment: bool = False):
    """
    Display a sample X-ray image with its YOLO bounding box annotations.

    Args:
        image_path (str): Path to the chest X-ray image file.
        label_path (str): Path to the corresponding YOLO .txt label file.
                          If None, no boxes are drawn.
        augment    (bool): Apply augmentation before displaying.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    h, w = image.shape[:2]

    # Optionally augment
    if augment:
        image = apply_augmentation(image)

    # Draw bounding boxes from YOLO label file
    if label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])

                # Convert normalized YOLO coords to pixel coords
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                # Draw box in green with class label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, config.CLASS_NAMES[cls_id],
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

    cv2.imshow("TB Dataset Sample", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TB Detection — Dataset Loader")
    print("=" * 60)

    print("\n[STEP 1] Verifying dataset structure...")
    verify_dataset_structure()

    print("\n[STEP 2] Generating dataset.yaml...")
    generate_dataset_yaml()

    print("\n[STEP 3] Dataset statistics:")
    get_dataset_statistics()

    print("\n[DONE] Dataset module ready.")
