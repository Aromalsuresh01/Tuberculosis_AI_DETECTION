"""
detect_tb.py
============
Module 3: TB Inference Engine — Run YOLO Detection on Chest X-rays.

This module loads a trained YOLO model and runs inference on a given chest X-ray image.
It extracts:
  - Bounding boxes for detected TB lesions
  - Confidence scores per lesion
  - Total number of lesions
  - Total infected area as a percentage of the estimated lung area

Medical Context:
  Tuberculosis lesions appear as opacifications (white cloudy areas) or cavities in
  chest X-rays. This detector targets those regions and quantifies their spatial extent,
  which is a key factor in severity assessment.

Usage:
  python inference/detect_tb.py --image path/to/xray.jpg

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys
import argparse
import json

import cv2
import numpy as np

# ── Project root on path ───────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TBDetector:
    """
    Encapsulates the YOLO model for TB lesion detection.

    Attributes:
        model     : Loaded YOLO model instance.
        conf_thres: Minimum confidence score to keep a detection.
        iou_thres : IoU threshold for NMS (non-maximum suppression).
    """

    def __init__(self,
                 weights_path: str = None,
                 conf_thres: float = None,
                 iou_thres: float  = None):
        """
        Initialize the TBDetector with a trained YOLO model.

        Args:
            weights_path (str):   Path to .pt model weights file.
            conf_thres   (float): Confidence threshold (0.0–1.0).
            iou_thres    (float): IoU threshold for NMS (0.0–1.0).

        Raises:
            FileNotFoundError: If the weights file does not exist.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.weights_path = weights_path or config.MODEL_WEIGHTS
        self.conf_thres   = conf_thres   or config.CONFIDENCE_THRESHOLD
        self.iou_thres    = iou_thres    or config.IOU_THRESHOLD

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}\n"
                "Train the model first using: python training/train_yolo.py"
            )

        print(f"[INFO] Loading YOLO model from: {self.weights_path}")
        self.model = YOLO(self.weights_path)
        print("[INFO] Model loaded successfully.")

    # ─────────────────────────────────────────────────────────────────────────
    def detect(self, image_path: str) -> dict:
        """
        Run YOLO inference on a single chest X-ray image.

        Process:
          1. Load and validate image
          2. Run YOLO forward pass
          3. Parse bounding boxes and confidence scores
          4. Calculate infected area as % of estimated lung area
          5. Return structured detection results

        Args:
            image_path (str): Absolute or relative path to the X-ray image.

        Returns:
            dict: Detection results with keys:
              - image_path    (str)
              - image_size    (tuple) — (width, height)
              - lesion_count  (int)
              - detections    (list of dicts with box, confidence, area_px)
              - total_area_px (float) — total lesion pixels
              - infected_area_percent (float) — % of estimated lung area
              - confidence_avg (float) — mean detection confidence

        Raises:
            FileNotFoundError: If image does not exist.
            ValueError: If image cannot be loaded.
        """
        # ── Validate input ─────────────────────────────────────────────────────
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image (corrupt or unsupported format): {image_path}")

        h, w = image.shape[:2]

        # ── Run YOLO inference ─────────────────────────────────────────────────
        results = self.model.predict(
            source = image_path,
            conf   = self.conf_thres,
            iou    = self.iou_thres,
            imgsz  = config.IMAGE_SIZE,
            verbose= False,
        )

        # ── Parse detections ───────────────────────────────────────────────────
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Pixel coordinates of the bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf            = float(box.conf[0])
                cls_id          = int(box.cls[0])

                # Lesion area in pixels
                box_area_px = (x2 - x1) * (y2 - y1)

                detections.append({
                    "class_name"  : config.CLASS_NAMES[cls_id] if cls_id < len(config.CLASS_NAMES) else "unknown",
                    "confidence"  : round(conf, 4),
                    "box_xyxy"    : [round(x1), round(y1), round(x2), round(y2)],
                    "area_pixels" : round(box_area_px, 2),
                })

        # ── Compute aggregate statistics ───────────────────────────────────────
        lesion_count   = len(detections)
        total_area_px  = sum(d["area_pixels"] for d in detections)
        confidence_avg = (sum(d["confidence"] for d in detections) / lesion_count
                          if lesion_count > 0 else 0.0)

        # Estimate lung area in pixels using config fraction
        # (In a production system, use a segmentation model for accurate lung masks)
        image_area_px  = w * h
        lung_area_px   = image_area_px * config.LUNG_AREA_FRACTION
        infected_pct   = min((total_area_px / lung_area_px) * 100, 100.0) if lung_area_px > 0 else 0.0

        return {
            "image_path"            : image_path,
            "image_name"            : os.path.basename(image_path),
            "image_size"            : (w, h),
            "lesion_count"          : lesion_count,
            "detections"            : detections,
            "total_area_pixels"     : round(total_area_px, 2),
            "infected_area_percent" : round(infected_pct, 2),
            "confidence_avg"        : round(confidence_avg, 4),
            "tb_detected"           : lesion_count > 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def detect_batch(self, image_dir: str) -> list:
        """
        Run detection on all images in a directory.

        Args:
            image_dir (str): Path to folder containing X-ray images.

        Returns:
            list: List of detection result dicts (one per image).
        """
        valid_exts  = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(valid_exts)
        ]

        if not image_files:
            print(f"[WARNING] No valid images found in: {image_dir}")
            return []

        print(f"[INFO] Running batch inference on {len(image_files)} images...")

        all_results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"  [{i:3d}/{len(image_files)}] {os.path.basename(img_path)}")
            try:
                result = self.detect(img_path)
                all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] Failed to process {img_path}: {e}")
                all_results.append({"image_path": img_path, "error": str(e)})

        return all_results


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_detection_summary(detection_result: dict):
    """
    Print a human-readable summary of detection results to the console.

    Args:
        detection_result (dict): Output from TBDetector.detect()
    """
    print("\n" + "=" * 55)
    print("  TB DETECTION RESULTS")
    print("=" * 55)
    print(f"  Image      : {detection_result.get('image_name', 'N/A')}")
    print(f"  TB Detected: {'YES ⚠️' if detection_result.get('tb_detected') else 'NO ✅'}")
    print(f"  Lesions    : {detection_result.get('lesion_count', 0)}")
    print(f"  Infected % : {detection_result.get('infected_area_percent', 0):.1f}%")
    print(f"  Avg Conf   : {detection_result.get('confidence_avg', 0):.3f}")
    print("-" * 55)

    for i, det in enumerate(detection_result.get("detections", []), 1):
        print(f"  Lesion {i:2d}: conf={det['confidence']:.3f}  "
              f"box={det['box_xyxy']}  area={det['area_pixels']:.0f}px²")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TB lesion detection on a chest X-ray image"
    )
    parser.add_argument("--image",   required=True, help="Path to input chest X-ray image")
    parser.add_argument("--weights", default=None,  help="Path to YOLO .pt weights file")
    parser.add_argument("--conf",    type=float, default=None, help="Confidence threshold")
    parser.add_argument("--output",  default=None,
                        help="Optional output JSON path to save detection results")
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    detector = TBDetector(weights_path=args.weights, conf_thres=args.conf)
    result   = detector.detect(args.image)

    print_detection_summary(result)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[INFO] Results saved to: {args.output}")
