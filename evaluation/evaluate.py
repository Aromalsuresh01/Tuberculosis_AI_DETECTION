"""
evaluate.py
===========
Module 8: Model Evaluation — Compute Detection and Severity Metrics.

This script evaluates the trained YOLO model on the test set and generates
a comprehensive evaluation report including:
  - Detection metrics: mAP50, mAP50-95, Precision, Recall
  - Severity classification metrics: Accuracy, Confusion Matrix
  - Visual plots saved to reports/ directory
  - Full evaluation report saved to reports/evaluation.json

Usage:
  python evaluation/evaluate.py --weights models/tb_yolo.pt

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate_detection(weights_path: str = None) -> dict:
    """
    Run YOLO validation on the test set to compute detection metrics.

    Metrics computed:
      - mAP50     : Mean Average Precision at IoU=0.50 (primary metric)
      - mAP50-95  : Mean Average Precision at IoU=0.50:0.95 (stricter)
      - Precision  : Fraction of detections that are true positives
      - Recall     : Fraction of actual lesions that are detected

    Args:
        weights_path (str): Path to model .pt file.

    Returns:
        dict: Detection metrics dictionary.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed.")

    weights_path = weights_path or config.MODEL_WEIGHTS

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not os.path.exists(config.DATASET_YAML):
        raise FileNotFoundError(f"dataset.yaml not found: {config.DATASET_YAML}")

    print("[INFO] Running YOLO detection evaluation on test set...")
    model   = YOLO(weights_path)

    # ── Use 'test' split for unbiased evaluation ───────────────────────────────
    metrics = model.val(
        data    = config.DATASET_YAML,
        split   = "test",           # Use held-out test set
        imgsz   = config.IMAGE_SIZE,
        conf    = config.CONFIDENCE_THRESHOLD,
        iou     = config.IOU_THRESHOLD,
        verbose = False,
    )

    detection_metrics = {
        "mAP50"     : round(float(metrics.box.map50), 4),
        "mAP50-95"  : round(float(metrics.box.map),   4),
        "precision" : round(float(metrics.box.p.mean()), 4),
        "recall"    : round(float(metrics.box.r.mean()), 4),
        "f1_score"  : 0.0,  # Computed below
    }

    # Compute F1 = 2×P×R / (P+R)
    p = detection_metrics["precision"]
    r = detection_metrics["recall"]
    detection_metrics["f1_score"] = round(
        2 * p * r / (p + r) if (p + r) > 0 else 0.0, 4
    )

    print("\n[DETECTION METRICS]")
    for k, v in detection_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    return detection_metrics


def evaluate_severity(weights_path: str = None,
                       test_image_dir: str = None) -> dict:
    """
    Evaluate severity classification accuracy on labeled test images.

    For each image in the test set (that has a corresponding severity ground-truth):
      1. Run YOLO detection
      2. Compute severity from detection results
      3. Compare against ground truth
      4. Compute accuracy and confusion matrix

    Note: This function requires a severity ground-truth file:
          data/test/severity_labels.json
          Format: {"image_name.jpg": "Mild", "image_name2.jpg": "Severe", ...}

    Args:
        weights_path   (str): Path to YOLO .pt file.
        test_image_dir (str): Path to test images directory.

    Returns:
        dict: Severity metrics with accuracy and per-class results.
    """
    from inference.detect_tb import TBDetector
    from severity.severity_calculator import SeverityCalculator
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    test_image_dir = test_image_dir or os.path.join(config.TEST_DIR, "images")
    labels_file    = os.path.join(config.TEST_DIR, "severity_labels.json")

    if not os.path.exists(labels_file):
        print(f"[WARNING] No severity ground-truth found: {labels_file}")
        print("          Skipping severity evaluation.")
        return {}

    with open(labels_file) as f:
        ground_truth = json.load(f)

    detector   = TBDetector(weights_path=weights_path)
    calculator = SeverityCalculator()

    y_true, y_pred = [], []

    for img_name, true_severity in ground_truth.items():
        img_path = os.path.join(test_image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            result = detector.detect(img_path)
            severity_result = calculator.calculate(
                infected_area_percent = result["infected_area_percent"],
                lesion_count          = result["lesion_count"],
                confidence_avg        = result["confidence_avg"],
            )
            y_true.append(true_severity)
            y_pred.append(severity_result["severity_level"])
        except Exception as e:
            print(f"  [ERROR] {img_name}: {e}")

    if not y_true:
        return {}

    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   labels=["Mild", "Moderate", "Severe"],
                                   output_dict=True, zero_division=0)

    severity_metrics = {
        "accuracy"               : round(acc, 4),
        "samples_evaluated"      : len(y_true),
        "per_class_report"       : report,
        "confusion_matrix_labels": ["Mild", "Moderate", "Severe"],
        "confusion_matrix"       : confusion_matrix(
                                       y_true, y_pred,
                                       labels=["Mild", "Moderate", "Severe"]
                                   ).tolist(),
    }

    print("\n[SEVERITY METRICS]")
    print(f"  Accuracy : {acc:.4f}  ({len(y_true)} samples)")
    print(f"  Mild     P={report.get('Mild', {}).get('precision',0):.3f}  "
          f"R={report.get('Mild', {}).get('recall',0):.3f}")
    print(f"  Moderate P={report.get('Moderate', {}).get('precision',0):.3f}  "
          f"R={report.get('Moderate', {}).get('recall',0):.3f}")
    print(f"  Severe   P={report.get('Severe', {}).get('precision',0):.3f}  "
          f"R={report.get('Severe', {}).get('recall',0):.3f}")

    return severity_metrics


def run_full_evaluation(weights_path: str = None) -> str:
    """
    Run both detection and severity evaluation, save full report.

    Args:
        weights_path (str): Path to YOLO .pt model file.

    Returns:
        str: Path to saved evaluation JSON report.
    """
    print("=" * 60)
    print("  AI TB Detection — Full Model Evaluation")
    print("=" * 60)

    weights_path = weights_path or config.MODEL_WEIGHTS

    # 1. Detection evaluation
    detection_metrics = evaluate_detection(weights_path)

    # 2. Severity evaluation (may skip if labels file missing)
    severity_metrics  = evaluate_severity(weights_path)

    # 3. Build full report
    eval_report = {
        "evaluation_timestamp" : datetime.now().isoformat(),
        "model_path"           : weights_path,
        "dataset_yaml"         : config.DATASET_YAML,
        "detection_metrics"    : detection_metrics,
        "severity_metrics"     : severity_metrics,
    }

    # 4. Save report to disk
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    report_path = config.EVAL_REPORT_FILE
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\n[INFO] Evaluation report saved to: {report_path}")

    # 5. Generate confusion matrix visualization (if severity evaluated)
    if severity_metrics.get("confusion_matrix"):
        from utils.visualization import plot_confusion_matrix
        y_true_expanded, y_pred_expanded = [], []
        cm     = severity_metrics["confusion_matrix"]
        labels = severity_metrics["confusion_matrix_labels"]
        for i, row in enumerate(cm):
            for j, count in enumerate(row):
                y_true_expanded.extend([labels[i]] * count)
                y_pred_expanded.extend([labels[j]] * count)
        if y_true_expanded:
            cm_save_path = os.path.join(config.REPORTS_DIR, "confusion_matrix.png")
            plot_confusion_matrix(y_true_expanded, y_pred_expanded, save_path=cm_save_path)

    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TB YOLO model on test set"
    )
    parser.add_argument("--weights", default=None,
                        help="Path to model weights (default: models/tb_yolo.pt)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_evaluation(weights_path=args.weights)
