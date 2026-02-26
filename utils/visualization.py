"""
visualization.py
================
Module: Visualization Utilities for TB Detection System.

Provides helper functions for:
  - Drawing bounding boxes on chest X-rays
  - Plotting training history curves (loss, mAP)
  - Generating severity distribution charts
  - Displaying confusion matrices
  - Side-by-side comparison (original vs annotated)

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# COLOR MAP
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_COLORS_BGR = {
    "None"     : (0, 200, 0),     # Green
    "Mild"     : (0, 200, 255),   # Yellow
    "Moderate" : (0, 140, 255),   # Orange
    "Severe"   : (0, 0, 255),     # Red
}

SEVERITY_COLORS_RGB = {
    "None"     : "#00C800",
    "Mild"     : "#FFD000",
    "Moderate" : "#FF8C00",
    "Severe"   : "#FF0000",
}


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDING BOX DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(image: np.ndarray,
                    detections: list,
                    severity_level: str = "Mild",
                    show_labels: bool   = True) -> np.ndarray:
    """
    Draw YOLO detection boxes on an X-ray image array.

    Args:
        image         (np.ndarray): BGR image loaded with OpenCV.
        detections    (list):       List of detection dicts from TBDetector.detect()
        severity_level(str):        Overall severity label for color selection.
        show_labels   (bool):       Whether to draw confidence labels above boxes.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = image.copy()
    color     = SEVERITY_COLORS_BGR.get(severity_level, (0, 200, 0))

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["box_xyxy"]
        conf            = det.get("confidence", 0.0)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label  = f"TB#{i} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(results_csv_path: str, save_path: str = None):
    """
    Plot training loss and mAP50 curves from a YOLO results.csv file.

    Ultralytics YOLO saves training progress to runs/train/<run_name>/results.csv

    Args:
        results_csv_path (str): Path to results.csv from a YOLO training run.
        save_path        (str): Where to save the plot PNG. If None, displays interactively.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[WARNING] pandas not installed. Install with: pip install pandas")
        return

    if not os.path.exists(results_csv_path):
        print(f"[ERROR] results.csv not found: {results_csv_path}")
        return

    df = pd.read_csv(results_csv_path, skipinitialspace=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TB YOLO Training History", fontsize=14, fontweight="bold")

    # ── Loss curves ───────────────────────────────────────────────────────────
    ax = axes[0]
    if "train/box_loss" in df.columns:
        ax.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="#1f77b4")
    if "val/box_loss" in df.columns:
        ax.plot(df["epoch"], df["val/box_loss"],   label="Val Box Loss",   color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Detection Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── mAP50 curve ───────────────────────────────────────────────────────────
    ax = axes[1]
    if "metrics/mAP50(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", color="#2ca02c")
    if "metrics/mAP50-95(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95",
                color="#d62728", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Detection mAP")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Training curves saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: list, y_pred: list, save_path: str = None):
    """
    Plot a severity-level confusion matrix.

    Args:
        y_true    (list): Ground truth severity labels.
        y_pred    (list): Predicted severity labels.
        save_path (str):  Path to save the plot PNG image.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    labels = ["Mild", "Moderate", "Severe"]
    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Severity Classification Confusion Matrix", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY PIE CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_severity_distribution(severity_counts: dict, save_path: str = None):
    """
    Plot a pie chart of severity distribution across a batch of predictions.

    Args:
        severity_counts (dict): E.g. {"Mild": 30, "Moderate": 15, "Severe": 5}
        save_path       (str):  Path to save the chart.
    """
    labels  = list(severity_counts.keys())
    sizes   = list(severity_counts.values())
    colors  = [SEVERITY_COLORS_RGB.get(l, "#888888") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 12}
    )
    ax.set_title("TB Severity Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Severity distribution chart saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SIDE BY SIDE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def show_comparison(original_path: str, annotated_path: str, save_path: str = None):
    """
    Display the original X-ray alongside the annotated (detected) version.

    Args:
        original_path  (str): Path to the original X-ray image.
        annotated_path (str): Path to the annotated output image.
        save_path      (str): Path to save the comparison image.
    """
    orig  = cv2.imread(original_path)
    annot = cv2.imread(annotated_path)

    if orig is None or annot is None:
        print("[ERROR] Could not load images for comparison.")
        return

    # Resize both to same height for clean side-by-side
    target_h = 512
    orig  = cv2.resize(orig,  (int(orig.shape[1]  * target_h / orig.shape[0]),  target_h))
    annot = cv2.resize(annot, (int(annot.shape[1] * target_h / annot.shape[0]), target_h))

    # Add labels
    cv2.putText(orig,  "ORIGINAL",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.putText(annot, "DETECTED",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255),   2)

    comparison = np.hstack([orig, annot])

    if save_path:
        cv2.imwrite(save_path, comparison)
        print(f"[INFO] Comparison image saved to: {save_path}")
    else:
        cv2.imshow("TB Detection Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# RL LEARNING CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_rl_learning_curve(feedback_json_path: str, save_path: str = None):
    """
    Plot the RL reward history over time to show learning progress.

    Args:
        feedback_json_path (str): Path to data/rl_feedback.json.
        save_path          (str): Path to save the chart.
    """
    import json

    if not os.path.exists(feedback_json_path):
        print(f"[WARNING] No RL feedback log found: {feedback_json_path}")
        return

    with open(feedback_json_path) as f:
        records = json.load(f)

    if not records:
        print("[WARNING] RL feedback log is empty.")
        return

    rewards   = [r["reward"] for r in records]
    cum_reward = [sum(rewards[:i+1]) for i in range(len(rewards))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Reinforcement Learning Progress", fontweight="bold")

    # Per-event reward
    axes[0].bar(range(len(rewards)), rewards,
                color=["green" if r > 0 else "red" for r in rewards])
    axes[0].set_xlabel("Feedback Event")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Per-Event Reward")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].grid(True, alpha=0.3)

    # Cumulative reward
    axes[1].plot(cum_reward, color="#1f77b4", marker="o", markersize=4)
    axes[1].set_xlabel("Feedback Event")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].set_title("Cumulative RL Reward")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] RL learning curve saved to: {save_path}")
    else:
        plt.show()
    plt.close()
