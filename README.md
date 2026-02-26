# AI Tuberculosis Detection & Severity Assessment System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-Research%20Only-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

**A production-ready AI system for TB lesion detection, severity grading, and RL-based continuous learning from chest X-rays.**

</div>

---

> ⚠️ **MEDICAL DISCLAIMER**: This system is developed for **RESEARCH AND EDUCATIONAL PURPOSES ONLY**. It must **NOT** be used for clinical diagnosis, treatment decisions, or any direct patient care. Always consult a qualified medical professional for TB diagnosis and treatment.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Folder Structure](#-folder-structure)
4. [Dataset Setup (Roboflow)](#-dataset-setup-roboflow)
5. [Installation](#-installation)
6. [Training Instructions](#-training-instructions)
7. [Inference Instructions](#-inference-instructions)
8. [Reinforcement Learning Workflow](#-reinforcement-learning-workflow)
9. [Fine-Tuning Guide](#-fine-tuning-guide)
10. [Model Evaluation](#-model-evaluation)
11. [Example Outputs](#-example-outputs)
12. [Configuration](#-configuration)

---

## 🔬 Project Overview

This system detects **Tuberculosis (TB) lesions** in chest X-ray images using a **YOLOv8** object detection model, and then:

- Classifies disease **severity** (Mild / Moderate / Severe) based on lesion area and count
- Computes a **clinical risk score** (0–100)
- Generates structured **medical reports** in JSON format
- Saves **annotated X-ray images** with bounding box overlays
- Improves severity classification over time using **Reinforcement Learning** from expert feedback
- Supports **continuous fine-tuning** with new labeled data

### Key Features

| Feature | Description |
|---|---|
| 🎯 **YOLO Detection** | YOLOv8 trained on annotated TB chest X-rays |
| 📊 **Severity Engine** | Weighted area/count/confidence risk scoring |
| 🤖 **Reinforcement Learning** | Adaptive threshold tuning from expert corrections |
| 🔄 **Fine-Tuning** | Resume training on new data without forgetting |
| 📋 **Medical Reports** | Structured JSON with clinical recommendations |
| 🖼️ **Visual Output** | Annotated X-rays with severity color-coded boxes |

---

## 🏗️ System Architecture

```
Chest X-ray Input
       │
       ▼
┌─────────────────┐
│  YOLOv8 Detector │  ← Detects TB lesion bounding boxes
└─────────────────┘
       │
       ▼
┌────────────────────────┐
│  Severity Calculator    │  ← Area%, lesion count, confidence → Mild/Moderate/Severe
└────────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Report Generator    │  ← JSON report + annotated image
└─────────────────────┘
       │
       ▼
  Expert Feedback?
       │ YES
       ▼
┌─────────────────────┐
│  RL Update Engine   │  ← Adjusts thresholds via Q-learning reward
└─────────────────────┘
```

---

## 📁 Folder Structure

```
tuberculosis/
│
├── data/
│   ├── train/               ← Training images + YOLO labels
│   │   ├── images/
│   │   └── labels/
│   ├── val/                 ← Validation images + labels
│   ├── test/                ← Test images + labels
│   ├── dataset.yaml         ← Auto-generated YOLO dataset config
│   └── rl_feedback.json     ← Reinforcement learning feedback log
│
├── models/
│   ├── tb_yolo.pt           ← Base trained model weights
│   └── tb_yolo_updated.pt   ← Fine-tuned model weights
│
├── training/
│   └── train_yolo.py        ← YOLO training script
│
├── inference/
│   └── detect_tb.py         ← TB lesion detection engine
│
├── severity/
│   └── severity_calculator.py ← Severity + risk score computation
│
├── reinforcement/
│   └── rl_update.py         ← RL feedback and threshold update engine
│
├── finetuning/
│   └── finetune.py          ← Fine-tuning on new labeled data
│
├── reports/
│   └── report_generator.py  ← Medical report + visual output generator
│
├── evaluation/
│   └── evaluate.py          ← mAP, precision, recall, confusion matrix
│
├── utils/
│   └── visualization.py     ← Plotting utilities
│
├── new_data/                ← Drop new X-ray images here for fine-tuning
│   ├── images/
│   └── labels/
│
├── output/                  ← Annotated output images saved here
├── main.py                  ← 🚀 Main entry point
├── config.py                ← All configuration settings
├── requirements.txt         ← Python dependencies
├── .gitignore
└── README.md
```

---

## 📦 Dataset Setup (Roboflow)

### Option A — Download via Roboflow API (Recommended)

1. Create a free account at [roboflow.com](https://roboflow.com)
2. Find or upload a Tuberculosis chest X-ray dataset with YOLO annotations
3. Get your API key from **Settings → API Keys**
4. Run in Python:

```python
from data.dataset_loader import download_roboflow_dataset

download_roboflow_dataset(
    api_key   = "YOUR_ROBOFLOW_API_KEY",
    workspace = "your-workspace-name",
    project   = "tuberculosis-chest-xray",
    version   = 1
)
```

### Option B — Manual Setup

Place your dataset in YOLO format:
```
data/
  train/
    images/  ← .jpg or .png X-ray files
    labels/  ← .txt files (one per image, YOLO format)
  val/
    images/
    labels/
  test/
    images/
    labels/
```

Then generate the dataset config YAML:
```bash
python data/dataset_loader.py
```

### YOLO Annotation Format

Each `.txt` label file contains one line per lesion:
```
0 0.512 0.438 0.180 0.240
│  │     │     │     └── box height (normalized)
│  │     │     └──────── box width (normalized)
│  │     └────────────── y-center (normalized)
│  └──────────────────── x-center (normalized)
└─────────────────────── class id (0 = TB_lesion)
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tuberculosis-ai.git
cd tuberculosis-ai

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, CUDA GPU recommended (CPU works but slower)

---

## 🏋️ Training Instructions

### 1. Verify dataset structure
```bash
python data/dataset_loader.py
```

### 2. Train YOLO model (transfer learning from COCO)
```bash
python training/train_yolo.py
```

### 3. Custom training options
```bash
python training/train_yolo.py --epochs 150 --batch 16 --model yolov8s
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 100 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--model` | yolov8n | YOLO variant (n/s/m/l/x) |
| `--resume` | False | Resume from existing weights |
| `--validate` | False | Run validation only |

Trained weights are saved to: `models/tb_yolo.pt`

Training plots are saved to: `runs/train/<run_name>/`

---

## 🔍 Inference Instructions

### Single image
```bash
python main.py --image path/to/xray.jpg
```

### Batch processing (entire folder)
```bash
python main.py --batch data/test/images/
```

### With expert feedback (triggers RL update)
```bash
python main.py --image xray.jpg --feedback Severe
```

### Save outputs
JSON reports → `reports/<image_name>_report.json`
Annotated images → `output/<image_name>_annotated.jpg`

---

## 🤖 Reinforcement Learning Workflow

The RL module learns from expert severity corrections to improve the severity thresholds over time.

### How it works

```
Model predicts: "Moderate"
Expert says:    "Mild"
Reward:         +0.5  (close but not exact)

→ Thresholds shift: mild_max slightly increases
→ Next similar case more likely to be classified correctly
```

### Usage

```python
from reinforcement.rl_update import RLUpdateEngine
from severity.severity_calculator import SeverityCalculator

calculator = SeverityCalculator()
rl_engine  = RLUpdateEngine(calculator=calculator)

rl_engine.process_feedback(
    detection_result   = detection_result,   # from TBDetector.detect()
    predicted_severity = "Moderate",
    expert_severity    = "Mild"
)
```

### Or via CLI
```bash
python main.py --image xray.jpg --feedback Mild
```

RL feedback log saved to: `data/rl_feedback.json`

### Reward Table

| Prediction vs Expert | Reward |
|---|---|
| Exact match (Mild→Mild) | +1.0 |
| Adjacent level (Mild→Moderate) | +0.5 |
| Two levels apart (Mild→Severe) | -1.0 |

---

## 🔄 Fine-Tuning Guide

Add new annotated X-ray images to continue training without forgetting old patterns.

### 1. Place new data
```
new_data/
  images/  ← New .jpg/.png chest X-ray images
  labels/  ← Corresponding YOLO .txt annotation files
```

### 2. Run fine-tuning
```bash
python main.py --finetune --epochs 20
```

Or directly:
```bash
python finetuning/finetune.py --epochs 20 --batch 8
```

Updated weights saved to: `models/tb_yolo_updated.pt`

**Anti-forgetting techniques used:**
- Lower learning rate (0.001 vs 0.01)
- First 10 backbone layers frozen
- Only detection head layers updated

---

## 📈 Model Evaluation

```bash
python main.py --evaluate
```

Or directly:
```bash
python evaluation/evaluate.py --weights models/tb_yolo.pt
```

### Metrics Generated

**Detection (YOLO):**
- mAP@0.5 — Primary detection accuracy
- mAP@0.5:0.95 — Stricter metric
- Precision, Recall, F1

**Severity Classification:**
- Accuracy
- Per-class Precision / Recall / F1
- Confusion Matrix (saved to `reports/confusion_matrix.png`)

Full report saved to: `reports/evaluation.json`

---

## 📊 Example Outputs

### JSON Report
```json
{
  "image_name": "patient_xray_001.jpg",
  "timestamp": "2026-02-25T18:54:00",
  "tb_detected": true,
  "lesion_count": 5,
  "infected_area_percent": 22.5,
  "severity_level": "Moderate",
  "confidence_average": 0.84,
  "clinical_risk_score": 47,
  "recommendation": "Clinical consultation advised. Refer to pulmonologist for sputum culture and sensitivity testing. Initiate first-line TB therapy (HRZE regimen) under physician supervision.",
  "disclaimer": "⚠️  RESEARCH USE ONLY — This AI-generated report is NOT a clinical diagnosis."
}
```

### Severity Levels

| Level | Infected Area | Risk Score | Recommendation |
|---|---|---|---|
| ✅ None | 0% | 0 | No TB detected |
| 🟡 Mild | < 10% | 1–25 | Routine monitoring |
| 🟠 Moderate | 10–30% | 26–65 | Clinical consultation |
| 🔴 Severe | > 30% | 66–100 | Immediate attention |

---

## ⚙️ Configuration

All settings are in [`config.py`](config.py). Key parameters:

```python
# Model
YOLO_MODEL_SIZE      = "yolov8n"    # Model variant
CONFIDENCE_THRESHOLD = 0.25          # Detection threshold

# Severity thresholds (updated by RL)
SEVERITY_THRESHOLDS = {
    "mild_max":     10.0,    # < 10% → Mild
    "moderate_max": 30.0,    # 10–30% → Moderate
}

# RL learning rate
RL_LEARNING_RATE = 0.05
```

---

## 📚 Technologies Used

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — Object detection
- **[OpenCV](https://opencv.org/)** — Image processing
- **[PyTorch](https://pytorch.org/)** — Deep learning backend
- **[scikit-learn](https://scikit-learn.org/)** — Evaluation metrics
- **[Roboflow](https://roboflow.com/)** — Dataset management

---

## 📄 License

This project is released for **Research and Educational Use Only**.
Not licensed for commercial or clinical use.

---

*Built with ❤️ for advancing AI in medical imaging research.*
