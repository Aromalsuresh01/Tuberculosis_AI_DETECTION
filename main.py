"""
main.py
=======
AI Tuberculosis Detection and Severity Assessment System — Main Entry Point.

This is the primary execution script that orchestrates the full TB detection
pipeline end-to-end:

  1. Load the trained YOLO model
  2. Run inference on a chest X-ray image
  3. Compute severity level and risk score
  4. Generate a structured medical report (JSON)
  5. Save an annotated visual output image
  6. (Optional) Accept expert feedback and update RL thresholds
  7. (Optional) Run fine-tuning on new labeled data

MEDICAL DISCLAIMER:
  This system is for RESEARCH AND EDUCATIONAL USE ONLY.
  It must NOT be used for clinical diagnosis or treatment decisions.
  Always consult a qualified medical professional.

Usage Examples:
  # Basic inference on a single image:
  python main.py --image data/test/images/xray001.jpg

  # With expert feedback for RL update:
  python main.py --image data/test/images/xray001.jpg --feedback Severe

  # Batch inference on a folder:
  python main.py --batch data/test/images/

  # Run full evaluation on test set:
  python main.py --evaluate

  # Fine-tune on new data:
  python main.py --finetune --epochs 20

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys
import json
import argparse

# ── Ensure project root is on Python path ─────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

# ── Import all modules ─────────────────────────────────────────────────────────
from inference.detect_tb         import TBDetector, print_detection_summary
from severity.severity_calculator import SeverityCalculator
from reports.report_generator    import ReportGenerator
from reinforcement.rl_update     import RLUpdateEngine


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE: Single Image
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(image_path: str,
                 expert_feedback: str = None,
                 save_report: bool    = True,
                 save_visual: bool    = True,
                 weights_path: str    = None) -> dict:
    """
    Execute the full TB detection pipeline on a single chest X-ray image.

    Pipeline Steps:
      1. YOLO detection → lesion boxes + area %
      2. Severity classification → Mild / Moderate / Severe
      3. Medical report generation → JSON
      4. Visual annotated output → saved image
      5. (Optional) RL feedback update if expert_feedback is provided

    Args:
        image_path     (str):  Path to the input chest X-ray image.
        expert_feedback(str):  Expert's correct severity ('Mild'/'Moderate'/'Severe').
                               If provided, triggers RL threshold update.
        save_report    (bool): Save JSON report to reports/ directory.
        save_visual    (bool): Save annotated image to output/ directory.
        weights_path   (str):  Path to YOLO model weights.

    Returns:
        dict: Full pipeline result containing detection, severity, and report.
    """
    print("\n" + "═" * 65)
    print("   🫁  AI TB DETECTION & SEVERITY ASSESSMENT SYSTEM")
    print("═" * 65)
    print(f"   Image: {os.path.basename(image_path)}")
    print("─" * 65)

    # ── STEP 1: Detection ──────────────────────────────────────────────────────
    print("\n[STEP 1] Running YOLO TB Detection...")
    detector = TBDetector(weights_path=weights_path)
    detection = detector.detect(image_path)
    print_detection_summary(detection)

    # ── STEP 2: Severity Estimation ────────────────────────────────────────────
    print("\n[STEP 2] Estimating Severity...")
    calculator = SeverityCalculator()

    if detection["tb_detected"]:
        severity_result = calculator.calculate(
            infected_area_percent = detection["infected_area_percent"],
            lesion_count          = detection["lesion_count"],
            confidence_avg        = detection["confidence_avg"],
        )
        severity_level = severity_result["severity_level"]
        risk_score     = severity_result["risk_score"]
        print(f"  ✔ Severity Level : {severity_level}")
        print(f"  ✔ Risk Score     : {risk_score}/100")
    else:
        severity_level = "None"
        risk_score     = 0
        print("  ✅ No TB lesions detected.")

    # ── STEP 3: Medical Report ─────────────────────────────────────────────────
    print("\n[STEP 3] Generating Medical Report...")
    generator = ReportGenerator()
    report    = generator.generate_report(detection)
    generator.print_report(report)

    if save_report:
        generator.save_report_json(report)

    # ── STEP 4: Visual Output ──────────────────────────────────────────────────
    if save_visual:
        print("\n[STEP 4] Saving Annotated Image...")
        generator.generate_visual_output(image_path, detection, report)

    # ── STEP 5: RL Feedback Update (optional) ─────────────────────────────────
    rl_record = None
    if expert_feedback:
        print(f"\n[STEP 5] Processing Expert Feedback: '{expert_feedback}'")
        rl_engine = RLUpdateEngine(calculator=calculator)
        rl_record = rl_engine.process_feedback(
            detection_result   = detection,
            predicted_severity = severity_level,
            expert_severity    = expert_feedback,
        )
        print(f"  ✔ RL update applied. New thresholds saved to: {config.RL_FEEDBACK_FILE}")

    # ── Compile full result ────────────────────────────────────────────────────
    pipeline_result = {
        "detection"     : detection,
        "severity_level": severity_level,
        "risk_score"    : risk_score,
        "report"        : report,
        "rl_update"     : rl_record,
    }

    print("\n" + "═" * 65)
    print("   ✅ Pipeline Complete")
    print("═" * 65)

    return pipeline_result


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE: Batch Processing
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_pipeline(image_dir: str, weights_path: str = None) -> list:
    """
    Run the full TB detection pipeline on all images in a directory.

    Args:
        image_dir    (str): Path to folder containing X-ray images.
        weights_path (str): Path to YOLO model weights.

    Returns:
        list: Pipeline results for each image.
    """
    valid_exts  = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(valid_exts)
    ]

    if not image_files:
        print(f"[WARNING] No valid images found in: {image_dir}")
        return []

    print(f"\n[INFO] Batch mode: processing {len(image_files)} images...\n")
    all_results   = []
    tb_count      = 0
    severity_dist = {"None": 0, "Mild": 0, "Moderate": 0, "Severe": 0}

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {os.path.basename(img_path)}")
        try:
            result = run_pipeline(img_path, weights_path=weights_path)
            all_results.append(result)

            if result["report"].get("tb_detected"):
                tb_count += 1
            sev = result["severity_level"]
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
        except Exception as e:
            print(f"  [ERROR] {e}")

    # ── Batch summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  BATCH SUMMARY")
    print("═" * 55)
    print(f"  Total Images  : {len(image_files)}")
    print(f"  TB Detected   : {tb_count} ({tb_count/len(image_files)*100:.1f}%)")
    print(f"  Severity Dist :")
    for sev, count in severity_dist.items():
        print(f"    {sev:10s}: {count}")
    print("═" * 55)

    # Optionally plot severity distribution
    try:
        from utils.visualization import plot_severity_distribution
        chart_path = os.path.join(config.REPORTS_DIR, "severity_distribution.png")
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        plot_severity_distribution({k: v for k, v in severity_dist.items() if v > 0},
                                   save_path=chart_path)
    except Exception:
        pass  # Non-critical if visualization fails

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Define all CLI arguments for the main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "AI Tuberculosis Detection & Severity Assessment System\n"
            "─────────────────────────────────────────────────────\n"
            "For RESEARCH USE ONLY. Not for clinical diagnosis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--image",    type=str,
                            help="Path to a single X-ray image for inference")
    mode_group.add_argument("--batch",    type=str,
                            help="Path to a folder of X-ray images for batch inference")
    mode_group.add_argument("--evaluate", action="store_true",
                            help="Run full model evaluation on test set")
    mode_group.add_argument("--finetune", action="store_true",
                            help="Fine-tune model on new_data/ images")
    mode_group.add_argument("--train",    action="store_true",
                            help="Train YOLO model from scratch on dataset")

    # Optional arguments
    parser.add_argument("--weights",  type=str, default=None,
                        help="Override model weights path")
    parser.add_argument("--feedback", type=str, default=None,
                        choices=["Mild", "Moderate", "Severe"],
                        help="Expert feedback label to trigger RL update")
    parser.add_argument("--epochs",   type=int, default=None,
                        help="Epochs for training or fine-tuning")
    parser.add_argument("--no-save",  action="store_true",
                        help="Do not save report/visual output to disk")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main dispatcher — routes CLI args to the appropriate pipeline.
    """
    args = parse_args()

    # ── Print system banner ────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("   🩺 AI TUBERCULOSIS DETECTION SYSTEM")
    print("   ⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS")
    print("═" * 65)

    # ── Route to correct mode ──────────────────────────────────────────────────

    if args.image:
        # Single image inference
        run_pipeline(
            image_path      = args.image,
            expert_feedback = args.feedback,
            save_report     = not args.no_save,
            save_visual     = not args.no_save,
            weights_path    = args.weights,
        )

    elif args.batch:
        # Batch inference on folder
        run_batch_pipeline(image_dir=args.batch, weights_path=args.weights)

    elif args.evaluate:
        # Model evaluation
        from evaluation.evaluate import run_full_evaluation
        run_full_evaluation(weights_path=args.weights)

    elif args.finetune:
        # Fine-tune model on new data
        from finetuning.finetune import finetune
        finetune(epochs=args.epochs)

    elif args.train:
        # Train from scratch
        from training.train_yolo import train_yolo
        train_yolo(epochs=args.epochs)

    else:
        print("\n[INFO] No mode specified. Use --help to see available options.")
        print("\nQuick Start:")
        print("  python main.py --image path/to/xray.jpg")
        print("  python main.py --batch data/test/images/")
        print("  python main.py --evaluate")
        print("  python main.py --train")
        print("  python main.py --finetune --epochs 20")


if __name__ == "__main__":
    main()
