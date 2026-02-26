"""
rl_update.py
============
Module 6: Reinforcement Learning — Learning from Expert Feedback.

This module implements a Q-learning-inspired feedback loop that enables the
TB detection system to improve its SEVERITY CLASSIFICATION over time.

How it works:
  1. The AI model predicts a severity level for an X-ray
  2. A doctor or expert provides the correct severity label
  3. A reward is calculated based on how accurate the prediction was
  4. The reward signal is used to adjust the severity thresholds
  5. Experience is stored in data/rl_feedback.json for audit trails

Why Reinforcement Learning?
  Traditional supervised learning only trains on fixed datasets.
  RL-based threshold tuning allows the system to continuously adapt its
  decision boundaries based on real-world expert corrections — improving
  accuracy over time without retraining the full neural network.

RL Strategy:
  We use a lightweight Q-learning style update:
    new_threshold = old_threshold + lr × reward × correction_direction
  This shifts the Mild/Moderate/Severe boundaries based on feedback.

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from severity.severity_calculator import SeverityCalculator


# ─────────────────────────────────────────────────────────────────────────────
# RL UPDATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RLUpdateEngine:
    """
    Reinforcement Learning engine for adaptive severity threshold tuning.

    Uses a Q-learning-style update rule applied to the Mild/Moderate thresholds,
    shifting them based on reward signals derived from expert feedback.

    Attributes:
        calculator    (SeverityCalculator): Shared calculator instance whose thresholds are updated.
        feedback_log  (list):              In-memory list of (state, prediction, label, reward) tuples.
        learning_rate (float):             Controls how aggressively thresholds shift.
    """

    def __init__(self, calculator: SeverityCalculator = None):
        """
        Initialize the RL engine.

        Args:
            calculator (SeverityCalculator): Optional shared instance. Creates new if None.
        """
        self.calculator    = calculator or SeverityCalculator()
        self.learning_rate = config.RL_LEARNING_RATE
        self.feedback_log  = self._load_feedback_log()

        print(f"[RL] Engine initialized. "
              f"{len(self.feedback_log)} past feedback records loaded.")

    # ─────────────────────────────────────────────────────────────────────────
    def process_feedback(self,
                          detection_result: dict,
                          predicted_severity: str,
                          expert_severity: str) -> dict:
        """
        Process a single expert feedback event and update severity thresholds.

        Steps:
          1. Calculate reward based on correctness
          2. Determine threshold adjustment direction
          3. Apply Q-learning update to thresholds
          4. Store experience in feedback log
          5. Persist log to disk

        Args:
            detection_result   (dict): Raw detection result from TBDetector.
            predicted_severity (str):  What the model predicted ('Mild'/'Moderate'/'Severe')
            expert_severity    (str):  Expert's correct severity label.

        Returns:
            dict: Feedback record with keys: timestamp, state, predicted, actual, reward,
                  old_thresholds, new_thresholds.
        """
        # ── Step 1: Calculate reward ───────────────────────────────────────────
        reward = self._calculate_reward(predicted_severity, expert_severity)
        print(f"[RL] Prediction: {predicted_severity}  |  "
              f"Expert: {expert_severity}  |  Reward: {reward:+.1f}")

        # ── Step 2: Record old thresholds ──────────────────────────────────────
        old_mild_max     = self.calculator.mild_max
        old_moderate_max = self.calculator.moderate_max

        # ── Step 3: Adjust thresholds using Q-learning update ─────────────────
        new_mild_max, new_moderate_max = self._update_thresholds(
            predicted_severity, expert_severity, reward,
            old_mild_max, old_moderate_max,
            detection_result.get("infected_area_percent", 0)
        )

        # ── Step 4: Apply updated thresholds ──────────────────────────────────
        self.calculator.update_thresholds(new_mild_max, new_moderate_max)

        # ── Step 5: Build experience record ───────────────────────────────────
        record = {
            "timestamp"             : datetime.now().isoformat(),
            "image_name"            : detection_result.get("image_name", "unknown"),
            "infected_area_percent" : detection_result.get("infected_area_percent", 0),
            "lesion_count"          : detection_result.get("lesion_count", 0),
            "confidence_avg"        : detection_result.get("confidence_avg", 0),
            "predicted_severity"    : predicted_severity,
            "expert_severity"       : expert_severity,
            "reward"                : reward,
            "old_thresholds"        : {"mild_max": old_mild_max,
                                       "moderate_max": old_moderate_max},
            "new_thresholds"        : {"mild_max": self.calculator.mild_max,
                                       "moderate_max": self.calculator.moderate_max},
        }

        # ── Step 6: Persist to disk ────────────────────────────────────────────
        self.feedback_log.append(record)
        self._save_feedback_log()

        return record

    # ─────────────────────────────────────────────────────────────────────────
    def _calculate_reward(self, predicted: str, actual: str) -> float:
        """
        Calculate a scalar reward based on prediction accuracy.

        Reward scheme:
          - Exact match  → +1.0 (correct prediction)
          - 1 level off  → +0.5 (close, e.g. Mild vs Moderate)
          - 2 levels off → -1.0 (wrong prediction, e.g. Mild vs Severe)

        Args:
            predicted (str): Model's severity prediction.
            actual    (str): Expert's correct severity label.

        Returns:
            float: Reward value.
        """
        distance = SeverityCalculator.severity_distance(predicted, actual)

        if distance == 0:
            return config.RL_REWARD_CORRECT    # Perfect match
        elif distance == 1:
            return config.RL_REWARD_CLOSE      # Adjacent level
        else:
            return config.RL_REWARD_WRONG      # Significantly wrong

    # ─────────────────────────────────────────────────────────────────────────
    def _update_thresholds(self,
                            predicted: str, actual: str, reward: float,
                            mild_max: float, moderate_max: float,
                            area_pct: float) -> tuple:
        """
        Apply Q-learning style threshold update.

        Logic:
          - If model over-predicts severity (e.g. said Severe, actual Mild):
              RAISE the threshold (make it harder to trigger higher severity)
          - If model under-predicts (e.g. said Mild, actual Severe):
              LOWER the threshold (make it easier to trigger higher severity)
          - Reward is negative for wrong predictions, which drives the correction

        The update magnitude is:  lr × |reward| × area_pct influence

        Args:
            predicted    (str):   Model's prediction.
            actual       (str):   Expert label.
            reward       (float): Calculated reward (negative = wrong).
            mild_max     (float): Current mild threshold.
            moderate_max (float): Current moderate threshold.
            area_pct     (float): Infected area % (used to scale the adjustment).

        Returns:
            tuple: (new_mild_max, new_moderate_max)
        """
        # No update needed for correct predictions
        if reward == config.RL_REWARD_CORRECT:
            return mild_max, moderate_max

        order    = SeverityCalculator.SEVERITY_ORDER
        pred_ord = order.get(predicted, 0)
        true_ord = order.get(actual, 0)

        # Direction: +1 means thresholds too high (overpredicted), -1 means too low
        direction = +1 if pred_ord > true_ord else -1

        # Scale adjustment by area_pct influence (larger areas → bigger adjustment)
        area_scale  = min(area_pct / 50.0, 1.0)  # Normalize to 0–1
        adjustment  = self.learning_rate * abs(reward) * (1 + area_scale)

        # Shift mild threshold
        new_mild_max = mild_max - direction * adjustment

        # Shift moderate threshold proportionally
        new_moderate_max = moderate_max - direction * adjustment * 1.5

        print(f"[RL] Threshold adjustment: Δ={direction * adjustment:.3f}  "
              f"Mild: {mild_max:.2f}→{new_mild_max:.2f}  "
              f"Moderate: {moderate_max:.2f}→{new_moderate_max:.2f}")

        return new_mild_max, new_moderate_max

    # ─────────────────────────────────────────────────────────────────────────
    def get_feedback_summary(self) -> dict:
        """
        Summarize the feedback history for analysis.

        Returns:
            dict: Summary with total count, reward distribution, accuracy rate.
        """
        if not self.feedback_log:
            return {"total": 0, "accuracy": 0.0}

        total     = len(self.feedback_log)
        correct   = sum(1 for r in self.feedback_log
                        if r["predicted_severity"] == r["expert_severity"])
        avg_reward = sum(r["reward"] for r in self.feedback_log) / total

        return {
            "total_feedback"       : total,
            "correct_predictions"  : correct,
            "accuracy_rate"        : round(correct / total, 3),
            "average_reward"       : round(avg_reward, 3),
            "current_mild_max"     : self.calculator.mild_max,
            "current_moderate_max" : self.calculator.moderate_max,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _load_feedback_log(self) -> list:
        """Load existing feedback log from disk, or return empty list if not found."""
        if os.path.exists(config.RL_FEEDBACK_FILE):
            try:
                with open(config.RL_FEEDBACK_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[WARNING] Could not load feedback log: {e}")
        return []

    def _save_feedback_log(self):
        """Persist the full feedback log to disk."""
        os.makedirs(os.path.dirname(config.RL_FEEDBACK_FILE), exist_ok=True)
        with open(config.RL_FEEDBACK_FILE, "w") as f:
            json.dump(self.feedback_log, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Reinforcement Learning Update Demo")
    print("=" * 60)

    rl_engine = RLUpdateEngine()

    # Simulate feedback sessions
    sessions = [
        # (area_pct, predicted, expert_label)
        (8.0,  "Moderate", "Mild"),     # Model over-predicted
        (22.0, "Moderate", "Moderate"), # Correct
        (45.0, "Moderate", "Severe"),   # Model under-predicted
        (5.0,  "Mild",     "Mild"),     # Correct
    ]

    for area, pred, expert in sessions:
        mock_detection = {
            "image_name"            : f"xray_{area}.jpg",
            "infected_area_percent" : area,
            "lesion_count"          : int(area / 3),
            "confidence_avg"        : 0.80,
        }
        rl_engine.process_feedback(mock_detection, pred, expert)
        print()

    summary = rl_engine.get_feedback_summary()
    print("\n[RL SUMMARY]")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")
