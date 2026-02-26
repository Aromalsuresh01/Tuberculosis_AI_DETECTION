"""
severity_calculator.py
======================
Module 4: TB Severity Estimation Engine.

This module takes detection results (lesion count, infected area %, confidence)
and computes:
  - A severity level: Mild / Moderate / Severe
  - A clinical risk score (0–100)
  - A clinical recommendation

Severity Logic (evidence-informed thresholds, adjustable via RL):
  Mild     : Lesion area < 10% of lung area
  Moderate : 10% – 30%
  Severe   : > 30%

Risk Score Formula (weighted combination):
  risk_score = (area_weight  × area_pct / 100)
             + (count_weight × min(lesion_count / max_norm, 1.0))
             + (conf_weight  × avg_confidence)
  (scaled to 0–100)

Medical Disclaimer:
  These thresholds are simplified for educational purposes. Clinical TB severity
  assessment depends on additional factors including sputum culture, patient history,
  and radiologist interpretation. Do NOT use for real diagnosis.

Author: AI TB Detection Team
License: Research/Educational Use Only
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY CALCULATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SeverityCalculator:
    """
    Computes TB severity level and clinical risk score from detection results.

    The severity thresholds are loaded from config and can be dynamically
    updated by the Reinforcement Learning module (rl_update.py).

    Attributes:
        mild_max     (float): Maximum infected_area_percent for 'Mild'.
        moderate_max (float): Maximum infected_area_percent for 'Moderate'.
    """

    # Maps severity levels to ordered integers for comparison (used in RL)
    SEVERITY_ORDER = {"Mild": 0, "Moderate": 1, "Severe": 2}

    def __init__(self,
                 mild_max: float     = None,
                 moderate_max: float = None):
        """
        Initialize with severity thresholds.

        Args:
            mild_max     (float): Maximum area% for Mild classification.
            moderate_max (float): Maximum area% for Moderate classification.
        """
        thresholds = config.SEVERITY_THRESHOLDS
        self.mild_max     = mild_max     or thresholds["mild_max"]
        self.moderate_max = moderate_max or thresholds["moderate_max"]

    # ─────────────────────────────────────────────────────────────────────────
    def calculate(self,
                  infected_area_percent: float,
                  lesion_count: int,
                  confidence_avg: float) -> dict:
        """
        Calculate TB severity and risk score from detection metrics.

        Args:
            infected_area_percent (float): % of estimated lung area with lesions.
            lesion_count          (int):   Number of detected lesions.
            confidence_avg        (float): Mean YOLO confidence across detections.

        Returns:
            dict with keys:
              - severity_level   (str):   'Mild', 'Moderate', or 'Severe'
              - risk_score       (int):   0–100 clinical risk score
              - severity_details (dict):  Breakdown of contributing factors
        """
        # ── Step 1: Classify severity level ───────────────────────────────────
        severity_level = self._classify_severity(infected_area_percent)

        # ── Step 2: Compute weighted risk score ────────────────────────────────
        risk_score = self._compute_risk_score(
            infected_area_percent, lesion_count, confidence_avg
        )

        # ── Step 3: Build detailed breakdown ──────────────────────────────────
        severity_details = {
            "infected_area_percent" : round(infected_area_percent, 2),
            "lesion_count"          : lesion_count,
            "confidence_avg"        : round(confidence_avg, 4),
            "mild_threshold"        : self.mild_max,
            "moderate_threshold"    : self.moderate_max,
        }

        return {
            "severity_level"   : severity_level,
            "risk_score"       : risk_score,
            "severity_details" : severity_details,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _classify_severity(self, infected_area_percent: float) -> str:
        """
        Apply threshold-based severity classification.

        Args:
            infected_area_percent (float): % of estimated lung area with lesions.

        Returns:
            str: 'Mild', 'Moderate', or 'Severe'.
        """
        if infected_area_percent < self.mild_max:
            return "Mild"
        elif infected_area_percent < self.moderate_max:
            return "Moderate"
        else:
            return "Severe"

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_risk_score(self,
                             infected_area_pct: float,
                             lesion_count: int,
                             confidence_avg: float) -> int:
        """
        Compute a weighted clinical risk score from 0 to 100.

        Formula:
          raw_score = (w_area  × area / 100)
                    + (w_count × min(count / max_norm, 1.0))
                    + (w_conf  × confidence)
          risk_score = round(raw_score × 100)

        Args:
            infected_area_pct (float): % of lung area infected (0–100).
            lesion_count      (int):   Number of detected lesions.
            confidence_avg    (float): Average detection confidence (0–1).

        Returns:
            int: Risk score clamped to [0, 100].
        """
        # Normalize each factor to 0–1 range
        area_norm  = min(infected_area_pct / 100.0, 1.0)
        count_norm = min(lesion_count / config.MAX_LESION_COUNT_NORM, 1.0)
        conf_norm  = min(confidence_avg, 1.0)

        # Weighted sum
        raw_score = (
            config.RISK_WEIGHT_AREA       * area_norm  +
            config.RISK_WEIGHT_COUNT      * count_norm +
            config.RISK_WEIGHT_CONFIDENCE * conf_norm
        )

        # Scale to 0–100 and clamp
        risk_score = int(round(raw_score * 100))
        return max(0, min(100, risk_score))

    # ─────────────────────────────────────────────────────────────────────────
    def update_thresholds(self, new_mild_max: float, new_moderate_max: float):
        """
        Update severity thresholds (called by the RL update module).

        Args:
            new_mild_max     (float): New mild upper bound.
            new_moderate_max (float): New moderate upper bound.
        """
        self.mild_max     = max(1.0, min(new_mild_max,     50.0))   # Clamp to sane range
        self.moderate_max = max(self.mild_max + 1.0,
                                min(new_moderate_max, 90.0))
        config.SEVERITY_THRESHOLDS["mild_max"]     = self.mild_max
        config.SEVERITY_THRESHOLDS["moderate_max"] = self.moderate_max

        print(f"[RL] Severity thresholds updated → Mild<{self.mild_max}%  "
              f"Moderate<{self.moderate_max}%")

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_recommendation(severity_level: str) -> str:
        """
        Return a clinical recommendation string based on severity.

        Args:
            severity_level (str): 'Mild', 'Moderate', or 'Severe'.

        Returns:
            str: Clinical recommendation text.
        """
        recommendations = {
            "Mild"     : ("Routine monitoring recommended. "
                          "Schedule a follow-up chest X-ray in 4–6 weeks. "
                          "Standard TB treatment protocol may be initiated."),
            "Moderate" : ("Clinical consultation advised. "
                          "Refer to pulmonologist for sputum culture and sensitivity testing. "
                          "Initiate first-line TB therapy (HRZE regimen) under physician supervision."),
            "Severe"   : ("IMMEDIATE medical attention required. "
                          "High lesion burden detected. "
                          "Hospitalization may be necessary. "
                          "Urgent infectious disease consultation and isolation protocol advised."),
        }
        return recommendations.get(severity_level,
                                    "Consult a qualified medical professional for evaluation.")

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def severity_distance(predicted: str, actual: str) -> int:
        """
        Compute the distance between two severity levels (used for RL reward).

        Returns:
            int: 0 = same, 1 = adjacent level, 2 = opposite ends.
        """
        order = SeverityCalculator.SEVERITY_ORDER
        return abs(order.get(predicted, 0) - order.get(actual, 0))


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Severity Calculator — Demo")
    print("=" * 55)

    calculator = SeverityCalculator()

    # Test cases covering all three severity levels
    test_cases = [
        {"infected_area_percent": 5.0,  "lesion_count": 2,  "confidence_avg": 0.72, "expected": "Mild"},
        {"infected_area_percent": 18.0, "lesion_count": 6,  "confidence_avg": 0.81, "expected": "Moderate"},
        {"infected_area_percent": 42.0, "lesion_count": 14, "confidence_avg": 0.89, "expected": "Severe"},
    ]

    for tc in test_cases:
        result = calculator.calculate(
            infected_area_percent = tc["infected_area_percent"],
            lesion_count          = tc["lesion_count"],
            confidence_avg        = tc["confidence_avg"],
        )
        match = "✅" if result["severity_level"] == tc["expected"] else "❌"
        print(f"\n  Area: {tc['infected_area_percent']:5.1f}%  "
              f"Lesions: {tc['lesion_count']:2d}  "
              f"Conf: {tc['confidence_avg']:.2f}")
        print(f"  → Severity: {result['severity_level']}  "
              f"Risk: {result['risk_score']}/100  {match}")
        print(f"  → {calculator.get_recommendation(result['severity_level'])}")
