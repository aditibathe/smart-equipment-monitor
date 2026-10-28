"""
Step 4 Verification Test (test_step4.py)
----------------------------------------
Tests Practicals 10, 11, and model inference:
- Linear Regression training and evaluation (Practical 10)
- Classification training and evaluation (Practical 11)
- Real-time prediction helper
- Target leakage verification
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import CLEAN_DATA_PATH, SENSOR_COLUMNS
from src.data_loader import load_dataset
from src.models import (
    train_linear_regression,
    train_classifier,
    predict_single_record,
)


class TestStep4MachineLearning(unittest.TestCase):
    """Test suite for Step 4: Practicals 10 & 11."""

    @classmethod
    def setUpClass(cls):
        """Load cleaned dataset once for ML tests."""
        cls.df = load_dataset(CLEAN_DATA_PATH)
        cls.reg_model, cls.reg_metrics = train_linear_regression(cls.df)
        cls.clf_model, cls.clf_metrics = train_classifier(cls.df)

    def test_no_target_leakage(self):
        """Verify training feature columns do NOT include target labels."""
        forbidden_targets = {"failure_risk_score", "failure_status", "equipment_status"}
        for feat in SENSOR_COLUMNS:
            self.assertNotIn(feat, forbidden_targets, f"Target leak detected in feature '{feat}'!")

    def test_practical10_linear_regression(self):
        """Verify Linear Regression performance and evaluation metrics."""
        self.assertGreater(self.reg_metrics["test_r2"], 0.80, "R2 score should exceed 0.80 on physical sensor data.")
        self.assertGreater(self.reg_metrics["test_mse"], 0.0, "MSE must be positive.")
        self.assertGreater(self.reg_metrics["test_rmse"], 0.0, "RMSE must be positive.")
        self.assertGreater(self.reg_metrics["test_mae"], 0.0, "MAE must be positive.")
        self.assertEqual(len(self.reg_metrics["coefficients"]), len(SENSOR_COLUMNS))

    def test_practical11_classification(self):
        """Verify Classification performance and confusion matrix."""
        self.assertGreater(self.clf_metrics["accuracy"], 0.85, "Accuracy should exceed 85%.")
        self.assertGreaterEqual(self.clf_metrics["f1_score"], 0.0)
        self.assertEqual(self.clf_metrics["confusion_matrix"].shape, (2, 2))
        self.assertIn("precision", self.clf_metrics)
        self.assertIn("recall", self.clf_metrics)

    def test_predict_single_record(self):
        """Verify real-time prediction pipeline for normal and fault conditions."""
        # Normal condition reading
        normal_pred = predict_single_record(
            self.reg_model,
            self.clf_model,
            operating_hours=500.0,
            temperature_c=55.0,
            vibration_mm_s=2.0,
            voltage_v=230.0,
            current_a=12.0,
        )
        self.assertIn(normal_pred["status_label"], ["NORMAL", "WARNING", "CRITICAL / FAULT"])
        self.assertGreaterEqual(normal_pred["predicted_risk_score"], 0.0)
        self.assertLessEqual(normal_pred["predicted_risk_score"], 100.0)
        self.assertAlmostEqual(normal_pred["power_kw"], 2.76, places=2)

        # Severe fault reading
        fault_pred = predict_single_record(
            self.reg_model,
            self.clf_model,
            operating_hours=4800.0,
            temperature_c=95.0,
            vibration_mm_s=11.5,
            voltage_v=240.0,
            current_a=21.0,
        )
        self.assertEqual(fault_pred["status_label"], "CRITICAL / FAULT")
        self.assertGreater(fault_pred["predicted_risk_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
