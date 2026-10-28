"""
Step 1 Verification Test (test_step1.py)
----------------------------------------
Verifies that:
1. config.py loads valid constants and directories exist.
2. generator.py generates the required synthetic sensor dataset.
3. Dataset contains the expected columns, physical values, missing values, and duplicates.
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

from src.config import (
    RAW_DATA_PATH,
    DATA_DIR,
    EQUIPMENT_IDS,
    ALL_COLUMNS,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    STATUS_LABEL_COL,
)
from src.generator import generate_sensor_dataset, save_dataset


class TestStep1DataGeneration(unittest.TestCase):
    """Test suite for Step 1: Configuration & Synthetic Data Generation."""

    @classmethod
    def setUpClass(cls):
        """Generate dataset once for tests."""
        cls.df = generate_sensor_dataset(num_samples=500, random_seed=42, missing_ratio=0.04, num_duplicates=15)
        save_dataset(cls.df, filepath=RAW_DATA_PATH)

    def test_directories_exist(self):
        """Verify data directory exists."""
        self.assertTrue(DATA_DIR.exists(), "Data directory should exist.")

    def test_raw_data_file_created(self):
        """Verify raw CSV file exists on disk and is non-empty."""
        self.assertTrue(RAW_DATA_PATH.exists(), "Raw sensor CSV file should exist.")
        self.assertGreater(RAW_DATA_PATH.stat().st_size, 0, "Raw CSV should not be empty.")

    def test_expected_columns_present(self):
        """Verify all expected columns match the schema."""
        for col in ALL_COLUMNS:
            self.assertIn(col, self.df.columns, f"Column '{col}' must be present in DataFrame.")

    def test_equipment_ids_valid(self):
        """Verify equipment IDs belong to the configured equipment set."""
        unique_ids = set(self.df["equipment_id"].unique())
        self.assertTrue(unique_ids.issubset(set(EQUIPMENT_IDS)), "Invalid equipment IDs generated.")

    def test_missing_values_injected(self):
        """Verify missing values (NaN) were injected for Practical 4 demonstration."""
        null_counts = self.df.isnull().sum()
        self.assertGreater(null_counts["temperature_c"], 0, "Missing values expected in temperature_c.")
        self.assertGreater(null_counts["vibration_mm_s"], 0, "Missing values expected in vibration_mm_s.")
        self.assertGreater(null_counts["voltage_v"], 0, "Missing values expected in voltage_v.")

    def test_duplicates_injected(self):
        """Verify duplicate records were injected for Practical 4 demonstration."""
        num_duplicates = self.df.duplicated().sum()
        self.assertEqual(num_duplicates, 15, "Expected exactly 15 duplicate rows to be injected.")

    def test_power_calculation_vectorization(self):
        """Verify vectorized power formula P = (V * I) / 1000 for Practical 1."""
        clean_subset = self.df.dropna(subset=["voltage_v", "current_a", "power_kw"]).head(20)
        expected_power = np.round((clean_subset["voltage_v"] * clean_subset["current_a"]) / 1000.0, 3)
        np.testing.assert_allclose(clean_subset["power_kw"].values, expected_power.values, rtol=1e-3)

    def test_targets_validity(self):
        """Verify regression and classification targets are bounded and properly formatted."""
        # Risk score bounded between 0 and 100
        self.assertTrue((self.df[REGRESSION_TARGET] >= 0.0).all())
        self.assertTrue((self.df[REGRESSION_TARGET] <= 100.0).all())

        # Classification binary targets: only 0 and 1
        unique_classes = set(self.df[CLASSIFICATION_TARGET].unique())
        self.assertTrue(unique_classes.issubset({0, 1}))

        # Status label categories
        unique_statuses = set(self.df[STATUS_LABEL_COL].unique())
        self.assertTrue(unique_statuses.issubset({"Normal", "Warning", "Fault"}))


if __name__ == "__main__":
    unittest.main()
