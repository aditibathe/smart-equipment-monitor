"""
Step 2 Verification Test (test_step2.py)
----------------------------------------
Tests Practicals 3, 4, 5, and 6:
- Dataset inspection output format (Practical 3)
- Duplicate removal and median imputation (Practical 4)
- Conditional filtering and sorting (Practical 5)
- Grouping and aggregation (Practical 6)
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH, EQUIPMENT_IDS
from src.data_loader import load_dataset, inspect_dataset
from src.preprocessor import (
    clean_and_restructure_data,
    filter_critical_equipment,
    summarize_equipment_health,
)


class TestStep2DataProcessing(unittest.TestCase):
    """Test suite for Step 2: Practicals 3, 4, 5, and 6."""

    @classmethod
    def setUpClass(cls):
        """Load raw dataset once for the test suite."""
        cls.raw_df = load_dataset(RAW_DATA_PATH)
        cls.clean_df, cls.report = clean_and_restructure_data(
            cls.raw_df, save_cleaned=True, output_path=CLEAN_DATA_PATH
        )

    # -------------------------------------------------------------
    # Tests for Practical 3: Inspection & Descriptive Statistics
    # -------------------------------------------------------------
    def test_practical3_dataset_inspection(self):
        """Verify inspection dictionary keys and metrics."""
        info = inspect_dataset(self.raw_df)
        self.assertIn("num_rows", info)
        self.assertIn("num_columns", info)
        self.assertIn("dtypes", info)
        self.assertIn("missing_counts", info)
        self.assertIn("descriptive_stats", info)
        self.assertGreater(info["duplicate_rows"], 0, "Raw dataset should report duplicates.")
        self.assertGreater(info["total_missing_cells"], 0, "Raw dataset should report missing cells.")

    # -------------------------------------------------------------
    # Tests for Practical 4: Cleaning, Imputation & Restructuring
    # -------------------------------------------------------------
    def test_practical4_original_data_preserved(self):
        """Verify cleaning does not mutate or alter the original raw DataFrame in memory."""
        self.assertGreater(self.raw_df.duplicated().sum(), 0, "Raw DataFrame must still contain duplicates.")
        self.assertGreater(self.raw_df.isnull().sum().sum(), 0, "Raw DataFrame must still contain NaNs.")

    def test_practical4_no_remaining_duplicates(self):
        """Verify cleaned DataFrame has zero duplicates."""
        self.assertEqual(self.clean_df.duplicated().sum(), 0, "Cleaned dataset must have 0 duplicates.")

    def test_practical4_no_remaining_missing_values(self):
        """Verify cleaned DataFrame has zero NaNs after median imputation."""
        self.assertEqual(self.clean_df.isnull().sum().sum(), 0, "Cleaned dataset must have 0 missing values.")

    def test_practical4_restructuring(self):
        """Verify timestamp parsing and index reset."""
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.clean_df["timestamp"]))
        # Index should be continuous 0 to N-1
        self.assertEqual(list(self.clean_df.index), list(range(len(self.clean_df))))
        # File was saved to disk
        self.assertTrue(CLEAN_DATA_PATH.exists(), "Cleaned CSV file must exist on disk.")

    # -------------------------------------------------------------
    # Tests for Practical 5: Filtering, Sorting & Conditional Selection
    # -------------------------------------------------------------
    def test_practical5_filtering_thresholds(self):
        """Verify conditional filtering by temperature and vibration thresholds."""
        temp_cut = 75.0
        vib_cut = 4.5
        filtered = filter_critical_equipment(
            self.clean_df, temp_threshold=temp_cut, vib_threshold=vib_cut
        )

        # Every filtered row must satisfy (temp > temp_cut) OR (vib > vib_cut)
        condition_check = (filtered["temperature_c"] > temp_cut) | (filtered["vibration_mm_s"] > vib_cut)
        self.assertTrue(condition_check.all(), "All filtered rows must exceed either threshold.")

    def test_practical5_equipment_filter_and_sorting(self):
        """Verify filtering for a specific machine and sorting descending."""
        target_eq = "EQ_101"
        sorted_filtered = filter_critical_equipment(
            self.clean_df,
            temp_threshold=60.0,
            vib_threshold=2.0,
            equipment_id=target_eq,
            sort_by="temperature_c",
            ascending=False,
        )

        # Check equipment ID is strictly target_eq
        self.assertTrue((sorted_filtered["equipment_id"] == target_eq).all())

        # Check descending sort order
        temps = sorted_filtered["temperature_c"].tolist()
        self.assertEqual(temps, sorted(temps, reverse=True), "Filtered rows must be sorted descending.")

    # -------------------------------------------------------------
    # Tests for Practical 6: Grouping and Aggregations
    # -------------------------------------------------------------
    def test_practical6_groupby_aggregation(self):
        """Verify grouping by equipment_id and calculating domain summaries."""
        summary = summarize_equipment_health(self.clean_df)

        # All equipment IDs present in index
        self.assertEqual(set(summary.index), set(EQUIPMENT_IDS))

        # Check required summary columns exist
        expected_cols = [
            "total_readings",
            "mean_temperature_c",
            "max_temperature_c",
            "mean_current_a",
            "max_vibration_mm_s",
            "mean_vibration_mm_s",
            "mean_power_kw",
            "max_operating_hours",
            "total_fault_alerts",
        ]
        for col in expected_cols:
            self.assertIn(col, summary.columns, f"Summary column '{col}' missing.")

        # Total readings per machine should sum to total rows in cleaned_df
        self.assertEqual(summary["total_readings"].sum(), len(self.clean_df))


if __name__ == "__main__":
    unittest.main()
