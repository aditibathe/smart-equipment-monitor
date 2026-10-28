"""
Step 3 Verification Test (test_step3.py)
----------------------------------------
Tests Practicals 1, 2, 7, 8, and 9:
- NumPy array indexing, slicing, reshaping, vectorization (Practical 1)
- NumPy matrix transpose, dot product, covariance, array statistics (Practical 2)
- Detailed statistics & Pearson correlation (Practical 9)
- Visualizations: line plot, bar chart, histogram, scatter, heatmap (Practicals 7 & 8)
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import CLEAN_DATA_PATH, FIGURES_DIR
from src.data_loader import load_dataset
from src.preprocessor import summarize_equipment_health
from src.analytics import (
    demonstrate_numpy_array_operations,
    demonstrate_numpy_matrix_operations,
    calculate_statistical_metrics,
)
from src.visualizer import (
    plot_sensor_time_series,
    plot_equipment_comparison_bar,
    plot_sensor_distribution_histogram,
    plot_vibration_vs_temperature_scatter,
    plot_correlation_heatmap,
)


class TestStep3AnalyticsAndVisualization(unittest.TestCase):
    """Test suite for Step 3: Practicals 1, 2, 7, 8, and 9."""

    @classmethod
    def setUpClass(cls):
        """Load clean dataset once for the test suite."""
        cls.df = load_dataset(CLEAN_DATA_PATH)
        cls.summary_df = summarize_equipment_health(cls.df)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Practical 1 Tests
    # -------------------------------------------------------------
    def test_practical1_numpy_operations(self):
        """Verify Practical 1 NumPy array mechanics."""
        res = demonstrate_numpy_array_operations(self.df)
        self.assertEqual(res["original_shape"], (len(self.df), 6))
        self.assertEqual(res["ndim"], 2)
        self.assertEqual(len(res["temp_slice_1d"]), 5)
        self.assertEqual(res["submatrix_2d"].shape, (3, 2))
        self.assertEqual(res["reshaped_3x4"].shape, (3, 4))
        self.assertEqual(res["standardized_sample"].shape, (3, 6))
        self.assertGreaterEqual(res["num_vectorized_anomalies"], 0)

    # -------------------------------------------------------------
    # Practical 2 Tests
    # -------------------------------------------------------------
    def test_practical2_matrix_operations(self):
        """Verify Practical 2 NumPy matrix math and array statistics."""
        res = demonstrate_numpy_matrix_operations(self.df)
        self.assertEqual(res["X_shape"], (len(self.df), 2))
        self.assertEqual(res["X_T_shape"], (2, len(self.df)))
        self.assertEqual(res["gram_matrix"].shape, (2, 2))
        self.assertEqual(res["cov_matrix"].shape, (2, 2))
        self.assertEqual(len(res["col_means"]), 2)
        self.assertEqual(len(res["col_medians"]), 2)
        self.assertEqual(len(res["col_variances"]), 2)
        self.assertEqual(len(res["col_std"]), 2)
        self.assertEqual(len(res["col_range"]), 2)

    # -------------------------------------------------------------
    # Practical 9 Tests
    # -------------------------------------------------------------
    def test_practical9_statistical_metrics(self):
        """Verify Practical 9 descriptive statistics and Pearson correlation."""
        stats_df, corr_df = calculate_statistical_metrics(self.df)

        # Check required statistical indicator columns
        for metric in ["Mean", "Median", "Variance", "Std_Dev", "Min", "Max", "IQR"]:
            self.assertIn(metric, stats_df.columns)

        # Check correlation matrix properties
        self.assertEqual(corr_df.shape[0], corr_df.shape[1])
        # Diagonal elements must be 1.0 (self-correlation)
        np.testing.assert_allclose(np.diag(corr_df.values), 1.0, rtol=1e-3)
        # Bounded between -1 and 1
        self.assertTrue((corr_df.values >= -1.0).all())
        self.assertTrue((corr_df.values <= 1.0).all())

    # -------------------------------------------------------------
    # Practical 7 & 8 Tests: Visualizations
    # -------------------------------------------------------------
    def test_practical7_visualizations(self):
        """Verify Practical 7 line plot, bar chart, and histogram creation."""
        # Line plot
        fig1 = plot_sensor_time_series(self.df, equipment_id="EQ_101")
        self.assertIsInstance(fig1, plt.Figure)
        plt.close(fig1)

        # Bar chart
        fig2 = plot_equipment_comparison_bar(self.summary_df, metric="max_vibration_mm_s")
        self.assertIsInstance(fig2, plt.Figure)
        plt.close(fig2)

        # Histogram
        fig3 = plot_sensor_distribution_histogram(self.df, column="temperature_c")
        self.assertIsInstance(fig3, plt.Figure)
        plt.close(fig3)

    def test_practical8_visualizations(self):
        """Verify Practical 8 scatter plot and correlation heatmap."""
        # Scatter plot
        fig4 = plot_vibration_vs_temperature_scatter(self.df)
        self.assertIsInstance(fig4, plt.Figure)
        plt.close(fig4)

        # Correlation heatmap
        fig5 = plot_correlation_heatmap(self.df)
        self.assertIsInstance(fig5, plt.Figure)
        plt.close(fig5)


if __name__ == "__main__":
    unittest.main()
