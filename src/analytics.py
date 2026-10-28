"""
Numerical Computing, Matrix Math & Statistical Analytics (analytics.py)
------------------------------------------------------------------------
Covers Syllabus Practicals:
- Practical 1: NumPy arrays, indexing, slicing, reshaping, and vectorized operations.
- Practical 2: NumPy matrix operations (dot product, transpose, covariance) & statistical functions.
- Practical 9: Detailed descriptive statistics (Mean, Median, Variance, Std Dev, Pearson Correlation).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Allow running directly as a script
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import SENSOR_COLUMNS, CLEAN_DATA_PATH
from src.data_loader import load_dataset


# =====================================================================
# PRACTICAL 1: NumPy Arrays, Slicing, Reshaping & Vectorization
# =====================================================================

def demonstrate_numpy_array_operations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Demonstrates foundational NumPy array mechanics on sensor signals:
    - Array extraction from tabular data
    - Array shape, dimensions, and data type inspection
    - Indexing and slicing (1D and 2D)
    - Array reshaping
    - Vectorized mathematical scaling (Z-score normalization)

    Demonstrates Practical 1.
    """
    # 1. Convert sensor columns to a 2D NumPy array
    feature_matrix = df[SENSOR_COLUMNS].to_numpy()

    # 2. Array inspection
    original_shape = feature_matrix.shape
    ndim = feature_matrix.ndim
    dtype = feature_matrix.dtype

    # 3. Indexing & Slicing
    # Extract first 5 readings of temperature (column index 1)
    temp_slice_1d = feature_matrix[:5, 1]
    # Slicing a 2D submatrix: first 3 rows, columns 1 to 3 (temperature and vibration)
    submatrix_2d = feature_matrix[:3, 1:3]

    # 4. Reshaping
    # Flatten the first 12 temperature readings and reshape to a 3x4 matrix
    sample_flat = feature_matrix[:12, 1]
    reshaped_3x4 = sample_flat.reshape(3, 4)

    # 5. Vectorized Operations (Z-Score Standardization: (X - mu) / sigma)
    # Computed purely via NumPy broadcasting across all sensor features simultaneously
    means = np.mean(feature_matrix, axis=0)
    std_devs = np.std(feature_matrix, axis=0)
    standardized_matrix = (feature_matrix - means) / std_devs

    # Vectorized Boolean Masking (Identify all rows where vibration > 2 standard deviations)
    vib_col = feature_matrix[:, 2]
    high_vib_mask = vib_col > (means[2] + 2 * std_devs[2])
    anomaly_indices = np.where(high_vib_mask)[0]

    return {
        "original_shape": original_shape,
        "ndim": ndim,
        "dtype": str(dtype),
        "temp_slice_1d": temp_slice_1d,
        "submatrix_2d": submatrix_2d,
        "reshaped_3x4": reshaped_3x4,
        "standardized_sample": np.round(standardized_matrix[:3], 3),
        "num_vectorized_anomalies": len(anomaly_indices),
        "anomaly_indices": anomaly_indices[:5].tolist(),
    }


# =====================================================================
# PRACTICAL 2: NumPy Matrix Math & Statistical Operations
# =====================================================================

def demonstrate_numpy_matrix_operations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Demonstrates NumPy matrix operations and array statistics:
    - Matrix transpose (X^T)
    - Matrix multiplication / Dot product (X^T @ X)
    - Covariance matrix computation
    - Array statistics along axes (axis=0)

    Demonstrates Practical 2.
    """
    # Extract temperature and vibration as a bivariate matrix X (N x 2)
    X = df[["temperature_c", "vibration_mm_s"]].to_numpy()

    # 1. Transpose
    X_T = X.T

    # 2. Matrix Multiplication / Dot Product: X^T @ X (Produces a 2x2 Gram matrix)
    gram_matrix = np.dot(X_T, X)

    # 3. Covariance Matrix: Measures joint variability between temperature and vibration
    cov_matrix = np.cov(X, rowvar=False)

    # 4. NumPy Statistical Operations along columns (axis=0)
    col_means = np.mean(X, axis=0)
    col_medians = np.median(X, axis=0)
    col_variances = np.var(X, axis=0, ddof=1)
    col_std = np.std(X, axis=0, ddof=1)
    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    col_range = np.ptp(X, axis=0)  # Peak-to-peak (max - min)

    return {
        "X_shape": X.shape,
        "X_T_shape": X_T.shape,
        "gram_matrix": np.round(gram_matrix, 2),
        "cov_matrix": np.round(cov_matrix, 4),
        "col_means": np.round(col_means, 2),
        "col_medians": np.round(col_medians, 2),
        "col_variances": np.round(col_variances, 4),
        "col_std": np.round(col_std, 4),
        "col_range": np.round(col_range, 2),
    }


# =====================================================================
# PRACTICAL 9: Comprehensive Statistical Metrics & Correlation
# =====================================================================

def calculate_statistical_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes rigorous descriptive statistical metrics across all numeric sensor channels:
    - Mean
    - Median
    - Sample Variance (ddof=1)
    - Standard Deviation (ddof=1)
    - Minimum, Maximum, Interquartile Range (IQR)
    - Pearson Correlation Coefficient Matrix

    Demonstrates Practical 9.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (statistics_table, correlation_matrix)
    """
    numeric_df = df.select_dtypes(include=[np.number])

    stats_dict = {}
    for col in numeric_df.columns:
        data_series = numeric_df[col].to_numpy()
        q75, q25 = np.percentile(data_series, [75, 25])
        stats_dict[col] = {
            "Mean": round(float(np.mean(data_series)), 3),
            "Median": round(float(np.median(data_series)), 3),
            "Variance": round(float(np.var(data_series, ddof=1)), 3),
            "Std_Dev": round(float(np.std(data_series, ddof=1)), 3),
            "Min": round(float(np.min(data_series)), 3),
            "Max": round(float(np.max(data_series)), 3),
            "IQR": round(float(q75 - q25), 3),
        }

    stats_table = pd.DataFrame(stats_dict).T

    # Pearson Correlation Coefficient Matrix
    correlation_matrix = numeric_df.corr(method="pearson").round(3)

    return stats_table, correlation_matrix


# =====================================================================
# Academic Demonstration Driver
# =====================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("DEMONSTRATION: PRACTICALS 1, 2, AND 9 (NUMERICAL & STATISTICAL COMPUTING)")
    print("=" * 75)

    # Load cleaned sensor dataset
    df = load_dataset(CLEAN_DATA_PATH)
    print(f"Loaded Cleaned Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # 1. Practical 1 Demo
    print("--- Practical 1: NumPy Arrays, Slicing, Reshaping & Vectorization ---")
    p1 = demonstrate_numpy_array_operations(df)
    print(f"* Feature Matrix Shape: {p1['original_shape']} (Dimensions: {p1['ndim']}, Dtype: {p1['dtype']})")
    print(f"* 1D Temperature Slice (First 5): {p1['temp_slice_1d']}")
    print(f"* 2D Submatrix Slice (First 3 rows of Temp & Vib):\n{p1['submatrix_2d']}")
    print(f"* Reshaped 3x4 Matrix (12 Temperature points):\n{p1['reshaped_3x4']}")
    print(f"* Standardized Sample (Z-score vectorized broadcasting):\n{p1['standardized_sample']}")
    print(f"* Vectorized Vibration Anomalies Detected (> 2-sigma): {p1['num_vectorized_anomalies']}")

    # 2. Practical 2 Demo
    print("\n--- Practical 2: NumPy Matrix Math & Statistical Operations ---")
    p2 = demonstrate_numpy_matrix_operations(df)
    print(f"* Matrix Shapes: X = {p2['X_shape']} | X.T = {p2['X_T_shape']}")
    print(f"* Gram Matrix (X.T @ X):\n{p2['gram_matrix']}")
    print(f"* Covariance Matrix [Temp, Vib]:\n{p2['cov_matrix']}")
    print(f"* Means [Temp, Vib]: {p2['col_means']}")
    print(f"* Medians [Temp, Vib]: {p2['col_medians']}")
    print(f"* Variances [Temp, Vib]: {p2['col_variances']}")
    print(f"* Standard Deviations [Temp, Vib]: {p2['col_std']}")
    print(f"* Peak-to-Peak Range [Temp, Vib]: {p2['col_range']}")

    # 3. Practical 9 Demo
    print("\n--- Practical 9: Statistical Metrics & Pearson Correlation ---")
    stats_df, corr_df = calculate_statistical_metrics(df)
    print("Summary Statistical Indicators (Mean, Median, Variance, Std Dev, Min, Max, IQR):")
    print(stats_df.to_string())

    print("\nPearson Correlation Matrix (First 5 Features vs Targets):")
    cols_to_show = ["temperature_c", "vibration_mm_s", "current_a", "failure_risk_score", "failure_status"]
    print(corr_df[cols_to_show].loc[cols_to_show].to_string())
    print("=" * 75)
