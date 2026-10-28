"""
Data Loader & Dataset Inspection Module (data_loader.py)
--------------------------------------------------------
Covers Syllabus Practical 3:
- Safe dataset import using Pandas
- Dataset inspection: shape, columns, data types, missing values, duplicates
- Descriptive statistics computation
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Allow running directly as a script
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_PATH


def load_dataset(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Safely loads a CSV dataset using Pandas with robust exception handling.

    Parameters:
        filepath (Path): Path to the CSV file to load.

    Returns:
        pd.DataFrame: Loaded Pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the file is completely empty.
    """
    path_obj = Path(filepath)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"[Error] The dataset file was not found at: {path_obj.resolve()}\n"
            "Please run 'src/generator.py' first to generate the raw dataset."
        )

    try:
        df = pd.read_csv(path_obj)
        if df.empty:
            raise pd.errors.EmptyDataError(f"[Error] Dataset at {path_obj} is empty.")
        return df
    except Exception as e:
        print(f"[Exception] Failed to load dataset: {e}")
        raise


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive inspection summary of a DataFrame.
    Demonstrates Practical 3: Inspection and Descriptive Statistics.

    Parameters:
        df (pd.DataFrame): DataFrame to inspect.

    Returns:
        Dict[str, Any]: Dictionary containing shape, data types, missing counts,
                        duplicate count, and descriptive statistics.
    """
    inspection_info = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "total_missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "descriptive_stats": df.describe().round(2),
    }
    return inspection_info


def display_inspection_report(df: pd.DataFrame, title: str = "DATASET INSPECTION REPORT") -> None:
    """
    Prints a clean, beginner-friendly inspection report for academic lab evaluation.
    """
    info = inspect_dataset(df)

    print("=" * 75)
    print(f" {title} (Practical 3)")
    print("=" * 75)
    print(f"* Total Rows (Records):      {info['num_rows']}")
    print(f"* Total Columns (Features):  {info['num_columns']}")
    print(f"* Duplicate Rows Detected:   {info['duplicate_rows']}")
    print(f"* Total Missing (NaN) Cells: {info['total_missing_cells']}")
    print("-" * 75)
    print("Columns and Data Types:")
    for col, dtype in info["dtypes"].items():
        missing = info["missing_counts"][col]
        print(f"  - {col:<22} | Type: {str(dtype):<10} | Missing: {missing}")

    print("-" * 75)
    print("Sample Records (Head - First 3 Rows):")
    print(df.head(3).to_string())

    print("-" * 75)
    print("Descriptive Statistics (Practical 3):")
    print(info["descriptive_stats"].to_string())
    print("=" * 75)


if __name__ == "__main__":
    print("Running Practical 3 Demonstration: Loading and Inspecting Sensor Data...")
    raw_df = load_dataset(RAW_DATA_PATH)
    display_inspection_report(raw_df, title="RAW SENSOR DATA INSPECTION")
