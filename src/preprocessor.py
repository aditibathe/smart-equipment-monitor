"""
Data Preprocessing, Cleaning & Aggregation Module (preprocessor.py)
------------------------------------------------------------------
Covers Syllabus Practicals:
- Practical 4: Missing-value handling (median imputation), duplicate removal, dataset restructuring.
- Practical 5: Filtering, sorting, and conditional selection.
- Practical 6: Grouping and aggregation using Pandas (groupby).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Allow running directly as a script
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    THRESHOLDS,
    EQUIPMENT_ID_COL,
    TIMESTAMP_COL,
)
from src.data_loader import load_dataset


# =====================================================================
# PRACTICAL 4: Missing Values, Duplicates & Restructuring
# =====================================================================

def clean_and_restructure_data(
    df: pd.DataFrame,
    save_cleaned: bool = True,
    output_path: Path = CLEAN_DATA_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw sensor data and restructures it:
    1. Preserves original DataFrame by operating on a deep copy.
    2. Identifies and removes duplicate records.
    3. Handles missing numerical values via Median Imputation (robust to sensor outliers).
    4. Restructures the dataset: parses timestamps, sorts chronologically per machine, resets index.
    5. Saves the cleaned data to CSV.

    Parameters:
        df (pd.DataFrame): Raw sensor DataFrame.
        save_cleaned (bool): Whether to persist cleaned data to disk.
        output_path (Path): Destination CSV path.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and a report dictionary.
    """
    # 1. Preserve original dataset
    cleaned_df = df.copy()

    report: Dict[str, Any] = {
        "initial_rows": len(cleaned_df),
        "initial_duplicates": int(cleaned_df.duplicated().sum()),
        "initial_missing_by_col": cleaned_df.isnull().sum()[cleaned_df.isnull().sum() > 0].to_dict(),
        "imputation_values": {},
    }

    # 2. Duplicate Removal
    cleaned_df = cleaned_df.drop_duplicates()
    report["rows_after_dedup"] = len(cleaned_df)
    report["duplicates_removed"] = report["initial_rows"] - report["rows_after_dedup"]

    # 3. Missing-Value Handling using Median Imputation
    # We impute continuous sensor signals with their median value (statistically robust against extreme spikes)
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_count = int(cleaned_df[col].isnull().sum())
        if missing_count > 0:
            median_val = round(float(cleaned_df[col].median()), 2)
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            report["imputation_values"][col] = {
                "imputed_cells": missing_count,
                "median_value_used": median_val,
            }

    # 4. Dataset Restructuring
    # Convert timestamp string to datetime object
    cleaned_df[TIMESTAMP_COL] = pd.to_datetime(cleaned_df[TIMESTAMP_COL])

    # Sort chronologically by equipment ID and timestamp
    cleaned_df = cleaned_df.sort_values(by=[EQUIPMENT_ID_COL, TIMESTAMP_COL])

    # Reset index to maintain consecutive 0..N row numbering
    cleaned_df = cleaned_df.reset_index(drop=True)

    report["final_rows"] = len(cleaned_df)
    report["remaining_missing_cells"] = int(cleaned_df.isnull().sum().sum())
    report["remaining_duplicates"] = int(cleaned_df.duplicated().sum())

    # 5. Persist to disk
    if save_cleaned:
        cleaned_df.to_csv(output_path, index=False)
        report["saved_to"] = str(output_path)

    return cleaned_df, report


# =====================================================================
# PRACTICAL 5: Filtering, Sorting and Conditional Selection
# =====================================================================

def filter_critical_equipment(
    df: pd.DataFrame,
    temp_threshold: Optional[float] = None,
    vib_threshold: Optional[float] = None,
    equipment_id: Optional[str] = None,
    sort_by: str = "temperature_c",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Performs conditional selection, filtering, and sorting on sensor data.
    Demonstrates Practical 5.

    Physical Units:
    - Temperature: Celsius (°C) [Threshold default: 75.0°C]
    - Vibration: Velocity in millimeters per second (mm/s) [Threshold default: 4.5 mm/s]

    Parameters:
        df (pd.DataFrame): Sensor DataFrame.
        temp_threshold (float, optional): Critical temperature cutoff in °C.
        vib_threshold (float, optional): Critical vibration cutoff in mm/s.
        equipment_id (str, optional): Target machine ID (e.g. 'EQ_101').
        sort_by (str): Column to sort filtered results by.
        ascending (bool): Sort direction (False = highest first).

    Returns:
        pd.DataFrame: Filtered and sorted subset of equipment records.
    """
    if temp_threshold is None:
        temp_threshold = THRESHOLDS["temperature_c"]["normal_max"]
    if vib_threshold is None:
        vib_threshold = THRESHOLDS["vibration_mm_s"]["normal_max"]

    # Boolean conditional masks (Practical 5)
    temp_mask = df["temperature_c"] > temp_threshold
    vib_mask = df["vibration_mm_s"] > vib_threshold

    # Compound condition using bitwise OR (|) to flag any critical reading
    critical_mask = temp_mask | vib_mask

    filtered_df = df[critical_mask]

    # Optional machine filter
    if equipment_id is not None:
        filtered_df = filtered_df[filtered_df[EQUIPMENT_ID_COL] == equipment_id]

    # Sorting (Practical 5)
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    return filtered_df.reset_index(drop=True)


# =====================================================================
# PRACTICAL 6: Grouping and Aggregation using Pandas
# =====================================================================

def summarize_equipment_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups sensor records by equipment_id and calculates key engineering summaries:
    - Mean temperature (°C)
    - Mean current (A)
    - Maximum vibration (mm/s)
    - Mean power consumption (kW)
    - Maximum operating hours (hrs)
    - Total detected failure alerts

    Demonstrates Practical 6 (Pandas GroupBy & Aggregations).

    Parameters:
        df (pd.DataFrame): Cleaned sensor DataFrame.

    Returns:
        pd.DataFrame: Aggregated summary table indexed by equipment_id.
    """
    # Grouping by equipment_id and calculating domain-specific aggregations
    grouped = df.groupby(EQUIPMENT_ID_COL).agg(
        total_readings=(TIMESTAMP_COL, "count"),
        mean_temperature_c=("temperature_c", "mean"),
        max_temperature_c=("temperature_c", "max"),
        mean_current_a=("current_a", "mean"),
        max_vibration_mm_s=("vibration_mm_s", "max"),
        mean_vibration_mm_s=("vibration_mm_s", "mean"),
        mean_power_kw=("power_kw", "mean"),
        max_operating_hours=("operating_hours", "max"),
        total_fault_alerts=("failure_status", "sum"),
    ).round(2)

    return grouped


# =====================================================================
# Academic Demonstration Driver
# =====================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("DEMONSTRATION: PRACTICALS 4, 5, AND 6")
    print("=" * 75)

    # 1. Load Raw Data
    raw_data = load_dataset(RAW_DATA_PATH)
    print(f"[Input] Raw Data Loaded: {len(raw_data)} rows, {raw_data.duplicated().sum()} duplicates")

    # 2. Execute Practical 4 (Cleaning & Restructuring)
    print("\n--- Practical 4: Data Cleaning & Restructuring ---")
    clean_data, clean_report = clean_and_restructure_data(raw_data)
    print(f"* Initial rows: {clean_report['initial_rows']}")
    print(f"* Duplicates removed: {clean_report['duplicates_removed']}")
    print("* Imputation summary (Median values used for missing cells):")
    for col, imp_info in clean_report["imputation_values"].items():
        print(f"    - {col}: {imp_info['imputed_cells']} missing values filled with median = {imp_info['median_value_used']}")
    print(f"* Final clean rows: {clean_report['final_rows']}")
    print(f"* Remaining missing cells: {clean_report['remaining_missing_cells']}")
    print(f"* Clean dataset saved to: {clean_report['saved_to']}")

    # 3. Execute Practical 5 (Filtering, Sorting, Conditional Selection)
    print("\n--- Practical 5: Filtering, Sorting & Conditional Selection ---")
    temp_limit = THRESHOLDS["temperature_c"]["normal_max"]   # 75.0 C
    vib_limit = THRESHOLDS["vibration_mm_s"]["normal_max"]   # 4.5 mm/s
    print(f"Filtering conditions: Temp > {temp_limit} deg C OR Vibration > {vib_limit} mm/s")

    critical_records = filter_critical_equipment(
        clean_data,
        temp_threshold=temp_limit,
        vib_threshold=vib_limit,
        sort_by="vibration_mm_s",
        ascending=False,
    )
    print(f"Total critical abnormal readings found: {len(critical_records)}")
    print("Top 3 most severe vibration records:")
    print(critical_records[["equipment_id", "timestamp", "temperature_c", "vibration_mm_s", "equipment_status"]].head(3).to_string())

    # 4. Execute Practical 6 (Grouping & Aggregation)
    print("\n--- Practical 6: Grouping & Aggregation by Equipment ID ---")
    equipment_summary = summarize_equipment_health(clean_data)
    print(equipment_summary.to_string())
    print("=" * 75)
