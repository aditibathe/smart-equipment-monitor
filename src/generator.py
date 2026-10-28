"""
Synthetic Sensor Data Generator (generator.py)
----------------------------------------------
Generates a realistic industrial sensor dataset for Smart Equipment Monitoring.
Demonstrates NumPy vectorized operations, array indexing, slicing,
and intentional injection of missing values and duplicates for data cleaning practicals.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Allow running directly as a script (e.g., `python src/generator.py`)
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    EQUIPMENT_IDS,
    RANDOM_SEED,
    TIMESTAMP_COL,
    EQUIPMENT_ID_COL,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    STATUS_LABEL_COL,
)


def generate_sensor_dataset(
    num_samples: int = 1200,
    random_seed: int = RANDOM_SEED,
    missing_ratio: float = 0.03,
    num_duplicates: int = 25,
) -> pd.DataFrame:
    """
    Generates a synthetic time-series sensor dataset with realistic physical correlations.

    Parameters:
        num_samples (int): Total number of base sensor observations to generate.
        random_seed (int): Seed for NumPy random number generation (ensures reproducibility).
        missing_ratio (float): Fraction of rows to inject with NaN (for Practical 4).
        num_duplicates (int): Number of duplicate sensor records to append (for Practical 4).

    Returns:
        pd.DataFrame: Raw synthetic sensor DataFrame with injected NaNs and duplicates.
    """
    rng = np.random.default_rng(seed=random_seed)

    # 1. Generate Timestamps (Hourly readings starting from a fixed start date)
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [
        (start_time + timedelta(hours=int(i))).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(num_samples)
    ]

    # 2. Assign Equipment IDs cyclically
    equipment_ids = [
        EQUIPMENT_IDS[i % len(EQUIPMENT_IDS)] for i in range(num_samples)
    ]

    # 3. Vectorized Generation of Physical Sensor Readings using NumPy arrays
    # Operating Hours: Gradually accumulates between 100 and 5000 hours
    base_hours = np.linspace(100.0, 4800.0, num_samples)
    noise_hours = rng.uniform(-50.0, 50.0, size=num_samples)
    operating_hours = np.clip(base_hours + noise_hours, 50.0, 5000.0).round(1)

    # Baseline Voltage (V): Centered around 230V with occasional voltage dips/spikes
    voltage_v = rng.normal(loc=230.0, scale=4.5, size=num_samples).round(2)

    # Motor Current (A): Typically 8A to 22A, higher under heavy loads
    current_a = rng.uniform(8.0, 22.0, size=num_samples).round(2)

    # Power (kW): Calculated using NumPy vectorized formula P = (V * I) / 1000
    # Demonstrates: Practical 1 (vectorized operations)
    power_kw = np.round((voltage_v * current_a) / 1000.0, 3)

    # Vibration (mm/s): Normal range 1.0 - 3.5 mm/s; increases with operating hours & load
    wear_factor = (operating_hours / 5000.0) * 3.5
    vibration_noise = rng.exponential(scale=1.0, size=num_samples)
    vibration_mm_s = np.clip(1.2 + wear_factor + vibration_noise, 0.5, 14.5).round(2)

    # Temperature (C): Base ~50C; increases with current load and friction from vibration
    temperature_base = 48.0 + (current_a * 0.8) + (vibration_mm_s * 2.5)
    temp_noise = rng.normal(loc=0.0, scale=2.5, size=num_samples)
    temperature_c = np.clip(temperature_base + temp_noise, 32.0, 108.0).round(2)

    # 4. Target Variables (No Target Leakage: computed purely as physical ground truth)
    # Failure Risk Score (Continuous 0.0 - 100.0) for Linear Regression (Practical 10)
    # A combination of temperature, vibration, hours, and electrical stress
    norm_temp = (temperature_c - 40.0) / 60.0
    norm_vib = (vibration_mm_s - 1.0) / 12.0
    norm_hrs = operating_hours / 5000.0
    norm_pwr = (power_kw - 1.5) / 4.0

    raw_risk = (
        (0.35 * norm_temp)
        + (0.35 * norm_vib)
        + (0.20 * norm_hrs)
        + (0.10 * norm_pwr)
    ) * 100.0
    risk_noise = rng.normal(loc=0.0, scale=3.0, size=num_samples)
    failure_risk_score = np.clip(raw_risk + risk_noise, 0.0, 100.0).round(2)

    # Status Label & Binary Failure Status for Classification (Practical 11)
    # Normal: risk < 45 | Warning: 45 <= risk < 65 | Fault: risk >= 65
    equipment_status = []
    failure_status = []
    for score in failure_risk_score:
        if score >= 65.0:
            equipment_status.append("Fault")
            failure_status.append(1)  # Positive failure class
        elif score >= 45.0:
            equipment_status.append("Warning")
            failure_status.append(0)  # Considered non-critical/operational
        else:
            equipment_status.append("Normal")
            failure_status.append(0)

    # Assemble into a Pandas DataFrame
    df = pd.DataFrame(
        {
            TIMESTAMP_COL: timestamps,
            EQUIPMENT_ID_COL: equipment_ids,
            "operating_hours": operating_hours,
            "temperature_c": temperature_c,
            "vibration_mm_s": vibration_mm_s,
            "voltage_v": voltage_v,
            "current_a": current_a,
            "power_kw": power_kw,
            REGRESSION_TARGET: failure_risk_score,
            CLASSIFICATION_TARGET: failure_status,
            STATUS_LABEL_COL: equipment_status,
        }
    )

    # 5. Inject Missing Values (NaN) for Practical 4 demonstration
    # Randomly introduce NaNs into sensor columns (~3% of rows)
    num_missing = int(num_samples * missing_ratio)
    missing_indices_temp = rng.choice(num_samples, size=num_missing, replace=False)
    missing_indices_vib = rng.choice(num_samples, size=num_missing, replace=False)
    missing_indices_volt = rng.choice(num_samples, size=num_missing, replace=False)

    df.loc[missing_indices_temp, "temperature_c"] = np.nan
    df.loc[missing_indices_vib, "vibration_mm_s"] = np.nan
    df.loc[missing_indices_volt, "voltage_v"] = np.nan

    # 6. Inject Duplicate Records for Practical 4 demonstration
    # Pick duplicate candidates and append them to simulate network retries
    duplicate_indices = rng.choice(num_samples, size=num_duplicates, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def save_dataset(df: pd.DataFrame, filepath=RAW_DATA_PATH) -> None:
    """Saves DataFrame to CSV file with UTF-8 encoding."""
    df.to_csv(filepath, index=False)
    print(f"[OK] Synthetic dataset successfully saved to: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Generating Raw Synthetic Sensor Dataset")
    print("=" * 60)

    dataset = generate_sensor_dataset(num_samples=1200, random_seed=RANDOM_SEED)
    save_dataset(dataset)

    print(f"Total Rows Generated (including duplicates): {len(dataset)}")
    print(f"Total Columns: {len(dataset.columns)}")
    print("\nMissing Values Injected:")
    print(dataset.isnull().sum()[dataset.isnull().sum() > 0])
    print(f"\nDuplicate Rows Injected: {dataset.duplicated().sum()}")
    print("=" * 60)
