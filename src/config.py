"""
Configuration Module (config.py)
--------------------------------
Centralizes file paths, feature definitions, operating thresholds,
and syllabus practical mappings for the Smart Equipment Monitor.
"""

from pathlib import Path

# =====================================================================
# 1. Directory and File Paths
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw_sensor_data.csv"
CLEAN_DATA_PATH = DATA_DIR / "cleaned_sensor_data.csv"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure necessary directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 2. Equipment Configuration & Thresholds
# =====================================================================
EQUIPMENT_IDS = ["EQ_101", "EQ_102", "EQ_103", "EQ_104", "EQ_105"]

# Operational Normal Limits for Industrial Motors/Pumps
THRESHOLDS = {
    "temperature_c": {"min": 35.0, "normal_max": 75.0, "critical_max": 90.0},
    "vibration_mm_s": {"min": 0.5, "normal_max": 4.5, "critical_max": 7.0},
    "voltage_v": {"min": 210.0, "nominal": 230.0, "max": 250.0},
    "current_a": {"min": 5.0, "nominal": 15.0, "max": 30.0},
}

# =====================================================================
# 3. Dataset Column Schema
# =====================================================================
TIMESTAMP_COL = "timestamp"
EQUIPMENT_ID_COL = "equipment_id"

SENSOR_COLUMNS = [
    "operating_hours",
    "temperature_c",
    "vibration_mm_s",
    "voltage_v",
    "current_a",
    "power_kw",
]

# Continuous Target (For Practical 10: Linear Regression)
REGRESSION_TARGET = "failure_risk_score"

# Categorical Target (For Practical 11: Classification)
CLASSIFICATION_TARGET = "failure_status"       # 0: Normal, 1: Faulty
STATUS_LABEL_COL = "equipment_status"           # 'Normal', 'Warning', 'Fault'

ALL_COLUMNS = (
    [TIMESTAMP_COL, EQUIPMENT_ID_COL]
    + SENSOR_COLUMNS
    + [REGRESSION_TARGET, CLASSIFICATION_TARGET, STATUS_LABEL_COL]
)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# =====================================================================
# 4. Syllabus Mapping Reference (12 Practicals)
# =====================================================================
PRACTICALS = {
    1: "NumPy arrays, indexing, slicing, reshaping and vectorized operations",
    2: "NumPy matrix and statistical operations",
    3: "Pandas dataset import, inspection and descriptive statistics",
    4: "Missing-value handling, duplicate removal and dataset restructuring",
    5: "Filtering, sorting and conditional selection",
    6: "Grouping and aggregation using Pandas",
    7: "Matplotlib line plots, bar charts and histograms",
    8: "Scatter plots and correlation heatmaps",
    9: "Mean, median, variance, standard deviation and correlation",
    10: "Scikit-learn linear regression for continuous-value prediction",
    11: "Scikit-learn classification for categorical prediction",
    12: "Integrated mini-project (end-to-end data pipeline & dashboard)",
}
