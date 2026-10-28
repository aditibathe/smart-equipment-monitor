# Smart Equipment Monitoring & Predictive Maintenance

An industrial IoT telemetry monitoring and predictive maintenance mini-project designed for Electronics & Communication Engineering (ECE) students.

This project demonstrates all **12 Python Practicals** from the university programming syllabus within a unified engineering application.

---

## 📑 12 Practicals Syllabus Mapping

| Practical # | Academic Concept | Implementation in this Project | Code Location |
| :---: | :--- | :--- | :--- |
| **1** | **NumPy arrays, indexing, slicing, reshaping & vectorized operations** | Feature matrix slicing, $(3 \times 4)$ reshaping, vectorized power calculation ($P = \frac{V \times I}{1000}$), and standard scaling ($Z = \frac{X - \mu}{\sigma}$). | [`src/analytics.py`](src/analytics.py) |
| **2** | **NumPy matrix & statistical operations** | Matrix transpose ($X^T$), Gram matrix ($X^T X$), bivariate covariance matrix, and column-wise statistics (`axis=0`). | [`src/analytics.py`](src/analytics.py) |
| **3** | **Pandas dataset import, inspection & descriptive statistics** | Safe CSV loading with error handling, shape/dtype inspection, missing counts, duplicate counts, and `.describe()` statistics. | [`src/data_loader.py`](src/data_loader.py) |
| **4** | **Missing-value handling, duplicate removal & dataset restructuring** | Median imputation for sensor channels, dropping duplicates, datetime parsing, chronological sorting, and index resetting. | [`src/preprocessor.py`](src/preprocessor.py) |
| **5** | **Filtering, sorting & conditional selection** | Boolean masks filtering high temperature (>75°C) or high vibration (>4.5 mm/s), equipment filtering, and descending severity sorting. | [`src/preprocessor.py`](src/preprocessor.py) |
| **6** | **Grouping and aggregation using Pandas** | `groupby("equipment_id")` computing mean temperature, mean current, max vibration, mean power, and fault counts. | [`src/preprocessor.py`](src/preprocessor.py) |
| **7** | **Matplotlib line plots, bar charts & histograms** | Dual-axis telemetry line plots with threshold lines, categorical comparison bar charts, and sensor frequency histograms with mean/median markers. | [`src/visualizer.py`](src/visualizer.py) |
| **8** | **Scatter plots and correlation heatmaps** | Bivariate scatter plots colored by health status (`Normal`, `Warning`, `Fault`) and Seaborn annotated Pearson correlation heatmap. | [`src/visualizer.py`](src/visualizer.py) |
| **9** | **Mean, median, variance, standard deviation & correlation** | Rigorous calculation of sample variance ($\text{ddof}=1$), std dev, IQR ($Q_3 - Q_1$), and Pearson correlation coefficients ($r$). | [`src/analytics.py`](src/analytics.py) |
| **10** | **Scikit-learn linear regression for continuous prediction** | Continuous `failure_risk_score` prediction using sensor features without target leakage; evaluated with MSE, RMSE, MAE, and $R^2$ score. | [`src/models.py`](src/models.py) |
| **11** | **Scikit-learn classification for categorical prediction** | Equipment failure status classification (`0: Normal`, `1: Fault`) using Logistic Regression; evaluated with Accuracy, Precision, Recall, F1, and Confusion Matrix. | [`src/models.py`](src/models.py) |
| **12** | **Integrated mini-project with real-world pipeline & dashboard** | End-to-end data processing pipeline, terminal evaluation runner (`main.py`), and interactive Streamlit Web Dashboard (`app.py`). | [`main.py`](main.py) & [`app.py`](app.py) |

---

## 📂 Project Structure

```text
smart equipment monitor/
│
├── .venv/                         # Pre-configured Python virtual environment
├── data/
│   ├── raw_sensor_data.csv        # Raw dataset with injected NaNs and duplicates (Practical 4)
│   └── cleaned_sensor_data.csv    # Cleaned, standardized production dataset
│
├── reports/
│   └── figures/                   # Exported Matplotlib & Seaborn visualizations (Practicals 7 & 8)
│       ├── practical7_time_series.png
│       ├── practical7_bar_chart.png
│       ├── practical7_histogram.png
│       ├── practical8_scatter_plot.png
│       └── practical8_correlation_heatmap.png
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── config.py                  # Thresholds, column names, paths, and practical metadata
│   ├── generator.py               # Synthetic sensor generator with physics-based correlations
│   ├── data_loader.py             # Safe CSV loading and inspection (Practical 3)
│   ├── preprocessor.py            # Cleaning, imputation, filtering, grouping (Practicals 4, 5, 6)
│   ├── analytics.py               # NumPy arrays, matrix math, descriptive stats (Practicals 1, 2, 9)
│   ├── visualizer.py              # Matplotlib and Seaborn plotting routines (Practicals 7, 8)
│   └── models.py                  # Linear Regression & Classification models (Practicals 10, 11)
│
├── tests/                         # Automated unit test suite
│   ├── __init__.py
│   ├── test_step1.py              # Step 1 tests (Data generator & schema)
│   ├── test_step2.py              # Step 2 tests (Inspection, cleaning, filtering, grouping)
│   ├── test_step3.py              # Step 3 tests (NumPy math, stats, visualizations)
│   └── test_step4.py              # Step 4 tests (Linear regression, classification, inference)
│
├── app.py                         # Interactive Streamlit Web Dashboard (Practical 12)
├── main.py                        # Terminal CLI demonstration for college viva evaluation
├── run_tests.py                   # Automated test runner script (all 25 tests)
├── requirements.txt               # Installed library dependencies
└── README.md                      # Comprehensive academic documentation
```

---

## 🚀 Quick Start Guide

### 1. Run the Sequential Practical Demonstration (Terminal CLI)
To demonstrate all 12 practicals consecutively for a lab evaluation or viva:
```powershell
.\.venv\Scripts\python.exe main.py
```

### 2. Launch the Interactive Web Dashboard (Streamlit)
To launch the interactive dashboard with telemetry charts and live slider-based predictive diagnostics:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
*The web browser will open automatically at `http://localhost:8501`.*

### 3. Run Automated Unit Tests
To verify all test cases across the entire project cleanly:
```powershell
.\.venv\Scripts\python.exe run_tests.py
```
*(Alternatively, using standard unittest discovery: `.\.venv\Scripts\python.exe -m unittest discover tests`)*

---

## ⚙️ Engineering Sensor Details

| Channel | Physical Meaning | Normal Range | Critical Threshold |
| :--- | :--- | :---: | :---: |
| `temperature_c` | Motor winding/bearing temperature | 45°C – 75°C | > 75.0°C |
| `vibration_mm_s` | Mechanical vibration velocity | 1.0 – 4.5 mm/s | > 4.5 mm/s |
| `voltage_v` | Line voltage | 220V – 240V | < 210V or > 250V |
| `current_a` | Motor operating current | 8A – 20A | > 22A |
| `power_kw` | Electrical power consumption ($P = \frac{V \times I}{1000}$) | 1.8 – 4.8 kW | N/A |
| `operating_hours`| Cumulative equipment run time | 50 – 5,000 hrs | > 4,000 hrs |
| `failure_risk_score` | Continuous wear metric (**Regression Target**) | 0.0 – 100.0 | > 65.0 |
| `failure_status` | Discrete health binary state (**Classification Target**) | 0 (Normal) | 1 (Fault) |
