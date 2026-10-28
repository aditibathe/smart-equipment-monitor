"""
Smart Equipment Monitoring & Predictive Maintenance - Main CLI Entry Point (main.py)
------------------------------------------------------------------------------------
Executes and demonstrates all 12 syllabus practicals sequentially:
  Practical 1:  NumPy arrays, indexing, slicing, reshaping & vectorized operations
  Practical 2:  NumPy matrix math (transpose, dot product, covariance) & statistics
  Practical 3:  Pandas dataset import, inspection & descriptive statistics
  Practical 4:  Missing-value handling (median imputation) & duplicate removal
  Practical 5:  Filtering, sorting & conditional selection
  Practical 6:  Grouping and aggregation using Pandas
  Practical 7:  Matplotlib line plots, bar charts & histograms
  Practical 8:  Scatter plots & correlation heatmaps
  Practical 9:  Descriptive statistics (Mean, Median, Variance, Std Dev) & correlation
  Practical 10: Scikit-learn linear regression for continuous-value prediction
  Practical 11: Scikit-learn classification for categorical prediction
  Practical 12: Integrated mini-project execution & Streamlit launch guide
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    FIGURES_DIR,
    THRESHOLDS,
)
from src.generator import generate_sensor_dataset, save_dataset
from src.data_loader import load_dataset, inspect_dataset
from src.preprocessor import (
    clean_and_restructure_data,
    filter_critical_equipment,
    summarize_equipment_health,
)
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
from src.models import (
    train_linear_regression,
    train_classifier,
    predict_single_record,
)


def run_all_practicals():
    """Sequentially executes and displays all 12 syllabus practicals."""
    print("*" * 80)
    print(" SMART EQUIPMENT MONITORING & PREDICTIVE MAINTENANCE")
    print(" Python Mini-Project: 12-Practical Syllabus Demonstration")
    print("*" * 80)

    # -------------------------------------------------------------
    # PRACTICAL 1: NumPy Arrays, Slicing, Reshaping & Vectorization
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 1] NumPy Arrays, Indexing, Slicing, Reshaping & Vectorization")
    print("=" * 80)
    # Ensure datasets exist or generate/clean them
    if not RAW_DATA_PATH.exists():
        print("Generating initial raw synthetic sensor dataset...")
        raw_df = generate_sensor_dataset(num_samples=1200)
        save_dataset(raw_df, filepath=RAW_DATA_PATH)
    else:
        raw_df = load_dataset(RAW_DATA_PATH)

    clean_df, clean_report = clean_and_restructure_data(raw_df, save_cleaned=True)

    p1_results = demonstrate_numpy_array_operations(clean_df)
    print(f"* 2D Feature Matrix Shape: {p1_results['original_shape']} (Dimensions: {p1_results['ndim']})")
    print(f"* 1D Slicing (First 5 Temperature readings): {p1_results['temp_slice_1d']}")
    print(f"* 2D Submatrix Slicing (First 3 rows of Temp & Vib):\n{p1_results['submatrix_2d']}")
    print(f"* Array Reshaping (Flattened 12 points into 3x4 Matrix):\n{p1_results['reshaped_3x4']}")
    print(f"* Vectorized Power Formula: P = (V * I) / 1000 computed across all rows simultaneously.")
    print(f"* Vectorized Vibration Anomalies (> 2-sigma): {p1_results['num_vectorized_anomalies']} readings")

    # -------------------------------------------------------------
    # PRACTICAL 2: NumPy Matrix Math & Statistical Operations
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 2] NumPy Matrix & Statistical Operations")
    print("=" * 80)
    p2_results = demonstrate_numpy_matrix_operations(clean_df)
    print(f"* Matrix Shapes: X = {p2_results['X_shape']} | Transpose X.T = {p2_results['X_T_shape']}")
    print(f"* Gram Matrix (X.T @ X):\n{p2_results['gram_matrix']}")
    print(f"* Covariance Matrix [Temp, Vib]:\n{p2_results['cov_matrix']}")
    print(f"* Column Statistics [Temp, Vib]:")
    print(f"    - Means:      {p2_results['col_means']}")
    print(f"    - Medians:    {p2_results['col_medians']}")
    print(f"    - Variances:  {p2_results['col_variances']}")
    print(f"    - Std Dev:    {p2_results['col_std']}")
    print(f"    - Peak-to-Peak Range (max - min): {p2_results['col_range']}")

    # -------------------------------------------------------------
    # PRACTICAL 3: Pandas Dataset Import, Inspection & Descriptive Stats
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 3] Pandas Dataset Import, Inspection & Descriptive Statistics")
    print("=" * 80)
    info = inspect_dataset(raw_df)
    print(f"* Loaded CSV: {RAW_DATA_PATH.name} ({info['num_rows']} rows, {info['num_columns']} columns)")
    print(f"* Detected Duplicates in Raw Data: {info['duplicate_rows']}")
    print(f"* Detected Missing Cells in Raw Data: {info['total_missing_cells']}")
    print("Columns & Missing Counts:")
    for col, count in info["missing_counts"].items():
        if count > 0:
            print(f"    - {col}: {count} missing (NaN)")
    print("\nDescriptive Statistics (Pandas .describe()):")
    print(info["descriptive_stats"][["temperature_c", "vibration_mm_s", "current_a", "failure_risk_score"]].to_string())

    # -------------------------------------------------------------
    # PRACTICAL 4: Missing-Value Handling, Duplicate Removal & Restructuring
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 4] Missing-Value Imputation, Duplicate Removal & Dataset Restructuring")
    print("=" * 80)
    clean_df, clean_report = clean_and_restructure_data(raw_df, save_cleaned=True)
    print(f"* Initial Rows: {clean_report['initial_rows']}")
    print(f"* Duplicates Dropped: {clean_report['duplicates_removed']}")
    print("* Median Imputation Summary:")
    for col, imp in clean_report["imputation_values"].items():
        print(f"    - {col}: Filled {imp['imputed_cells']} NaNs with median = {imp['median_value_used']}")
    print(f"* Dataset Restructuring: Parsed datetime, sorted chronologically, reset index.")
    print(f"* Final Clean Rows: {clean_report['final_rows']} (Remaining NaNs: {clean_report['remaining_missing_cells']})")
    print(f"* Cleaned Dataset Saved: {CLEAN_DATA_PATH.name}")

    # -------------------------------------------------------------
    # PRACTICAL 5: Filtering, Sorting and Conditional Selection
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 5] Filtering, Sorting and Conditional Selection")
    print("=" * 80)
    temp_cut = THRESHOLDS["temperature_c"]["normal_max"]
    vib_cut = THRESHOLDS["vibration_mm_s"]["normal_max"]
    print(f"Condition: Temperature > {temp_cut}°C OR Vibration > {vib_cut} mm/s")
    critical_records = filter_critical_equipment(
        clean_df, temp_threshold=temp_cut, vib_threshold=vib_cut, sort_by="vibration_mm_s", ascending=False
    )
    print(f"* Critical Sensor Anomalies Filtered: {len(critical_records)} records")
    print("Top 3 Highest Vibration Events:")
    print(critical_records[["equipment_id", "timestamp", "temperature_c", "vibration_mm_s", "equipment_status"]].head(3).to_string())

    # -------------------------------------------------------------
    # PRACTICAL 6: Grouping and Aggregation using Pandas
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 6] Grouping and Aggregation using Pandas")
    print("=" * 80)
    grouped_summary = summarize_equipment_health(clean_df)
    print("Equipment Aggregations Grouped by equipment_id:")
    print(grouped_summary.to_string())

    # -------------------------------------------------------------
    # PRACTICAL 7: Matplotlib Line Plots, Bar Charts & Histograms
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 7] Matplotlib Line Plots, Bar Charts and Histograms")
    print("=" * 80)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    f1 = FIGURES_DIR / "practical7_time_series.png"
    f2 = FIGURES_DIR / "practical7_bar_chart.png"
    f3 = FIGURES_DIR / "practical7_histogram.png"
    plot_sensor_time_series(clean_df, equipment_id="EQ_101", save_path=f1)
    plot_equipment_comparison_bar(grouped_summary, metric="max_vibration_mm_s", save_path=f2)
    plot_sensor_distribution_histogram(clean_df, column="temperature_c", save_path=f3)
    print(f"* [Saved] Line Plot (Time Series):      {f1}")
    print(f"* [Saved] Bar Chart (Equipment Health): {f2}")
    print(f"* [Saved] Histogram (Temperature Dist): {f3}")

    # -------------------------------------------------------------
    # PRACTICAL 8: Scatter Plots and Correlation Heatmaps (Seaborn)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 8] Scatter Plots & Correlation Heatmaps (Seaborn)")
    print("=" * 80)
    f4 = FIGURES_DIR / "practical8_scatter_plot.png"
    f5 = FIGURES_DIR / "practical8_correlation_heatmap.png"
    plot_vibration_vs_temperature_scatter(clean_df, save_path=f4)
    plot_correlation_heatmap(clean_df, save_path=f5)
    print(f"* [Saved] Scatter Plot (Vib vs Temp): {f4}")
    print(f"* [Saved] Heatmap (Sensor Pearson r): {f5}")

    # -------------------------------------------------------------
    # PRACTICAL 9: Mean, Median, Variance, Std Dev & Correlation
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 9] Mean, Median, Variance, Standard Deviation & Correlation")
    print("=" * 80)
    stats_table, corr_matrix = calculate_statistical_metrics(clean_df)
    print("Detailed Statistical Indicators:")
    print(stats_table.to_string())
    print("\nCorrelation with Continuous Risk Target (failure_risk_score):")
    print(corr_matrix["failure_risk_score"].sort_values(ascending=False).to_string())

    # -------------------------------------------------------------
    # PRACTICAL 10: Linear Regression for Continuous Prediction
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 10] Scikit-Learn Linear Regression (Continuous Risk Prediction)")
    print("=" * 80)
    reg_model, reg_metrics = train_linear_regression(clean_df)
    print(f"* Target: failure_risk_score (0.0 to 100.0) | Features: 6 physical sensor channels")
    print(f"* Train Samples: {reg_metrics['train_samples']} | Test Samples: {reg_metrics['test_samples']}")
    print(f"* Coefficients: {reg_metrics['coefficients']}")
    print(f"* Intercept: {reg_metrics['intercept']}")
    print(f"* Model Performance:")
    print(f"    - R-Squared (R2 Score):          {reg_metrics['test_r2']} (~{reg_metrics['test_r2']*100:.1f}% explained)")
    print(f"    - Mean Squared Error (MSE):      {reg_metrics['test_mse']}")
    print(f"    - Root Mean Squared Error (RMSE): {reg_metrics['test_rmse']}")
    print(f"    - Mean Absolute Error (MAE):     {reg_metrics['test_mae']}")

    # -------------------------------------------------------------
    # PRACTICAL 11: Scikit-Learn Classification (Categorical Prediction)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 11] Scikit-Learn Classification (Equipment Failure Prediction)")
    print("=" * 80)
    clf_model, clf_metrics = train_classifier(clean_df)
    print(f"* Target: failure_status (0: Normal, 1: Fault) | Model: Logistic Regression")
    print(f"* Accuracy:  {clf_metrics['accuracy'] * 100:.2f}%")
    print(f"* Precision: {clf_metrics['precision'] * 100:.2f}%")
    print(f"* Recall:    {clf_metrics['recall'] * 100:.2f}%")
    print(f"* F1-Score:  {clf_metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(clf_metrics["confusion_matrix"])

    # -------------------------------------------------------------
    # PRACTICAL 12: Integrated Mini-Project Execution
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PRACTICAL 12] Integrated Mini-Project: Real-Time Inference Demonstration")
    print("=" * 80)
    sample_inference = predict_single_record(
        regressor=reg_model,
        classifier=clf_model,
        operating_hours=3850.0,
        temperature_c=84.2,
        vibration_mm_s=6.8,
        voltage_v=226.5,
        current_a=18.4,
    )
    print("Simulated Field Sensor Telemetry:")
    print("  - Operating Hours: 3850.0 hrs | Temp: 84.2°C | Vibration: 6.8 mm/s | Voltage: 226.5V | Current: 18.4A")
    print(f"  - Real-time Power Computed: {sample_inference['power_kw']} kW")
    print(f"  - Predicted Risk Score:    {sample_inference['predicted_risk_score']} / 100.0")
    print(f"  - Machine Health State:    {sample_inference['status_label']}")
    print(f"  - Recommended Action:      {sample_inference['recommended_action']}")

    print("\n" + "*" * 80)
    print("[COMPLETED] All 12 syllabus practicals executed successfully!")
    print("To launch the interactive Web Dashboard, run:")
    print("  .\\.venv\\Scripts\\streamlit.exe run app.py")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    run_all_practicals()
