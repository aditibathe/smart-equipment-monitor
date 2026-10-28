"""
Smart Equipment Monitoring & Predictive Maintenance Dashboard (app.py)
------------------------------------------------------------------------
Practical 12: Integrated End-to-End Application
An interactive Streamlit web dashboard connecting all 12 syllabus practicals:
- Data ingestion, cleaning, restructuring, filtering & grouping (Practicals 3-6)
- NumPy array mechanics, matrix math & statistical analytics (Practicals 1, 2, 9)
- Diagnostic visualizations (Practicals 7, 8)
- Real-time Linear Regression & Classification predictive maintenance (Practicals 10, 11, 12)
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    EQUIPMENT_IDS,
    THRESHOLDS,
    PRACTICALS,
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

# Streamlit Page Configuration
st.set_page_config(
    page_title="Smart Equipment Monitor & Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_all_data():
    """Loads or generates raw and cleaned datasets."""
    if not RAW_DATA_PATH.exists():
        raw_df = generate_sensor_dataset(num_samples=1200)
        save_dataset(raw_df, filepath=RAW_DATA_PATH)
    else:
        raw_df = load_dataset(RAW_DATA_PATH)

    clean_df, clean_report = clean_and_restructure_data(raw_df, save_cleaned=True)
    summary_df = summarize_equipment_health(clean_df)
    return raw_df, clean_df, clean_report, summary_df


@st.cache_resource
def load_trained_models(clean_df):
    """Trains and caches the ML models."""
    reg_model, reg_metrics = train_linear_regression(clean_df)
    clf_model, clf_metrics = train_classifier(clean_df)
    return reg_model, reg_metrics, clf_model, clf_metrics


# Ingest Data and Train Models
raw_data, clean_data, cleaning_report, eq_summary = load_all_data()
reg_model, reg_metrics, clf_model, clf_metrics = load_trained_models(clean_data)


# =====================================================================
# Sidebar Navigation & Practical Index
# =====================================================================
st.sidebar.title("⚙️ Equipment Monitor")
st.sidebar.markdown("**Predictive Maintenance System**")
st.sidebar.caption("ECE Student Lab Mini-Project (12 Practicals)")

nav_choice = st.sidebar.radio(
    "Go to Module:",
    [
        "1. Overview & Health KPI",
        "2. Dataset & Preprocessing (Prac 3, 4, 5, 6)",
        "3. NumPy & Matrix Math (Prac 1, 2, 9)",
        "4. Visualizations & Telemetry (Prac 7, 8)",
        "5. ML Models & Live Predictor (Prac 10, 11, 12)",
    ],
)

with st.sidebar.expander("📚 Syllabus Practicals Index (1-12)"):
    for num, title in PRACTICALS.items():
        st.write(f"**Prac {num}:** {title}")


# =====================================================================
# MODULE 1: Overview & Health KPI
# =====================================================================
if nav_choice == "1. Overview & Health KPI":
    st.title("🏭 Smart Equipment Monitoring & Predictive Maintenance")
    st.markdown(
        "A full-stack industrial telemetry monitoring system designed to prevent machine breakdown "
        "through real-time sensor analytics and Scikit-learn predictive maintenance models."
    )

    # Top KPI Metrics
    total_records = len(clean_data)
    total_machines = len(EQUIPMENT_IDS)
    total_faults = int(clean_data["failure_status"].sum())
    avg_temp = round(clean_data["temperature_c"].mean(), 1)
    avg_vib = round(clean_data["vibration_mm_s"].mean(), 2)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Active Machines", f"{total_machines}")
    kpi2.metric("Telemetry Records", f"{total_records:,}")
    kpi3.metric("Avg Operating Temp", f"{avg_temp} °C")
    kpi4.metric("Avg Vibration", f"{avg_vib} mm/s")
    kpi5.metric("Detected Fault Events", f"{total_faults}", delta=f"{total_faults} Critical", delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Equipment Fleet Health Summary (Practical 6)")
        st.dataframe(eq_summary, use_container_width=True)

    with col_right:
        st.subheader("Core Project Architecture")
        st.info(
            "• **Data Pipeline:** Ingests raw telemetry, cleans missing records via median imputation, drops duplicates, sorts chronologically.\n\n"
            "• **Analytics Engine:** Vectorized operations, array statistics, and Pearson correlation coefficients.\n\n"
            "• **Machine Learning:** Continuous failure risk prediction ($R^2 > 0.85$) and binary fault classification ($> 90\\%$ accuracy)."
        )


# =====================================================================
# MODULE 2: Dataset & Preprocessing (Practicals 3, 4, 5, 6)
# =====================================================================
elif nav_choice == "2. Dataset & Preprocessing (Prac 3, 4, 5, 6)":
    st.title("📊 Dataset Management, Cleaning & Filtering")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Practical 3: Import & Inspection",
        "Practical 4: Cleaning & Imputation",
        "Practical 5: Filtering & Thresholds",
        "Practical 6: Grouping & Aggregations",
    ])

    with tab1:
        st.subheader("Practical 3: Dataset Import & Descriptive Statistics")
        raw_inspect = inspect_dataset(raw_data)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows", raw_inspect["num_rows"])
        c2.metric("Features", raw_inspect["num_columns"])
        c3.metric("Missing Cells (NaN)", raw_inspect["total_missing_cells"])
        c4.metric("Duplicate Rows", raw_inspect["duplicate_rows"])

        st.markdown("**Sample Raw Sensor Records:**")
        st.dataframe(raw_data.head(10), use_container_width=True)

        st.markdown("**Descriptive Statistics:**")
        st.dataframe(raw_inspect["descriptive_stats"], use_container_width=True)

    with tab2:
        st.subheader("Practical 4: Missing-Value Handling & Duplicate Removal")
        st.success(
            f"✓ Removed **{cleaning_report['duplicates_removed']}** duplicate records.\n\n"
            f"✓ Used **Median Imputation** for sensor channels to resist transient outlier spikes.\n\n"
            f"✓ Restructured timestamp into datetime objects and sorted records chronologically per machine."
        )
        st.markdown("**Imputation Details:**")
        for col, imp in cleaning_report["imputation_values"].items():
            st.write(f"- `{col}`: Imputed **{imp['imputed_cells']}** missing values with median = **{imp['median_value_used']}**")

        st.markdown("**Cleaned Production Dataset:**")
        st.dataframe(clean_data.head(10), use_container_width=True)

    with tab3:
        st.subheader("Practical 5: Filtering, Sorting & Conditional Selection")
        col_thresh1, col_thresh2 = st.columns(2)
        with col_thresh1:
            temp_filter = st.slider("Temperature Threshold (°C):", 50.0, 95.0, 75.0, step=1.0)
        with col_thresh2:
            vib_filter = st.slider("Vibration Threshold (mm/s):", 2.0, 10.0, 4.5, step=0.5)

        filtered = filter_critical_equipment(
            clean_data, temp_threshold=temp_filter, vib_threshold=vib_filter, sort_by="vibration_mm_s", ascending=False
        )
        st.write(f"Found **{len(filtered)}** records exceeding either condition:")
        st.dataframe(filtered, use_container_width=True)

    with tab4:
        st.subheader("Practical 6: Grouping & Aggregation by Equipment ID")
        st.dataframe(eq_summary, use_container_width=True)


# =====================================================================
# MODULE 3: NumPy & Matrix Math (Practicals 1, 2, 9)
# =====================================================================
elif nav_choice == "3. NumPy & Matrix Math (Prac 1, 2, 9)":
    st.title("🧮 Numerical Computing, Matrix Math & Statistics")

    tab_np1, tab_np2, tab_stat = st.tabs([
        "Practical 1: NumPy Arrays & Vectorization",
        "Practical 2: Matrix Math & Operations",
        "Practical 9: Descriptive Statistics & Correlation",
    ])

    with tab_np1:
        st.subheader("Practical 1: NumPy Arrays, Slicing, Reshaping & Vectorization")
        np_res = demonstrate_numpy_array_operations(clean_data)
        st.write(f"• **2D Matrix Shape:** `{np_res['original_shape']}` (Rank: `{np_res['ndim']}`, Dtype: `{np_res['dtype']}`)")
        st.write("• **1D Slicing (First 5 Temperature readings):**")
        st.code(str(np_res["temp_slice_1d"]))
        st.write("• **2D Slicing (First 3 rows of Temp & Vibration):**")
        st.code(str(np_res["submatrix_2d"]))
        st.write("• **Reshaping (12 values into 3x4 Matrix):**")
        st.code(str(np_res["reshaped_3x4"]))
        st.write("• **Vectorized Standard Scaling (Z-Score Sample):**")
        st.code(str(np_res["standardized_sample"]))

    with tab_np2:
        st.subheader("Practical 2: Matrix Multiplication, Transpose & Covariance")
        m_res = demonstrate_numpy_matrix_operations(clean_data)
        st.write(f"• **Matrix Shapes:** $X = {m_res['X_shape']}$, Transpose $X^T = {m_res['X_T_shape']}$")
        st.write("• **Gram Matrix ($X^T X$):**")
        st.code(str(m_res["gram_matrix"]))
        st.write("• **Bivariate Covariance Matrix [Temperature, Vibration]:**")
        st.code(str(m_res["cov_matrix"]))
        st.write(f"• **Column Means:** {m_res['col_means']}")
        st.write(f"• **Column Variances:** {m_res['col_variances']}")
        st.write(f"• **Peak-to-Peak Range:** {m_res['col_range']}")

    with tab_stat:
        st.subheader("Practical 9: Comprehensive Statistical Metrics & Pearson Correlation")
        stats_df, corr_df = calculate_statistical_metrics(clean_data)
        st.markdown("**Statistical Indicators (Mean, Median, Variance, Std Dev, Min, Max, IQR):**")
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("**Pearson Correlation Matrix:**")
        st.dataframe(corr_df, use_container_width=True)


# =====================================================================
# MODULE 4: Visualizations & Telemetry (Practicals 7, 8)
# =====================================================================
elif nav_choice == "4. Visualizations & Telemetry (Prac 7, 8)":
    st.title("📈 Sensor Visualizations & Diagnostic Telemetry")

    tab_vis1, tab_vis2 = st.tabs(["Practical 7: Matplotlib Plots", "Practical 8: Seaborn Heatmap & Scatter"])

    with tab_vis1:
        st.subheader("Practical 7: Time-Series, Bar Charts & Histograms")
        sel_machine = st.selectbox("Select Equipment for Time-Series Line Plot:", EQUIPMENT_IDS, index=0)
        fig_ts = plot_sensor_time_series(clean_data, equipment_id=sel_machine)
        st.pyplot(fig_ts)

        col_b, col_h = st.columns(2)
        with col_b:
            st.markdown("**Equipment Comparison Bar Chart:**")
            fig_bar = plot_equipment_comparison_bar(eq_summary, metric="max_vibration_mm_s")
            st.pyplot(fig_bar)

        with col_h:
            st.markdown("**Sensor Distribution Histogram:**")
            fig_hist = plot_sensor_distribution_histogram(clean_data, column="temperature_c")
            st.pyplot(fig_hist)

    with tab_vis2:
        st.subheader("Practical 8: Bivariate Scatter & Correlation Heatmap")
        col_s, col_m = st.columns(2)
        with col_s:
            st.markdown("**Vibration vs. Temperature Scatter Plot:**")
            fig_scat = plot_vibration_vs_temperature_scatter(clean_data)
            st.pyplot(fig_scat)

        with col_m:
            st.markdown("**Sensor Correlation Heatmap (Seaborn):**")
            fig_heat = plot_correlation_heatmap(clean_data)
            st.pyplot(fig_heat)


# =====================================================================
# MODULE 5: ML Models & Live Predictor (Practicals 10, 11, 12)
# =====================================================================
elif nav_choice == "5. ML Models & Live Predictor (Prac 10, 11, 12)":
    st.title("🤖 Machine Learning & Live Equipment Diagnostic Predictor")

    st.subheader("Model Evaluation Summary")
    c_m1, c_m2 = st.columns(2)

    with c_m1:
        st.markdown("#### Practical 10: Linear Regression (Continuous Target)")
        st.write(f"• **Test $R^2$ Score:** `{reg_metrics['test_r2']}`")
        st.write(f"• **Mean Squared Error (MSE):** `{reg_metrics['test_mse']}`")
        st.write(f"• **Root Mean Squared Error (RMSE):** `{reg_metrics['test_rmse']}`")
        st.write(f"• **Mean Absolute Error (MAE):** `{reg_metrics['test_mae']}`")
        st.caption("Target: failure_risk_score (0.0 to 100.0) | No target leakage")

    with c_m2:
        st.markdown("#### Practical 11: Classification (Categorical Target)")
        st.write(f"• **Accuracy:** `{clf_metrics['accuracy'] * 100:.2f}%`")
        st.write(f"• **Precision:** `{clf_metrics['precision'] * 100:.2f}%`")
        st.write(f"• **Recall:** `{clf_metrics['recall'] * 100:.2f}%`")
        st.write(f"• **F1-Score:** `{clf_metrics['f1_score']:.4f}`")
        st.caption("Target: failure_status (0: Normal, 1: Fault) | Model: Logistic Regression")

    st.divider()

    # Practical 12: Interactive Real-Time Prediction
    st.subheader("Practical 12: Real-Time Field Telemetry Predictor")
    st.markdown("Simulate sensor telemetry on a physical machine and evaluate health status:")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        in_hours = st.slider("Operating Hours (hrs):", 50.0, 5000.0, 2400.0, step=50.0)
        in_temp = st.slider("Temperature (°C):", 35.0, 105.0, 68.0, step=0.5)

    with col_in2:
        in_vib = st.slider("Vibration Velocity (mm/s):", 0.5, 12.0, 3.2, step=0.1)
        in_volt = st.slider("Electrical Voltage (V):", 200.0, 260.0, 230.0, step=1.0)

    with col_in3:
        in_curr = st.slider("Electrical Current (A):", 5.0, 30.0, 15.0, step=0.5)
        computed_pwr = round((in_volt * in_curr) / 1000.0, 3)
        st.metric("Computed Electrical Power", f"{computed_pwr} kW")

    # Run Prediction
    prediction = predict_single_record(
        regressor=reg_model,
        classifier=clf_model,
        operating_hours=in_hours,
        temperature_c=in_temp,
        vibration_mm_s=in_vib,
        voltage_v=in_volt,
        current_a=in_curr,
    )

    st.markdown("### Equipment Health Diagnosis")
    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
        st.metric("Predicted Failure Risk Score", f"{prediction['predicted_risk_score']} / 100.0")
        st.progress(prediction["predicted_risk_score"] / 100.0)
        st.caption(f"Fault Probability: {prediction['failure_probability']}%")

    with res_col2:
        if prediction["badge_color"] == "red":
            st.error(f"🚨 **Status: {prediction['status_label']}**")
        elif prediction["badge_color"] == "orange":
            st.warning(f"⚠️ **Status: {prediction['status_label']}**")
        else:
            st.success(f"✅ **Status: {prediction['status_label']}**")

        st.markdown(f"**Recommended Action:** {prediction['recommended_action']}")
