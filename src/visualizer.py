"""
Data Visualization Module (visualizer.py)
-----------------------------------------
Covers Syllabus Practicals:
- Practical 7: Matplotlib line plots (time series), bar charts (equipment comparison), and histograms (distribution).
- Practical 8: Scatter plots (vibration vs temperature) and correlation heatmaps (Seaborn).

All functions return a Matplotlib Figure object so they can be rendered in both
standalone image files (reports/figures/) and the interactive Streamlit dashboard.
"""

import sys
from pathlib import Path
from typing import Optional
import matplotlib
# Use non-interactive backend for headless and server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Allow running directly as a script
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    FIGURES_DIR,
    CLEAN_DATA_PATH,
    THRESHOLDS,
    TIMESTAMP_COL,
    EQUIPMENT_ID_COL,
)
from src.data_loader import load_dataset
from src.preprocessor import summarize_equipment_health


# =====================================================================
# PRACTICAL 7: Line Plots, Bar Charts, and Histograms (Matplotlib)
# =====================================================================

def plot_sensor_time_series(
    df: pd.DataFrame,
    equipment_id: str = "EQ_101",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots a dual-axis time-series line chart for a specific machine:
    - Left Y-Axis: Temperature (°C)
    - Right Y-Axis: Vibration (mm/s)
    - Threshold alarm lines indicating critical operating boundaries.

    Demonstrates Practical 7 (Matplotlib Line Plots).
    """
    machine_df = df[df[EQUIPMENT_ID_COL] == equipment_id].copy()
    if machine_df.empty:
        raise ValueError(f"No records found for equipment_id '{equipment_id}'.")

    machine_df[TIMESTAMP_COL] = pd.to_datetime(machine_df[TIMESTAMP_COL])
    machine_df = machine_df.sort_values(by=TIMESTAMP_COL)

    fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=100)

    # Left Axis: Temperature
    color_temp = "#d9534f"  # Red
    ax1.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Temperature (°C)", color=color_temp, fontsize=11, fontweight="bold")
    line1 = ax1.plot(
        machine_df[TIMESTAMP_COL],
        machine_df["temperature_c"],
        color=color_temp,
        linewidth=1.8,
        label="Temperature (°C)",
    )
    # Critical Temperature Limit Line
    temp_crit = THRESHOLDS["temperature_c"]["normal_max"]
    ax1.axhline(temp_crit, color="#b52b27", linestyle="--", linewidth=1.2, label=f"Temp Limit ({temp_crit}°C)")
    ax1.tick_params(axis="y", labelcolor=color_temp)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right Axis: Vibration
    ax2 = ax1.twinx()
    color_vib = "#0275d8"  # Blue
    ax2.set_ylabel("Vibration (mm/s)", color=color_vib, fontsize=11, fontweight="bold")
    line2 = ax2.plot(
        machine_df[TIMESTAMP_COL],
        machine_df["vibration_mm_s"],
        color=color_vib,
        linewidth=1.8,
        alpha=0.85,
        label="Vibration (mm/s)",
    )
    # Critical Vibration Limit Line
    vib_crit = THRESHOLDS["vibration_mm_s"]["normal_max"]
    ax2.axhline(vib_crit, color="#014c8c", linestyle=":", linewidth=1.2, label=f"Vib Limit ({vib_crit} mm/s)")
    ax2.tick_params(axis="y", labelcolor=color_vib)

    plt.title(f"Time-Series Sensor Telemetry - {equipment_id} (Practical 7)", fontsize=12, fontweight="bold", pad=12)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_equipment_comparison_bar(
    summary_df: pd.DataFrame,
    metric: str = "max_vibration_mm_s",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots a categorical bar chart comparing equipment metrics:
    - Demonstrates Practical 7 (Matplotlib Bar Charts).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    machines = summary_df.index.tolist()
    values = summary_df[metric].tolist()

    metric_labels = {
        "max_vibration_mm_s": "Max Vibration Velocity (mm/s)",
        "mean_temperature_c": "Mean Operating Temperature (°C)",
        "max_temperature_c": "Peak Operating Temperature (°C)",
        "mean_power_kw": "Mean Power Consumption (kW)",
        "total_fault_alerts": "Total Fault Alerts Count",
    }
    y_label = metric_labels.get(metric, metric)

    # Color bars with highlight for the highest value
    colors = ["#4575b4" if v < max(values) else "#d73027" for v in values]

    bars = ax.bar(machines, values, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    # Add numeric labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Equipment ID", fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
    ax.set_title(f"Equipment Comparison: {y_label} (Practical 7)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_sensor_distribution_histogram(
    df: pd.DataFrame,
    column: str = "temperature_c",
    bins: int = 25,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots a distribution histogram with mean and median markers:
    - Demonstrates Practical 7 (Matplotlib Histograms).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    data = df[column].dropna()
    mean_val = data.mean()
    median_val = data.median()

    # Histogram
    n, bin_edges, patches = ax.hist(
        data,
        bins=bins,
        color="#3182bd",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        density=False,
    )

    # Vertical markers for Mean and Median (Linking with Practical 9)
    ax.axvline(mean_val, color="#de2d26", linestyle="--", linewidth=2.0, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="#31a354", linestyle="-.", linewidth=2.0, label=f"Median: {median_val:.2f}")

    unit_map = {"temperature_c": "°C", "vibration_mm_s": "mm/s", "voltage_v": "V", "current_a": "A"}
    unit = unit_map.get(column, "")

    ax.set_xlabel(f"{column} ({unit})", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency (Counts)", fontsize=11, fontweight="bold")
    ax.set_title(f"Distribution Histogram of {column} (Practical 7)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


# =====================================================================
# PRACTICAL 8: Scatter Plots and Correlation Heatmap (Seaborn)
# =====================================================================

def plot_vibration_vs_temperature_scatter(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generates a bivariate scatter plot of Vibration vs. Temperature:
    - Color-coded by machine status ('Normal', 'Warning', 'Fault')
    - Includes horizontal and vertical operational threshold boundaries
    - Demonstrates Practical 8 (Scatter Plots).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=100)

    palette = {"Normal": "#2ca02c", "Warning": "#ff7f0e", "Fault": "#d62728"}

    sns.scatterplot(
        data=df,
        x="temperature_c",
        y="vibration_mm_s",
        hue="equipment_status",
        palette=palette,
        style="equipment_status",
        s=65,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
        ax=ax,
    )

    # Critical boundary lines
    temp_crit = THRESHOLDS["temperature_c"]["normal_max"]
    vib_crit = THRESHOLDS["vibration_mm_s"]["normal_max"]
    ax.axvline(temp_crit, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Temp Limit ({temp_crit}°C)")
    ax.axhline(vib_crit, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label=f"Vib Limit ({vib_crit} mm/s)")

    ax.set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Vibration Velocity (mm/s)", fontsize=11, fontweight="bold")
    ax.set_title("Scatter Plot: Vibration vs. Temperature (Practical 8)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", title="Status", frameon=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generates an annotated correlation heatmap using Seaborn:
    - Visualizes Pearson correlation across sensor channels and risk targets
    - Demonstrates Practical 8 (Correlation Heatmaps).
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=100)

    # Focus on key physical sensor features and targets
    cols = [
        "operating_hours",
        "temperature_c",
        "vibration_mm_s",
        "voltage_v",
        "current_a",
        "power_kw",
        "failure_risk_score",
        "failure_status",
    ]
    numeric_subset = df[cols]
    corr_matrix = numeric_subset.corr(method="pearson")

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.8,
        cbar_kws={"label": "Pearson Correlation Coefficient (r)"},
        square=True,
        ax=ax,
    )

    ax.set_title("Sensor Feature Correlation Heatmap (Practical 8)", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


# =====================================================================
# Academic Demonstration Driver
# =====================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("DEMONSTRATION: PRACTICALS 7 AND 8 (DATA VISUALIZATION)")
    print("=" * 75)

    df_clean = load_dataset(CLEAN_DATA_PATH)
    summary_data = summarize_equipment_health(df_clean)

    # Ensure figures output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Line Plot (Practical 7)
    line_file = FIGURES_DIR / "practical7_time_series.png"
    plot_sensor_time_series(df_clean, equipment_id="EQ_101", save_path=line_file)
    print(f"* Practical 7 Line Plot saved to: {line_file}")

    # 2. Bar Chart (Practical 7)
    bar_file = FIGURES_DIR / "practical7_bar_chart.png"
    plot_equipment_comparison_bar(summary_data, metric="max_vibration_mm_s", save_path=bar_file)
    print(f"* Practical 7 Bar Chart saved to: {bar_file}")

    # 3. Histogram (Practical 7)
    hist_file = FIGURES_DIR / "practical7_histogram.png"
    plot_sensor_distribution_histogram(df_clean, column="temperature_c", save_path=hist_file)
    print(f"* Practical 7 Histogram saved to: {hist_file}")

    # 4. Scatter Plot (Practical 8)
    scatter_file = FIGURES_DIR / "practical8_scatter_plot.png"
    plot_vibration_vs_temperature_scatter(df_clean, save_path=scatter_file)
    print(f"* Practical 8 Scatter Plot saved to: {scatter_file}")

    # 5. Correlation Heatmap (Practical 8)
    heatmap_file = FIGURES_DIR / "practical8_correlation_heatmap.png"
    plot_correlation_heatmap(df_clean, save_path=heatmap_file)
    print(f"* Practical 8 Correlation Heatmap saved to: {heatmap_file}")

    print("=" * 75)
    print("[SUCCESS] All 5 figures generated and saved successfully!")
    print("=" * 75)
