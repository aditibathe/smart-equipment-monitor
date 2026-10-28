"""
Machine Learning Models Module (models.py)
------------------------------------------
Covers Syllabus Practicals:
- Practical 10: Scikit-learn Linear Regression for continuous risk score prediction.
- Practical 11: Scikit-learn Classification (Logistic Regression) for equipment failure prediction.
- Practical 12 (Partial): Reusable predictive maintenance model pipeline with no target leakage.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Allow running directly as a script
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CLEAN_DATA_PATH,
    SENSOR_COLUMNS,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    RANDOM_SEED,
)
from src.data_loader import load_dataset


# =====================================================================
# PRACTICAL 10: Linear Regression for Continuous-Value Prediction
# =====================================================================

def train_linear_regression(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[LinearRegression, Dict[str, Any]]:
    """
    Trains a Linear Regression model to predict continuous equipment failure_risk_score.

    Ensures NO Target Leakage:
    - Features (X) only include physical sensor readings:
      [operating_hours, temperature_c, vibration_mm_s, voltage_v, current_a, power_kw]
    - Target (y) is the continuous failure_risk_score (0.0 to 100.0).

    Demonstrates Practical 10.

    Returns:
        Tuple[LinearRegression, Dict[str, Any]]: Trained model and evaluation metrics.
    """
    X = df[SENSOR_COLUMNS]
    y = df[REGRESSION_TARGET]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Instantiate and fit Linear Regression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predictions
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)

    # Evaluation Metrics (Practical 10)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mse_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    # Model coefficients
    coefficients = dict(zip(SENSOR_COLUMNS, [round(c, 4) for c in regressor.coef_]))

    metrics = {
        "model_type": "Linear Regression",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "intercept": round(float(regressor.intercept_), 4),
        "coefficients": coefficients,
        "train_r2": round(float(r2_train), 4),
        "test_r2": round(float(r2_test), 4),
        "test_mse": round(float(mse_test), 4),
        "test_rmse": round(rmse_test, 4),
        "test_mae": round(float(mae_test), 4),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
    }

    return regressor, metrics


# =====================================================================
# PRACTICAL 11: Classification for Categorical Prediction
# =====================================================================

def train_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Trains a Logistic Regression classifier to predict equipment failure status:
    - 0: Normal / Safe
    - 1: Faulty / High Risk

    Ensures NO Target Leakage:
    - Features (X) only include physical sensor readings.
    - Target (y) is failure_status (discrete categorical state).

    Demonstrates Practical 11.

    Returns:
        Tuple[LogisticRegression, Dict[str, Any]]: Trained classifier and evaluation metrics.
    """
    X = df[SENSOR_COLUMNS]
    y = df[CLASSIFICATION_TARGET]

    # Stratified split to maintain class balance in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Instantiate and fit Logistic Regression
    classifier = LogisticRegression(max_iter=1000, random_state=random_state)
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    y_prob_test = classifier.predict_proba(X_test)[:, 1]

    # Evaluation Metrics (Practical 11)
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)
    report_text = classification_report(
        y_test, y_pred_test, target_names=["Normal (0)", "Faulty (1)"], zero_division=0
    )

    metrics = {
        "model_type": "Logistic Regression",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "classification_report": report_text,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "y_prob_test": y_prob_test,
    }

    return classifier, metrics


# =====================================================================
# Integrated Prediction Helper (For Dashboard and Real-Time Use)
# =====================================================================

def predict_single_record(
    regressor: LinearRegression,
    classifier: LogisticRegression,
    operating_hours: float,
    temperature_c: float,
    vibration_mm_s: float,
    voltage_v: float,
    current_a: float,
) -> Dict[str, Any]:
    """
    Performs real-time inference on a single sensor reading:
    - Calculates electrical power P = (V * I) / 1000 (Practical 1)
    - Predicts continuous risk score (Practical 10)
    - Predicts categorical failure status & probability (Practical 11)

    Returns:
        Dict[str, Any]: Consolidated predictions and health status recommendation.
    """
    # 1. Compute electrical power
    power_kw = round((voltage_v * current_a) / 1000.0, 3)

    # 2. Build feature vector in exact training column order
    input_vector = pd.DataFrame(
        [
            {
                "operating_hours": operating_hours,
                "temperature_c": temperature_c,
                "vibration_mm_s": vibration_mm_s,
                "voltage_v": voltage_v,
                "current_a": current_a,
                "power_kw": power_kw,
            }
        ]
    )

    # 3. Continuous Prediction (Linear Regression)
    risk_score = float(regressor.predict(input_vector)[0])
    risk_score = round(max(0.0, min(100.0, risk_score)), 2)

    # 4. Categorical Prediction (Classifier)
    failure_class = int(classifier.predict(input_vector)[0])
    failure_prob = float(classifier.predict_proba(input_vector)[0][1])

    # 5. Status Label & Maintenance Recommendation
    if risk_score >= 65.0 or failure_class == 1:
        status_label = "CRITICAL / FAULT"
        action = "Schedule Immediate Overhaul (High wear/vibration detected)"
        badge_color = "red"
    elif risk_score >= 45.0:
        status_label = "WARNING"
        action = "Increase Inspection Frequency (Approaching thermal/vibration limits)"
        badge_color = "orange"
    else:
        status_label = "NORMAL"
        action = "Equipment Operating Normally (Continue scheduled operation)"
        badge_color = "green"

    return {
        "power_kw": power_kw,
        "predicted_risk_score": risk_score,
        "predicted_failure_class": failure_class,
        "failure_probability": round(failure_prob * 100.0, 2),
        "status_label": status_label,
        "recommended_action": action,
        "badge_color": badge_color,
    }


# =====================================================================
# Academic Demonstration Driver
# =====================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("DEMONSTRATION: PRACTICALS 10 AND 11 (MACHINE LEARNING MODELS)")
    print("=" * 75)

    df_clean = load_dataset(CLEAN_DATA_PATH)

    # 1. Linear Regression Demonstration (Practical 10)
    print("--- Practical 10: Scikit-learn Linear Regression (Continuous Prediction) ---")
    reg_model, reg_metrics = train_linear_regression(df_clean)
    print(f"* Model: {reg_metrics['model_type']}")
    print(f"* Train Samples: {reg_metrics['train_samples']} | Test Samples: {reg_metrics['test_samples']}")
    print(f"* Coefficients: {reg_metrics['coefficients']}")
    print(f"* Intercept: {reg_metrics['intercept']}")
    print(f"* Evaluation Metrics on Test Set:")
    print(f"    - R-Squared Score (R2):       {reg_metrics['test_r2']} (Explains ~{reg_metrics['test_r2']*100:.1f}% variance)")
    print(f"    - Mean Squared Error (MSE):    {reg_metrics['test_mse']}")
    print(f"    - Root Mean Squared Error (RMSE): {reg_metrics['test_rmse']}")
    print(f"    - Mean Absolute Error (MAE):   {reg_metrics['test_mae']}")

    # 2. Classification Demonstration (Practical 11)
    print("\n--- Practical 11: Scikit-learn Classification (Categorical Prediction) ---")
    clf_model, clf_metrics = train_classifier(df_clean)
    print(f"* Model: {clf_metrics['model_type']}")
    print(f"* Evaluation Metrics on Test Set:")
    print(f"    - Accuracy:  {clf_metrics['accuracy'] * 100:.2f}%")
    print(f"    - Precision: {clf_metrics['precision'] * 100:.2f}%")
    print(f"    - Recall:    {clf_metrics['recall'] * 100:.2f}%")
    print(f"    - F1-Score:  {clf_metrics['f1_score']:.4f}")
    print("\n* Confusion Matrix:")
    print(clf_metrics["confusion_matrix"])
    print("\n* Classification Report:")
    print(clf_metrics["classification_report"])

    # 3. Real-Time Inference Demo
    print("\n--- Real-Time Equipment Status Inference Demo ---")
    test_reading = predict_single_record(
        regressor=reg_model,
        classifier=clf_model,
        operating_hours=4200.0,
        temperature_c=86.5,
        vibration_mm_s=8.2,
        voltage_v=228.0,
        current_a=19.5,
    )
    print(f"* Input: Hours=4200h, Temp=86.5°C, Vib=8.2 mm/s, V=228V, I=19.5A")
    print(f"* Computed Power: {test_reading['power_kw']} kW")
    print(f"* Predicted Failure Risk Score: {test_reading['predicted_risk_score']} / 100.0")
    print(f"* Predicted Health Status: {test_reading['status_label']}")
    print(f"* Recommended Action: {test_reading['recommended_action']}")
    print("=" * 75)
