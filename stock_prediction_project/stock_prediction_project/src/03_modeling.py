"""
03_modeling.py
---------------
Two prediction tasks on the feature-engineered dataset:
  (A) Regression: predict next-day return
  (B) Classification: predict next-day direction (up/down)

Uses a TIME-BASED split (never shuffle time series data) and a naive
baseline for honest comparison, since price movement is notoriously hard
to beat and a portfolio project should be upfront about that.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA = "/home/claude/stock_prediction_project/data/features.csv"
PLOTS = "/home/claude/stock_prediction_project/plots"

df = pd.read_csv(DATA, parse_dates=["date"])
df = df.dropna().reset_index(drop=True)  # drop rows with NaN from rolling windows/lags

feature_cols = [
    "rsi_14", "macd", "macd_signal", "volatility_10", "volume_change",
    "momentum_5", "momentum_10", "return_lag_1", "return_lag_2",
    "return_lag_3", "return_lag_5", "close_to_sma20", "close_to_sma50",
]

X = df[feature_cols]
y_reg = df["target_return"]
y_clf = df["target_direction"]

# --- Time-based split: last 20% of dates held out as test set ------------
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
dates_test = df["date"].iloc[split_idx:]

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# ============================================================
# TASK A: Regression -- predict next-day return
# ============================================================
y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=15, random_state=42
    ),
}

# Naive baseline: predict tomorrow's return = 0 (i.e. "no change")
baseline_pred = np.zeros(len(y_reg_test))
baseline_rmse = mean_squared_error(y_reg_test, baseline_pred) ** 0.5

reg_results = {"Naive (predict 0)": {"rmse": round(baseline_rmse, 5), "mae": round(
    mean_absolute_error(y_reg_test, baseline_pred), 5), "r2": None}}

plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_reg_test.values, label="Actual next-day return", alpha=0.6)

for name, model in reg_models.items():
    model.fit(X_train_s, y_reg_train)
    pred = model.predict(X_test_s)
    rmse = mean_squared_error(y_reg_test, pred) ** 0.5
    mae = mean_absolute_error(y_reg_test, pred)
    r2 = r2_score(y_reg_test, pred)
    reg_results[name] = {"rmse": round(rmse, 5), "mae": round(mae, 5), "r2": round(r2, 4)}
    plt.plot(dates_test, pred, label=f"{name} prediction", alpha=0.8, linewidth=1)

plt.title("Next-Day Return: Actual vs. Predicted (Test Period)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/04_regression_actual_vs_predicted.png")
plt.close()

results["regression"] = reg_results

# ============================================================
# TASK B: Classification -- predict next-day direction
# ============================================================
y_clf_train, y_clf_test = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=15, random_state=42
    ),
}

# Naive baseline: always predict the majority class from training data
majority_class = y_clf_train.mode()[0]
baseline_acc = (y_clf_test == majority_class).mean()

clf_results = {"Naive (majority class)": {"accuracy": round(baseline_acc, 4)}}

plt.figure(figsize=(6, 5))
for name, model in clf_models.items():
    model.fit(X_train_s, y_clf_train)
    proba = model.predict_proba(X_test_s)[:, 1]
    preds = model.predict(X_test_s)

    acc = accuracy_score(y_clf_test, preds)
    auc = roc_auc_score(y_clf_test, proba)
    fpr, tpr, _ = roc_curve(y_clf_test, proba)

    clf_results[name] = {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
    }
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    if name == "Random Forest":
        rf_clf_preds = preds
        rf_clf_model = model

plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Direction Prediction — ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/05_classification_roc.png")
plt.close()

results["classification"] = clf_results

# Confusion matrix for Random Forest classifier
cm = confusion_matrix(y_clf_test, rf_clf_preds)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
plt.title("Random Forest — Direction Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{PLOTS}/06_classification_confusion_matrix.png")
plt.close()

# Feature importance
imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_clf_model.feature_importances_,
}).sort_values("importance", ascending=False)

plt.figure(figsize=(7, 6))
sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
plt.title("Feature Importance — Direction Classifier (Random Forest)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/07_feature_importance.png")
plt.close()

with open("/home/claude/stock_prediction_project/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSaved regression, ROC, confusion matrix, and feature importance plots.")
