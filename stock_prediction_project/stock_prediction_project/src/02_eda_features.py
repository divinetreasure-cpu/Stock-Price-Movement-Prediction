"""
02_eda_features.py
--------------------
Exploratory analysis of the price series, plus construction of standard
technical-indicator features used as model inputs: moving averages, RSI,
MACD, rolling volatility, and lagged returns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA = "/home/claude/stock_prediction_project/data/stock_prices.csv"
PLOTS = "/home/claude/stock_prediction_project/plots"

df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# --- Daily return ------------------------------------------------------
df["return"] = df["close"].pct_change()

print("=" * 60)
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Rows: {len(df)}")
print(f"Annualized volatility: {df['return'].std() * np.sqrt(252):.2%}")
print(f"Total return over period: {df['close'].iloc[-1] / df['close'].iloc[0] - 1:.2%}")
print("=" * 60)

# --- 1. Price with moving averages --------------------------------------
df["sma_20"] = df["close"].rolling(20).mean()
df["sma_50"] = df["close"].rolling(50).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["close"], label="Close", linewidth=1)
plt.plot(df["date"], df["sma_20"], label="SMA 20", linewidth=1)
plt.plot(df["date"], df["sma_50"], label="SMA 50", linewidth=1)
plt.title("Price History with Moving Averages")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_price_history.png")
plt.close()

# --- 2. Return distribution & volatility clustering ----------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df["return"].dropna(), bins=60, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Daily Return Distribution")

df["rolling_vol_20"] = df["return"].rolling(20).std() * np.sqrt(252)
axes[1].plot(df["date"], df["rolling_vol_20"], color="#C44E52")
axes[1].set_title("Rolling 20-Day Annualized Volatility")
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_returns_and_volatility.png")
plt.close()

# --- Technical indicator feature engineering -----------------------------
def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["rsi_14"] = rsi(df["close"], 14)

ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

df["volatility_10"] = df["return"].rolling(10).std()
df["volume_change"] = df["volume"].pct_change()
df["momentum_5"] = df["close"].pct_change(5)
df["momentum_10"] = df["close"].pct_change(10)

# Lagged returns (yesterday's, 2-days-ago, etc. -- avoids look-ahead bias)
for lag in [1, 2, 3, 5]:
    df[f"return_lag_{lag}"] = df["return"].shift(lag)

# Price relative to moving averages (normalized, model-friendly)
df["close_to_sma20"] = df["close"] / df["sma_20"] - 1
df["close_to_sma50"] = df["close"] / df["sma_50"] - 1

# --- Targets ---------------------------------------------------------
# Regression target: next day's return
df["target_return"] = df["return"].shift(-1)
# Classification target: will price go up tomorrow?
df["target_direction"] = (df["target_return"] > 0).astype(int)

# --- 3. Correlation of features with next-day return ---------------------
feature_cols = [
    "rsi_14", "macd", "macd_signal", "volatility_10", "volume_change",
    "momentum_5", "momentum_10", "return_lag_1", "return_lag_2",
    "return_lag_3", "return_lag_5", "close_to_sma20", "close_to_sma50",
]
corr_with_target = df[feature_cols + ["target_return"]].corr()["target_return"].drop("target_return")
plt.figure(figsize=(7, 5))
corr_with_target.sort_values().plot(kind="barh", color="#55A868")
plt.title("Feature Correlation with Next-Day Return")
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_feature_correlation.png")
plt.close()

# Save the feature-engineered dataset for the modeling step
out_path = "/home/claude/stock_prediction_project/data/features.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved feature-engineered dataset ({df.shape[1]} columns) to {out_path}")
print("Saved 3 EDA/feature plots to", PLOTS)
