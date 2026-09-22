"""
01_generate_data.py
--------------------
Generates a synthetic daily OHLCV price series for one ticker, built with
a GARCH-like volatility-clustering process plus a drift and mild seasonal
component, so it behaves like a real equity series (fat-tailed returns,
volatility clusters, momentum). This stands in for real data pulled from
a source like Yahoo Finance / Alpha Vantage -- the rest of the pipeline
works unchanged on real OHLCV data with the same column names.
"""

import numpy as np
import pandas as pd

np.random.seed(7)

N_DAYS = 1500  # ~6 years of trading days
start_price = 120.0

dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=N_DAYS)

# --- GARCH(1,1)-like volatility clustering for daily returns --------------
omega, alpha, beta = 2e-6, 0.06, 0.90
sigma2 = np.zeros(N_DAYS)
returns = np.zeros(N_DAYS)
sigma2[0] = omega / (1 - alpha - beta)

for t in range(1, N_DAYS):
    sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    shock = np.random.standard_t(df=6) * np.sqrt(sigma2[t])
    shock = np.clip(shock, -0.07, 0.07)  # cap single-day moves at +/-7%
    drift = 0.00028 + 0.0001 * np.sin(2 * np.pi * t / 252)  # mild seasonality
    returns[t] = drift + shock

prices = start_price * np.exp(np.cumsum(returns))

# --- Build OHLCV around the close price ------------------------------------
close = prices
open_ = close * (1 + np.random.normal(0, 0.003, N_DAYS))
intraday_range = np.abs(np.random.normal(0, 0.008, N_DAYS)) * close
high = np.maximum(open_, close) + intraday_range
low = np.minimum(open_, close) - intraday_range
volume = np.random.lognormal(mean=14.5, sigma=0.35, size=N_DAYS) * (
    1 + 2 * np.abs(returns)  # volume spikes on big moves
)

df = pd.DataFrame({
    "date": dates,
    "open": open_.round(2),
    "high": high.round(2),
    "low": low.round(2),
    "close": close.round(2),
    "volume": volume.round(0).astype(int),
})

out_path = "/home/claude/stock_prediction_project/data/stock_prices.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.head())
print(df.tail())
