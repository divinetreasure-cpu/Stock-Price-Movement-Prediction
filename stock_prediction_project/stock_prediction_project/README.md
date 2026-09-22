# Stock Price Movement Prediction

A full pipeline for predicting short-term stock price behavior from
technical indicators — and an honest evaluation of *how well that actually
works*, which is itself the main finding of this project.

## Project Structure

```
stock_prediction_project/
├── data/
│   ├── stock_prices.csv        # daily OHLCV data (~6 years)
│   └── features.csv             # feature-engineered dataset
├── src/
│   ├── 01_generate_data.py      # data generation
│   ├── 02_eda_features.py       # EDA + technical indicator engineering
│   └── 03_modeling.py           # regression + classification + evaluation
├── plots/                        # 7 PNGs
├── results.json
└── README.md
```

**Note on data:** this uses a synthetically generated daily price series
(GARCH-style volatility clustering + drift, so it behaves like a real
equity — fat-tailed returns, volatility clustering, no look-ahead bias).
The pipeline runs unchanged on real OHLCV data pulled from a source like
Yahoo Finance (`yfinance`) or Alpha Vantage — just replace
`01_generate_data.py`'s output with a real CSV using the same column
names (`date, open, high, low, close, volume`).

## 1. Data & EDA

~1,500 trading days (~6 years) of daily OHLCV data. EDA covers price
history with moving averages, return distribution, and rolling volatility
(`plots/01`, `plots/02`) — annualized volatility came out around 25%,
in line with a typical single-stock series.

## 2. Feature Engineering

Standard technical indicators, all built only from *past* data to avoid
look-ahead bias:
- **RSI (14-day)**, **MACD** and its signal line
- **Rolling volatility** (10-day) and **volume change**
- **Momentum** (5-day, 10-day price change)
- **Lagged returns** (1, 2, 3, 5 days back)
- **Price relative to SMA-20 / SMA-50**

## 3. Modeling — Two Tasks, Time-Based Split

Since this is time-series data, the train/test split is **chronological**
(first 80% of dates train, last 20% test) — never shuffled, to avoid
leaking future information into training.

**Task A — Regression: predict next-day return**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Naive (predict 0% change) | 0.01545 | 0.01098 | — |
| Linear Regression | 0.01560 | 0.01120 | -0.020 |
| Random Forest | 0.01563 | 0.01110 | -0.023 |

**Task B — Classification: predict next-day direction (up/down)**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Naive (majority class) | 48.97% | — |
| Logistic Regression | 48.97% | 0.475 |
| Random Forest | 51.38% | 0.499 |

## 4. The Honest Finding

Both models perform **essentially no better than chance**, and the naive
"predict no change" baseline beats both regressors on RMSE. This is not a
bug — it is the expected result, and it is the most defensible outcome a
project like this can show.

Daily stock returns are close to a random walk; if 13 basic technical
indicators reliably predicted next-day direction, that edge would already
be arbitraged away by every quant desk using the same public data (the
core idea behind the Efficient Market Hypothesis). A portfolio project
that reports 65%+ next-day accuracy from RSI/MACD alone is a red flag to
anyone with finance experience — it almost always means leaked information
or an overfit model, not real skill.

**What this project demonstrates instead:** correct time-series
methodology (no shuffling, no leakage), a full technical-indicator feature
set, proper baselines, and — most importantly — the judgment to report a
null result honestly rather than manufacture a misleadingly good number.

## 5. How to Extend (where real edge tends to come from)

- Predict **volatility** instead of direction (volatility clusters and is
  genuinely more predictable — see `plots/02`)
- Use **longer horizons** (weekly/monthly) or **multi-asset** features
  (sector indices, macro variables, sentiment data)
- Try **portfolio-level** signals rather than single-stock direction
- If pursuing direction prediction seriously: alternative data (news
  sentiment, order-book microstructure) tends to matter far more than
  additional technical indicators on the same price series

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python src/01_generate_data.py
python src/02_eda_features.py
python src/03_modeling.py
```
