# NiftyPulse — AI Market Intelligence Platform

> **CDAC Practical Machine Learning (PML) Project**  
> A research and decision-support platform for analysing the Nifty 50 market, forecasting short-term volatility, and evaluating next-day market direction.

NiftyPulse goes beyond a simple price forecast. It studies market behaviour using Nifty 50 and India VIX data, creates leak-safe technical and statistical features, compares machine-learning models with simple baselines, and evaluates a volatility-targeting approach through historical backtesting.

**Disclaimer:** NiftyPulse is an educational research and decision-support project. It is not financial advice, does not guarantee returns, and is not currently designed to place real-money trades.

---

## Table of Contents

- [Project Objectives](#project-objectives)
- [Current Project Status](#current-project-status)
- [Data Sources](#data-sources)
- [Project Workflow](#project-workflow)
- [Machine-Learning Tasks](#machine-learning-tasks)
- [Final Test Results](#final-test-results)
- [Backtesting Results](#backtesting-results)
- [Key Findings](#key-findings)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Current Pipeline](#running-the-current-pipeline)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)

---

## Project Objectives

The project addresses three related market-analysis questions:

1. **Market direction:** Can historical Nifty 50 and India VIX information help estimate whether the next trading day's close will be higher or lower?
2. **Volatility forecasting:** Can the available market information estimate the realised volatility over the next five trading days?
3. **Risk evaluation:** Does adjusting market exposure using a volatility estimate improve risk-adjusted performance compared with a simple Buy-and-Hold benchmark?

The project uses supervised learning for direction classification and volatility regression, together with baseline comparisons and time-aware evaluation.

## Current Project Status

**Completed:** Core project work through Step 5.4, including the final sealed test and historical backtest.

| Phase | Work | Status |
|---|---|---|
| 0 | Problem identification and project planning | Complete |
| 1 | Data pipeline, collection and validation | Complete |
| 2 | Exploratory Data Analysis (EDA) | Complete |
| 3 | Cleaning, target creation and feature engineering | Complete |
| 4 | Baselines, model training and model comparison | Complete |
| 5 | Evaluation, walk-forward analysis, backtesting and final sealed test | Complete |
| 6 | SHAP explainability | Planned |
| 7 | K-Means market-regime detection | Planned |
| 8 | Streamlit dashboard and MLflow tracking | Planned |
| 9 | Final report and viva preparation | Planned |

The final test period has already been evaluated. **It must remain sealed and must not be reused for model selection or hyperparameter tuning.** Any future strategy experiments should use new data or a separate paper-trading period.

## Data Sources

The current research pipeline uses daily historical data for:

| Instrument | Yahoo Finance ticker | Use |
|---|---|---|
| Nifty 50 | `^NSEI` | Daily OHLC price history and market returns |
| India VIX | `^INDIAVIX` | Market-implied volatility and changes in risk sentiment |

The data client uses `yfinance` for the documented daily historical workflow. The project also contains preliminary Angel One SmartAPI and Upstox client modules; broker authentication, connectivity and live/intraday data retrieval are not treated as completed features in this version.

### Data handling

- Raw downloads are kept separate from processed data.
- Data is sorted by date and checked for missing values, duplicates, invalid prices and OHLC consistency.
- Nifty 50 is used as the master trading calendar when aligning India VIX observations.
- India VIX gaps are forward-filled for a limited number of days and flagged for audit.
- Unreliable index volume fields are not used as model inputs.
- Genuine extreme market events, including the COVID-19 crash, are retained rather than removed as erroneous outliers.

## Project Workflow

```text
Nifty 50 + India VIX daily data
              |
              v
       Data collection
              |
              v
     Data validation and cleaning
              |
              v
      Exploratory Data Analysis
              |
              v
       Target construction
              |
              v
      Feature engineering
              |
              v
  Chronological train/validation/test split
              |
              v
       Baseline comparisons
              |
              v
  Direction and volatility modelling
              |
              v
 Walk-forward evaluation and final test
              |
              v
       Historical backtesting
```

### Feature engineering

The feature-engineering stage created **31 candidate features**, covering:

- Returns and momentum: daily returns, lagged returns and multi-period returns.
- Candle characteristics: opening gap, daily range, candle body and close position.
- Trend: moving-average distances, moving-average relationship and drawdown.
- Volatility: rolling realised volatility, volatility ratios and ATR percentage.
- Momentum oscillator: RSI.
- India VIX: level, changes, moving-average relationship and relative volatility measures.

Feature selection retained **27 features** for the model input set. Features are calculated using information available on or before the prediction date. Future target columns are excluded from model inputs to prevent data leakage.

## Machine-Learning Tasks

### Task A — Next-day direction classification

The target is defined as:

- `1`: next trading day's Nifty 50 close is higher than the current day's close.
- `0`: next trading day's Nifty 50 close is not higher than the current day's close.

Models compared:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Evaluation includes accuracy and ROC-AUC, with comparison against a simple always-up baseline.

### Task B — Next-five-day volatility regression

The target is the annualised standard deviation of Nifty 50 daily returns over the next five trading days. The project compares:

- Rescaled India VIX baseline
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

The volatility model is trained on the logarithm of the target and predictions are converted back to volatility units for evaluation.

### Time-aware evaluation

The project uses chronological training, validation and test periods instead of randomly shuffling time-series observations. Model selection and tuning are performed without using the final test set. Walk-forward evaluation is also used to examine performance across different market periods.

## Final Test Results

**Final sealed test period: January 2025 – November 2026**

These are the final results recorded for Step 5.4. They are reported as obtained; the test results are not used to retune the models.

### Task B — Volatility

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Rescaled VIX baseline** | **3.657** | **5.366** | **0.304** |
| Ridge Regression | 3.691 | 5.423 | 0.289 |

**Interpretation:** Ridge did not outperform the rescaled VIX baseline on the final test. The difference is small, so the practical result is close to a tie. The rescaled VIX baseline is retained as the primary volatility forecast because it is simpler; Ridge remains a challenger model.

### Task A — Direction

| Model | ROC-AUC | Reported 95% range | Accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.526 | 0.471–0.579 | 0.492 |
| Random Forest | **0.565** | 0.510–0.621 | 0.549 |
| XGBoost | 0.564 | 0.507–0.618 | 0.556 |
| Always-up baseline | — | — | 0.501 |

**Interpretation:** Random Forest and XGBoost performed modestly better than the always-up accuracy baseline in this final test. However, the evidence is not conclusive: the reported ranges use a resampling assumption that does not fully account for time dependence, walk-forward AUC was weak, and only a limited test period is available. Direction predictability is therefore treated as **weak and unproven**, not as a reliable trading signal.

## Backtesting Results

**Backtest period: January 2025 – November 2026**

| Strategy | CAGR % | Volatility % | Sharpe | Maximum drawdown % | Average exposure |
|---|---:|---:|---:|---:|---:|
| **Buy and Hold** | **4.57** | 13.43 | **0.40** | -15.56 | 1.00 |
| VIX-rescaled targeting | 2.72 | 12.25 | 0.28 | -16.66 | 0.96 |
| Ridge targeting | 2.99 | 12.35 | 0.30 | -16.16 | 0.97 |
| Constant exposure | 4.47 | 13.07 | 0.40 | **-15.15** | 0.97 |

**Interpretation:** Volatility targeting did not improve performance over Buy and Hold in this test period. It produced lower CAGR and Sharpe ratios, and did not reduce maximum drawdown. The results suggest that volatility targeting may act like insurance during severe market stress, but its benefit was not demonstrated in this particular test period.

## Key Findings

1. **Volatility:** The rescaled India VIX baseline slightly outperformed Ridge on the final test. The simpler baseline is preferred for now.
2. **Direction:** The final test showed modest classification performance, but validation and walk-forward evidence was weak. The direction signal is not considered sufficiently reliable for trading.
3. **Backtesting:** Volatility-targeting strategies underperformed Buy and Hold on the final test period.
4. **Market behaviour:** Volatility was more forecastable than next-day direction in the experiments conducted.
5. **Testing discipline:** The final test set is considered used. Future improvements must not be justified by repeatedly tuning against these test results.

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data handling | Pandas, NumPy |
| Data source | yfinance / Yahoo Finance |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn, XGBoost |
| Statistical analysis | Statsmodels, SciPy-compatible statistical methods |
| Explainability dependency | SHAP (planned for next phase) |
| Notebook environment | Jupyter Notebook, VS Code |
| Broker API adapters | Angel One SmartAPI and Upstox (preliminary; not live-validated in this release) |
| Planned app and experiment tracking | Streamlit, MLflow |

## Project Structure

```text
NiftyPulse/
│
├── data/
│   ├── raw/                  # Downloaded source data (not committed)
│   └── processed/            # Cleaned data and model features (not committed)
│
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
│
├── src/
│   ├── config.py             # Project paths and environment configuration
│   ├── yfinance_client.py    # Yahoo Finance daily data client
│   ├── angelone_client.py   # Preliminary Angel One client
│   ├── upstox_client.py     # Preliminary Upstox client
│   ├── data_loader.py       # Data-source fallback loader
│   ├── download_data.py     # Historical data download
│   ├── validate_data.py     # Data-quality checks
│   ├── cleaning.py          # Data cleaning and alignment
│   ├── targets.py           # Prediction target creation
│   ├── features.py          # Feature engineering
│   ├── split.py             # Chronological data split
│   ├── feature_selection.py # Feature selection
│   ├── baselines.py         # Baseline evaluation
│   └── models_vol.py        # Volatility regression models
│
├── reports/
│   ├── figures/              # EDA and validation charts
│   ├── feature_importance_mi.csv
│   └── vol_validation_predictions.csv
│
├── setup_project.py          # Project setup helper
├── test_config.py            # Configuration check
├── test_loader.py            # Data-loader check
├── test_yfinance.py          # Yahoo Finance client check
├── test_leakage.py           # Feature leakage check
├── requirements.txt
├── .env.example              # Example environment variables only
├── .gitignore
└── README.md
```

> The uploaded ZIP snapshot contains the source modules and validation scripts listed above. If your latest local version has additional final direction-model, walk-forward or backtesting scripts, include those files in the repository as well so the final results can be reproduced directly from code.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-github-username>/NiftyPulse.git
cd NiftyPulse
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables (only if using broker clients)

Create a local `.env` file from `.env.example` and enter your own credentials if you plan to test the broker API adapters. Keep `.env` private and never commit it to GitHub. The daily Yahoo Finance data workflow does not require broker credentials.

## Running the Current Pipeline

Run commands from the repository root.

### 1. Download daily data

```bash
python -m src.download_data
```

The current download script is configured to use Yahoo Finance for Nifty 50 and India VIX daily history.

### 2. Validate downloaded data

```bash
python -m src.validate_data
```

### 3. Run the EDA notebook

Open `notebooks/01_eda.ipynb` in Jupyter or VS Code and select the project's virtual environment as the notebook kernel. Run the notebook cells in order.

### 4. Run the implemented processing and baseline modules

```bash
python -m src.cleaning
python -m src.targets
python -m src.features
python -m src.split
python -m src.feature_selection
python -m src.baselines
python -m src.models_vol
```

The final sealed-test and backtesting results above are recorded from the completed project experiments. Check that the corresponding final experiment scripts and outputs are included in your latest repository version if you want other users to reproduce every final table directly from code.

## Limitations

- The current research dataset is daily data for one market index and India VIX.
- Direction prediction showed weak and inconsistent results and is not treated as a dependable trading signal.
- Volatility targeting did not outperform Buy and Hold in the final test period.
- Backtest performance depends on assumptions about execution, costs and market conditions; it does not establish live profitability.
- Broker API adapters have not been verified as a stable live-data service in this release.
- The current release does not execute real-money trades.

## Future Enhancements

The following items are planned and are not presented as completed features:

1. **SHAP explainability:** explain the contribution of input features to model predictions.
2. **Market-regime detection:** use K-Means to explore trending, sideways and volatile market conditions.
3. **Streamlit dashboard:** present market data, model forecasts, evaluation charts and risk information.
4. **MLflow tracking:** record experiments, model versions and evaluation results.
5. **Intraday data:** validate broker data connections and collect 5-minute, 15-minute, 1-hour and 4-hour candles.
6. **Strategy research:** evaluate ORB, VWAP and Supertrend independently and in combination.
7. **Paper trading:** build a simulated order engine with position sizing, transaction costs, stop-loss, trade logs and a kill switch.

Real-money auto-trading is outside the scope of this release. Any future execution capability would require separate testing, security controls and compliance review.

---

## Acknowledgement

Developed as a Practical Machine Learning project for CDAC, with the objective of applying data engineering, exploratory analysis, feature engineering, supervised machine learning, time-series evaluation and financial backtesting to a real-world market-data problem.

**NiftyPulse — Measure the market. Understand the risk. Evaluate the signal.**
