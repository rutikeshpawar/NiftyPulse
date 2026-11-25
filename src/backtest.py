import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, BASE_DIR
from src.walk_forward import FOLD_YEARS, GAP, VOL

TARGET_VOL = 12.0            # annualised volatility we aim for, in %
STEP = 0.25                  # exposure moves in quarter steps
MAIN_COST_BPS = 5            # assumed cost per unit traded (5 bps = 0.05%)
COSTS_BPS = [0, 5, 10, 20]   # cost sensitivity test


def walk_forward_forecasts(df, features, alpha, years=FOLD_YEARS):
    """Volatility forecasts for the given years, each made by a model trained only on earlier data."""
    pieces = []
    for year in years:
        train = df[df["date"] < f"{year}-01-01"].iloc[:-GAP].dropna(subset=[VOL])
        test = df[df["date"].dt.year == year].copy()

        ratio = train[VOL].mean() / train["vix_close"].mean()
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train[features], np.log(train[VOL] / train["vix_close"]))

        test["forecast_ridge"] = test["vix_close"] * np.exp(model.predict(test[features]))
        test["forecast_vix"] = test["vix_close"] * ratio
        pieces.append(test)
    return pd.concat(pieces, ignore_index=True)


def exposure_from_forecast(forecast):
    raw = (TARGET_VOL / forecast).clip(0, 1)           # less risk when forecast volatility is high
    return (raw / STEP).round() * STEP


def run_backtest(next_ret, exposure, cost_bps):
    """Net daily return = exposure x next-day return - trading cost."""
    turnover = exposure.diff().abs().fillna(exposure.abs())   # first day: building the position costs too
    cost = turnover * cost_bps / 10_000
    return exposure * next_ret - cost, turnover


def metrics(net, exposure, turnover):
    equity = (1 + net).cumprod()
    years = len(net) / 252
    vol = net.std() * np.sqrt(252)
    return {
        "CAGR_%": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "Vol_%": vol * 100,
        "Sharpe": net.mean() * 252 / vol if vol > 0 else np.nan,   # risk-free rate taken as 0
        "MaxDD_%": (equity / equity.cummax() - 1).min() * 100,
        "Avg_exposure": exposure.mean(),
        "Turnover_per_yr": turnover.sum() / years,
    }


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    with open(PROCESSED_DIR / "selected_features.json") as f:
        features = json.load(f)["features"]
    with open(PROCESSED_DIR / "best_params_vol.json") as f:
        alpha = json.load(f)["Ridge"]["alpha"]

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    df = df[df["date"] <= "2024-12-31"]                         # the 2025+ test set stays sealed

    bt = walk_forward_forecasts(df, features, alpha).set_index("date")
    bt = bt.iloc[:-1]                                           # last row's next-day return belongs to 2025
    next_ret = bt["target_next_ret"]

    ridge_exp = exposure_from_forecast(bt["forecast_ridge"])
    exposures = {
        "Buy and hold": pd.Series(1.0, index=bt.index),
        "VIX-rescaled vol targeting": exposure_from_forecast(bt["forecast_vix"]),
        "Ridge vol targeting": ridge_exp,
        "Constant exposure (same average)": pd.Series(ridge_exp.mean(), index=bt.index),
    }

    # ---------- Main comparison ----------
    rows, curves, nets = [], {}, {}
    for name, exposure in exposures.items():
        net, turnover = run_backtest(next_ret, exposure, MAIN_COST_BPS)
        rows.append({"strategy": name, **metrics(net, exposure, turnover)})
        curves[name] = (1 + net).cumprod()
        nets[name] = net
    print(f"BACK-TEST {bt.index.min().date()} to {bt.index.max().date()} "
          f"(costs {MAIN_COST_BPS} bps, target vol {TARGET_VOL}%)")
    print(pd.DataFrame(rows).set_index("strategy").round(2))

    # ---------- Cost sensitivity ----------
    sens = []
    for bps in COSTS_BPS:
        net, turnover = run_backtest(next_ret, ridge_exp, bps)
        m = metrics(net, ridge_exp, turnover)
        sens.append({"cost_bps": bps, "CAGR_%": m["CAGR_%"], "Sharpe": m["Sharpe"], "MaxDD_%": m["MaxDD_%"]})
    print("\nRidge vol targeting: sensitivity to trading costs")
    print(pd.DataFrame(sens).set_index("cost_bps").round(2))

    # ---------- Year by year ----------
    yearly = pd.DataFrame({
        name: nets[name].groupby(nets[name].index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
        for name in ["Buy and hold", "VIX-rescaled vol targeting", "Ridge vol targeting"]
    })
    print("\nNet return by calendar year (%)")
    print(yearly.round(1))

    # ---------- Chart ----------
    (BASE_DIR / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    for name, curve in curves.items():
        plt.plot(curve, label=name, linewidth=1.2)
    plt.title("Growth of 1 rupee, net of costs (out-of-sample, 2019-2024)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "reports" / "figures" / "13_backtest_equity.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()