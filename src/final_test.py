import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.backtest import (
    MAIN_COST_BPS, TARGET_VOL, exposure_from_forecast, metrics, run_backtest,
    walk_forward_forecasts,
)
from src.config import PROCESSED_DIR
from src.models_direction import auc_interval, get_models as direction_models
from src.models_vol import score
from src.walk_forward import DIR, VOL, make_fold

TEST_YEARS = [2025, 2026]


def volatility_final(bt):
    sub = bt.dropna(subset=[VOL])
    rows = []
    for name, col in [("VIX rescaled (baseline)", "forecast_vix"), ("Ridge (final model)", "forecast_ridge")]:
        rows.append({"model": name, **score(sub[VOL], sub[col])})
    return pd.DataFrame(rows).set_index("model").round(3), len(sub)


def direction_final(df, features):
    names = [n for n in direction_models() if not n.startswith("Dummy")]
    probs, ys = {n: [] for n in names}, []

    for year in TEST_YEARS:
        train, test = make_fold(df, year)       # train: before the year (minus purge); test: the year
        ys.append(test[DIR].astype(int))
        for name, model in direction_models().items():
            if name.startswith("Dummy"):
                continue
            model.fit(train[features], train[DIR].astype(int))
            probs[name].append(pd.Series(model.predict_proba(test[features])[:, 1], index=test.index))

    y = pd.concat(ys)
    rows = []
    for name in names:
        p = pd.concat(probs[name])
        low, high = auc_interval(y, p)
        rows.append({"model": name, "AUC": roc_auc_score(y, p), "AUC_low": low, "AUC_high": high,
                     "accuracy": accuracy_score(y, p >= 0.5)})
    return pd.DataFrame(rows).set_index("model").round(3), y.mean()


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("FINAL SEALED TEST. Run once. Do not change settings after seeing the results.\n")

    with open(PROCESSED_DIR / "selected_features.json") as f:
        features = json.load(f)["features"]
    with open(PROCESSED_DIR / "best_params_vol.json") as f:
        alpha = json.load(f)["Ridge"]["alpha"]

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    bt = walk_forward_forecasts(df, features, alpha, years=TEST_YEARS)
    print(f"Test period: {bt['date'].min().date()} to {bt['date'].max().date()}\n")

    # ----- Task B: volatility -----
    vol_table, n = volatility_final(bt)
    print(f"TASK B: volatility forecasts on the final test ({n} days)")
    print(vol_table)

    # ----- Task A: direction -----
    dir_table, up_share = direction_final(df, features)
    print(f"\nTASK A: direction on the final test (always-up accuracy = {up_share:.3f})")
    print(dir_table)

    # ----- Back-test -----
    bt = bt.set_index("date").dropna(subset=["target_next_ret"])
    next_ret = bt["target_next_ret"]
    ridge_exp = exposure_from_forecast(bt["forecast_ridge"])
    exposures = {
        "Buy and hold": pd.Series(1.0, index=bt.index),
        "VIX-rescaled vol targeting": exposure_from_forecast(bt["forecast_vix"]),
        "Ridge vol targeting": ridge_exp,
        "Constant exposure (same average)": pd.Series(ridge_exp.mean(), index=bt.index),
    }

    rows, nets = [], {}
    for name, exposure in exposures.items():
        net, turnover = run_backtest(next_ret, exposure, MAIN_COST_BPS)
        rows.append({"strategy": name, **metrics(net, exposure, turnover)})
        nets[name] = net
    print(f"\nBACK-TEST {bt.index.min().date()} to {bt.index.max().date()} "
          f"(costs {MAIN_COST_BPS} bps, target vol {TARGET_VOL}%)")
    print(pd.DataFrame(rows).set_index("strategy").round(2))

    sens = []
    for bps in [0, 5, 10, 20]:
        net, turnover = run_backtest(next_ret, ridge_exp, bps)
        m = metrics(net, ridge_exp, turnover)
        sens.append({"cost_bps": bps, "CAGR_%": m["CAGR_%"], "Sharpe": m["Sharpe"], "MaxDD_%": m["MaxDD_%"]})
    print("\nRidge vol targeting: sensitivity to costs")
    print(pd.DataFrame(sens).set_index("cost_bps").round(2))

    yearly = pd.DataFrame({
        name: nets[name].groupby(nets[name].index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
        for name in ["Buy and hold", "VIX-rescaled vol targeting", "Ridge vol targeting"]
    })
    print("\nNet return by calendar year (%; 2026 is partial)")
    print(yearly.round(1))


if __name__ == "__main__":
    main()