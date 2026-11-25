import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import PROCESSED_DIR, BASE_DIR
from src.models_direction import get_models as direction_models
from src.models_vol import score
from src.models_vol_residual import candidates as vol_candidates

VOL = "target_vol_5d"
DIR = "target_direction"
FOLD_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
GAP = 5                      # purge rows, because targets look 5 days ahead
BASELINE = "VIX rescaled (baseline)"


def make_fold(df, year):
    """Train on everything before `year`; test on `year` itself."""
    train = df[df["date"] < f"{year}-01-01"].iloc[:-GAP]       # purge the end of training
    test = df[df["date"].dt.year == year].iloc[:-GAP]          # purge the end of the test year
    return train.dropna(subset=[VOL, DIR]), test.dropna(subset=[VOL, DIR])


def volatility_fold(train, test, features):
    ratio = train[VOL].mean() / train["vix_close"].mean()      # baseline factor, learned on train only
    forecasts = {BASELINE: test["vix_close"] * ratio}

    y_train = np.log(train[VOL] / train["vix_close"])
    for name, (model, cols) in vol_candidates(features).items():
        model.fit(train[cols], y_train)
        forecasts[name] = test["vix_close"] * np.exp(model.predict(test[cols]))

    return {name: score(test[VOL], pred) for name, pred in forecasts.items()}


def direction_fold(train, test, features):
    y_train, y_test = train[DIR].astype(int), test[DIR].astype(int)
    out = {}
    for name, model in direction_models().items():
        if name.startswith("Dummy"):
            continue
        model.fit(train[features], y_train)
        proba = model.predict_proba(test[features])[:, 1]
        out[name] = {
            "AUC": roc_auc_score(y_test, proba),
            "accuracy": accuracy_score(y_test, proba >= 0.5),
            "always_up_accuracy": y_test.mean(),
        }
    return out


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    with open(PROCESSED_DIR / "selected_features.json") as f:
        features = json.load(f)["features"]

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    df = df[df["date"] <= "2024-12-31"]                        # the 2025+ test set stays sealed
    assert df["date"].max() <= pd.Timestamp("2024-12-31")

    vol_rows, dir_rows = [], []
    for year in FOLD_YEARS:
        train, test = make_fold(df, year)
        print(f"Fold {year}: train {len(train)} rows, test {len(test)} rows")

        for name, s in volatility_fold(train, test, features).items():
            vol_rows.append({"year": year, "model": name, **s})
        for name, s in direction_fold(train, test, features).items():
            dir_rows.append({"year": year, "model": name, **s})

    # ---------- TASK B: volatility ----------
    vol = pd.DataFrame(vol_rows)
    mae = vol.pivot(index="model", columns="year", values="MAE")
    summary = vol.groupby("model")[["MAE", "RMSE", "R2"]].mean()
    summary["folds_beating_baseline_MAE"] = [(mae.loc[m] < mae.loc[BASELINE]).sum() for m in summary.index]

    print("\nTASK B: MAE by test year")
    print(mae.round(2))
    print("\nTASK B: average across 6 folds")
    print(summary.round(3).sort_values("MAE"))

    # ---------- TASK A: direction ----------
    dr = pd.DataFrame(dir_rows)
    auc = dr.pivot(index="model", columns="year", values="AUC")
    d_summary = dr.groupby("model")[["AUC", "accuracy", "always_up_accuracy"]].mean()
    d_summary["AUC_std"] = auc.std(axis=1)
    d_summary["folds_AUC_above_0.5"] = (auc > 0.5).sum(axis=1)

    print("\nTASK A: ROC-AUC by test year")
    print(auc.round(3))
    print("\nTASK A: average across 6 folds")
    print(d_summary.round(3).sort_values("AUC", ascending=False))

    (BASE_DIR / "reports").mkdir(exist_ok=True)
    vol.to_csv(BASE_DIR / "reports" / "walk_forward_volatility.csv", index=False)
    dr.to_csv(BASE_DIR / "reports" / "walk_forward_direction.csv", index=False)


if __name__ == "__main__":
    main()