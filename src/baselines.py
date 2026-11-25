import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from src.config import PROCESSED_DIR
from src.split import time_split


def direction_baselines(train, val) -> pd.DataFrame:
    """Simple up/down rules scored on the validation set."""
    y = val["target_direction"].astype(int)
    train_up = int(train["target_direction"].mean() >= 0.5)   # the majority class in training

    rules = {
        "Always majority class (up)": (np.full(len(val), train_up), np.zeros(len(val))),
        "Same as today (momentum)": ((val["ret_1d"] > 0).astype(int), val["ret_1d"]),
        "Opposite of today (reversal)": ((val["ret_1d"] <= 0).astype(int), -val["ret_1d"]),
    }

    rows = []
    for name, (pred, score) in rules.items():
        rows.append({
            "baseline": name,
            "accuracy": accuracy_score(y, pred),
            "balanced_acc": balanced_accuracy_score(y, pred),
            "f1_up": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score),
        })
    return pd.DataFrame(rows).set_index("baseline").round(3)


def volatility_baselines(train, val) -> pd.DataFrame:
    """Simple volatility forecasts scored on the validation set."""
    y = val["target_vol_5d"]
    ratio = train["target_vol_5d"].mean() / train["vix_close"].mean()   # learned on training only

    forecasts = {
        "Training average": np.full(len(val), train["target_vol_5d"].mean()),
        "Past 5-day volatility": val["vol_5d"],
        "Past 21-day volatility": val["vol_21d"],
        "VIX as-is": val["vix_close"],
        f"VIX rescaled (x{ratio:.2f})": val["vix_close"] * ratio,
    }

    rows = []
    for name, pred in forecasts.items():
        rows.append({
            "baseline": name,
            "MAE": mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred),
        })
    return pd.DataFrame(rows).set_index("baseline").round(3)


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    train, val, _test = time_split(df)           # the test set is NOT used here

    print("TASK A: direction baselines (validation set)")
    print(direction_baselines(train, val))
    print("\nTASK B: volatility baselines (validation set)")
    print(volatility_baselines(train, val))


if __name__ == "__main__":
    main()