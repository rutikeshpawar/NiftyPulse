import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.config import PROCESSED_DIR, BASE_DIR
from src.features import FEATURE_COLUMNS
from src.split import time_split

CORR_THRESHOLD = 0.90   # a pair above this is considered redundant


def drop_correlated(X: pd.DataFrame, threshold: float):
    """Repeatedly remove one feature from the most-correlated pair."""
    corr = X.corr(method="spearman").abs()
    kept, dropped = list(X.columns), []

    while True:
        arr = corr.loc[kept, kept].to_numpy(copy=True)
        np.fill_diagonal(arr, 0.0)
        if arr.max() <= threshold:
            break
        i, j = np.unravel_index(arr.argmax(), arr.shape)
        a, b = kept[i], kept[j]
        # drop the one that overlaps more with all the other features
        drop = a if arr[i].mean() >= arr[j].mean() else b
        partner = b if drop == a else a
        dropped.append({"dropped": drop, "kept_instead": partner, "correlation": round(float(arr[i, j]), 2)})
        kept.remove(drop)

    return kept, dropped


def importance_table(train: pd.DataFrame, features: list) -> pd.DataFrame:
    """Mutual information: how much each feature tells us about each target (non-linear aware)."""
    sub = train.dropna(subset=["target_direction", "target_vol_5d"])
    X = sub[features]
    mi_dir = mutual_info_classif(X, sub["target_direction"].astype(int), random_state=42)
    mi_vol = mutual_info_regression(X, sub["target_vol_5d"], random_state=42)
    table = pd.DataFrame({"mi_direction": mi_dir, "mi_volatility": mi_vol}, index=features)
    return table.sort_values("mi_volatility", ascending=False).round(3)


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    train, val, test = time_split(df)          # selection uses the TRAINING period only

    kept, dropped = drop_correlated(train[FEATURE_COLUMNS], CORR_THRESHOLD)

    print(f"Features before: {len(FEATURE_COLUMNS)} | after correlation filter: {len(kept)}\n")
    print("Dropped as redundant (correlation above", CORR_THRESHOLD, "):")
    print(pd.DataFrame(dropped).to_string(index=False) if dropped else "  none")

    table = importance_table(train, kept)
    print("\nMutual information with each target (training period only):")
    print(table)

    (BASE_DIR / "reports").mkdir(exist_ok=True)
    table.to_csv(BASE_DIR / "reports" / "feature_importance_mi.csv")
    with open(PROCESSED_DIR / "selected_features.json", "w") as f:
        json.dump({"features": kept, "dropped": dropped}, f, indent=2)
    print("\nSaved selected_features.json and reports/feature_importance_mi.csv")


if __name__ == "__main__":
    main()