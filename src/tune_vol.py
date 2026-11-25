import itertools
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import PROCESSED_DIR, BASE_DIR
from src.models_vol import score
from src.walk_forward import make_fold, FOLD_YEARS, VOL

GRIDS = {
    "Ridge": {"alpha": [1, 10, 30, 100, 300, 1000, 3000]},
    "Random Forest": {"max_depth": [2, 3, 4, 6], "min_samples_leaf": [20, 40, 80]},
    "XGBoost": {"max_depth": [2, 3], "learning_rate": [0.02, 0.05], "n_estimators": [100, 200, 400]},
}

# The settings we used in Step 5.1 (for the "before vs after" comparison)
INITIAL = {
    "Ridge": {"alpha": 100},
    "Random Forest": {"max_depth": 3, "min_samples_leaf": 40},
    "XGBoost": {"max_depth": 2, "learning_rate": 0.02, "n_estimators": 200},
}


def build(name, params):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, **params)
    return XGBRegressor(min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                        random_state=42, **params)


def cv_mae(name, params, folds, features):
    """Average MAE (in volatility points) across the walk-forward folds."""
    maes = []
    for train, test in folds:
        model = build(name, params)
        model.fit(train[features], np.log(train[VOL] / train["vix_close"]))
        forecast = test["vix_close"] * np.exp(model.predict(test[features]))
        maes.append(score(test[VOL], forecast)["MAE"])
    return float(np.mean(maes))


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    with open(PROCESSED_DIR / "selected_features.json") as f:
        features = json.load(f)["features"]

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    df = df[df["date"] <= "2024-12-31"]                      # the test set stays sealed
    folds = [make_fold(df, year) for year in FOLD_YEARS]

    best_params, summary = {}, []
    for name, grid in GRIDS.items():
        keys = list(grid)
        results = []
        for values in itertools.product(*grid.values()):
            params = dict(zip(keys, values))
            results.append({**params, "avg_MAE": cv_mae(name, params, folds, features)})

        table = pd.DataFrame(results).sort_values("avg_MAE").reset_index(drop=True)
        print(f"\n===== {name}: top 5 settings =====")
        print(table.head(5).round(3))

        best = {k: table.loc[0, k] for k in keys}
        best = {k: (int(v) if float(v).is_integer() and k != "learning_rate" else float(v))
                for k, v in best.items()}
        best_params[name] = best

        summary.append({
            "model": name,
            "initial_MAE": cv_mae(name, INITIAL[name], folds, features),
            "tuned_MAE": float(table.loc[0, "avg_MAE"]),
            "best_settings": best,
        })

    print("\n===== Initial vs tuned (average MAE across 6 folds) =====")
    out = pd.DataFrame(summary).set_index("model")
    out["improvement_%"] = (1 - out["tuned_MAE"] / out["initial_MAE"]) * 100
    print(out.round(3))

    with open(PROCESSED_DIR / "best_params_vol.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("\nSaved best_params_vol.json")


if __name__ == "__main__":
    main()