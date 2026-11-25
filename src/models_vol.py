import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.baselines import volatility_baselines
from src.config import PROCESSED_DIR, BASE_DIR
from src.split import time_split

TARGET = "target_vol_5d"
FIG_DIR = BASE_DIR / "reports" / "figures"
REPORT_DIR = BASE_DIR / "reports"


def get_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42),
    }


def score(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    with open(PROCESSED_DIR / "selected_features.json") as f:
        features = json.load(f)["features"]

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    train, val, _test = time_split(df)               # the test set stays sealed
    train = train.dropna(subset=[TARGET])
    val = val.dropna(subset=[TARGET])

    X_train, X_val = train[features], val[features]
    y_train_log = np.log(train[TARGET])              # the models learn the LOG of volatility
    y_val = val[TARGET]

    rows, preds, fitted = [], {}, {}
    for name, model in get_models().items():
        model.fit(X_train, y_train_log)              # learn from the training period
        pred = np.exp(model.predict(X_val))          # convert back to normal volatility units
        preds[name], fitted[name] = pred, model
        rows.append({"model": name, **score(y_val, pred)})

    results = pd.DataFrame(rows).set_index("model")
    baseline = volatility_baselines(train, val).iloc[[-1]]      # the "VIX rescaled" row
    table = pd.concat([baseline, results]).round(3)
    table.index.name = "model / baseline"

    print("TASK B: volatility models vs best baseline (validation set)")
    print(table)

    # Which features do the tree models rely on?
    for name in ["Random Forest", "XGBoost"]:
        imp = pd.Series(fitted[name].feature_importances_, index=features)
        print(f"\nTop 8 features for {name}:")
        print(imp.sort_values(ascending=False).head(8).round(3))

    # Chart: actual vs baseline vs best model
    best = results["MAE"].idxmin()
    ratio = train[TARGET].mean() / train["vix_close"].mean()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4.5))
    plt.plot(val["date"], y_val, label="Actual next-5-day volatility", linewidth=1)
    plt.plot(val["date"], val["vix_close"] * ratio, label="VIX rescaled (baseline)", linewidth=1)
    plt.plot(val["date"], preds[best], label=f"{best} (model)", linewidth=1)
    plt.title("Validation period: forecast vs actual volatility")
    plt.ylabel("Annualised volatility (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_volatility_validation.png", dpi=150)
    plt.close()

    # Save predictions for later analysis
    out = pd.DataFrame({"date": val["date"].values, "actual": y_val.values})
    for name, p in preds.items():
        out[name] = p
    out.to_csv(REPORT_DIR / "vol_validation_predictions.csv", index=False)
    print(f"\nBest model by MAE: {best}. Saved chart and predictions to reports/.")


if __name__ == "__main__":
    main()