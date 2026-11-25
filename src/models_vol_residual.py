import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.baselines import volatility_baselines
from src.config import PROCESSED_DIR
from src.models_vol import score, TARGET
from src.split import time_split

# A small, EDA-motivated feature set: VIX premium, recent volatility, trend stress
SMALL_SET = ["vix_vs_ma20", "vix_z_63", "vol_21d", "ret_5d", "drawdown", "vix_minus_vol21"]


def candidates(all_features):
    """name -> (model, features used)"""
    return {
        "Constant ratio (sanity check)": (DummyRegressor(strategy="mean"), all_features),
        "Ridge (alpha=100)": (make_pipeline(StandardScaler(), Ridge(alpha=100.0)), all_features),
        "Ridge (6 EDA features)": (make_pipeline(StandardScaler(), Ridge(alpha=10.0)), SMALL_SET),
        "Random Forest (shallow)": (RandomForestRegressor(
            n_estimators=300, max_depth=3, min_samples_leaf=40,
            random_state=42, n_jobs=-1), all_features),
        "XGBoost (shallow)": (XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.02, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, random_state=42), all_features),
    }


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    with open(PROCESSED_DIR / "selected_features.json") as f:
        features = json.load(f)["features"]

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    train, val, _test = time_split(df)                 # test set stays sealed
    train = train.dropna(subset=[TARGET])
    val = val.dropna(subset=[TARGET])

    # The model learns how far realised volatility sits from VIX, in logs
    y_train = np.log(train[TARGET] / train["vix_close"])

    rows = []
    for name, (model, cols) in candidates(features).items():
        model.fit(train[cols], y_train)
        ratio_pred = np.exp(model.predict(val[cols]))
        forecast = val["vix_close"] * ratio_pred       # forecast = VIX x predicted ratio
        rows.append({"model": name, **score(val[TARGET], forecast)})

    results = pd.DataFrame(rows).set_index("model")
    baseline = volatility_baselines(train, val).iloc[[-1]]
    table = pd.concat([baseline, results]).round(3)
    table.index.name = "model / baseline"

    print("TASK B (residual-on-VIX): validation set")
    print(table)


if __name__ == "__main__":
    main()