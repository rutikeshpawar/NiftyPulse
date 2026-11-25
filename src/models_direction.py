import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import PROCESSED_DIR
from src.split import time_split

TARGET = "target_direction"


def get_models():
    return {
        "Dummy (no skill)": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=40,
            random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.02, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, random_state=42),
    }


def auc_interval(y_true, score, n_boot=1000, seed=42):
    """Rough 95% range of AUC by resampling days (shows how much luck is involved)."""
    rng = np.random.default_rng(seed)
    y_true, score = np.asarray(y_true), np.asarray(score)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], score[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    with open(PROCESSED_DIR / "selected_features.json") as f:
        features = json.load(f)["features"]

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    train, val, _test = time_split(df)                # the test set stays sealed
    train = train.dropna(subset=[TARGET])
    val = val.dropna(subset=[TARGET])

    X_train, y_train = train[features], train[TARGET].astype(int)
    X_val, y_val = val[features], val[TARGET].astype(int)

    rows, probas = [], {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]      # probability that tomorrow is UP
        pred = (proba >= 0.5).astype(int)
        probas[name] = pred
        low, high = auc_interval(y_val, proba)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_val, pred),
            "balanced_acc": balanced_accuracy_score(y_val, pred),
            "f1_up": f1_score(y_val, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, proba),
            "auc_low": low,
            "auc_high": high,
            "pred_up_share": pred.mean(),
        })

    results = pd.DataFrame(rows).set_index("model").round(3)
    print("TASK A: direction models (validation set)")
    print(f"Reference: always-up accuracy = {y_val.mean():.3f}; "
          f"best simple rule (momentum) AUC = 0.529\n")
    print(results)

    best = results.drop(index="Dummy (no skill)")["roc_auc"].idxmax()
    cm = confusion_matrix(y_val, probas[best])
    print(f"\nConfusion matrix for {best} (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=["actual down", "actual up"],
                       columns=["pred down", "pred up"]))


if __name__ == "__main__":
    main()