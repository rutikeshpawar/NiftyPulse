import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.features import add_features, FEATURE_COLUMNS

TARGET_COLUMNS = ["target_direction", "target_next_ret", "target_vol_5d"]


def test_truncation():
    """Features on a date must be identical whether or not future data exists."""
    clean = pd.read_csv(PROCESSED_DIR / "market_clean.csv", parse_dates=["date"])
    full = add_features(clean).set_index("date")

    rng = np.random.default_rng(42)
    cut_positions = rng.integers(300, len(clean) - 1, size=20)   # 20 random cut points

    for pos in cut_positions:
        truncated = add_features(clean.iloc[: pos + 1]).set_index("date")
        last_date = truncated.index[-1]

        a = truncated.loc[last_date, FEATURE_COLUMNS].astype(float).values
        b = full.loc[last_date, FEATURE_COLUMNS].astype(float).values
        same = np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)

        bad = [f for f, ok in zip(FEATURE_COLUMNS, same) if not ok]
        assert not bad, f"LEAKAGE on {last_date.date()} in features: {bad}"

    print(f"PASS | truncation test: {len(cut_positions)} random dates x {len(FEATURE_COLUMNS)} features")


def test_target_correlation():
    """A feature almost identical to a target is a red flag."""
    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    corr = df[FEATURE_COLUMNS + TARGET_COLUMNS].corr().loc[FEATURE_COLUMNS, TARGET_COLUMNS].abs()

    worst_feature = corr.max(axis=1).idxmax()
    worst_value = corr.max(axis=1).max()
    assert worst_value < 0.9, f"Suspiciously high correlation: {worst_feature} = {worst_value:.2f}"

    print(f"PASS | highest |correlation| with any target: {worst_value:.2f} ({worst_feature})")


def test_no_target_in_features():
    leaked = [c for c in FEATURE_COLUMNS if c.startswith("target_")]
    assert not leaked, f"Targets used as features: {leaked}"
    print("PASS | no target columns among the features")


if __name__ == "__main__":
    test_no_target_in_features()
    test_truncation()
    test_target_correlation()
    print("\nAll leakage tests passed.")