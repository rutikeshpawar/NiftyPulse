import pandas as pd

TRAIN_END = "2022-12-31"
VAL_END = "2024-12-31"
GAP_ROWS = 5     # targets look 5 trading days ahead, so purge the last 5 rows of train and validation


def time_split(df: pd.DataFrame):
    """Split chronologically into train / validation / test with a purge gap."""
    df = df.sort_values("date").reset_index(drop=True)

    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END]

    # Purge: their targets reach into the next period
    train = train.iloc[:-GAP_ROWS]
    val = val.iloc[:-GAP_ROWS]
    return train, val, test


def main():
    from src.config import PROCESSED_DIR

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    train, val, test = time_split(df)

    assert train["date"].max() < val["date"].min() < test["date"].min(), "Periods overlap"

    for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"{name:<11}: {len(part):>4} rows | {part['date'].min().date()} to {part['date'].max().date()} "
              f"| up-days {part['target_direction'].mean():.1%} "
              f"| avg future vol {part['target_vol_5d'].mean():.1f}")


if __name__ == "__main__":
    main()