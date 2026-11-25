import json
from datetime import datetime

from src.config import RAW_DIR
from src.data_loader import get_history

START_DATE = "2015-01-01"
SYMBOLS = ["NIFTY50", "INDIAVIX"]

# For now only yfinance (brokers need fixing). Later change this to None
# to use the full priority chain: Angel One -> Upstox -> yfinance.
SOURCES = ["yfinance"]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {}

    for name in SYMBOLS:
        df = get_history(name, start=START_DATE, sources=SOURCES)

        file_path = RAW_DIR / f"{name.lower()}_daily.csv"
        df.to_csv(file_path, index=False)

        metadata[name] = {
            "file": file_path.name,
            "source": df["source"].iloc[0],
            "rows": int(len(df)),
            "first_date": str(df["date"].min().date()),
            "last_date": str(df["date"].max().date()),
            "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        print(f"Saved {name}: {len(df)} rows -> {file_path}")

    with open(RAW_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata.json")


if __name__ == "__main__":
    main()