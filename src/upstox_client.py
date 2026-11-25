import os
from urllib.parse import quote

import pandas as pd
import requests

SOURCE_NAME = "upstox"
COLUMNS = ["date", "open", "high", "low", "close", "volume", "source"]

# Upstox instrument keys for NSE indices
KEYS = {"NIFTY50": "NSE_INDEX|Nifty 50", "INDIAVIX": "NSE_INDEX|India VIX"}

BASE_URL = "https://api.upstox.com/v2/historical-candle"
CHUNK_DAYS = 365  # download one year at a time


def fetch_history(name: str, start: str, end: str | None = None) -> pd.DataFrame:
    if name not in KEYS:
        raise ValueError(f"Unknown symbol '{name}'. Choose from {list(KEYS)}")

    headers = {"Accept": "application/json"}
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    key = quote(KEYS[name], safe="")

    rows = []
    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS), end_dt)
        url = f"{BASE_URL}/{key}/day/{chunk_end:%Y-%m-%d}/{chunk_start:%Y-%m-%d}"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Upstox request failed ({resp.status_code}): {resp.text[:200]}")
        rows.extend(resp.json().get("data", {}).get("candles", []))
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not rows:
        raise RuntimeError(f"Upstox returned no data for {name}")

    # Each candle: [timestamp, open, high, low, close, volume, open_interest]
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["source"] = SOURCE_NAME
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[COLUMNS]