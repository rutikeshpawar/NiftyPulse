import pandas as pd
import pyotp
from SmartApi import SmartConnect

from src.config import get_env

SOURCE_NAME = "angelone"
COLUMNS = ["date", "open", "high", "low", "close", "volume", "source"]

# Angel One instrument tokens (NSE indices)
TOKENS = {"NIFTY50": "99926000", "INDIAVIX": "99926017"}

CHUNK_DAYS = 1500  # download in pieces to stay under API date-range limits


def _get_totp_code() -> str:
    """Create the 6-digit code from the secret stored in .env."""
    secret = get_env("ANGEL_TOTP_SECRET").replace(" ", "").replace('"', "").replace("'", "").upper()
    try:
        return pyotp.TOTP(secret).now()
    except Exception:
        raise ValueError(
            "ANGEL_TOTP_SECRET in .env is not a valid TOTP secret. "
            "It must be the long text secret (letters A-Z and digits 2-7), not the 6-digit code."
        )


def _login() -> SmartConnect:
    """Log in with API key + client code + PIN + a fresh TOTP code."""
    api = SmartConnect(api_key=get_env("ANGEL_API_KEY"))
    session = api.generateSession(
        get_env("ANGEL_CLIENT_CODE"), get_env("ANGEL_PIN"), _get_totp_code()
    )
    if not session or not session.get("status"):
        msg = session.get("message") if session else "no response"
        raise RuntimeError(f"Angel One login failed: {msg}")
    return api


def fetch_history(name: str, start: str, end: str | None = None) -> pd.DataFrame:
    if name not in TOKENS:
        raise ValueError(f"Unknown symbol '{name}'. Choose from {list(TOKENS)}")

    api = _login()
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    rows = []
    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS), end_dt)
        resp = api.getCandleData({
            "exchange": "NSE",
            "symboltoken": TOKENS[name],
            "interval": "ONE_DAY",
            "fromdate": chunk_start.strftime("%Y-%m-%d 09:15"),
            "todate": chunk_end.strftime("%Y-%m-%d 15:30"),
        })
        if not resp or not resp.get("status"):
            raise RuntimeError(f"Angel One data request failed: {resp}")
        rows.extend(resp.get("data") or [])
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not rows:
        raise RuntimeError(f"Angel One returned no data for {name}")

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["source"] = SOURCE_NAME
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[COLUMNS]