from src.yfinance_client import fetch_history

for name in ["NIFTY50", "INDIAVIX"]:
    df = fetch_history(name, start="2015-01-01")
    print(f"\n===== {name} =====")
    print("Shape:", df.shape)
    print(df.head(3))
    print(df.tail(3))
    print(df.dtypes)