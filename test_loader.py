from src.data_loader import get_history

# Full priority chain: expect Angel One and Upstox to fail (blocked), yfinance to succeed
df = get_history("NIFTY50", start="2015-01-01")

print(df.shape)
print(df.tail(3))
print("Source used:", df["source"].unique())