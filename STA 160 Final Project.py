import yfinance as yf
import pandas as pd
import requests
import numpy as np

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

sp500_table = pd.read_html(response.text)[0]

print(sp500_table.head())
print("Rows:", len(sp500_table))


# 1) Create tickers list from the table
tickers = sp500_table["Symbol"].astype(str).tolist()

# 2) Fix tickers with periods for Yahoo Finance (BRK.B -> BRK-B, BF.B -> BF-B)
tickers = [t.replace(".", "-") for t in tickers]

print("Number of tickers:", len(tickers))
print("First 10 tickers:", tickers[:10])

start_date = "2020-01-01"
end_date = "2025-12-31"

# 3) Download Adjusted Close prices for all tickers with retry logic
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    group_by="column",
    auto_adjust=False,
    threads=True,
    timeout=30  # Increase timeout to 30 seconds
)["Adj Close"]

# Retry failed tickers individually
failed_tickers = [col for col in tickers if col not in data.columns]
if failed_tickers:
    print(f"Retrying failed tickers: {failed_tickers}")
    for ticker in failed_tickers:
        try:
            retry_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                group_by="column",
                auto_adjust=False,
                timeout=60  # Longer timeout for retries
            )["Adj Close"]
            data[ticker] = retry_data
            print(f"Successfully downloaded {ticker}")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            # Fill with NaN if still failing
            data[ticker] = pd.Series(dtype=float)

print(data.head())

# 4) Compute daily log returns
log_returns = np.log(data / data.shift(1)).dropna(how="all")

print(log_returns.head())

print("Number of firms (columns):", log_returns.shape[1])
print("Number of observations (rows):", log_returns.shape[0])

print(log_returns.describe())
