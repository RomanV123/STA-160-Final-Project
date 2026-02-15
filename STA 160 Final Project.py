import yfinance as yf
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("Fetching S&P 500 constituent list from Wikipedia...")
# Use requests to fetch, then parse with pandas
response = requests.get(url, headers=headers)
response.raise_for_status()  # Raise an error for bad status codes

print("Parsing HTML tables...")
# Read HTML tables - need to use StringIO to pass HTML string properly
html_io = StringIO(response.text)
tables = pd.read_html(html_io, header=0)
    
sp500_table = tables[0]  # First table is the S&P 500 constituents
print("Successfully parsed S&P 500 table!")

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

# Remove firms with excessive missing observations (e.g., IPOs)
# Drop columns with more than 10% missing values
missing_threshold = 0.10
missing_pct = log_returns.isna().sum() / len(log_returns)
firms_to_drop = missing_pct[missing_pct > missing_threshold].index
log_returns = log_returns.drop(columns=firms_to_drop)

print(f"\nDropped {len(firms_to_drop)} firms with >{missing_threshold*100}% missing data")
print(f"Remaining firms: {log_returns.shape[1]}")

# Drop any remaining rows with NaN values
log_returns = log_returns.dropna()

print(f"Final data shape: {log_returns.shape}")

# ============================================================================
# Step 4: Exploratory Data Analysis (EDA)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("="*80)

# 1. Summary statistics of returns (mean, standard deviation, min, max)
print("\n1. Summary Statistics of Returns:")
print("-" * 80)
summary_stats = log_returns.describe()
print(summary_stats)

# Additional summary statistics
print("\nAdditional Summary Statistics:")
print(f"Overall mean return: {log_returns.values.mean():.6f}")
print(f"Overall std deviation: {log_returns.values.std():.6f}")
print(f"Overall min return: {log_returns.values.min():.6f}")
print(f"Overall max return: {log_returns.values.max():.6f}")

# 2. Histogram of daily returns
print("\n2. Creating histogram of daily returns...")
plt.figure(figsize=(12, 6))
plt.hist(log_returns.values.flatten(), bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Daily Log Return', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Daily Log Returns (All Stocks)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histogram_daily_returns.png', dpi=300, bbox_inches='tight')
print("Saved: histogram_daily_returns.png")
plt.close()

# 3. Correlation matrix heatmap
print("\n3. Computing correlation matrix and creating heatmap...")
# Compute correlation matrix
corr_matrix = log_returns.corr()

# Create heatmap (sample for visualization - full matrix is too large)
# For visualization, we'll show a sample or use a smaller subset
# Option 1: Show first 50 stocks for clarity
n_sample = min(50, corr_matrix.shape[0])
sample_tickers = corr_matrix.index[:n_sample]
corr_sample = corr_matrix.loc[sample_tickers, sample_tickers]

plt.figure(figsize=(14, 12))
sns.heatmap(corr_sample, 
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation'},
            xticklabels=False,
            yticklabels=False)
plt.title(f'Correlation Matrix Heatmap (Sample: First {n_sample} Stocks)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: correlation_heatmap.png (showing first {n_sample} stocks)")
plt.close()

# 4. Distribution of pairwise correlations
print("\n4. Creating distribution of pairwise correlations...")
# Extract upper triangle of correlation matrix (excluding diagonal)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
pairwise_corrs = corr_matrix.where(mask).stack().values

plt.figure(figsize=(12, 6))
plt.hist(pairwise_corrs, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Pairwise Correlation', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Pairwise Stock Return Correlations', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Correlation')
plt.axvline(x=pairwise_corrs.mean(), color='green', linestyle='--', linewidth=2, 
            label=f'Mean: {pairwise_corrs.mean():.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('distribution_pairwise_correlations.png', dpi=300, bbox_inches='tight')
print("Saved: distribution_pairwise_correlations.png")
plt.close()

# Print correlation distribution statistics
print("\nPairwise Correlation Statistics:")
print(f"  Mean: {pairwise_corrs.mean():.4f}")
print(f"  Median: {np.median(pairwise_corrs):.4f}")
print(f"  Std Dev: {pairwise_corrs.std():.4f}")
print(f"  Min: {pairwise_corrs.min():.4f}")
print(f"  Max: {pairwise_corrs.max():.4f}")
print(f"  25th percentile: {np.percentile(pairwise_corrs, 25):.4f}")
print(f"  75th percentile: {np.percentile(pairwise_corrs, 75):.4f}")

print("\n" + "="*80)
print("EDA Complete! This confirms that the market exhibits meaningful correlation structure.")
print("="*80)
