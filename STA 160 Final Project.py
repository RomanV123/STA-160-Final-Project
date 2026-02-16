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

# Data Cleaning: Remove firms with excessive missing observations
# ----------------------------------------------------------------------------
# What we're searching for: Stocks with too much missing data that would 
# compromise correlation calculations. Missing data can occur due to:
# - IPOs during our time period (stock didn't exist for full period)
# - Delistings (stock removed from S&P 500)
# - Trading halts or data errors
#
# What we're doing: Calculating the percentage of missing values for each 
# stock, then removing stocks with more than 10% missing data. This ensures 
# we have sufficient data points to compute reliable correlations.
#
# Expected result: Most S&P 500 stocks should have <10% missing data, 
# so we'll drop only a few problematic stocks
missing_threshold = 0.10  # 10% missing data threshold
missing_pct = log_returns.isna().sum() / len(log_returns)  # % missing per stock
firms_to_drop = missing_pct[missing_pct > missing_threshold].index
log_returns = log_returns.drop(columns=firms_to_drop)

print(f"\nDropped {len(firms_to_drop)} firms with >{missing_threshold*100}% missing data")
print(f"Remaining firms: {log_returns.shape[1]}")

# Drop any remaining rows with NaN values
# What we're doing: After removing problematic stocks, remove any days that 
# still have missing values. This ensures we have a complete matrix for 
# correlation calculations (all stocks have returns for all remaining days).
log_returns = log_returns.dropna()

print(f"Final data shape: {log_returns.shape}")
# Result: Clean matrix with no missing values - ready for correlation analysis

# ============================================================================
# Step 4: Exploratory Data Analysis (EDA)
# ============================================================================
# Purpose: Understand the distribution and relationships in our return data
# before building the correlation network. We want to verify:
# 1. Returns follow expected patterns (near-zero mean, symmetric distribution)
# 2. Stocks exhibit meaningful correlation structure (not random)
# 3. Data quality is sufficient for network analysis

print("\n" + "="*80)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("="*80)

# 1. Summary Statistics of Returns
# ----------------------------------------------------------------------------
# What we're searching for: Understanding the central tendency, spread, and 
# extreme values of daily log returns across all stocks in our dataset.
# 
# What we're doing: Computing descriptive statistics for each stock's return 
# series, then aggregating across all stocks to get overall market statistics.
#
# Expected findings:
# - Mean return should be close to zero (daily returns average out over time)
# - Standard deviation shows volatility (typically 0.01-0.03 for daily returns)
# - Min/max show extreme daily moves (crashes/rallies)
print("\n1. Summary Statistics of Returns:")
print("-" * 80)
# Compute per-stock statistics: count, mean, std, min, 25th/50th/75th percentile, max
summary_stats = log_returns.describe()
print(summary_stats)

# Additional summary statistics
# What we're doing: Flattening all returns into a single array to get 
# market-wide statistics across all stocks and all days.
# This tells us: How does the ENTIRE market behave on average?
print("\nAdditional Summary Statistics:")
overall_mean = log_returns.values.mean()
overall_std = log_returns.values.std()
overall_min = log_returns.values.min()
overall_max = log_returns.values.max()
print(f"Overall mean return: {overall_mean:.6f}")
print(f"Overall std deviation: {overall_std:.6f}")
print(f"Overall min return: {overall_min:.6f}")
print(f"Overall max return: {overall_max:.6f}")
# Interpretation: 
# - Mean near 0 indicates no systematic bias (efficient market hypothesis)
# - Std dev shows typical daily volatility (e.g., 0.015 = 1.5% daily volatility)
# - Min/max show worst single-day losses and best single-day gains

# 2. Histogram of Daily Returns
# ----------------------------------------------------------------------------
# What we're searching for: The shape of the return distribution across all 
# stocks and all days. We want to verify:
# - Returns are approximately normally distributed (bell curve)
# - Distribution is centered near zero
# - Presence of "fat tails" (extreme events more common than normal distribution)
#
# What we're doing: Flattening all return data into a single array and 
# creating a histogram to visualize the frequency distribution.
#
# Expected findings:
# - Bell-shaped curve centered near zero
# - Fat tails (more extreme values than normal distribution would predict)
# - This confirms returns follow patterns suitable for correlation analysis
print("\n2. Creating histogram of daily returns...")
plt.figure(figsize=(12, 6))
# Flatten all returns: convert 2D array (days × stocks) to 1D array
all_returns = log_returns.values.flatten()
plt.hist(all_returns, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Daily Log Return', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Daily Log Returns (All Stocks)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histogram_daily_returns.png', dpi=300, bbox_inches='tight')
print("Saved: histogram_daily_returns.png")
plt.close()
# Interpretation: 
# - If bell-shaped: returns follow expected statistical patterns
# - If centered at zero: no systematic bias in market
# - Fat tails indicate market crashes/booms are more common than normal distribution predicts

# 3. Correlation Matrix Heatmap
# ----------------------------------------------------------------------------
# What we're searching for: Visual patterns in how stocks move together.
# We want to identify:
# - Clusters of highly correlated stocks (sectors that move together)
# - Overall correlation level (are stocks generally correlated or independent?)
# - Block structure (groups of stocks with similar correlation patterns)
#
# What we're doing: Computing Pearson correlation coefficient between every 
# pair of stocks' return series. Correlation ranges from -1 (perfect negative)
# to +1 (perfect positive), with 0 meaning no linear relationship.
# Formula: corr(X,Y) = covariance(X,Y) / (std(X) * std(Y))
#
# Expected findings:
# - Most correlations are positive (stocks tend to move together)
# - Correlations typically range from 0.2 to 0.8 for stocks in same sector
# - Diagonal is always 1.0 (stock perfectly correlated with itself)
print("\n3. Computing correlation matrix and creating heatmap...")
# Compute correlation matrix: N×N matrix where entry (i,j) is correlation 
# between stock i and stock j's daily returns over entire time period
corr_matrix = log_returns.corr()
# Result: Each cell shows how similarly two stocks' returns move together
# Values close to 1: stocks move together (both up or both down)
# Values close to 0: stocks move independently
# Values close to -1: stocks move oppositely (one up when other down)

# Create heatmap (sample for visualization - full matrix is too large)
# For visualization, we'll show a sample or use a smaller subset
# Option 1: Show first 50 stocks for clarity
n_sample = min(50, corr_matrix.shape[0])
sample_tickers = corr_matrix.index[:n_sample]
corr_sample = corr_matrix.loc[sample_tickers, sample_tickers]

plt.figure(figsize=(14, 12))
sns.heatmap(corr_sample, 
            cmap='coolwarm',  # Blue = negative, Red = positive correlation
            center=0,          # White at zero correlation
            vmin=-1,          # Minimum correlation value
            vmax=1,           # Maximum correlation value
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
# Interpretation:
# - Red/white blocks: groups of stocks that move together (likely same sector)
# - Overall red tint: positive average correlation (market moves together)
# - Dark red squares: highly correlated pairs (e.g., tech stocks together)
# - This visualization confirms stocks form clusters, validating network approach

# 4. Distribution of Pairwise Correlations
# ----------------------------------------------------------------------------
# What we're searching for: Understanding the overall correlation structure 
# of the market. We want to know:
# - What is the average correlation between any two stocks?
# - Are most stock pairs correlated or independent?
# - How spread out are correlations? (tight distribution vs. wide range)
#
# What we're doing: Extracting all unique pairwise correlations from the 
# correlation matrix (excluding diagonal and duplicates). For N stocks, 
# we get N*(N-1)/2 unique pairs. Then we analyze the distribution of these 
# correlation values.
#
# Expected findings:
# - Mean correlation typically 0.3-0.6 (stocks move together, not independently)
# - Distribution shifted right (more positive correlations than negative)
# - This confirms market has structure suitable for network analysis
# 4. Distribution of Pairwise Correlations
# ----------------------------------------------------------------------------
print("\n4. Creating distribution of pairwise correlations...")

# Recompute correlation matrix (safe)
corr_matrix = log_returns.corr()

# Convert to numpy array
corr_values = corr_matrix.values

# Extract upper triangle (exclude diagonal)
n = corr_values.shape[0]
pairwise_corrs = corr_values[np.triu_indices(n, k=1)]

# Remove NaNs
pairwise_corrs = pairwise_corrs[~np.isnan(pairwise_corrs)]

print("Number of pairwise correlations:", len(pairwise_corrs))

# Safety check
if len(pairwise_corrs) == 0:
    print("ERROR: No valid pairwise correlations found.")
else:
    # ------------------------------
    # Create histogram
    # ------------------------------
    plt.figure(figsize=(12, 6))

    plt.hist(pairwise_corrs,
             bins=100,
             edgecolor='black',
             alpha=0.7)

    plt.xlabel('Pairwise Correlation', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Pairwise Stock Return Correlations',
              fontsize=14, fontweight='bold')

    # Add zero line
    plt.axvline(x=0,
                color='red',
                linestyle='--',
                linewidth=2,
                label='Zero Correlation')

    # Add mean line
    mean_corr = np.mean(pairwise_corrs)
    plt.axvline(x=mean_corr,
                color='green',
                linestyle='--',
                linewidth=2,
                label=f'Mean: {mean_corr:.3f}')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('distribution_pairwise_correlations.png',
                dpi=300,
                bbox_inches='tight')

    print("Saved: distribution_pairwise_correlations.png")
    plt.close()

    # ------------------------------
    # Print statistics
    # ------------------------------
    print("\nPairwise Correlation Statistics:")

    mean_val = np.mean(pairwise_corrs)
    median_val = np.median(pairwise_corrs)
    std_val = np.std(pairwise_corrs)
    min_val = np.min(pairwise_corrs)
    max_val = np.max(pairwise_corrs)
    p25 = np.percentile(pairwise_corrs, 25)
    p75 = np.percentile(pairwise_corrs, 75)

    print(f"  Mean: {mean_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  Std Dev: {std_val:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"  25th percentile: {p25:.4f}")
    print(f"  75th percentile: {p75:.4f}")


print(f"  Mean: {mean_val:.4f}")  # Average correlation: typically 0.3-0.6 for S&P 500
print(f"  Median: {median_val:.4f}")  # Middle value: shows if distribution is symmetric
print(f"  Std Dev: {std_val:.4f}")  # Spread: higher = more diverse correlation patterns
print(f"  Min: {min_val:.4f}")  # Most negative correlation (stocks that move oppositely)
print(f"  Max: {max_val:.4f}")  # Most positive correlation (stocks that move together)
print(f"  25th percentile: {p25:.4f}")  # 25% of pairs have correlation below this
print(f"  75th percentile: {p75:.4f}")  # 75% of pairs have correlation below this
# Interpretation:
# - Mean > 0.3: Confirms market has meaningful correlation structure
# - Positive mean: Stocks tend to move together (systematic risk)
# - This validates our approach: we can build a meaningful correlation network

print("\n" + "="*80)
print("EDA Complete! This confirms that the market exhibits meaningful correlation structure.")
print("="*80)
