from sklearn.model_selection import train_test_split
import yfinance as yf
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
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
sectors = sp500_table[['Symbol', 'GICS Sector']].copy()
sectors['Symbol'] = sectors['Symbol'].str.replace('.', '-', regex=False)
sectors = sectors.set_index('Symbol')
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


# Sector-level correlation heatmap
# Join returns with sector labels
sectors_aligned = sectors.reindex(log_returns.columns).dropna()

# Compute mean return per sector per day
sector_returns = (
    log_returns[sectors_aligned.index]
    .T
    .assign(sector=sectors_aligned['GICS Sector'].values)
    .groupby('sector')
    .mean()
    .T
)

# Compute sector-level correlation matrix
sector_corr = sector_returns.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    sector_corr,
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,          # show correlation values in each cell
    fmt='.2f',
    square=True,
    cbar_kws={'label': 'Correlation'},
    xticklabels=True,
    yticklabels=True
)
plt.title('Sector-Level Return Correlation Heatmap', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('sector_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: sector_correlation_heatmap.png")
plt.close()

#Step 5: Build Correlation Network

# Step 5: Build Correlation Network
corr_matrix = log_returns.corr()
adjacency = corr_matrix.abs().to_numpy().copy()
np.fill_diagonal(adjacency, 0)

print("Adjacency matrix shape:", adjacency.shape)
print("Diagonal sum (should be 0):", np.diag(adjacency).sum())
print("Min value (should be >=0):", adjacency.min())
print("Max value (should be <=1):", adjacency.max())

print("Adjacency matrix shape:", adjacency.shape)
print("Diagonal sum (should be 0):", np.diag(adjacency).sum())
print("Min value (should be >=0):", adjacency.min())
print("Max value (should be <=1):", adjacency.max())

tech_ticker = sectors[sectors['GICS Sector'] == 'Information Technology'].index
tech_ticker = [t for t in tech_ticker if t in log_returns.columns]
tech_corr = log_returns[tech_ticker].corr()
plt.figure(figsize=(18, 16))
sns.heatmap(
    tech_corr,
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,          # show correlation values in each cell
    fmt='.2f',
    square=True,
    cbar_kws={'label': 'Correlation'},
    xticklabels=True,
    yticklabels=True
)
plt.title('Information Technology Sector Correlation Heatmap', fontsize=14, fontweight='bold')

# Tech sector vs all other sectors correlation heatmap
sector_avg_returns = (
    log_returns[sectors_aligned.index]
    .T
    .assign(sector=sectors_aligned['GICS Sector'].values)
    .groupby('sector')
    .mean()
    .T
)

tech_vs_sectors = log_returns[tech_ticker].corrwith(sector_avg_returns.mean(axis=1))
tech_sector_corr = pd.DataFrame({sector: log_returns[tech_ticker].corrwith(sector_avg_returns[sector]) for sector in sector_avg_returns.columns}
)

plt.figure(figsize=(16, 12))
sns.heatmap(
    tech_sector_corr,
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    annot=False,
    square=False,
    cbar_kws={'label': 'Correlation'},
    xticklabels=True,
    yticklabels=True
)
plt.title('Tech Stocks vs All Sectors — Correlation Heatmap',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig('tech_vs_sectors_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: tech_vs_sectors_correlation_heatmap.png")
plt.close()

plt.figure(figsize=(16, 12))
sns.heatmap(
    tech_corr,
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    annot=False,
    square=True,
    cbar_kws={'label': 'Correlation'},
    xticklabels=True,
    yticklabels=True
)
plt.title('Information Technology Sector Correlation Heatmap', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig('tech_sector_correlation_heatmap.png', dpi=300, bbox_inches='tight')  
print("Saved: tech_sector_correlation_heatmap.png")
plt.close()  


#Step 6

# Step 6: Power Method for Eigenvector Centrality
def power_method(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for i in range(max_iter):
        v_next = A @ v
        v_next = v_next / np.linalg.norm(v_next)
        if np.linalg.norm(v_next - v) < tol:
            print(f"Converged after {i+1} iterations")
            break
        v = v_next
    
    eigenvalue = (v_next @ A @ v_next) / (v_next @ v_next)
    return eigenvalue, v_next

dominant_eigenvalue, dominant_eigenvector = power_method(adjacency)
print(f"Dominant Eigenvalue: {dominant_eigenvalue:.4f}")

# Store as Series with ticker labels
centrality_series = pd.Series(dominant_eigenvector, index=log_returns.columns, name='centrality')
print("\nTop 10 most central stocks:")
print(centrality_series.sort_values(ascending=False).head(10))

mag7 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']
print("\nMag 7 Centrality Rankings:")
print(centrality_series[mag7].sort_values(ascending=False))

print(f"\nMag 7 average centrality rank out of {len(centrality_series)} stocks:")
ranked = centrality_series.rank(ascending=False)
print(ranked[mag7].sort_values())

# Centrality for last year only (2025)
log_returns_2025 = log_returns['2025-01-01':'2025-12-31']

corr_matrix_2025 = log_returns_2025.corr()
adjacency_2025 = corr_matrix_2025.abs().to_numpy().copy()
np.fill_diagonal(adjacency_2025, 0)

_, eigenvector_2025 = power_method(adjacency_2025)
centrality_2025 = pd.Series(eigenvector_2025, index=log_returns_2025.columns, name='centrality')

print("\nTop 10 most central stocks (2025):")
print(centrality_2025.sort_values(ascending=False).head(10))

print("\nMag 7 Centrality Rankings (2025):")
print(centrality_2025[mag7].sort_values(ascending=False))

print(f"\nMag 7 centrality ranks out of {len(centrality_2025)} stocks (2025):")
ranked_2025 = centrality_2025.rank(ascending=False)
print(ranked_2025[mag7].sort_values())

# Time window analysis
windows = {
    'Pre-COVID (2019)': ('2019-01-01', '2019-12-31'),
    'COVID Crash (2020 Q1)': ('2020-01-01', '2020-06-30'),
    'Recovery (2020 Q3-Q4)': ('2020-07-01', '2020-12-31'),
    'Bull Market (2021)': ('2021-01-01', '2021-12-31'),
    'Rate Hikes (2022)': ('2022-01-01', '2022-12-31'),
    'AI Boom (2023-2024)': ('2023-01-01', '2024-12-31'),
    '2025': ('2025-01-01', '2025-12-31')
}

mag7 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']

window_ranks = {}

for label, (start, end) in windows.items():
    window_returns = log_returns[start:end]
    
    # Skip if not enough data
    if len(window_returns) < 30:
        print(f"Skipping {label} — not enough data")
        continue
    
    # Only use tickers available in this window
    valid_tickers = window_returns.dropna(axis=1, thresh=int(0.8 * len(window_returns))).columns
    window_returns = window_returns[valid_tickers].dropna()
    
    # Build adjacency and compute centrality
    adj = window_returns.corr().abs().to_numpy().copy()
    np.fill_diagonal(adj, 0)
    _, eigvec = power_method(adj)
    centrality_window = pd.Series(eigvec, index=valid_tickers)
    
    # Store mag 7 ranks (only ones available in this window)
    ranked = centrality_window.rank(ascending=False)
    available_mag7 = [t for t in mag7 if t in ranked.index]
    window_ranks[label] = ranked[available_mag7]

# Build comparison DataFrame
rank_df = pd.DataFrame(window_ranks).T
print("\nMag 7 Centrality Ranks Across Time Windows:")
print(rank_df)

# Plot
plt.figure(figsize=(14, 7))
for ticker in mag7:
    if ticker in rank_df.columns:
        plt.plot(rank_df.index, rank_df[ticker], marker='o', label=ticker)

plt.gca().invert_yaxis()  # Lower rank = more central, so invert so "better" is higher
plt.title("Mag 7 Centrality Rank Over Time\n(Lower = More Central)", fontsize=14, fontweight='bold')
plt.ylabel("Centrality Rank")
plt.xlabel("Time Period")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mag7_centrality_over_time.png', dpi=300)
print("Saved: mag7_centrality_over_time.png")
plt.close()

# Market cap weight of Mag 7 vs rest of S&P 500
mag7_info = {}
for ticker in mag7:
    try:
        mag7_info[ticker] = yf.Ticker(ticker).info.get('marketCap', np.nan)
    except:
        mag7_info[ticker] = np.nan

mag7_caps = pd.Series(mag7_info)

# Get all S&P 500 market caps
all_caps = {}
for ticker in log_returns.columns:
    try:
        all_caps[ticker] = yf.Ticker(ticker).info.get('marketCap', np.nan)
    except:
        all_caps[ticker] = np.nan

all_caps_series = pd.Series(all_caps)

mag7_total = mag7_caps.sum()
sp500_total = all_caps_series.sum()
others_total = sp500_total - mag7_total

plt.figure(figsize=(8, 8))
plt.pie(
    [mag7_total, others_total],
    labels=['Mag 7', 'Rest of S&P 500'],
    autopct='%1.1f%%',
    colors=['#e74c3c', '#3498db'],
    startangle=90
)
plt.title('Mag 7 vs Rest of S&P 500 — Market Cap Weight', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mag7_marketcap_weight.png', dpi=300)
print("Saved: mag7_marketcap_weight.png")
plt.close()

# Correlation of each Mag 7 stock with the equal-weighted market return
market_return = log_returns.mean(axis=1)  # proxy for market

mag7_market_corr = {}
for ticker in mag7:
    if ticker in log_returns.columns:
        mag7_market_corr[ticker] = log_returns[ticker].corr(market_return)

# Do the same for all stocks
all_market_corr = log_returns.corrwith(market_return)

mag7_corr_series = pd.Series(mag7_market_corr)

plt.figure(figsize=(10, 5))
plt.hist(all_market_corr, bins=50, alpha=0.6, label='All S&P 500 stocks', color='steelblue')
for ticker, corr in mag7_corr_series.items():
    plt.axvline(x=corr, linestyle='--', linewidth=2, label=f'{ticker} ({corr:.2f})')
plt.xlabel('Correlation with Market Return')
plt.ylabel('Frequency')
plt.title('Individual Stock Correlation with Market\nMag 7 Highlighted', fontsize=14, fontweight='bold')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('mag7_market_correlation.png', dpi=300)
print("Saved: mag7_market_correlation.png")
plt.close()

# Build a comparison table
cap_rank = all_caps_series.rank(ascending=False)
cent_rank = centrality_series.rank(ascending=False)

comparison = pd.DataFrame({
    'Market Cap Rank': cap_rank[mag7],
    'Centrality Rank': cent_rank[mag7]
}).sort_values('Market Cap Rank')

print("\nMag 7 — Market Cap Rank vs Centrality Rank:")
print(comparison)

# Scatter plot for all stocks
common = all_caps_series.dropna().index.intersection(centrality_series.index)
plt.figure(figsize=(10, 7))
plt.scatter(
    np.log(all_caps_series[common]),
    centrality_series[common],
    alpha=0.3, color='steelblue', label='All stocks'
)
for ticker in mag7:
    if ticker in common:
        plt.scatter(
            np.log(all_caps_series[ticker]),
            centrality_series[ticker],
            color='red', s=100, zorder=5
        )
        plt.annotate(ticker,
            (np.log(all_caps_series[ticker]), centrality_series[ticker]),
            textcoords='offset points', xytext=(5, 5), fontsize=9
        )
plt.xlabel('Log Market Cap')
plt.ylabel('Eigenvector Centrality')
plt.title('Market Cap vs Network Centrality\nMag 7 Highlighted in Red', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('marketcap_vs_centrality.png', dpi=300)
print("Saved: marketcap_vs_centrality.png")
plt.close()


#Step 7

# 1. Convert centrality series to DataFrame with ticker column
centrality_df = centrality_series.reset_index()
centrality_df.columns = ['Symbol', 'centrality']

# 2. Join with sectors on Symbol
centrality_df = centrality_df.join(sectors, on='Symbol')

# 3. Group by sector, take mean, sort
sector_centrality = centrality_df.groupby('GICS Sector')['centrality'].mean().sort_values(ascending=False)

# 4. Print and plot

print(sector_centrality)

#Step 8 Linear Regression Analysis


# 1. Fetch market caps
market_caps = {}
for i, ticker in enumerate(centrality_df['Symbol']):
    if i % 50 == 0:
        print(f"Fetching {i}/{len(centrality_df)}...")
    try:
        market_caps[ticker] = yf.Ticker(ticker).info.get('marketCap', np.nan)
    except:
        market_caps[ticker] = np.nan

# 2. Add market cap and volatility to centrality_df
centrality_df['market_cap'] = centrality_df['Symbol'].map(market_caps)
centrality_df['volatility'] = log_returns.std().values

# 3. Drop missing values
reg_df = centrality_df.dropna()

# 4. Build X and y
# X = market cap, volatility, sector dummies
# y = centrality
X = pd.get_dummies(reg_df[['market_cap', 'volatility', 'GICS Sector']], drop_first=True)
y = reg_df['centrality']

X_scaled = StandardScaler().fit_transform(X)
model = LinearRegression().fit(X_scaled, y)

# 6. Results
print(f"R²: {model.score(X_scaled, y):.4f}")
coef_df = pd.Series(model.coef_, index=X.columns).sort_values()
print(coef_df)

plt.figure(figsize=(12, 7))
colors = ['#e74c3c' if x > 0 else '#3498db' for x in coef_df.values]
coef_df.plot(kind='barh', color=colors)

plt.axvline(x=0, color='black', linewidth=1.5, linestyle='--')
plt.xlabel('Coefficient (Standardized)', fontsize=12)
plt.title('Regression Coefficients — What Drives Network Centrality?\n(Red = More Central, Blue = Less Central)', 
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_coefficients.png', dpi=300)
print("Saved: regression_coefficients.png")
plt.close()