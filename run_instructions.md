# How to Run the Script

## Step 1: Install Required Packages

Open your terminal and run:

```bash
pip3 install yfinance pandas numpy matplotlib seaborn requests
```

Or if you're using a virtual environment:

```bash
python3 -m pip install yfinance pandas numpy matplotlib seaborn requests
```

## Step 2: Run the Script

Navigate to the project directory and run:

```bash
cd /Users/shotaruo/STA-160-Final-Project
python3 "STA 160 Final Project.py"
```

## What to Expect

The script will:
1. Download S&P 500 constituent list from Wikipedia
2. Download historical price data for all stocks (2020-2025)
3. Compute log returns
4. Perform EDA and generate visualizations:
   - `histogram_daily_returns.png`
   - `correlation_heatmap.png`
   - `distribution_pairwise_correlations.png`

**Note:** The data download may take several minutes depending on your internet connection.
