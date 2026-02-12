

# README

## S&P 500 Correlation Network and Sector Centrality Analysis

### Project Overview

This project investigates the question:

**Which sectors are most structurally central in the stock return correlation network of the S&P 500?**

Rather than measuring influence using market capitalization or index weight, we model the S&P 500 as a network of stocks connected through return correlations. We compute eigenvector centrality using the power method to identify structurally influential stocks and then aggregate centrality scores at the sector level to determine which sectors occupy the most central positions in the network.

The analysis covers daily data from January 2020 through December 2025.

---

## Conceptual Framework

1. Each stock is treated as a node in a network.
2. Edges between nodes are defined by the absolute value of return correlations.
3. A stock is considered influential if it is strongly connected to other highly connected stocks.
4. Eigenvector centrality captures this recursive definition of influence.
5. Sector-level influence is computed by averaging centrality scores within each sector.

This allows us to quantify structural importance in the market without assuming causality.

---

## Data Sources

* Daily adjusted closing prices are obtained using the Yahoo Finance API (`yfinance`).
* S&P 500 constituent list and sector classifications are retrieved from publicly available index constituent data.
* Market capitalization data is retrieved from Yahoo Finance when needed for regression analysis.

Time Period:

* January 1, 2020 through December 31, 2025.

---

## Software Requirements

Python 3.9+

Required libraries:

```
pip install yfinance pandas numpy matplotlib seaborn requests
```

---

## Step-by-Step Instructions

### Step 1: Retrieve S&P 500 Constituents

* Load the list of S&P 500 companies.
* Extract ticker symbols.
* Adjust ticker format for Yahoo Finance compatibility (replace "." with "-").

Output:

* A list of approximately 500 tickers.

---

### Step 2: Download Historical Price Data

* Use `yfinance.download()` to retrieve daily adjusted close prices.
* Specify start and end dates (2020–2025).
* Store results in a DataFrame where:

  * Rows = dates
  * Columns = tickers

---

### Step 3: Compute Daily Log Returns

For each stock:

log_return_t = log(P_t / P_{t-1})

* Drop missing values.
* Remove firms with excessive missing observations (e.g., IPOs).

Output:

* T × N matrix of daily log returns.

---

### Step 4: Exploratory Data Analysis (EDA)

Perform the following:

1. Summary statistics of returns (mean, standard deviation, min, max).
2. Histogram of daily returns.
3. Correlation matrix heatmap.
4. Distribution of pairwise correlations.

This confirms that the market exhibits meaningful correlation structure.

---

### Step 5: Construct Correlation Network

* Compute correlation matrix of returns.
* Take absolute value of correlations to form weighted adjacency matrix.

This matrix represents the financial network.

---

### Step 6: Compute Eigenvector Centrality (Power Method)

We compute the leading eigenvector of the adjacency matrix using the power method:

Initialize v randomly.
Iteratively compute:

v_(k+1) = A v_k / ||A v_k||

Repeat until convergence.

Output:

* Centrality score for each stock.

---

### Step 7: Sector-Level Aggregation

* Merge stock centrality scores with sector classifications.
* Compute average centrality per sector.
* Rank sectors by mean centrality.

This identifies which sectors are structurally central in the market network.

---

### Step 8: Regression Analysis (Interpretation)

Run cross-sectional regression:

Centrality_i = β₀ + β₁ MarketCap_i + β₂ Volatility_i + SectorIndicators_i + ε_i

Purpose:

* Determine whether structural centrality is explained by traditional measures.
* Test whether specific sectors exhibit statistically higher centrality.

Regression does not determine influence; it explains observed patterns.

---

## Expected Outputs

1. Ranked list of stocks by eigenvector centrality.
2. Ranked list of sectors by average centrality.
3. Visualization of correlation structure.
4. Regression output explaining drivers of centrality.

---

## Interpretation

High centrality indicates that a stock (and its sector) is deeply embedded in the market’s return structure. Such stocks may play an important role in systemic comovement and market synchronization.

This project does not claim causality or provide investment advice. It measures structural position within a financial network.

---

## Limitations

* Uses current S&P 500 constituents rather than historical membership.
* Correlation does not imply causation.
* Centrality measures structural importance, not predictive power.

---

## Research Contribution

This project contributes by:

* Modeling the S&P 500 as a financial network.
* Quantifying structural influence using eigenvector centrality.
* Comparing sector-level dominance to public perceptions of market concentration.

---

If you'd like, I can now:

* Write a shorter, submission-ready version
* Or create a README formatted specifically for GitHub
* Or help you implement the power method step cleanly next

You’re building a real quantitative research project now.
