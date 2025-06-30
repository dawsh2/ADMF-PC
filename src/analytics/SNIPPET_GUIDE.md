# Analytics Snippets Guide

This guide explains the analytics "lego blocks" system for interactive Jupyter analysis.

## Overview

The system consists of:
1. **Papermill Template** (`templates/universal_analysis.ipynb`) - Automated baseline analysis
2. **SQL Queries** (`queries/`) - Complex, reusable SQL queries
3. **Python Snippets** (`snippets/`) - Analysis code blocks you can load and customize
4. **Helper Functions** (`snippets/helpers.py`) - Shared utilities

## Usage Workflow

### Step 1: Run the Template
```bash
python main.py config/your_sweep.yaml --notebook
```
This generates a notebook with comprehensive baseline analysis.

### Step 2: Open the Generated Notebook
The notebook will be at `results/run_*/analysis_universal.ipynb`

### Step 3: Load and Customize Snippets
In new cells, use `%load` to insert analysis code:

```python
%load src/analytics/snippets/exploratory/signal_frequency.py
```

Edit the parameters at the top, then run the cell.

## Available Snippets

### Exploratory Analysis

#### `signal_frequency.py`
Analyzes trading frequency patterns across strategies.
- Parameters: `MIN_SIGNALS`, `MIN_SHARPE`, `TIMEFRAME`
- Outputs: Signal frequency statistics, visualizations by strategy type

#### `top_performers.py`
Finds and analyzes top performing strategies.
- Parameters: `MIN_SHARPE`, `MIN_TRADES`, `MAX_DRAWDOWN_LIMIT`, `TOP_N`
- Outputs: Top strategies table, risk-return scatter plots

#### `parameter_sweep.py`
Analyzes parameter sensitivity for a specific strategy type.
- Parameters: `STRATEGY_TYPE`, `PARAM1`, `PARAM2`, `MIN_INSTANCES`
- Outputs: Parameter heatmaps, correlation analysis

### Ensemble Building

#### `find_uncorrelated.py`
Identifies uncorrelated strategies for diversification.
- Parameters: `MIN_SHARPE`, `CORRELATION_THRESHOLD`, `TOP_N_CANDIDATES`
- Outputs: Correlation matrix, selected ensemble

#### `optimize_weights.py`
Optimizes portfolio weights using mean-variance optimization.
- Parameters: `RISK_FREE_RATE`, `MAX_WEIGHT`, `TARGET_VOL`
- Outputs: Optimal weights, efficient frontier plot

### Regime Analysis

#### `volatility_regimes.py`
Analyzes performance in different volatility environments.
- Parameters: `VOLATILITY_LOOKBACK`, `VOL_PERCENTILES`, `MIN_SHARPE`
- Outputs: Regime performance tables, specialization analysis

## SQL Queries

Pure SQL files that can be loaded by snippets:

- `top_strategies.sql` - Parameterized query for finding top performers
- `correlation_pairs.sql` - Find correlated/uncorrelated strategy pairs
- `regime_performance.sql` - Complex regime analysis query
- `signal_patterns.sql` - Detect signal patterns and clustering

## Helper Functions

Load with: `%load src/analytics/snippets/helpers.py`

Key functions:
- `load_sql_query(filename)` - Load SQL from queries directory
- `get_top_sharpe(con, n)` - Quick function for top strategies
- `plot_strategy_comparison(df, metric)` - Comparison visualizations
- `export_ensemble_config(ensemble_df)` - Export for production

## Tips for Effective Use

### 1. Parameter Editing
Each snippet has parameters at the top. Always review and adjust these before running:

```python
# Edit these parameters as needed:
MIN_SHARPE = 1.5  # Change from 1.0 to 1.5
TOP_N = 20        # Change from 10 to 20
```

### 2. Building on Previous Results
Snippets often use variables created by previous analysis:

```python
# After running find_uncorrelated.py, you'll have:
# - uncorrelated_ensemble (DataFrame)
# - ensemble_pairs (correlation data)

# Then optimize_weights.py can use these:
%load src/analytics/snippets/ensembles/optimize_weights.py
```

### 3. Custom SQL Queries
You can always write custom DuckDB queries:

```python
# Direct query on the connection
custom_df = con.execute("""
    SELECT * FROM strategies 
    WHERE strategy_type = 'momentum' 
    AND sharpe_ratio > 2.0
""").df()
```

### 4. Combining Snippets
Mix and match snippets for comprehensive analysis:

```python
# 1. Find top performers
%load src/analytics/snippets/exploratory/top_performers.py

# 2. Analyze their correlations
%load src/analytics/snippets/ensembles/find_uncorrelated.py

# 3. Check regime performance
%load src/analytics/snippets/regime/volatility_regimes.py

# 4. Optimize final weights
%load src/analytics/snippets/ensembles/optimize_weights.py
```

## Creating Your Own Snippets

To add a new snippet:

1. Create a `.py` file in the appropriate subdirectory
2. Put editable parameters at the top
3. Include descriptive comments
4. Print summary results
5. Create visualizations where helpful
6. Store results in variables for downstream use

Example structure:
```python
# My custom analysis - description
# Parameters:
PARAM1 = 10
PARAM2 = 'value'

# Analysis code
print(f"Running analysis with {PARAM1}...")
results_df = con.execute("SELECT ...").df()

# Visualization
plt.figure(figsize=(10, 6))
# ... plotting code ...

# Store for reuse
my_analysis_results = results_df
print(f"Results stored in 'my_analysis_results'")
```

## Troubleshooting

### "No module named 'analytics'"
The notebook's path setup cell should handle this, but if not:
```python
import sys
sys.path.append('/path/to/project/root')
```

### "NameError: name 'con' is not defined"
Make sure you've run the Setup cells from the template first. The DuckDB connection is created there.

### "No market_data found"
Some snippets expect `market_data` DataFrame. Ensure the template found and loaded market data, or load it manually.

### Performance Issues
For large backtests, consider:
- Using the `MIN_STRATEGIES` parameter in the template
- Filtering strategies before correlation calculation
- Using sampling for visualization