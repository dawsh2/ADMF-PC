# ADMF-PC Analytics Library

Interactive analysis tools for sparse signal traces, designed for Jupyter notebook workflows.

## Quick Start

```python
from analytics import quick_analysis

# Load latest results
ta, pd = quick_analysis('config/keltner/results/latest')

# View strategy summary
ta.summary()

# Compare filter effectiveness
ta.compare_filters()

# Discover patterns
pd.find_signal_sequences()
```

## Core Components

### TraceAnalysis
Main interface for analyzing trace data:
- `summary()` - Get overview of all strategies
- `compare_filters()` - Analyze filter effectiveness
- `find_patterns()` - Discover common signal patterns
- `get_trades()` - Extract trade sequences
- `sql(query)` - Run custom DuckDB queries

### PatternDiscovery
Tools for finding and saving trading patterns:
- `find_signal_sequences()` - Common signal patterns
- `find_profitable_conditions()` - Market conditions analysis
- `create_pattern()` - Create reusable patterns
- `save_pattern()` - Save to pattern library
- `suggest_explorations()` - Get analysis ideas

### TradeAnalyzer
Reconstruct and analyze trades from sparse signals:
- `summary_stats()` - Overall performance metrics
- `by_strategy()` - Per-strategy breakdown
- `to_dataframe()` - Export trades for analysis

## Key Features

### 1. Works with Sparse Format
The library understands the sparse signal format (idx, val, px) and automatically handles signal reconstruction.

### 2. Pre-built Queries
Common analysis queries are pre-built and parameterized:
```python
# Use pre-built queries
from analytics import TraceQueries
print(TraceQueries.TOP_PERFORMERS)
```

### 3. Pattern Library
Save and reuse discovered patterns:
```python
# Create pattern from successful analysis
pattern = pd.create_pattern(
    name="Volatility Breakout",
    description="Works in high volatility",
    query="SELECT ...",
    tags=['volatility', 'breakout']
)
pd.save_pattern(pattern)

# Search patterns later
patterns = pd.library.search(tags=['volatility'])
```

### 4. Interactive Exploration
Designed for Jupyter notebooks with DataFrame outputs:
```python
# Returns DataFrames for easy plotting
ta.summary().plot.scatter('total_signals', 'filter')
ta.get_trades()['return_pct'].hist(bins=50)
```

## Workflow

1. **Load Data**
   ```python
   ta, pd = quick_analysis('path/to/results')
   ```

2. **Explore**
   ```python
   # Overview
   ta.summary()
   
   # Custom queries
   ta.sql("SELECT * FROM signals WHERE ...")
   ```

3. **Discover Patterns**
   ```python
   # Find what works
   pd.find_profitable_conditions()
   pd.find_signal_sequences()
   ```

4. **Save Findings**
   ```python
   # Save for reuse
   pattern = pd.create_pattern(...)
   pd.save_pattern(pattern)
   ```

5. **Export if Needed**
   ```python
   # For external tools
   ta.export_for_mining()
   ```

## Design Philosophy

- **Interactive First**: Built for exploratory analysis in Jupyter
- **SQL Powered**: DuckDB for fast queries on parquet files
- **No CLI**: Direct Python API, no command-line complexity
- **Pattern Memory**: Save discoveries for institutional knowledge
- **Sparse Aware**: Native understanding of sparse signal format

## Requirements

- Python 3.8+
- duckdb
- pandas
- numpy
- pyyaml

## Examples

See `example_usage.py` for detailed examples of all features.