# ADMF-PC Analytics

Minimal analytics for exploring backtest results using DuckDB + SQL.

## Quick Start

```python
from analytics import TraceAnalysis

# Auto-finds latest results
ta = TraceAnalysis()

# Query with SQL
df = ta.sql("SELECT * FROM traces WHERE sharpe > 1.5")
df.plot.scatter('period', 'sharpe')
```

## Workflow Example

After running a backtest with thousands of strategies:

```python
import duckdb
import matplotlib.pyplot as plt

# Connect to your results
ta = TraceAnalysis('config/keltner/results/latest')

# 1. Overview
ta.sql("SELECT COUNT(*), AVG(sharpe), MAX(sharpe) FROM traces").T

# 2. Find winners  
winners = ta.sql("""
    SELECT * FROM traces 
    WHERE sharpe > 1.5 AND max_drawdown < 0.1
    ORDER BY sharpe DESC
""")

# 3. Visualize
winners.plot.scatter('max_drawdown', 'sharpe')

# 4. Parameter analysis
params = ta.sql("""
    SELECT 
        params->>'period' as period,
        AVG(sharpe) as avg_sharpe,
        COUNT(*) as count
    FROM traces
    GROUP BY period
""")
params.plot.bar(x='period', y='avg_sharpe')

# 5. Export best for production
best = ta.sql("SELECT * FROM traces ORDER BY sharpe DESC LIMIT 10")
best.to_json('production_strategies.json')
```

## Core Concept

Everything is just SQL on parquet files:
- `traces` view = all your signal traces
- `metadata` view = strategy metadata (if available)
- Query returns pandas DataFrames
- Plot/analyze/export as needed

### Sparse Signal Format

The trace data uses a sparse format that only stores signal changes:

**Raw columns in parquet files:**
- `idx`: Bar index when signal changed
- `val`: Signal value (-1, 0, 1)
- `px`: Price at that bar

**Mapped to friendly names in traces view:**
- `strategy_id`: Extracted from filename (e.g., strategy_0.parquet → 0)
- `bar_idx`: Same as idx
- `signal_value`: Same as val (-1=short, 0=flat, 1=long)
- `price`: Same as px

This sparse format is highly efficient - only recording when signals change rather than every bar. For example, if a strategy holds a position for 100 bars, only 2 records are stored (entry and exit).

## Installation

```bash
pip install duckdb pandas
```

## Common Queries

```python
# Top performers
"SELECT * FROM traces ORDER BY sharpe DESC LIMIT 100"

# Filter by criteria
"SELECT * FROM traces WHERE sharpe > 1.5 AND num_trades > 100"

# Parameter sensitivity
"""
SELECT 
    params->>'period' as period,
    AVG(sharpe) as avg_sharpe
FROM traces
GROUP BY period
"""

# Find stable parameters
"""
SELECT t1.*, AVG(t2.sharpe) as neighbor_avg
FROM traces t1
JOIN traces t2 ON ABS(t1.period - t2.period) <= 5
WHERE t1.sharpe > 1.5
GROUP BY t1.strategy_id
HAVING STDDEV(t2.sharpe) < 0.3
"""
```

## Files

- `simple_analytics.py` - Core TraceAnalysis class
- `trace_analysis.py` - Extended version with more features
- `pattern_discovery.py` - Pattern library management
- `trade_metrics.py` - Trade reconstruction utilities

For most use cases, just `TraceAnalysis` and SQL is all you need.