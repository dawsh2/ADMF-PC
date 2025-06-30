# Keltner Strategy Analysis Guide

## Quick Start

1. **Run the quick analysis script**:
   ```bash
   cd config/keltner
   python quick_analysis.py
   ```

2. **Open Jupyter for interactive analysis**:
   ```bash
   jupyter lab analysis.ipynb
   ```

## What's Available

### Data Structure
- **Traces**: Sparse signal data in parquet format
  - `idx`: Bar index
  - `val`: Signal value (-1, 0, 1)
  - `px`: Price at signal change
  - `strategy_id`: Strategy identifier

### Key Analyses

1. **Signal Activity**
   - How many signals each strategy generates
   - Long vs short signal distribution
   - Signal frequency patterns

2. **Trade Performance**
   - Extract trades from signal changes
   - Calculate returns, Sharpe ratio, win rate
   - Identify best performing strategies

3. **Production Selection**
   - Filter by multiple criteria (Sharpe, trades, win rate)
   - Export best strategies for live trading

## Custom SQL Queries

The power is in SQL. Here are some useful queries:

```python
# Find strategies with low drawdown
ta.sql("""
    SELECT strategy_id, MIN(cumulative_return) as max_drawdown
    FROM (
        SELECT *, SUM(return_pct) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as cumulative_return
        FROM trades
    )
    GROUP BY strategy_id
    HAVING max_drawdown > -0.10
""")

# Analyze performance by time of day (assuming 78 bars per day for 5min data)
ta.sql("""
    SELECT 
        (idx % 78) as intraday_bar,
        AVG(CASE WHEN val > 0 THEN 1 ELSE -1 END) as avg_signal_direction,
        COUNT(*) as signal_count
    FROM traces
    GROUP BY intraday_bar
    ORDER BY intraday_bar
""")

# Find parameter sweet spots
ta.sql("""
    SELECT 
        strategy_id,
        COUNT(*) as trades,
        AVG(return_pct) as avg_return,
        STDDEV(return_pct) as volatility
    FROM trades
    GROUP BY strategy_id
    HAVING trades BETWEEN 100 AND 200
    ORDER BY avg_return DESC
""")
```

## Understanding the Results

### Metadata
The `metadata.json` file contains:
- Total bars analyzed
- Signal compression ratios
- Strategy parameters (though these are compiled in this run)

### Sparse Format Benefits
- Only stores signal changes, not every bar
- Compression ratio shows efficiency (e.g., 2.01 means 50% space saved)
- Can reconstruct full signal series when needed

## Next Steps

1. **Identify patterns**: Which signal frequencies work best?
2. **Filter analysis**: Compare strategies with different signal counts
3. **Parameter optimization**: Find optimal Keltner band settings
4. **Risk analysis**: Check drawdowns and consistency

## Tips

- Start with `quick_analysis.py` for overview
- Use Jupyter notebook for visualization
- Write custom SQL queries for specific questions
- Export best strategies as JSON/CSV for production

The key insight: Everything is just SQL on parquet files!