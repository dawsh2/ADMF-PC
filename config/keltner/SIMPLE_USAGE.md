# Just Use SQL!

You're right - no scripts needed. Just open Python/Jupyter and query:

```python
# In Python REPL or Jupyter
import sys
sys.path.append('../../src')
from analytics import TraceAnalysis

# Load data
ta = TraceAnalysis('results/20250622_155944')

# See what you have
ta.sql("SELECT * FROM traces LIMIT 5")

# Count strategies
ta.sql("SELECT COUNT(DISTINCT strategy_id) as num_strategies FROM traces")

# Signal activity
ta.sql("""
    SELECT 
        strategy_id,
        COUNT(*) as signals,
        SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END) as longs,
        SUM(CASE WHEN signal_value < 0 THEN 1 ELSE 0 END) as shorts
    FROM traces
    GROUP BY strategy_id
    ORDER BY signals DESC
    LIMIT 10
""")

# Extract trades (entry when signal != 0, exit when signal = 0)
ta.sql("""
    WITH changes AS (
        SELECT *,
            LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as prev_signal
        FROM traces
    ),
    trades AS (
        SELECT 
            strategy_id,
            bar_idx as entry_bar,
            price as entry_price,
            signal_value as direction,
            LEAD(bar_idx) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as exit_bar,
            LEAD(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as exit_price
        FROM changes
        WHERE signal_value != 0 AND (prev_signal = 0 OR prev_signal IS NULL)
    )
    SELECT 
        strategy_id,
        COUNT(*) as num_trades,
        AVG((exit_price - entry_price) / entry_price * direction) as avg_return
    FROM trades
    WHERE exit_bar IS NOT NULL
    GROUP BY strategy_id
    HAVING COUNT(*) > 20
    ORDER BY avg_return DESC
    LIMIT 10
""")
```

That's it! Everything is just SQL on the sparse parquet files.

## The Sparse Format

Your trace files have just 3 columns:
- `idx`: Bar index (when the signal changed)
- `val`: Signal value (-1, 0, 1)
- `px`: Price at that bar

The `TraceAnalysis` class adds:
- `strategy_id`: Extracted from filename
- Renamed columns for clarity: `bar_idx`, `signal_value`, `price`

## No Code Needed

Just write SQL queries to answer your questions:
- Which strategies trade most? GROUP BY and COUNT
- What's the win rate? Calculate returns between entries/exits
- Best performers? ORDER BY your metric

The whole point is: **It's just SQL!**