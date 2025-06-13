# DuckDB Metrics Extraction Guide

## Quick Start: Connect to Your Analytics Database

```bash
# Interactive mode
python duckdb_cli.py workspaces/your_workspace_id/analytics.duckdb

# Single query mode
python duckdb_cli.py workspaces/your_workspace_id/analytics.duckdb "YOUR_SQL_QUERY"

# Or install DuckDB CLI and use directly
brew install duckdb
duckdb workspaces/your_workspace_id/analytics.duckdb
```

## 1. **Basic Signal Analysis**

### Signal Frequency by Strategy Type
```sql
SELECT 
    REGEXP_EXTRACT(strat, '[a-z_]+') as strategy_type,
    COUNT(*) as total_signal_changes,
    COUNT(DISTINCT strat) as strategy_variants,
    AVG(CASE WHEN val = 1 THEN 1.0 ELSE 0.0 END) as long_signal_ratio,
    AVG(CASE WHEN val = -1 THEN 1.0 ELSE 0.0 END) as short_signal_ratio
FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet')
GROUP BY REGEXP_EXTRACT(strat, '[a-z_]+')
ORDER BY total_signal_changes DESC;
```

### Top Performing Signal Generators (by frequency)
```sql
SELECT 
    strat as strategy_id,
    COUNT(*) as signal_changes,
    MIN(idx) as first_signal_bar,
    MAX(idx) as last_signal_bar,
    tf as timeframe,
    src_file
FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet')
GROUP BY strat, tf, src_file
ORDER BY signal_changes DESC
LIMIT 20;
```

## 2. **Cross-Strategy Comparison**

### Compare RSI vs Momentum vs Breakout
```sql
WITH strategy_stats AS (
    SELECT 
        CASE 
            WHEN strat LIKE '%rsi%' THEN 'RSI'
            WHEN strat LIKE '%momentum%' THEN 'Momentum'  
            WHEN strat LIKE '%breakout%' THEN 'Breakout'
            WHEN strat LIKE '%ma_crossover%' THEN 'MA_Crossover'
            WHEN strat LIKE '%mean_reversion%' THEN 'Mean_Reversion'
            ELSE 'Other'
        END as strategy_family,
        strat,
        COUNT(*) as signal_changes,
        COUNT(DISTINCT idx) as unique_bars_with_signals
    FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet')
    GROUP BY strategy_family, strat
)
SELECT 
    strategy_family,
    COUNT(*) as strategy_variants,
    AVG(signal_changes) as avg_signal_changes,
    MAX(signal_changes) as max_signal_changes,
    SUM(signal_changes) as total_signals
FROM strategy_stats
GROUP BY strategy_family
ORDER BY avg_signal_changes DESC;
```

## 3. **Signal Timing Analysis**

### Signal Distribution by Bar Index
```sql
SELECT 
    idx as bar_index,
    COUNT(*) as signal_count,
    COUNT(DISTINCT strat) as strategies_signaling,
    STRING_AGG(DISTINCT 
        CASE WHEN val = 1 THEN 'LONG' 
             WHEN val = -1 THEN 'SHORT' 
             ELSE 'FLAT' END, ', ') as signal_types
FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet')
GROUP BY idx
HAVING COUNT(*) > 5  -- Only show bars with multiple signals
ORDER BY signal_count DESC
LIMIT 20;
```

### Strategy Signal Clustering
```sql
SELECT 
    idx,
    ts,
    COUNT(*) as concurrent_signals,
    STRING_AGG(strat, ', ') as strategies
FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet')
GROUP BY idx, ts
HAVING COUNT(*) >= 3  -- 3+ strategies signaling at same time
ORDER BY concurrent_signals DESC, idx;
```

## 4. **Parameter Optimization Analysis**

### RSI Parameter Analysis
```sql
WITH rsi_params AS (
    SELECT 
        strat,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_(\d+)_', 1) AS INTEGER) as rsi_period,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_\d+_(\d+)_', 1) AS INTEGER) as oversold_threshold,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_\d+_\d+_(\d+)', 1) AS INTEGER) as overbought_threshold,
        COUNT(*) as signal_changes
    FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/rsi_grid/*.parquet')
    GROUP BY strat
)
SELECT 
    rsi_period,
    oversold_threshold,
    overbought_threshold,
    AVG(signal_changes) as avg_signals,
    MAX(signal_changes) as max_signals,
    COUNT(*) as parameter_combinations
FROM rsi_params
GROUP BY rsi_period, oversold_threshold, overbought_threshold
ORDER BY avg_signals DESC;
```

### Moving Average Crossover Optimization
```sql
WITH ma_params AS (
    SELECT 
        strat,
        CAST(REGEXP_EXTRACT(strat, 'ma_crossover_grid_(\d+)_', 1) AS INTEGER) as fast_period,
        CAST(REGEXP_EXTRACT(strat, 'ma_crossover_grid_\d+_(\d+)_', 1) AS INTEGER) as slow_period,
        COUNT(*) as signal_changes
    FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/ma_crossover_grid/*.parquet')
    GROUP BY strat
)
SELECT 
    fast_period,
    slow_period,
    slow_period - fast_period as period_difference,
    AVG(signal_changes) as avg_signals,
    COUNT(*) as combinations
FROM ma_params
GROUP BY fast_period, slow_period
ORDER BY avg_signals DESC;
```

## 5. **Classifier Analysis**

### Market Regime Detection Summary
```sql
SELECT 
    REGEXP_EXTRACT(strat, '[a-z_]+') as classifier_type,
    val as detected_regime,
    COUNT(*) as detection_count,
    COUNT(DISTINCT strat) as classifier_variants
FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/classifiers/*/*.parquet')
GROUP BY REGEXP_EXTRACT(strat, '[a-z_]+'), val
ORDER BY classifier_type, detection_count DESC;
```

### Regime Change Timeline
```sql
SELECT 
    idx,
    ts,
    strat as classifier,
    val as regime,
    LAG(val) OVER (PARTITION BY strat ORDER BY idx) as previous_regime
FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/classifiers/*/*.parquet')
ORDER BY idx, strat;
```

## 6. **Source Data Integration**

Since each signal includes source metadata, you can join with actual market data:

### Example: Calculate Signal Performance (requires source data)
```sql
-- First, load your market data
CREATE TEMPORARY TABLE market_data AS 
SELECT * FROM read_csv('data/SPY_1m.csv');

-- Then join signals with prices for performance calculation
WITH signal_prices AS (
    SELECT 
        s.strat,
        s.idx,
        s.val as signal,
        s.ts as signal_time,
        m.close as signal_price,
        LEAD(m.close) OVER (ORDER BY s.idx) as next_price
    FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet') s
    JOIN market_data m ON s.idx = m.bar_index  -- Adjust join condition as needed
    WHERE s.val != 0  -- Only actual signals, not flat
)
SELECT 
    strat,
    COUNT(*) as trades,
    AVG(CASE WHEN signal = 1 THEN (next_price - signal_price) / signal_price 
             WHEN signal = -1 THEN (signal_price - next_price) / signal_price 
             ELSE 0 END) as avg_return_per_trade
FROM signal_prices
WHERE next_price IS NOT NULL
GROUP BY strat
ORDER BY avg_return_per_trade DESC;
```

## 7. **Advanced Metrics**

### Signal Correlation Analysis
```sql
WITH strategy_signals AS (
    SELECT 
        idx,
        strat,
        val
    FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet')
),
signal_matrix AS (
    SELECT 
        idx,
        SUM(CASE WHEN strat LIKE '%rsi%' AND val = 1 THEN 1 ELSE 0 END) as rsi_long,
        SUM(CASE WHEN strat LIKE '%momentum%' AND val = 1 THEN 1 ELSE 0 END) as momentum_long,
        SUM(CASE WHEN strat LIKE '%breakout%' AND val = 1 THEN 1 ELSE 0 END) as breakout_long
    FROM strategy_signals
    GROUP BY idx
)
SELECT 
    CORR(rsi_long, momentum_long) as rsi_momentum_correlation,
    CORR(rsi_long, breakout_long) as rsi_breakout_correlation,
    CORR(momentum_long, breakout_long) as momentum_breakout_correlation
FROM signal_matrix;
```

## 8. **Export Results**

### Save Query Results to CSV
```sql
COPY (
    SELECT 
        strat,
        COUNT(*) as signal_changes,
        src_file
    FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet')
    GROUP BY strat, src_file
    ORDER BY signal_changes DESC
) TO 'strategy_performance.csv' (HEADER, DELIMITER ',');
```

## 9. **Interactive Mode Examples**

Once connected to DuckDB:

```sql
-- Show all available parquet files
.tables

-- Set up convenient views
CREATE VIEW all_signals AS 
SELECT * FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet');

CREATE VIEW all_classifiers AS 
SELECT * FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/classifiers/*/*.parquet');

-- Now query the views
SELECT * FROM all_signals LIMIT 10;
SELECT DISTINCT val FROM all_classifiers;
```

## Quick Command Reference

```bash
# Connect to workspace
python duckdb_cli.py workspaces/your_workspace_id/analytics.duckdb

# Example workspace path
python duckdb_cli.py workspaces/expansive_grid_search_db1cfd51/analytics.duckdb

# Quick signal summary
python duckdb_cli.py workspaces/your_workspace/analytics.duckdb "
SELECT 
    COUNT(*) as total_signals,
    COUNT(DISTINCT strat) as unique_strategies,
    MIN(idx) as first_bar,
    MAX(idx) as last_bar
FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet')
"
```

The key insight is that with source metadata in each parquet file, you can now do meaningful performance analysis by joining signal data with the original market data!