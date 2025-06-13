# Efficient Signal-Price Joining Guide

## ✅ Correct Architecture: Single Source Data File

```
project/
├── data/
│   ├── SPY_1m.parquet          # ✅ Single master data file
│   ├── QQQ_1m.parquet          # ✅ Single master data file  
│   └── SPY_5m.parquet          # ✅ Single master data file
└── workspaces/
    ├── workspace_1/
    │   ├── analytics.duckdb
    │   └── traces/SPY_1m/signals/  # ✅ References ../../../data/SPY_1m.parquet
    └── workspace_2/
        ├── analytics.duckdb  
        └── traces/SPY_1m/signals/  # ✅ References same data file
```

## Why This Works Efficiently

### **1. Sparse Storage = Minimal Data**
Your signal files only store changes:
```
SPY_rsi_grid_7_20_70.parquet: 22 signal changes (instead of 102,236 bars)
Compression ratio: 99.98% space savings
```

### **2. On-Demand Joins = No Duplication**
```sql
-- This query only touches relevant data
SELECT s.strat, m.close 
FROM sparse_signals s
JOIN master_data m ON s.idx = m.bar_index
WHERE s.val != 0  -- Only 22 rows, not 102,236
```

### **3. Master Data File Benefits**
- **Single source of truth**: All workspaces reference the same data
- **No duplication**: 2.9MB master file vs copying into every workspace
- **Easy updates**: Update master file, all analyses use new data
- **Cross-workspace comparisons**: Same data baseline

## Practical Examples

### **Performance Calculation (No Data Duplication)**
```sql
-- Connect to any workspace
python duckdb_cli.py workspaces/your_workspace/analytics.duckdb

-- Calculate performance using master data file
WITH signals AS (
    SELECT * FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/rsi_grid/*.parquet')
    WHERE val != 0
),
market_data AS (
    SELECT * FROM read_parquet('data/SPY_1m.parquet')  -- Single master file
)
SELECT 
    s.strat,
    COUNT(*) as trades,
    AVG((m2.close - m1.close) / m1.close * 100) as avg_return_pct
FROM signals s
JOIN market_data m1 ON s.idx = m1.bar_index
JOIN market_data m2 ON s.idx + 1 = m2.bar_index  -- Next bar for exit
WHERE s.val = 1  -- Long signals only
GROUP BY s.strat
ORDER BY avg_return_pct DESC;
```

### **Cross-Workspace Analysis**
```sql
-- Compare strategies across different grid searches
WITH workspace1_signals AS (
    SELECT strat, COUNT(*) as signals1 
    FROM read_parquet('workspaces/grid_search_1/traces/SPY_1m/signals/*/*.parquet')
    GROUP BY strat
),
workspace2_signals AS (
    SELECT strat, COUNT(*) as signals2
    FROM read_parquet('workspaces/grid_search_2/traces/SPY_1m/signals/*/*.parquet') 
    GROUP BY strat
)
SELECT 
    COALESCE(w1.strat, w2.strat) as strategy,
    COALESCE(w1.signals1, 0) as experiment1_signals,
    COALESCE(w2.signals2, 0) as experiment2_signals
FROM workspace1_signals w1
FULL OUTER JOIN workspace2_signals w2 ON w1.strat = w2.strat;
```

## File Size Comparison

### **Without Sparse Storage (Bad)**
```
workspace_1/
├── SPY_1m_copy.parquet         # 2.9MB duplicated
└── traces/signals/...          # + signal data

workspace_2/  
├── SPY_1m_copy.parquet         # 2.9MB duplicated again
└── traces/signals/...

Total: 5.8MB+ per workspace
```

### **With Sparse Storage (Good)**
```
data/SPY_1m.parquet             # 2.9MB (shared)

workspace_1/traces/signals/     # ~50KB total (sparse)
workspace_2/traces/signals/     # ~50KB total (sparse)

Total: 2.9MB + 0.1MB = 3.0MB for everything
```

## Query Performance Tips

### **1. Use Indexed Joins**
```sql
-- Efficient: Join on bar_index (integer)
JOIN market_data m ON s.idx = m.bar_index

-- Less efficient: Join on timestamp (string)
JOIN market_data m ON s.ts = m.timestamp
```

### **2. Filter Signals First**
```sql
-- Good: Filter sparse signals first
WITH active_signals AS (
    SELECT * FROM read_parquet('traces/SPY_1m/signals/*/*.parquet')
    WHERE val != 0 AND strat LIKE '%rsi%'
)
SELECT ... FROM active_signals s JOIN market_data m ...

-- Less efficient: Join everything then filter
SELECT ... FROM signals s JOIN market_data m ... WHERE s.val != 0
```

### **3. Use Column Selection**
```sql
-- Good: Only select needed columns
SELECT s.strat, m.close, m.volume
FROM signals s JOIN market_data m ON s.idx = m.bar_index

-- Wasteful: Select all columns  
SELECT * FROM signals s JOIN market_data m ON s.idx = m.bar_index
```

## Storage Requirements

For a typical grid search with 200 strategies over 100K bars:

| Component | Size | Notes |
|-----------|------|-------|
| Master data (SPY_1m.parquet) | 2.9MB | Shared across all workspaces |
| Signal traces (sparse) | ~100KB | Only signal changes stored |
| Analytics database | ~1MB | Metadata and summary tables |
| **Total per workspace** | **~1.1MB** | **99.96% compression** |

## Best Practices

### **✅ Do This:**
- Keep master data files in `./data/`
- Reference master data with relative paths in queries
- Use sparse signal storage in workspaces
- Join on integer bar_index for performance

### **❌ Don't Do This:**
- Copy master data into each workspace
- Store full signal series (defeats sparse storage purpose)
- Join on string timestamps
- Create separate data files per strategy

## Example: Complete Performance Analysis

```sql
-- Single query that calculates strategy performance across your entire grid search
-- Uses master data file + sparse signals efficiently

WITH strategy_performance AS (
    SELECT 
        s.strat,
        s.val as signal,
        m1.close as entry_price,
        m2.close as exit_price,
        (m2.close - m1.close) / m1.close as return_1bar
    FROM read_parquet('workspaces/your_workspace/traces/SPY_1m/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    WHERE s.val != 0  -- Only actual signals
)
SELECT 
    strat,
    COUNT(*) as total_trades,
    AVG(CASE WHEN signal = 1 THEN return_1bar ELSE -return_1bar END) * 100 as avg_return_pct,
    STDDEV(CASE WHEN signal = 1 THEN return_1bar ELSE -return_1bar END) * 100 as volatility_pct,
    (AVG(CASE WHEN signal = 1 THEN return_1bar ELSE -return_1bar END) / 
     STDDEV(CASE WHEN signal = 1 THEN return_1bar ELSE -return_1bar END)) as sharpe_ratio
FROM strategy_performance
GROUP BY strat
ORDER BY sharpe_ratio DESC
LIMIT 20;
```

This approach gives you maximum efficiency: **one master data file** + **minimal sparse signals** + **fast analytics** = optimal storage and performance!