# SQL Analytics Integration Guide

## Overview

Your ADMF-PC system now automatically creates SQL analytics workspaces when topology runs complete successfully. This provides powerful analytics capabilities for your grid search results.

## How It Works

**Automatic Integration**: When you run commands like:
```bash
python main.py --config config/expansive_grid_search.yaml --signal-generation --bars 50
```

The system will:
1. Execute your topology as normal
2. **Automatically create a SQL analytics workspace** when the run completes successfully
3. Store all strategy and classifier data in a structured SQL database
4. Provide a universal SQL interface for analysis

## Example Workflow (Pure Lazy)

### 1. Run Your Grid Search
```bash
python main.py --config config/expansive_grid_search.yaml --signal-generation --bars 2000
```

### 2. Check for Analytics Workspace
Look for this log message:
```
📊 SQL analytics workspace created: 20250611_154523_expansive_grid_v1_SPY
```

### 3. Explore Your Catalog with SQL
```python
from analytics import AnalyticsWorkspace

# Connect to your workspace
workspace = AnalyticsWorkspace('workspaces/20250611_154523_expansive_grid_v1_SPY')

# Get workspace overview
summary = workspace.summary()
print(f"Workspace contains {summary['total_strategies']} strategies")

# Browse strategy catalog (NO performance metrics yet)
strategies = workspace.sql("""
    SELECT strategy_type, strategy_id, parameters
    FROM strategies 
    WHERE strategy_type = 'simple_momentum'
    LIMIT 10
""")
print(strategies)

# Parameter distribution analysis
param_dist = workspace.sql("""
    SELECT 
        JSON_EXTRACT(parameters, '$.sma_period') as sma_period,
        COUNT(*) as strategy_count
    FROM strategies 
    WHERE strategy_type = 'simple_momentum'
    GROUP BY JSON_EXTRACT(parameters, '$.sma_period')
    ORDER BY sma_period
""")
print(param_dist)

# Strategy type distribution
strategy_types = workspace.sql("""
    SELECT 
        strategy_type,
        COUNT(*) as total_strategies
    FROM strategies
    GROUP BY strategy_type
    ORDER BY total_strategies DESC
""")
print(strategy_types)

### 4. Calculate Performance On-Demand
```python
# Select interesting strategies based on parameters/type
candidates = workspace.sql("""
    SELECT strategy_id 
    FROM strategies 
    WHERE strategy_type = 'momentum' 
    AND JSON_EXTRACT(parameters, '$.sma_period') = 20
    LIMIT 5
""")

# Calculate performance for selected strategies
for strategy_id in candidates['strategy_id']:
    perf = workspace.calculate_performance(strategy_id)
    print(f"{strategy_id}: Sharpe={perf['sharpe_ratio']}, Drawdown={perf['max_drawdown']}")
```

## Available Tables

Your analytics workspace contains these tables:

### `runs`
- Run metadata (symbols, timeframes, duration, etc.)

### `strategies` (Pure Catalog)
- Strategy definitions and parameter combinations
- Signal file paths (pointers to sparse Parquet files)
- NO pre-computed performance metrics (calculated on-demand)

### `classifiers` (Pure Catalog)
- Classifier definitions and parameter combinations  
- States file paths (pointers to sparse Parquet files)
- NO pre-computed statistics (calculated on-demand)

### `event_archives`
- Event trace file catalog (if event tracing enabled)

## Common Queries (Pure Lazy)

### Browse Parameter Combinations
```sql
-- See what parameter values were tested
SELECT 
    JSON_EXTRACT(parameters, '$.sma_period') as sma_period,
    JSON_EXTRACT(parameters, '$.rsi_threshold_long') as rsi_threshold,
    COUNT(*) as count
FROM strategies 
WHERE strategy_type = 'simple_momentum'
GROUP BY JSON_EXTRACT(parameters, '$.sma_period'), 
         JSON_EXTRACT(parameters, '$.rsi_threshold_long')
ORDER BY sma_period, rsi_threshold;
```

### Find Specific Parameter Combinations
```sql
-- Find strategies with specific parameters
SELECT strategy_id, parameters
FROM strategies 
WHERE strategy_type = 'momentum'
AND JSON_EXTRACT(parameters, '$.sma_period') = 20
AND JSON_EXTRACT(parameters, '$.rsi_threshold_long') = 30;
```

### Count Strategy Variations
```sql
-- How many variations of each strategy type
SELECT 
    strategy_type,
    COUNT(*) as total_variations,
    COUNT(DISTINCT parameters) as unique_configs
FROM strategies
GROUP BY strategy_type
ORDER BY total_variations DESC;
```

## Export Results

```python
# Export top strategies to CSV
workspace.export_results(
    "SELECT * FROM strategies WHERE sharpe_ratio > 1.5",
    "top_strategies.csv"
)

# Export parameter analysis to Excel
workspace.export_results(
    "SELECT * FROM parameter_analysis ORDER BY avg_sharpe DESC",
    "parameter_analysis.xlsx",
    format="excel"
)
```

## Interactive Analysis

### Jupyter Notebook
```python
import sys
sys.path.append('/path/to/ADMF-PC/src')

from analytics import AnalyticsWorkspace
import plotly.express as px

workspace = AnalyticsWorkspace('workspaces/your_workspace')

# Parameter sensitivity visualization
data = workspace.sql("""
    SELECT 
        CAST(JSON_EXTRACT(parameters, '$.sma_period') AS INT) as sma_period,
        AVG(sharpe_ratio) as avg_sharpe,
        STDDEV(sharpe_ratio) as sharpe_std
    FROM strategies 
    WHERE strategy_type = 'simple_momentum'
    GROUP BY CAST(JSON_EXTRACT(parameters, '$.sma_period') AS INT)
""")

fig = px.line(data, x='sma_period', y='avg_sharpe', 
              error_y='sharpe_std', title='SMA Period Sensitivity')
fig.show()
```

### DuckDB CLI Direct Access
```bash
# Connect directly to any workspace
duckdb workspaces/20250611_154523_expansive_grid_v1_SPY/analytics.duckdb

# Run SQL directly
D SELECT COUNT(*) FROM strategies;
D SELECT strategy_type, AVG(sharpe_ratio) FROM strategies GROUP BY strategy_type;
```

## Migration from Existing Workspaces

Convert your existing UUID-based workspaces:
```python
from analytics import migrate_workspace

# Migrate existing workspace
migrate_workspace(
    source_path='workspaces/old-uuid-workspace',
    destination_path='workspaces/migrated-sql-workspace'
)

# Then analyze normally
workspace = AnalyticsWorkspace('workspaces/migrated-sql-workspace')
```

## Benefits

- **No Learning Curve**: Standard SQL works with any tool
- **Scalable**: Handles thousands of strategies efficiently  
- **Universal**: Works with Python, Jupyter, CLI, any SQL client
- **Powerful**: Full database capabilities (joins, window functions, etc.)
- **Automatic**: Zero configuration - just run your existing commands

Your existing workflow remains exactly the same, but now you get powerful SQL analytics automatically!