# SQL Query Library

Pre-built SQL queries for common analytics tasks.

## Usage

```bash
# Execute any query
./analytics query workspaces/your_workspace -f src/analytics/queries/strategy_performance.sql

# Or with DuckDB CLI directly
python duckdb_cli.py workspaces/your_workspace/analytics.duckdb < src/analytics/queries/simple_join.sql
```

## Available Queries

### Basic Analysis
- `simple_join.sql` - Join signals with price data
- `calculate_returns.sql` - Calculate individual trade returns

### Performance Analysis  
- `strategy_performance.sql` - Full performance metrics with Sharpe
- `compare_exit_strategies.sql` - Compare different exit methods

### Advanced Analysis
- `regime_performance_analysis.sql` - Performance by market regime
- `parameter_sensitivity_analysis.sql` - Find stable parameter neighborhoods

## Query Patterns

### Joining Sparse Signals with Market Data
```sql
FROM read_parquet('traces/*/signals/*/*.parquet') s
JOIN read_parquet('data/SPY_1m.parquet') m ON s.idx = m.bar_index
WHERE s.val != 0  -- Only actual signals
```

### Calculating Returns
```sql
CASE 
    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100   -- Long
    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100  -- Short
END as return_pct
```

### Wildcard Patterns
- `signals/*/*.parquet` - All strategy types
- `signals/rsi_grid/*.parquet` - Only RSI strategies
- `traces/*/signals/*/*.parquet` - All symbols and strategies