# Interactive Analysis Guide for ADMF-PC

## Overview

The interactive analysis module provides a Python-first approach to exploring backtest results, moving away from CLI-centric workflows to a more natural exploratory pattern suitable for Jupyter notebooks and interactive Python sessions.

## Quick Start

```python
from src.analytics.interactive import AnalysisWorkspace

# Create workspace and load a run
workspace = AnalysisWorkspace()
run = workspace.load_run("results/run_20250623_143030")

# Get top strategies
top = workspace.top_strategies(run, n=10)

# Find optimal ensemble
ensemble = workspace.find_ensemble(run, size=5)
```

## Core Components

### 1. AnalysisWorkspace

The main entry point for interactive analysis:

```python
workspace = AnalysisWorkspace()

# List all available runs
runs = workspace.list_runs()

# Load specific run
run = workspace.load_run("results/latest")

# Compare multiple runs
comparison = workspace.compare_runs([
    "results/run_20250623_143030",
    "results/run_20250624_090000"
])
```

### 2. BacktestRun

Represents a single backtest run with convenient data access:

```python
# Access run data
run.summary          # Run metadata
run.strategies       # All strategies as DataFrame
run.query(sql)       # Execute arbitrary DuckDB queries

# Example query
profitable = run.query("""
    SELECT * FROM strategies 
    WHERE total_return > 0.1 
    AND sharpe_ratio > 1.5
    ORDER BY sharpe_ratio DESC
""")
```

### 3. QueryLibrary

Pre-built queries for common analysis patterns:

```python
from src.analytics.interactive import QueryLibrary

# Signal frequency analysis
signal_freq = QueryLibrary.signal_frequency(run)

# Intraday patterns
patterns = QueryLibrary.intraday_patterns(run)

# Regime performance
regimes = QueryLibrary.regime_performance(run)
```

### 4. PatternLibrary

Save and retrieve discovered patterns:

```python
pattern_lib = PatternLibrary()

# Save a pattern
pattern_lib.save_pattern("high_sharpe_momentum", {
    'strategy_type': 'momentum',
    'parameters': {'fast_period': 10, 'slow_period': 20},
    'performance': {'sharpe': 2.5}
})

# Retrieve later
pattern = pattern_lib.get_pattern("high_sharpe_momentum")

# List all patterns
patterns = pattern_lib.list_patterns()
```

## Common Workflows

### 1. Strategy Discovery

```python
# Find strategies with specific characteristics
high_performers = run.query("""
    SELECT 
        strategy_type,
        strategy_hash,
        sharpe_ratio,
        total_return,
        max_drawdown
    FROM strategies
    WHERE sharpe_ratio > 2.0
    AND max_drawdown > -0.1
    ORDER BY sharpe_ratio DESC
    LIMIT 20
""")
```

### 2. Ensemble Building

```python
# Find uncorrelated strategies
ensemble = workspace.find_ensemble(
    run, 
    size=5,
    correlation_threshold=0.7
)

# Analyze ensemble
print(f"Average Sharpe: {ensemble['avg_sharpe']:.2f}")
print(f"Max Correlation: {ensemble['max_correlation']:.2f}")

# Get correlation matrix
corr = workspace.correlation_matrix(run, ensemble['strategies'])
```

### 3. Parameter Optimization

```python
# Analyze parameter sensitivity
param_analysis = workspace.analyze_parameters(run, 'momentum')

# Custom parameter sweep analysis
param_sweep = run.query("""
    SELECT 
        param_fast_period,
        param_slow_period,
        AVG(sharpe_ratio) as avg_sharpe,
        COUNT(*) as count
    FROM strategies
    WHERE strategy_type = 'momentum'
    GROUP BY param_fast_period, param_slow_period
    HAVING COUNT(*) >= 3
    ORDER BY avg_sharpe DESC
""")
```

### 4. Time-Based Analysis

```python
# Signal distribution by hour
hourly = run.query("""
    SELECT 
        EXTRACT(HOUR FROM ts) as hour,
        COUNT(*) as signals,
        AVG(CASE WHEN val > 0 THEN 1 ELSE -1 END) as direction_bias
    FROM signals
    WHERE val != 0
    GROUP BY hour
    ORDER BY hour
""")

# Most active trading times
peak_hours = hourly.nlargest(3, 'signals')
```

## Integration with Notebook Cells

Use the reusable cells from `notebook_cells` module:

```python
from src.analytics.notebook_cells import performance, correlation

# Execute a performance analysis cell
exec(performance.sharpe_calculation_cell())

# Execute correlation analysis
exec(correlation.correlation_matrix_cell())
```

## Advanced DuckDB Queries

### Cross-Strategy Signal Analysis

```python
# Find strategies that trade at similar times
concurrent_trading = run.query("""
    WITH signal_times AS (
        SELECT 
            strategy_hash,
            DATE_TRUNC('minute', ts) as minute,
            val
        FROM signals
        WHERE val != 0
    )
    SELECT 
        s1.strategy_hash as strategy1,
        s2.strategy_hash as strategy2,
        COUNT(*) as concurrent_signals
    FROM signal_times s1
    JOIN signal_times s2 
        ON s1.minute = s2.minute 
        AND s1.strategy_hash < s2.strategy_hash
    GROUP BY s1.strategy_hash, s2.strategy_hash
    HAVING COUNT(*) > 100
    ORDER BY concurrent_signals DESC
""")
```

### Performance Attribution

```python
# Attribute performance to different factors
attribution = run.query("""
    WITH strategy_signals AS (
        SELECT 
            s.strategy_hash,
            s.ts,
            s.val,
            st.strategy_type,
            st.sharpe_ratio,
            EXTRACT(HOUR FROM s.ts) as hour,
            EXTRACT(DOW FROM s.ts) as day_of_week
        FROM signals s
        JOIN strategies st ON s.strategy_hash = st.strategy_hash
        WHERE s.val != 0
    )
    SELECT 
        strategy_type,
        hour,
        COUNT(*) as signal_count,
        AVG(sharpe_ratio) as avg_sharpe
    FROM strategy_signals
    GROUP BY strategy_type, hour
    ORDER BY strategy_type, hour
""")
```

## Exporting Results

### Save Analysis Results

```python
# Export top strategies
top_strategies.to_csv('top_strategies.csv', index=False)

# Export ensemble configuration
import json
ensemble_config = {
    'strategies': ensemble['strategies'].to_dict('records'),
    'weights': 'equal',  # or optimize weights
    'correlation_threshold': 0.7
}
with open('ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)
```

### Generate Reports

```python
# Create summary report
report = f"""
# Backtest Analysis Report

Run ID: {run.summary['run_id']}
Config: {run.summary['config_name']}
Total Strategies: {run.summary['total_strategies']}

## Top Performers
{top_strategies[['strategy_type', 'sharpe_ratio']].head(10).to_markdown()}

## Ensemble Analysis
- Size: {len(ensemble['strategies'])}
- Avg Sharpe: {ensemble['avg_sharpe']:.2f}
- Max Correlation: {ensemble['max_correlation']:.2f}
"""

with open('analysis_report.md', 'w') as f:
    f.write(report)
```

## Tips for Effective Analysis

1. **Start Broad, Then Focus**: Begin with high-level summaries, then drill down into specific patterns.

2. **Use Views for Complex Queries**: Create DuckDB views for frequently used query patterns:
   ```python
   run.db.execute("""
       CREATE VIEW profitable_strategies AS
       SELECT * FROM strategies 
       WHERE total_return > 0.05 
       AND sharpe_ratio > 1.0
   """)
   ```

3. **Leverage Correlation Analysis**: Low correlation is key for ensemble diversification.

4. **Save Successful Patterns**: Use PatternLibrary to build a knowledge base of successful configurations.

5. **Combine with Market Data**: Join signal data with market data for regime-based analysis.

## Example Jupyter Workflow

See `templates/interactive_analysis_example.ipynb` for a complete example notebook demonstrating these concepts.