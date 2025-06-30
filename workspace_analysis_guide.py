#!/usr/bin/env python3
"""
Guide for analyzing P&L and performance metrics from ADMF-PC workspace
"""

print("""
============================================================
ADMF-PC Workspace Performance Analysis Guide
============================================================

Your workspace at 'workspaces/20250617_194112_signal_generation_SPY' contains
an analytics.duckdb file with strategy performance data.

To analyze P&L and performance metrics, you have several options:

1. USING PYTHON WITH REQUIRED PACKAGES
======================================

First, ensure you have the required packages installed:
    pip install pandas numpy duckdb

Then use this code:

```python
import sys
sys.path.append('/Users/daws/ADMF-PC')

from src.analytics.workspace import AnalyticsWorkspace

# Connect to workspace
ws = AnalyticsWorkspace('workspaces/20250617_194112_signal_generation_SPY')

# Get workspace summary
summary = ws.summary()
print(f"Total strategies: {summary['total_strategies']}")
print(f"Strategy types: {summary['strategy_types']}")

# Query strategy performance (if pre-calculated)
strategies = ws.sql('''
    SELECT 
        strategy_id,
        strategy_name,
        strategy_type,
        signal_file_path
    FROM strategies
    ORDER BY strategy_name
    LIMIT 10
''')

# Calculate performance for specific strategy
if not strategies.empty:
    strategy_id = strategies.iloc[0]['strategy_id']
    perf = ws.calculate_performance(strategy_id)
    print(f"Performance for {strategy_id}:")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']}")
    print(f"  Total Return: {perf['total_return']}")
```

2. USING DUCKDB CLI DIRECTLY
============================

Install DuckDB CLI:
    brew install duckdb  # macOS
    # or download from https://duckdb.org/docs/installation/

Then query the database:

```bash
duckdb workspaces/20250617_194112_signal_generation_SPY/analytics.duckdb

-- Show all tables
.tables

-- Get strategy count
SELECT COUNT(*) FROM strategies;

-- View strategy types
SELECT strategy_type, COUNT(*) as count 
FROM strategies 
GROUP BY strategy_type;

-- Get top strategies (if performance is stored)
SELECT * FROM strategies LIMIT 10;
```

3. USING SIGNAL PERFORMANCE ANALYZER
====================================

If the workspace contains signal/event files:

```python
from src.analytics.signal_performance_analyzer import SignalPerformanceAnalyzer

analyzer = SignalPerformanceAnalyzer('workspaces/20250617_194112_signal_generation_SPY')

# Load and analyze signals
analyzer.load_signal_events()
analyzer.pair_signals()
metrics = analyzer.calculate_performance()

print(analyzer.get_summary_report())
```

4. CUSTOM ANALYSIS SCRIPTS
=========================

Several pre-built analysis scripts are available:

- src/analytics/queries/analyze_all_strategies_batch.py
- src/analytics/queries/analyze_strategies_batch.py
- analyze_signal_generation_results.py
- analyze_final_workspace.py

Modify the workspace path in these scripts to point to your workspace.

5. EXPECTED PERFORMANCE METRICS
==============================

When properly analyzed, you should see metrics like:

- Total Return (%)
- Sharpe Ratio (annualized)
- Max Drawdown (%)
- Win Rate (%)
- Average Trade Return
- Number of Trades
- Profit Factor
- Risk-adjusted returns by market regime

NEXT STEPS
==========

1. Install required packages (pandas, numpy, duckdb)
2. Use the Python code examples above to connect to your workspace
3. The analytics.duckdb file contains the strategy catalog
4. Performance metrics may need to be calculated from stored signals

For immediate results without dependencies, check if these files exist:
- workspaces/*/analysis/signal_performance.json
- workspaces/*/results/*.json
- workspaces/*/metrics.json

============================================================
""")