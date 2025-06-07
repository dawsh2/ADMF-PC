# Analytics Module

This module provides data mining, visualization, and analysis capabilities for ADMF-PC optimization results.

## Structure

```
analytics/
├── README.md
├── __init__.py
├── core/                    # Core analytics infrastructure
│   ├── __init__.py
│   ├── storage.py          # Event persistence (Parquet/SQL)
│   ├── etl.py              # Extract-Transform-Load pipelines
│   └── query.py            # Query interfaces
├── mining/                  # Data mining and pattern discovery
│   ├── __init__.py
│   ├── patterns.py         # Pattern discovery algorithms
│   ├── correlation.py      # Cross-strategy analysis
│   └── validation.py       # Pattern validation
├── visualization/           # Dashboards and reports
│   ├── __init__.py
│   ├── reports.py          # HTML/PDF report generation
│   ├── dashboards.py       # Interactive dashboards
│   └── plots.py            # Plotting utilities
└── monitoring/             # Real-time monitoring
    ├── __init__.py
    ├── alerts.py           # Alert system
    ├── live_patterns.py    # Live pattern matching
    └── metrics.py          # Real-time metrics

```

## Why Not in `src/`?

1. **Different lifecycle**: Analytics code analyzes results AFTER optimization runs, while `src/` contains the core trading logic
2. **External dependencies**: May use different libraries (pandas, plotly, dash) that aren't needed for core trading
3. **Deployment separation**: Core trading engine can be deployed without analytics overhead
4. **Clear boundaries**: Analytics reads from event stores but doesn't participate in trading decisions

## Integration Points

- Reads events from coordinator's event tracing
- Connects to results stored in `results/` directory
- Can be run as separate service or on-demand analysis

## Usage

```python
from analytics.mining import PatternDiscovery
from analytics.visualization import create_performance_report

# Analyze recent optimization
patterns = PatternDiscovery().mine_optimization_results(
    start_date="2024-01-01",
    min_sharpe=1.5
)

# Generate report
report = create_performance_report(patterns)
report.save("results/analysis_report.html")
```