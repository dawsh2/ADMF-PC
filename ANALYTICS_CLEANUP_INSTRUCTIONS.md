# Analytics Module Cleanup Instructions

## Overview
The analytics module has been refactored with a new, cleaner implementation. This document provides instructions for completing the cleanup.

## Cleanup Script
A cleanup script has been created at `/Users/daws/ADMF-PC/run_analytics_cleanup.py` that will:

1. Remove all old/deprecated files
2. Remove old directories (cli/, mining/, queries/, sparse_trace_analysis/, storage/)
3. Clean up __pycache__ directories
4. Rename README_NEW.md to README.md
5. Remove the old README.md and cleanup_list.txt

## To Run the Cleanup

```bash
cd /Users/daws/ADMF-PC
python3 run_analytics_cleanup.py
```

## Files That Will Be Removed

### Python Files (Old Implementation)
- backtesting_framework.py
- calculate_log_returns.py
- correlation_filter.py
- exceptions.py
- execution_cost_analyzer.py
- fast_correlation_filter.py
- fixed_backtesting_example.py
- functions.py
- grid_search_analyzer.py
- integration.py
- metrics.py
- migration.py
- parameter_export.py
- patterns.py
- populate_from_metadata.py
- reports.py
- schema.py
- signal_context_analysis.py
- signal_performance_analyzer.py
- signal_reconstruction.py
- strategy_filter.py
- strategy_filter_optimized.py
- workspace.py

### Other Files
- cheap-filters.md
- README.md (old version)
- cleanup_list.txt

### Directories
- cli/ (and all contents)
- mining/ (and all contents)
- queries/ (and all contents)
- sparse_trace_analysis/ (and all contents)
- storage/ (and all contents)
- All __pycache__ directories

## Files That Will Remain (New Implementation)

- __init__.py (updated with new imports)
- trace_analysis.py (new core analysis module)
- pattern_discovery.py (new pattern discovery module)
- trade_metrics.py (new trade analysis module)
- queries.py (pre-built SQL queries)
- example_usage.py (usage examples)
- README.md (renamed from README_NEW.md)
- saved_patterns/ (directory for saving discovered patterns)

## After Cleanup

Once the cleanup is complete:

1. Remove the cleanup scripts:
   ```bash
   python3 cleanup_temp_files.py
   rm run_analytics_cleanup.py
   rm ANALYTICS_CLEANUP_INSTRUCTIONS.md
   ```

2. The analytics module will be ready to use with the new, cleaner API:
   ```python
   from analytics import quick_analysis
   ta, pd = quick_analysis("path/to/results")
   ```

## Git Commit

After cleanup, you can commit the changes:

```bash
git add -A src/analytics/
git commit -m "Refactor analytics module with cleaner, notebook-friendly API

- Remove old implementation files and directories
- New modular design with TraceAnalysis, PatternDiscovery, and TradeAnalyzer
- Simplified imports and better documentation
- Interactive notebook-first design"
```