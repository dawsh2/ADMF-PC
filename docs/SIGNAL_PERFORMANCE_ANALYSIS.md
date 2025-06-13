# Signal Performance Analysis

## Overview

The signal performance analysis system allows you to evaluate strategy performance using only generated signals, without requiring actual trade execution. This is particularly useful for:

- Backtesting signal quality before implementing execution logic
- Comparing strategy performance across different parameter sets
- Understanding signal patterns and timing
- Validating strategy logic

## Architecture

### 1. Signal Storage (Existing)

Signals are automatically stored by the `HierarchicalEventStorage` system:

```
workspaces/
└── {workflow_id}/
    └── {portfolio_container}/
        ├── events.parquet      # All events including signals
        ├── signals.parquet     # Signal index for fast lookup
        └── metrics.json        # Container metrics
```

### 2. Signal Performance Analyzer (New)

The `SignalPerformanceAnalyzer` class reads stored signals and:

1. **Loads signal events** from parquet files
2. **Pairs entry/exit signals** using implicit exit logic (opposite direction = exit)
3. **Calculates performance metrics** including:
   - Win rate
   - Profit factor
   - Sharpe ratio
   - P&L statistics
   - Per-strategy breakdown
   - Per-symbol breakdown

### 3. Enhanced Signal Metrics Observer

The `SignalMetricsObserver` now includes:
- Real-time signal pairing during generation
- Performance calculation using `SignalOnlyPerformance` calculator
- Integration with signal matching logic

## Usage

### 1. Generate Signals with Storage

```bash
# Run signal generation with hierarchical storage
python main.py --signal-generation --config config/test_signal_performance.yaml
```

### 2. Analyze Performance

```bash
# Analyze stored signals
python test_signal_performance_analysis.py
```

### 3. View Results

Results are saved to:
- `workspaces/{workflow_id}/analysis/signal_performance.json` - Detailed metrics
- `workspaces/{workflow_id}/analysis/signal_pairs.parquet` - Signal pairs data
- `results/signal_performance_summary.json` - Summary report

## Example Configuration

```yaml
# Enable hierarchical storage for signal analysis
execution:
  enable_event_tracing: true
  trace_settings:
    storage_backend: hierarchical
    batch_size: 100
    auto_flush_on_cleanup: true
    
# Strategy configuration
strategies:
  - name: momentum_fast
    type: momentum
    params:
      sma_period: 10
      rsi_period: 14
```

## Performance Metrics

The analyzer calculates:

### Overall Metrics
- **Total trades**: Number of completed signal pairs
- **Win rate**: Percentage of profitable trades
- **Profit factor**: Gross profit / Gross loss
- **Sharpe ratio**: Risk-adjusted return metric
- **Average P&L**: Mean profit/loss per trade
- **Max drawdown**: Largest peak-to-trough decline

### Breakdown Analysis
- **Per-strategy metrics**: Performance by strategy name
- **Per-symbol metrics**: Performance by trading symbol
- **Time-based analysis**: Holding periods and timing

## Signal Pairing Logic

The analyzer uses implicit exit logic:

1. **Entry signal** opens a position
2. **Opposite direction signal** closes existing position and opens new one
3. **Same direction signals** are ignored (keep first position)

Example:
```
Time 1: LONG signal at $100   → Open long position
Time 2: LONG signal at $102   → Ignored (already long)
Time 3: SHORT signal at $105  → Close long (+$5), open short
Time 4: LONG signal at $103   → Close short (+$2), open long
```

## API Reference

### SignalPerformanceAnalyzer

```python
analyzer = SignalPerformanceAnalyzer(workspace_path)

# Load signals
signals_df = analyzer.load_signal_events()

# Pair signals
pairs = analyzer.pair_signals()

# Calculate metrics
metrics = analyzer.calculate_performance()

# Save analysis
analyzer.save_analysis()

# Get text report
report = analyzer.get_summary_report()
```

### Convenience Function

```python
from src.analytics.signal_performance_analyzer import analyze_signal_performance

# One-line analysis
metrics = analyze_signal_performance('workspaces/my_workspace')
```

## Benefits

1. **No execution required** - Evaluate strategies before implementing execution
2. **Fast iteration** - Test multiple parameter sets quickly
3. **Detailed analysis** - Understand signal patterns and performance
4. **Storage efficient** - Uses existing sparse storage system
5. **Post-processing** - Analyze historical runs anytime

## Future Enhancements

1. **Strategy-specific storage paths** - Organize by strategy instead of portfolio
2. **Advanced pairing logic** - Support stop-loss and take-profit levels
3. **Regime analysis** - Performance breakdown by market conditions
4. **Multi-timeframe analysis** - Cross-timeframe signal correlation
5. **Visualization** - Charts and graphs of signal performance