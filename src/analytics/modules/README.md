# Analytics Modules

Organized, reusable analysis components for ADMF-PC analytics.

## Quick Start

```python
# In any Jupyter notebook:
from analytics.modules.trade_analysis import extract_trades, calculate_trade_frequency
from analytics.modules.regime_analysis import add_regime_indicators, analyze_by_regime
from analytics.modules.optimization import optimize_stops_by_regime

# Load your data
signals_df = pd.read_parquet('path/to/signals.parquet')

# Extract trades
trades = extract_trades(signals_df)

# Filter by frequency
freq_stats = calculate_trade_frequency(trades)
qualified_trades = filter_by_frequency(trades, min_per_day=2, max_per_day=50)

# Add regime indicators
signals_with_regimes = add_regime_indicators(signals_df)

# Analyze performance by regime
regime_stats = analyze_by_regime(qualified_trades, signals_with_regimes, 'volatility_regime')

# Optimize stops by regime
stop_results = optimize_stops_by_regime(qualified_trades, signals_with_regimes, 'trend_regime')
```

## Module Structure

### core/
Core utilities and helpers
- `data_loading.py` - Load traces, signals, and market data
- `helpers.py` - General utility functions
- `metrics.py` - Performance calculations (Sharpe, returns, etc.)

### trade_analysis/
Trade extraction and analysis
- `extraction.py` - Extract trades from signals
- `frequency.py` - Trade frequency analysis and filtering
- `duration.py` - Trade duration analysis

### regime_analysis/
Market regime analysis
- `volatility.py` - Volatility regime detection
- `trend.py` - Trend regime detection
- `volume.py` - Volume regime detection
- `combined.py` - Combined regime filters

### optimization/
Parameter optimization
- `stop_loss.py` - Stop loss optimization
- `targets.py` - Profit target optimization
- `regime_specific.py` - Regime-specific optimization

### performance/
Performance analysis
- `sharpe.py` - Sharpe ratio calculations (traditional and compound)
- `equity_curves.py` - Equity curve analysis
- `comparison.py` - Strategy comparison utilities

### visualization/
Plotting utilities
- `regimes.py` - Regime visualization
- `performance.py` - Performance plots

## Common Workflows

### 1. Filter Out Infrequent Traders

```python
from analytics.modules.trade_analysis import extract_trades, filter_by_frequency

# Extract trades from signals
trades = extract_trades(signals_df)

# Filter strategies trading 2-50 times per day
qualified_trades = filter_by_frequency(
    trades, 
    min_per_day=2, 
    max_per_day=50
)
```

### 2. Analyze Performance by Market Regime

```python
from analytics.modules.regime_analysis import (
    add_volatility_regime, 
    add_trend_regime,
    analyze_by_regime
)

# Add regime indicators
signals_df = add_volatility_regime(signals_df)
signals_df = add_trend_regime(signals_df)

# Analyze performance
vol_stats = analyze_by_regime(trades, signals_df, 'volatility_regime')
trend_stats = analyze_by_regime(trades, signals_df, 'trend_regime')
```

### 3. Optimize Stops by Regime

```python
from analytics.modules.optimization import optimize_stops_by_regime

# Find optimal stops for each volatility regime
stop_results = optimize_stops_by_regime(
    trades, 
    signals_df, 
    'volatility_regime',
    stop_range=[0.1, 0.2, 0.3, 0.5, 1.0],
    target_range=[0.2, 0.5, 1.0, 2.0]
)

# Display results
for regime, params in stop_results.items():
    print(f"{regime}: Stop={params['stop']}%, Target={params['target']}%")
```

## Migration from Snippets

The old snippets are still available in `src/analytics/snippets/`. 
This new module structure consolidates and improves upon that functionality.

Key improvements:
- Proper function signatures with type hints
- Comprehensive docstrings
- Error handling
- Unit tests (coming soon)
- No global variable dependencies

## Contributing

When adding new functionality:
1. Place it in the appropriate module
2. Include docstrings and type hints
3. Make it self-contained (no globals)
4. Add usage examples in docstring
5. Update this README if needed