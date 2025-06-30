# ADMF-PC Configuration Guide

This guide demonstrates the configuration system from basic strategies to production-ready ensembles.

## Directory Structure

Each strategy lives in its own directory:

```
configs/
├── strategy_name/
│   ├── config.yaml                    # Strategy configuration
│   ├── results/                       # Timestamped runs
│   │   └── run_20241224_143022/
│   │       ├── metadata.json          # Run metadata
│   │       ├── strategy_index.parquet # Queryable strategy catalog
│   │       └── traces/                # Sparse signal traces
│   └── notebooks/                     # Analysis notebooks
```

## Core Concepts

### 1. Basic Strategy

```yaml
# configs/simple_ma/config.yaml
name: simple_ma_crossover
symbols: ["SPY"]
timeframe: "5m"

strategy:
  sma_crossover:
    fast_period: 10
    slow_period: 30
```

Run: `python main.py -c configs/simple_ma/config.yaml`

### 2. Parameter Sweeps

Test multiple parameter combinations:

```yaml
# configs/ma_research/config.yaml
name: ma_crossover_research
symbols: ["SPY"]
timeframe: "5m"

strategy:
  sma_crossover:
    fast_period: [5, 10, 15, 20]      # List of values
    slow_period: [20, 30, 50, 100]
  
  bollinger_bands:
    period: range(15, 30, 5)          # Range syntax
    std_dev: [1.5, 2.0, 2.5]
```

This generates all combinations (16 for SMA, 12 for Bollinger).

### 3. Filters and Thresholds

Add conditions to filter signals:

```yaml
# Simple filter
strategy:
  sma_crossover:
    fast_period: 10
    slow_period: 30
    threshold: "volume > sma(volume, 20) * 1.2"

# Complex filter with boolean logic
strategy:
  momentum:
    period: 14
    threshold: |
      momentum > 0 AND 
      volume > sma(volume, 20) * 1.5 AND
      rsi(14) < 70 AND
      market_hours() == 'regular'

# Directional thresholds
strategy:
  macd:
    fast: 12
    slow: 26
    signal: 9
    threshold:
      long: "macd > 0 AND rsi(14) < 70"
      short: "macd < 0 AND rsi(14) > 30"
```

### 4. Weighted Ensembles

Combine multiple strategies:

```yaml
# Simple ensemble
strategy: [
  {sma_crossover: {fast_period: 10, slow_period: 30, weight: 0.4}},
  {bollinger_bands: {period: 20, std_dev: 2.0, weight: 0.3}},
  {rsi_bands: {period: 14, oversold: 30, overbought: 70, weight: 0.3}},
  {threshold: "0.5"}  # Minimum combined weight for signal
]

# Testing different thresholds
strategy: [
  {momentum: {period: 14, weight: 0.6}},
  {mean_reversion: {period: 20, weight: 0.4}},
  {
    threshold: [
      "0.5",                              # Pure weighted
      "0.5 AND volume > sma(volume, 20)", # With volume filter
      "0.3 AND atr(14) > 0.001"          # Lower threshold with volatility
    ]
  }
]
```

### 5. Nested Compositions

Build complex hierarchical strategies:

```yaml
strategy: [
  # Trend following group (60% weight)
  {
    weight: 0.6,
    strategy: [
      {ma_crossover: {fast_period: 10, slow_period: 30, weight: 0.5}},
      {momentum: {period: 14, weight: 0.5}}
    ],
    threshold: "0.3 AND adx(14) > 25"
  },
  
  # Mean reversion group (40% weight)  
  {
    weight: 0.4,
    strategy: [
      {bollinger_bands: {period: 20, std_dev: 2.0, weight: 0.6}},
      {rsi_extreme: {period: 14, oversold: 30, weight: 0.4}}
    ],
    threshold: "0.5 AND adx(14) < 20"
  },
  
  # Overall threshold
  {threshold: "0.5"}
]
```

### 6. Risk Management

Add stops and position sizing:

```yaml
# Global risk settings
risk:
  position_size: 0.1    # 10% per trade
  max_positions: 3
  max_drawdown: 0.10    # 10% maximum

strategy:
  sma_crossover:
    fast_period: 10
    slow_period: 30
    threshold: "volume > sma(volume, 20) * 1.2"
    
    # Strategy-specific risk
    risk:
      stop_loss: 0.002        # 0.2%
      take_profit: 0.004      # 0.4%
      max_holding_period: 100 # bars
      
      # Dynamic exits
      exit_filters: [
        {
          condition: "position_pnl > 0.002"
          action: {reduce_position: 0.5}  # Take half profit
        },
        {
          condition: "atr(14) > entry_atr * 2"
          stop_loss: "2 * atr(14)"        # Widen stop in high volatility
        }
      ]
```

## Advanced Features

### Parameter Optimization

```yaml
# configs/optimize_ma/config.yaml
name: optimize_ma
symbols: ["SPY", "QQQ"]
timeframe: "5m"

strategy:
  sma_crossover:
    fast_period: range(5, 30, 5)
    slow_period: range(20, 100, 10)

# Run optimization
# python main.py -c configs/optimize_ma/config.yaml --optimize
```

### Wildcard Discovery

Automatically test all available strategies:

```yaml
parameter_space:
  # Test all indicators
  indicators: "*"
  
  # Or specific categories
  indicators:
    momentum: "*"      # All momentum strategies
    volatility: "*"    # All volatility strategies
  
  # Control granularity
  optimization:
    granularity: 5     # Sample points for ranges
```

### Multi-Timeframe

```yaml
strategy: [
  {
    weight: 0.3,
    timeframe: "1h",
    trend_following: {ma_period: 20},
    threshold: "trend_following AND rsi(14, '1h') > 50"
  },
  {
    weight: 0.7,
    timeframe: "5m",  # Default
    momentum: {period: 14},
    threshold: "momentum AND volume > sma(volume, 20, '5m') * 1.5"
  }
]
```

## Working with Results

### Load Results
```python
import json
import pandas as pd

# Load metadata
with open('results/latest/metadata.json') as f:
    meta = json.load(f)

# Load strategy index (all strategies tested)
strategy_index = pd.read_parquet('results/latest/strategy_index.parquet')

# Query specific strategies
bollinger_20 = strategy_index[
    (strategy_index['strategy_type'] == 'bollinger_bands') &
    (strategy_index['period'] == 20)
]
```

### Analyze Signals
```python
import duckdb

con = duckdb.connect()

# Query sparse signal traces directly
signals = con.execute("""
    SELECT 
        strategy_hash,
        COUNT(*) as num_signals,
        COUNT(DISTINCT DATE(ts)) as trading_days
    FROM read_parquet('results/latest/traces/**/*.parquet')
    WHERE val != 0
    GROUP BY strategy_hash
""").df()
```

## Command Line Usage

```bash
# Signal generation (backtest)
python main.py -c config.yaml --signal-generation --dataset train

# With auto-generated notebook
python main.py -c config.yaml --signal-generation --notebook --launch-notebook

# Optimization
python main.py -c config.yaml --optimize

# Walk-forward validation
python main.py -c config.yaml --wfv-windows 10 --wfv-window 1
```

## Key Design Principles

1. **Composability**: Strategies can contain strategies recursively
2. **Parameter Expansion**: Any parameter can be a single value, list, or range
3. **Sparse Storage**: Only signal changes are stored (50-200x compression)
4. **Self-Documenting**: Each trace file contains its configuration
5. **Strategy Hashing**: Identical configurations get the same hash across runs

## Filter Expression Reference

Available in threshold expressions:
- **Variables**: `signal`, `price`, `open`, `high`, `low`, `close`, `volume`
- **Functions**: `sma()`, `ema()`, `rsi()`, `atr()`, `vwap()`, `adx()`
- **Market state**: `market_hours()`, `volatility_regime()`, `trend_strength()`
- **Math**: `abs()`, `min()`, `max()`

## Production Workflow

1. **Research**: Test wide parameter ranges
2. **Refine**: Focus on promising parameters
3. **Validate**: Walk-forward or out-of-sample testing
4. **Deploy**: Fixed parameters in production config

```yaml
# Production config - no parameter lists
name: production_bollinger
symbols: ["SPY"]
timeframe: "5m"

strategy:
  bollinger_bands:
    period: 20        # Fixed values only
    std_dev: 2.0
    threshold: "volume > sma(volume, 20) * 1.2"
    risk:
      stop_loss: 0.002
      position_size: 0.05
```

This configuration system provides a smooth path from research to production while maintaining clarity and flexibility.
