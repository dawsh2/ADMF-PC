# Trading Strategy Configuration Syntax Guide

## Overview
This guide documents the YAML configuration syntax for defining trading strategies, from simple indicators to complex regime-adaptive ensembles.

## Basic Configuration Structure

```yaml
# Required fields
name: my_trading_system
symbols: ["SPY", "QQQ"]
timeframes: ["1m", "5m"]
data_source: file
data_dir: ./data
start_date: "2023-01-01"
end_date: "2023-12-31"

# Strategy definition
strategy:
  # Your strategy here
```

## Strategy Syntax

### 1. Simple Strategy (Atomic)

A single strategy with parameters:

```yaml
strategy:
  sma_crossover:
    params: 
      fast_period: 10
      slow_period: 30
```

### 2. Composite Strategy (List)

Multiple strategies combined:

```yaml
strategy: [
  {sma_crossover: {params: {fast_period: 10, slow_period: 30}, weight: 0.5}},
  {rsi_threshold: {params: {period: 14, threshold: 70}, weight: 0.5}}
]
```

### 3. Filtered Strategies

Strategies with entry conditions:

```yaml
strategy:
  momentum:
    params: {period: 14}
    filter: |
      signal > 0.5 and 
      volume > ma(volume, 20) and
      price > vwap()
```

### 4. Conditional Strategies

Different strategies based on market conditions:

```yaml
strategy: [
  {
    filter: volatility_percentile(20) > 70
    momentum: {params: {period: 14}}
  },
  {
    filter: volatility_percentile(20) <= 70
    mean_reversion: {params: {period: 20}}
  }
]
```

### 5. Nested Strategies

Strategies within strategies:

```yaml
strategy: [
  {
    filter: market_hours() == 'regular'
    weight: 0.7
    strategy: [
      {sma_crossover: {weight: 0.6}},
      {momentum: {weight: 0.4}}
    ]
  },
  {
    filter: market_hours() == 'extended'
    weight: 0.3
    mean_reversion: {params: {period: 20}}
  }
]
```

### 6. Parameter Optimization

Define parameter ranges for optimization:

```yaml
strategy:
  momentum:
    params:
      period: [10, 14, 20]  # Grid search
      threshold: range(0.3, 0.8, 0.1)  # Range with step
      lookback: range(50, 200, 25)  # From 50 to 200, step 25
      
# Multi-strategy optimization
strategies: [
  {
    sma_crossover:
      params:
        fast_period: [5, 10, 15, 20]
        slow_period: [20, 30, 50, 100]
        # Constraint: slow > fast automatically enforced
  },
  {
    rsi_threshold:
      params:
        period: range(10, 20, 2)
        oversold: range(20, 40, 5)
        overbought: range(60, 80, 5)
  }
]
```

## Filter Expressions

Filters determine when strategies execute. They have access to:
- `signal` - The strategy's output (-1, 0, 1)
- `price` - Current price
- `volume` - Current volume
- All computed features and indicators

### Basic Filters

```yaml
# Single line
filter: signal > 0 and price > vwap()

# Multi-line with | (literal block)
filter: |
  signal > 0 and
  price > vwap() and
  volume > ma(volume, 20)
```

### Complex Filter Examples

```yaml
# Directional filters
filter: |
  (signal > 0 and price > vwap()) or  # Long above VWAP
  (signal < 0 and price < vwap())     # Short below VWAP

# Time-based filters
filter: |
  time_until_close() > 60 and
  market_hours() == 'regular' and
  not is_friday()

# Indicator-based filters
filter: |
  rsi(14) < 70 and
  volatility_percentile(20) > 30 and
  abs(price - vwap()) / vwap() > 0.002
```

### Available Filter Functions

```yaml
# Market microstructure
filter: |
  bid_ask_spread() < 0.001 and
  order_book_imbalance() > 0.2 and
  tick_direction() == 'uptick'

# Technical indicators
filter: |
  bollinger_band_width(20) < 0.02 and
  keltner_channel_position(20, 2) == 'below_lower' and
  atr(14) > atr_percentile(14, 20, 80)

# Price patterns
filter: |
  swing_high(5) or swing_low(5) and
  support_resistance_distance() < 0.001 and
  price_pattern('double_bottom', 20)

# Volume analysis
filter: |
  volume_profile_poc() > price and
  cumulative_delta() > 0 and
  relative_volume(20) > 1.5

# Market regime
filter: |
  volatility_regime() == 'high' and
  trend_regime() == 'strong_up' and
  correlation('SPY', 'VIX', 20) < -0.5
```

### Parameterized Filters

Filters can include parameters for optimization:

```yaml
strategy:
  momentum:
    params: {period: 14}
    filter: |
      signal > ${signal_threshold} and 
      volume > ma(volume, ${volume_period})
    filter_params:
      signal_threshold: range(0.3, 0.8)
      volume_period: range(10, 30)
```

## Risk Configuration

Risk settings can be global or strategy-specific:

```yaml
# Global risk settings
risk:
  stop_loss: 0.02
  position_size: 0.1
  max_holding_period: 100

strategy: [
  {
    momentum: {params: {period: 14}}
    risk:  # Override for this strategy
      stop_loss: 0.01
      trailing_stop: {activate: 0.02, distance: 0.01}
  }
]
```

## Research and Production Modes

The same configuration file can contain both research (parameter optimization) and production (live trading) sections:

### Research Mode

Define strategies to explore and optimize:

```yaml
# Research section - what we're exploring
research:
  strategies:
    # Test all mean reversion strategies
    - mean_reversion/*
    
    # Test specific strategy with parameter ranges
    - bollinger_bands:
        period: range(15, 30, 5)
        std: range(1.5, 2.5, 0.5)
    
    # Test with default parameter expansions
    - keltner_bands: default
    
    # Test multiple strategies with filters
    - type: rsi_threshold
      param_overrides:
        period: [10, 14, 20]
        oversold: [20, 30, 40]
        overbought: [60, 70, 80]
      filter: "volume > sma(volume, 20) * 1.2"
```

### Production Mode

Define the strategy to trade:

```yaml
# Production section - what we're trading now
production:
  strategy:
    bollinger_bands(20, 2.0):
      filter: "volume > sma(volume, 20) * 1.2"
      risk:
        stop_loss: 0.002
        max_holding_period: 100
```

### Running Different Modes

```bash
# Run research/optimization
python main.py --config my_strategy.yaml --research

# Run production trading
python main.py --config my_strategy.yaml --production

# Default (no flag) runs production
python main.py --config my_strategy.yaml
```

### Complete Research-to-Production Example

```yaml
name: bollinger_system
symbols: ["SPY"]
timeframes: ["5m"]

# Research - exploring variations
research:
  strategies:
    # Baseline search
    - bollinger_bands:
        period: [15, 20, 25, 30]
        std: [1.5, 2.0, 2.5]
    
    # With volume filter
    - type: bollinger_bands
      param_overrides:
        period: [20, 25]
        std: [2.0, 2.5]
      filter: "volume > sma(volume, 20) * ${vol_threshold}"
      filter_params:
        vol_threshold: [1.1, 1.2, 1.5]

# Production - current best
production:
  strategy:
    bollinger_bands(20, 2.0):
      filter: "volume > sma(volume, 20) * 1.2"
```

This approach keeps research and production in sync, allowing you to explore improvements while trading your current best strategy.

## Execution Configuration

Configure transaction costs and execution parameters:

```yaml
# Execution settings
execution:
  commission: 0.0001  # 1 bp per side
  slippage: 0.0001   # 1 bp slippage
  initial_capital: 100000
  data_frequency: "1m"
  fill_price: "close"  # or "open", "mid", "worst"
  
  # Advanced execution
  market_impact:
    model: "linear"  # or "square_root", "constant"
    coefficient: 0.1
  order_size_limit: 0.01  # Max 1% of volume
```

## Multi-Timeframe and Multi-Symbol

Strategies inherit timeframe/symbol from parent:

```yaml
symbols: ["SPY", "QQQ"]
timeframes: ["5m"]

strategy: [
  {
    # Inherits 5m timeframe
    sma_crossover: {params: {fast: 10, slow: 30}}
  },
  {
    # Override to 1m
    timeframe: "1m"
    filter: volatility_percentile(20) > 80
    scalping_strategy: {params: {threshold: 0.001}}
  },
  {
    # Multi-symbol strategy
    symbols: ["SPY", "QQQ"]
    pairs_trading: {params: {zscore_threshold: 2}}
  }
]
```

## Complete Example

A production-ready regime-adaptive strategy:

```yaml
name: adaptive_momentum_system
symbols: ["SPY"]
timeframes: ["1m"]
data_source: file

# Define risk parameters
risk:
  stop_loss: 0.001  # 0.1%
  position_size: 0.1
  min_bars_between_trades: 30

# Main strategy
strategy: [
  {
    # High volatility momentum
    filter: |
      volatility_regime == 'high' and
      trend_strength(50) > 0.5
    weight: 0.4
    risk:
      stop_loss: 0.0015  # Wider stop in high vol
    momentum:
      params: {period: 14}
      filter: signal > 0.5 and volume > ma(volume, 20)
  },
  {
    # Low volatility mean reversion
    filter: volatility_regime == 'low'
    weight: 0.3
    bollinger_bands_squeeze:
      params: {period: 20, std_dev: 2}
      filter: |
        abs(price - vwap()) / vwap() > 0.002 and
        time_until_close() > 60
  },
  {
    # Baseline trend following
    weight: 0.3
    filter: volume > ma(volume, 50) * 0.8  # Liquidity filter
    sma_crossover:
      params: {fast_period: 10, slow_period: 30}
      filter: signal != 0  # Take all crossover signals
  }
]
```

## Special Syntax Notes

### YAML Block Scalars
- `|` - Preserves line breaks (literal)
- `>` - Folds line breaks into spaces
- `|-` - Literal without trailing newline
- `|+` - Literal keeping trailing newlines

### Weight Normalization
Weights are automatically normalized if they don't sum to 1.0

### Signal Values
- `1` - Long signal
- `-1` - Short signal  
- `0` - Flat/no position

### Filter Evaluation
- Filters are evaluated before strategy execution
- Failed filters result in signal = 0
- Filters can reference any computed feature
