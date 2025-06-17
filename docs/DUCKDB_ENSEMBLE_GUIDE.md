# DuckDB Ensemble Strategy Guide

## Overview

The DuckDB Ensemble is an adaptive strategy that dynamically switches between different trading strategies based on detected market regimes. It was designed based on comprehensive backtesting analysis of 1,235+ strategies across different market conditions.

## Key Features

1. **Regime-Adaptive**: Automatically switches strategies based on market conditions
2. **Equal Weighting**: Uses simple 1/n weighting for all active strategies
3. **Configurable**: Easily customize strategies per regime
4. **Transparent**: Tracks which strategies contribute to each signal

## How It Works

### 1. Regime Detection
The strategy monitors a classifier (default: `volatility_momentum_classifier`) that categorizes market conditions into regimes:
- `low_vol_bullish`: Low volatility with bullish momentum
- `low_vol_bearish`: Low volatility with bearish momentum
- `neutral`: Normal market conditions
- `high_vol_bullish`: High volatility with bullish momentum
- `high_vol_bearish`: High volatility with bearish momentum

### 2. Strategy Selection
Based on the current regime, the ensemble activates a specific set of strategies optimized for that condition:

#### Low Volatility Bullish (Best Performers)
- **DEMA Crossover (19,35)**: Sharpe 3.389, Return 22.58%
- **MACD Crossover (12,35,9)**: Sharpe 3.013, Return 21.10%
- **DEMA Crossover (7,35)**: Sharpe 3.191, Return 22.45%
- **MACD Crossover (15,35,7)**: Sharpe 2.976, Return 20.76%
- **CCI Threshold (11,-40)**: Sharpe 2.925, Return 19.89%

#### Low Volatility Bearish (Defensive Strategies)
- **Stochastic Crossover (27,5)**: Sharpe 2.051, Return 15.03%
- **CCI Threshold (11,-20)**: Sharpe 1.962, Return 14.66%
- **EMA-SMA Crossover (11,15)**: Sharpe 1.886, Return 13.04%
- **Keltner Breakout (11,1.5)**: Sharpe 2.020, Return 5.80%
- **RSI Bands (7,25,70)**: Sharpe 2.071, Return 7.00%

#### Neutral (Balanced Strategies)
- **Stochastic RSI (21,21,15,80)**: Sharpe 2.686, Return 6.63%
- **Stochastic RSI (21,21,15,75)**: Sharpe 2.590, Return 6.22%
- **DEMA Crossover (19,15)**: Sharpe 2.405, Return 7.49%
- **Vortex Crossover (27)**: Sharpe 2.071, Return 5.94%

### 3. Signal Aggregation
The ensemble collects signals from all active strategies and aggregates them:
- Counts bullish (+1) and bearish (-1) signals
- Requires minimum agreement (default 30%) to generate a signal
- Returns the majority direction if agreement threshold is met

## Configuration

### Basic Usage
```yaml
strategies:
  - type: duckdb_ensemble
    name: adaptive_ensemble
    params:
      classifier_name: volatility_momentum_classifier
      aggregation_method: equal_weight
      min_agreement: 0.3
```

### Custom Strategy Mapping
```yaml
strategies:
  - type: duckdb_ensemble
    name: custom_ensemble
    params:
      regime_strategies:
        low_vol_bullish:
          - name: dema_crossover
            params:
              fast_dema_period: 19
              slow_dema_period: 35
          - name: macd_crossover
            params:
              fast_ema: 12
              slow_ema: 35
              signal_ema: 9
        low_vol_bearish:
          - name: stochastic_crossover
            params:
              k_period: 27
              d_period: 5
```

## Parameters

- **classifier_name**: Which classifier to use for regime detection
- **aggregation_method**: How to combine signals (currently only 'equal_weight')
- **min_agreement**: Minimum fraction of strategies that must agree (0-1)
- **regime_strategies**: Custom mapping of regimes to strategy configurations

## Output Signal

The strategy returns a signal with:
- **signal_value**: -1 (short), 0 (neutral), or 1 (long)
- **metadata**:
  - Current regime
  - Number of active strategies
  - Number of signals generated
  - Bullish/bearish signal counts
  - Agreement ratio
  - Details of contributing strategies

## Best Practices

1. **Start Conservative**: Use higher `min_agreement` (0.5+) to reduce false signals
2. **Monitor Regime Transitions**: The strategy adapts immediately to regime changes
3. **Backtest Thoroughly**: Test with your specific data and time periods
4. **Consider Transaction Costs**: More strategies mean more potential trades

## Example Configurations

### Conservative Ensemble
```python
CONSERVATIVE_ENSEMBLE = {
    'type': 'duckdb_ensemble',
    'params': {
        'min_agreement': 0.5,  # Require 50% agreement
        'regime_strategies': {
            'low_vol_bullish': [
                # Only top 2 performers
                {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 35}},
                {'name': 'macd_crossover', 'params': {'fast_ema': 12, 'slow_ema': 35, 'signal_ema': 9}}
            ]
        }
    }
}
```

### Aggressive Ensemble
```python
AGGRESSIVE_ENSEMBLE = {
    'type': 'duckdb_ensemble',
    'params': {
        'min_agreement': 0.2,  # Only 20% agreement needed
        # Uses all default strategies per regime
    }
}
```

## Performance Notes

Based on our analysis:
- **Low Vol Bullish** regime showed the best performance (avg Sharpe: 3.099)
- **Neutral** regime had moderate performance (avg Sharpe: 2.438)
- **Low Vol Bearish** regime focused on capital preservation (avg Sharpe: 1.967)

Remember: Past performance doesn't guarantee future results. Always validate with your own data and risk parameters.