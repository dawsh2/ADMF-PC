# Wildcard Discovery Guide for ADMF-PC

## Overview

The ADMF-PC framework supports powerful wildcard discovery for automatic strategy testing and optimization. This feature allows you to:

1. Automatically discover strategies without manually listing them
2. Test all strategies in a category or with specific tags
3. Generate parameter combinations for optimization
4. Run comprehensive backtests with minimal configuration

## Configuration Syntax

### 1. Test All Indicators

```yaml
parameter_space:
  indicators: "*"
```

This discovers ALL indicator strategies in `src/strategy/strategies/indicators/`.

### 2. Test by Category

```yaml
parameter_space:
  indicators:
    crossover: "*"      # All crossover strategies
    momentum: "*"       # All momentum strategies
    oscillator: "*"     # All oscillator strategies
    trend: "*"          # All trend strategies
    volatility: "*"     # All volatility strategies
    volume: "*"         # All volume strategies
    structure: "*"      # All structure strategies
```

### 3. Test by Strategy Tags

```yaml
parameter_space:
  indicators:
    mean_reversion: "*"    # All strategies tagged with 'mean_reversion'
    trend_following: "*"   # All strategies tagged with 'trend_following'
```

### 4. Mix Wildcards and Specific Strategies

```yaml
parameter_space:
  indicators:
    crossover: "*"                        # All crossovers
    momentum: ["rsi", "macd_crossover"]   # Specific momentum strategies
    volatility: "*"                       # All volatility strategies
```

## How It Works

1. **Discovery Phase**: The system scans registered strategies based on your wildcard patterns
2. **Parameter Extraction**: Extracts parameter spaces from `@strategy` decorators
3. **Combination Generation**: Creates parameter combinations based on ranges (using granularity setting)
4. **Execution**: Runs signal generation or backtesting for each combination

## Important Requirements

### Base Strategy Required

Even when using wildcards, you must define a base strategy:

```yaml
# Base strategy (used when not optimizing)
strategy:
  sma_crossover:
    params:
      fast_period: 10
      slow_period: 30

# Wildcard discovery
parameter_space:
  indicators: "*"
```

### Optimization Flag

Wildcard discovery only works with the `--optimize` flag:

```bash
python main.py --config config/my_wildcard_config.yaml --signal-generation --optimize
```

## Granularity Control

Control the number of parameter samples for ranges:

```yaml
optimization:
  granularity: 3  # Default: 5
                  # Lower = faster but less thorough
                  # Higher = slower but more comprehensive
```

## Example Configs

### 1. Test All Mean Reversion Strategies

```yaml
# config/test_mean_reversion_wildcard.yaml
name: test_mean_reversion_wildcard
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

strategy:
  rsi_bands:
    params:
      rsi_period: 14
      oversold: 30
      overbought: 70

parameter_space:
  indicators:
    mean_reversion: "*"

optimization:
  granularity: 3
```

### 2. Test Specific Categories

```yaml
# config/test_categories_wildcard.yaml
parameter_space:
  indicators:
    oscillator: "*"     # RSI, CCI, Williams %R, etc.
    volatility: "*"     # Bollinger, Keltner, ATR bands
    structure: "*"      # Pivots, trendlines, channels
```

### 3. Test Everything

```yaml
# config/test_all_wildcard.yaml
parameter_space:
  indicators: "*"         # All indicators
  classifiers: "*"        # All classifiers
```

## Strategy Tags

Strategies can be tagged in their `@strategy` decorator:

```python
@strategy(
    name="rsi_bands",
    tags=["oscillator", "mean_reversion"],
    parameter_space={
        "rsi_period": (7, 21),
        "oversold": (20, 30),
        "overbought": (70, 80)
    }
)
```

Common tags:
- `mean_reversion` - Mean reverting strategies
- `trend_following` - Trend following strategies
- `breakout` - Breakout strategies
- `oscillator` - Oscillator-based
- `volatility` - Volatility-based
- `volume` - Volume-based
- `structure` - Market structure-based

## Category Patterns

The system uses pattern matching for categories:

- `crossover` → Matches: `*_crossover`, `*_cross`
- `momentum` → Matches: `momentum`, `rsi`, `macd`, `roc`
- `oscillator` → Matches: `rsi`, `cci`, `stochastic`, `williams`
- `trend` → Matches: `adx`, `aroon`, `supertrend`, `sar`
- `volatility` → Matches: `atr`, `bollinger`, `keltner`
- `volume` → Matches: `obv`, `vwap`, `chaikin`, `mfi`
- `structure` → Matches: `pivot`, `support`, `trendline`

## Running Wildcard Configs

```bash
# Signal generation with optimization
python main.py --config config/test_mean_reversion_wildcard.yaml --signal-generation --bars 200 --optimize

# Backtesting with optimization
python main.py --config config/test_all_wildcard.yaml --backtest --optimize

# With custom granularity
python main.py --config config/test_categories_wildcard.yaml --signal-generation --optimize --granularity 10
```

## Troubleshooting

### Error: 'NoneType' object has no attribute 'get'

This usually means:
1. Missing base strategy definition
2. Invalid wildcard syntax
3. No strategies found matching the pattern

### No Strategies Discovered

Check:
1. Strategy registration in `__init__.py`
2. Correct tags in `@strategy` decorator
3. Valid category names

### Too Many Combinations

Reduce:
1. Granularity setting
2. Number of categories
3. Parameter ranges in strategies

## Best Practices

1. **Start Small**: Test one category first before "*"
2. **Use Tags**: More precise than category patterns
3. **Monitor Progress**: Watch logs for discovered strategies
4. **Save Results**: Use workspace features for analysis
5. **Iterate**: Refine based on initial results

## Summary

Wildcard discovery enables:
- Comprehensive strategy testing without manual listing
- Tag-based strategy grouping
- Automatic parameter space exploration
- Efficient optimization workflows

This feature is essential for:
- Strategy development and validation
- Finding optimal parameter combinations
- Discovering effective strategy ensembles
- Systematic backtesting