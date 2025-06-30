# Clean Config Syntax Guide

The clean config syntax provides a more concise and readable way to define trading strategies and their parameter sweeps.

## Basic Structure

```yaml
name: my_strategy_config
symbols: ["SPY"]
timeframe: "5m"

strategy:
  - strategy_name:
      param1: value_or_list
      param2: value_or_list
      filter: filter_spec
```

## Key Differences from Original Syntax

### Before (Original)
```yaml
parameter_space:
  strategies:
    - type: keltner_bands
      param_overrides:
        period: [10, 20, 30]
        multiplier: [1.5, 2.0]
```

### After (Clean)
```yaml
strategy:
  - keltner_bands:
      period: [10, 20, 30]
      multiplier: [1.5, 2.0]
```

## Filter Syntax

Filters are defined as part of the strategy configuration. The `filter` parameter accepts various formats:

### 1. No Filter
```yaml
filter: null
```

### 2. Single Filter
```yaml
filter: {rsi_below: {threshold: 50}}
```

### 3. Multiple Filters (Array = AND logic)
```yaml
filter: [
  {rsi_below: {threshold: 40}},
  {volume_above: {multiplier: 1.2}}
]
```

### 4. Directional Filters
```yaml
filter: {
  long: {rsi_below: {threshold: 30}},
  short: {rsi_above: {threshold: 70}}
}
```

### 5. Multiple Filter Variants
To test different filter combinations, provide an array:
```yaml
filter: [
  null,  # No filter
  {rsi_below: {threshold: 50}},  # Single RSI filter
  {volume_above: {multiplier: 1.2}},  # Single volume filter
  [  # Combined filters
    {rsi_below: {threshold: 40}},
    {volume_above: {multiplier: 1.1}}
  ]
]
```

## Built-in Filter Types

### Momentum/Oscillator Filters
- `rsi_below: {threshold: value}` - RSI less than threshold
- `rsi_above: {threshold: value}` - RSI greater than threshold

### Volume Filters
- `volume_above: {multiplier: value}` - Volume above SMA(20) * multiplier
- `volume_below: {multiplier: value}` - Volume below SMA(20) * multiplier

### Volatility Filters
- `volatility_above: {threshold: value}` - ATR(14) > ATR_SMA(50) * threshold
- `volatility_below: {threshold: value}` - ATR(14) < ATR_SMA(50) * threshold

### Price/VWAP Filters
- `price_below_vwap: {buffer: value}` - Close < VWAP * (1 - buffer)
- `price_above_vwap: {buffer: value}` - Close > VWAP * (1 + buffer)
- `price_distance_vwap: {min: value}` - |Close - VWAP| / VWAP > min

### Time Filters
- `time_exclude: {start: "HH:MM", end: "HH:MM"}` - Exclude time range
- `time_include: {start: "HH:MM", end: "HH:MM"}` - Only trade in time range

## Parameter Sweeps

Any parameter can be a list to create a parameter sweep:

```yaml
strategy:
  - keltner_bands:
      period: [10, 15, 20, 30, 50]  # Tests 5 values
      multiplier: [1.0, 1.5, 2.0, 2.5, 3.0]  # Tests 5 values
      # Creates 25 combinations (5 × 5)
```

Filter parameters also support sweeps:

```yaml
filter: {rsi_below: {threshold: [30, 40, 50, 60]}}
# Tests 4 different RSI thresholds
```

## Complete Example

```yaml
name: optimize_keltner_clean
symbols: ["SPY", "QQQ"]
timeframe: "5m"

strategy:
  - keltner_bands:
      period: [10, 20, 30]
      multiplier: [1.5, 2.0]
      filter: [
        null,  # Baseline: no filter
        
        # Single filters
        {rsi_below: {threshold: [40, 50]}},
        {volume_above: {multiplier: 1.2}},
        
        # Directional filter
        {
          long: {rsi_below: {threshold: 30}},
          short: {rsi_above: {threshold: 70}}
        },
        
        # Combined filters
        [
          {volatility_above: {threshold: 1.1}},
          {time_exclude: {start: "12:00", end: "14:30"}},
          {
            long: {price_below_vwap: {}},
            short: {price_above_vwap: {}}
          }
        ],
        
        # Long-only variant
        {
          long: {price_below_vwap: {buffer: 0.001}},
          short: false  # Disable short signals
        }
      ]

# Standard optimization settings
optimization:
  granularity: 10
  
execution:
  trace_settings:
    storage:
      base_dir: ./configs
```

This example creates:
- 2 symbols × 3 periods × 2 multipliers = 12 baseline combinations
- Plus various filtered versions for comprehensive testing

## Migration from Original Syntax

To convert existing configs:

1. Replace `parameter_space.strategies` with `strategy`
2. Remove `type:` and use strategy name as key
3. Remove `param_overrides:` and put parameters directly
4. Remove `filter_params:` - filter parameters are now inline
5. Remove `signal == 0 or` from filters (automatically added)

The clean syntax parser automatically converts to the internal format, so existing code continues to work unchanged.