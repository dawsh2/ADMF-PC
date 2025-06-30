# Wildcard Strategy Selection with Exclude Functionality

## Overview

The parameter expander now supports excluding specific strategies when using wildcard selection. This allows you to select "everything except xyz" as requested.

## Usage

### Basic Wildcard (all strategies in a category)
```yaml
parameter_space:
  indicators:
    mean_reversion: "*"  # All mean reversion strategies
```

### Wildcard with Exclusions
```yaml
parameter_space:
  indicators:
    mean_reversion: "*"
    exclude: ["diagonal_channel_reversion"]  # All except this one
```

### Multiple Exclusions
```yaml
parameter_space:
  indicators:
    mean_reversion: "*"
    exclude: 
      - "diagonal_channel_reversion"
      - "trendline_bounces"
      - "swing_pivot_bounce"
```

### Mixed Categories with Exclusions
```yaml
parameter_space:
  indicators:
    mean_reversion: "*"      # All mean reversion
    trend_following: "*"     # All trend following
    exclude: ["diagonal_channel_reversion", "parabolic_sar"]
```

## How It Works

1. The parameter expander first collects all strategies matching the wildcard patterns
2. Then it filters out any strategies listed in the `exclude` array
3. The remaining strategies are expanded with their parameter spaces

## Examples

### Example 1: All Oscillators Except RSI
```yaml
parameter_space:
  indicators:
    oscillator: "*"
    exclude: ["rsi_bands", "stochastic_rsi"]
```

### Example 2: All Volume Strategies Except VWAP
```yaml
parameter_space:
  indicators:
    volume: "*"
    exclude: ["vwap_deviation", "session_vwap_reversion"]
```

### Example 3: All Strategies Except Specific Ones
```yaml
parameter_space:
  indicators: "*"  # Global wildcard
  exclude: 
    - "diagonal_channel_reversion"
    - "experimental_strategy"
    - "legacy_strategy"
```

## Benefits

1. **Cleaner Configs**: No need to manually list all strategies when you want most of them
2. **Easier Maintenance**: Add new strategies without updating configs
3. **Flexible Testing**: Quickly exclude problematic strategies during debugging
4. **Future-Proof**: New strategies automatically included unless explicitly excluded

## Implementation Details

The exclude functionality is implemented in `src/core/coordinator/config/parameter_expander.py` in the `_expand_indicators` method. It:

1. Checks for an 'exclude' key in the indicators specification
2. Treats it as a list of strategy names to exclude
3. Filters the expanded strategy list to remove excluded items
4. Logs the exclusions for transparency