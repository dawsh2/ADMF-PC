# Wildcard Strategy Discovery Guide

## Listing Available Strategies

Simple approach using grep:
```bash
# List all strategies
python list_strategies.py

# Filter for mean reversion strategies
python list_strategies.py | grep -i mean
python list_strategies.py | grep -i reversion
python list_strategies.py | grep -i bands

# Filter by category
python list_strategies.py | grep -A10 "OSCILLATOR:"
python list_strategies.py | grep -A10 "VOLATILITY:"
python list_strategies.py | grep -A10 "STRUCTURE:"
```

Or use the built-in CLI option:
```bash
python main.py --list-strategies
python main.py --list-strategies --strategy-filter oscillator
python main.py --list-strategies --verbose  # Shows parameters
```

## Using Wildcards in Configs

### 1. Discover ALL Strategies
```yaml
# Discover all indicator strategies
parameter_space:
  indicators: "*"
  
optimization:
  granularity: 3  # Parameter samples per strategy
```

### 2. Discover by Category
```yaml
# Mean reversion strategies
parameter_space:
  oscillator: "*"    # RSI, CCI, Williams %R, etc.
  volatility: "*"    # Bollinger, Keltner, Donchian
  structure: "*"     # Pivots, trendlines, channels
  
optimization:
  granularity: 3
```

### 3. Multiple Specific Categories
```yaml
# Mix of categories
parameter_space:
  oscillator: ["rsi_bands", "cci_bands"]
  volatility: ["bollinger_bands", "keltner_bands"]
  
optimization:
  granularity: 5
```

### 4. Single Category Wildcard
```yaml
# Just volatility strategies
parameter_space:
  volatility: "*"
  
optimization:
  granularity: 5
```

## Running with Wildcards

### Mean Reversion Research Example
```bash
# Run all mean reversion strategies with optimization
python main.py --signal-generation \
  --config config/mean_reversion_research/config.yaml \
  --optimize \
  --bars 500
```

This will:
1. Discover all indicator strategies (using `indicators: "*"`)
2. Filter for mean reversion types (based on category patterns)
3. Generate parameter combinations (3 samples per parameter)
4. Run signal generation for each combination
5. Store results in `configs/mean_reversion_research/results/<timestamp>/`

### Category Patterns
The system recognizes these patterns when using wildcards:

- **momentum**: momentum, rsi, macd, roc
- **oscillator**: rsi, cci, stochastic, williams
- **trend**: adx, aroon, supertrend, sar
- **volatility**: atr, bollinger, keltner
- **volume**: obv, vwap, chaikin, mfi
- **structure**: pivot, support, trendline

## Example Strategies by Category

### OSCILLATOR (Mean Reversion)
- rsi_bands
- rsi_threshold
- cci_bands
- cci_threshold
- williams_r
- stochastic_rsi
- ultimate_oscillator

### VOLATILITY (Mean Reversion)
- bollinger_bands
- bollinger_breakout
- keltner_bands
- keltner_breakout
- donchian_bands
- donchian_breakout

### STRUCTURE (Mean Reversion)
- pivot_points
- pivot_bounces
- swing_pivot_bounce
- support_resistance_breakout
- trendline_bounces
- trendline_breaks
- diagonal_channel_reversion

### VOLUME (Mean Reversion)
- vwap_deviation
- mfi_bands
- chaikin_money_flow

## Results Organization

When using wildcards, results are organized by strategy:
```
configs/mean_reversion_research/results/20241220_143022/
├── metadata.json
└── traces/
    ├── bollinger_bands/
    │   ├── period_10_std_1.5.parquet
    │   ├── period_20_std_2.0.parquet
    │   └── period_30_std_2.5.parquet
    ├── keltner_bands/
    │   ├── period_10_multiplier_1.5.parquet
    │   └── period_20_multiplier_2.0.parquet
    └── rsi_bands/
        ├── period_7_oversold_20_overbought_80.parquet
        └── period_14_oversold_30_overbought_70.parquet
```

## Tips

1. **Start Small**: Use `granularity: 2` for initial tests
2. **Filter by Category**: Use specific categories to focus research
3. **Check Discovery**: Use `--list-strategies` to verify what will be discovered
4. **Monitor Progress**: Strategies are logged as they're discovered and run
5. **Analyze Results**: Each parquet file contains sparse signal data for one parameter combination