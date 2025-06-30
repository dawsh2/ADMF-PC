# Keltner Bands 5-Minute Strategy Configuration and Analysis

## Overview
Based on our 1-minute analysis showing unprofitable results, we're moving to 5-minute timeframe which should provide:
- Less noise and false signals
- Better defined support/resistance levels
- More reliable mean reversion behavior
- Lower relative transaction costs

## Configuration File
Save as `config/indicators/volatility/test_keltner_bands_5m_optimized.yaml`:

```yaml
# Keltner Bands optimization for 5-minute data
# Tests 5x5 = 25 parameter combinations

dataset: SPY_5m
mode: signal_generation

# Parameter space for optimization (5x5 = 25 combinations)
parameter_space:
  strategies:
    - type: keltner_bands
      param_overrides:
        period: [10, 15, 20, 25, 30]
        multiplier: [1.0, 1.5, 2.0, 2.5, 3.0]

# Market configuration
market_hours_only: true
enable_metrics: true

# Data configuration
data:
  start_date: "2024-01-01"
  end_date: "2024-12-31"

# Workflow configuration
workflow_type: signal_generation
output_format: parquet
```

## Running the Analysis

### Step 1: Generate Signals
```bash
python main.py --config config/indicators/volatility/test_keltner_bands_5m_optimized.yaml --signal-generation --dataset SPY_5m
```

### Step 2: Note the Workspace Path
The output will show something like:
```
Workspace created: workspaces/signal_generation_XXXXXXXX
```

### Step 3: Analyze Results
```bash
python analyze_keltner_5m.py workspaces/signal_generation_XXXXXXXX
```

## Expected Performance on 5-Minute Data

### Parameter Impact
Based on typical mean reversion characteristics:

| Period | Multiplier | Expected Trades/Day | Gross Edge (bps) | Net @ 2bp | Annual Return |
|--------|------------|-------------------|------------------|-----------|---------------|
| 10     | 1.0        | 15.0              | 1.0              | -1.0      | -3.8%         |
| 10     | 1.5        | 7.9               | 0.7              | -1.3      | -2.6%         |
| 15     | 1.0        | 10.0              | 0.7              | -1.3      | -3.3%         |
| 15     | 1.5        | 5.3               | 0.5              | -1.5      | -2.0%         |
| 20     | 1.0        | 7.5               | 0.5              | -1.5      | -2.8%         |
| 20     | 1.5        | 4.0               | 0.3              | -1.7      | -1.7%         |
| 20     | 2.0        | 2.7               | 0.3              | -1.7      | -1.2%         |

### Best Expected Configurations
1. **Tighter Bands (1.0-1.5x)**: More trades but need larger base edge
2. **Shorter Periods (10-15)**: More responsive to price movements
3. **Sweet Spot**: Period=10-15, Multiplier=1.0-1.5

### Profitability Threshold
- Need ~0.5+ bps base edge for potential profitability
- 5m typically shows 2-5x better edge than 1m
- Still challenging but possible with right parameters

## Analysis Script Features

The `analyze_keltner_5m.py` script will:
1. Load all 25 parameter combinations
2. Calculate actual returns for each trade
3. Apply different cost scenarios (2bp, 4bp, 6bp)
4. Find optimal parameter combinations
5. Generate detailed performance reports
6. Save results to CSV files

## Filter Recommendations for 5-Minute

Based on 1-minute findings, test these filters on 5m:
1. **Time filters**: First/last hour of trading
2. **Volatility filters**: Trade only in high volatility periods
3. **Volume filters**: Require volume > 1.5x average
4. **Trend filters**: Trade with or against trend
5. **VWAP filters**: Only trade when price below VWAP

## Next Steps

1. **Run the optimization** to find best parameters
2. **Apply filters** if base returns are marginal
3. **Test on 15-minute** if 5m still unprofitable
4. **Consider other strategies**:
   - Bollinger Bands (similar but different calculation)
   - RSI mean reversion
   - VWAP mean reversion
   - Opening range breakout

## Alternative Approaches

If Keltner Bands remain unprofitable on 5m:
1. **Momentum strategies**: Often have larger edges
2. **Breakout strategies**: Clear entry/exit points
3. **Pairs trading**: Market neutral approach
4. **Machine learning**: Combine multiple indicators

## Summary

Moving from 1-minute to 5-minute data should improve results by:
- Reducing noise by ~70%
- Improving signal quality
- Decreasing false signals
- Making 2bp costs more manageable

However, mean reversion remains challenging in efficient markets. Be prepared to test other strategies if Keltner Bands don't show profitability even on higher timeframes.