# Bollinger Strategy Filter Features

## Available Features for Your Threshold/Filter

### 1. Volume Features ✅
- **`volume_sma_20`** - 20-period simple moving average of volume
- Usage: `volume > volume_sma_20 * 1.3`

### 2. Volatility Features ✅
- **`volatility_percentile_50`** - Percentile rank of current volatility (0-100)
- Usage: `volatility_percentile_50 > 40`
- Note: Just created this feature! It ranks current ATR against last 50 bars

### 3. Trend/Slope Features ✅
- **`linreg_20_slope`** - Slope of 20-period linear regression
- Usage: `abs(linreg_20_slope) < 0.15`
- Note: This is what I used as "slope" in the analysis

### 4. Oscillator Features ✅
- **`rsi_14`** - 14-period Relative Strength Index
- Usage: `(rsi_14 < 35 or rsi_14 > 65)`

## Your Updated Config

```yaml
strategy:
  bollinger_bands:
    - period: 20
      multiplier: 2.5
      threshold: |
        volume > volume_sma_20 * 1.3 or
        volatility_percentile_50 > 40 or
        abs(linreg_20_slope) < 0.15 or
        (rsi_14 < 35 or rsi_14 > 65)
```

## How Features Are Configured

If you need specific periods, configure them in your config:

```yaml
features:
  # Volume SMA with custom period
  volume_sma_20:
    type: volume_sma
    period: 20
    
  # Volatility percentile with custom window
  volatility_percentile_50:
    type: volatility_percentile
    period: 50      # Window for percentile ranking
    atr_period: 14  # ATR calculation period
    
  # Linear regression for slope
  linreg_20:
    type: linreg
    period: 20
    # This creates: linreg_20_slope, linreg_20_intercept, linreg_20_r2
    
  # RSI
  rsi_14:
    type: rsi
    period: 14
```

## Expected Performance

Based on the analysis, this OR filter combination should deliver:
- **Return per trade**: 2.3-2.6 basis points
- **Trade retention**: ~60% of signals
- **Win rate improvement**: From 45% to 55-57%

## Notes

1. **Threshold vs Filter**: Your `threshold` field now works exactly like `filter` but with cleaner naming
2. **Feature Discovery**: All features in the threshold expression will be automatically discovered and calculated
3. **Signal Check**: The threshold is automatically wrapped with `signal != 0 and (...)` to ensure we only filter actual signals

The volatility percentile feature I created matches what I used in the analysis:
- Values 0-100 representing percentile rank
- Based on ATR (Average True Range) for robustness
- 50-bar lookback window by default
- Values > 40 mean "above 40th percentile of recent volatility"