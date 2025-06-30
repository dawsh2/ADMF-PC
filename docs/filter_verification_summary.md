# Filter Verification Guide

## How to Verify Filters are Working

### 1. Check Signal Counts

The most direct way to verify filters are working is to compare signal counts between strategies:

```python
# Run the verification script
python verify_filters.py
```

Expected output:
- **Baseline strategies** (0-24): Should have the most signals
- **Filtered strategies** (25+): Should have fewer signals due to filtering

### 2. Expected Filter Effects

Based on your config, here's what each filter should do:

#### Strategy Ranges:
- **0-24**: Baseline (no filter) - All Keltner band signals
- **25-60**: RSI filter - Only signals when RSI < threshold
- **61-96**: Volume filter - Only signals when volume > average
- **97-105**: Combined RSI+Volume - Both conditions must be true
- **106-121**: Directional RSI - Different thresholds for long/short
- **122-124**: Volatility filter - Only in high volatility
- **125-133**: VWAP positioning - Price relative to VWAP
- **134**: Time filter - Exclude lunch hours
- **135+**: Complex combinations

### 3. Verification Methods

#### Method 1: Signal Count Comparison
```python
# Compare baseline vs filtered
baseline_signals = 3262  # From strategy 0
rsi_filtered = 1500      # From strategy 25 (example)
reduction = (1 - rsi_filtered/baseline_signals) * 100
# Should see 40-60% reduction with filters
```

#### Method 2: Direct Signal Inspection
Look at specific bars where signals differ:
- Baseline has signal, filtered doesn't → Filter blocked it
- Both have same signal → Filter condition wasn't met

#### Method 3: Check Metadata
The metadata.json shows signal_changes for each strategy:
- Lower signal_changes = More restrictive filter
- signal_frequency shows what % of bars have signals

### 4. Common Filter Behaviors

**RSI Filters**:
- `rsi_below: 50` → Blocks ~50% of long signals
- `rsi_below: 30` → Blocks ~80% of long signals (more restrictive)

**Volume Filters**:
- `volume_above: 1.2` → Only high volume bars (20%+ above average)
- Typically reduces signals by 30-50%

**Time Filters**:
- `time_exclude: 12:00-14:30` → No signals during lunch
- Should see 0 signals during excluded hours

**Directional Filters**:
- Different rules for long vs short
- May completely eliminate one direction

### 5. Quick Verification Checklist

✅ **Signal Reduction**: Filtered strategies have fewer signals than baseline
✅ **Consistent Patterns**: Same base parameters show similar patterns
✅ **Zero Signals**: Some highly restrictive filters may produce no signals
✅ **Directional Bias**: Long-only filters should have no short signals

### 6. Troubleshooting

If all strategies have the same signal count:
1. Check that filter expressions were properly generated
2. Verify features (RSI, volume, etc.) are being calculated
3. Check for syntax errors in filter expressions

If filters seem too restrictive (very few signals):
1. This is often normal - filters are meant to be selective
2. Check filter thresholds aren't too extreme
3. Verify data has the required features (volume, ATR, etc.)

### 7. Manual Verification

To manually check a specific strategy:

```python
import pandas as pd

# Load two strategies to compare
baseline = pd.read_parquet("traces/keltner_bands/SPY_5m_compiled_strategy_0.parquet")
filtered = pd.read_parquet("traces/keltner_bands/SPY_5m_compiled_strategy_25.parquet")

# Find differences
diff = baseline['signal'] != filtered['signal']
print(f"Signals differ on {diff.sum()} bars out of {len(baseline)}")

# Check specific examples
different_bars = baseline[diff].head()
print(different_bars)
```

## Summary

Your clean syntax config generated 275 strategies with various filter combinations. The filters should be reducing signal counts by anywhere from 20% to 90% depending on how restrictive they are. Use the scripts provided to verify the exact impact of each filter type.