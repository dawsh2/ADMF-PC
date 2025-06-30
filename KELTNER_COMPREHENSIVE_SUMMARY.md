# Keltner Bands Strategy Comprehensive Analysis Summary

## Key Findings

### 1. Filter System Implementation
- Successfully added support for `volume_ratio()`, `volatility_percentile()`, and `vwap_distance()` functions to the ConfigSignalFilter
- Fixed the critical issue where filters were blocking exit signals by using `signal == 0 or ...` pattern
- Resolved feature configuration override issue in topology.py where manual features were being replaced

### 2. 1-Minute Keltner Bands Performance

#### Best Performers (by avg bps before costs):
1. **RSI + Volume Filter** (kb_rsi_volume): 0.21 bps, 6.44 trades/day
2. **RSI < 70 Filter** (kb_rsi_entry): 0.18 bps, 10.05 trades/day  
3. **Fast Period (10)** (kb_fast_10): 0.10 bps, 3.71 trades/day
4. **Volume Filter** (kb_volume): 0.09 bps, 7.59 trades/day

**None achieved profitability after 1bp transaction costs**

#### Key Insights:
- Filters improve per-trade edge but reduce frequency
- Tighter bands (1.5x) increase frequency but reduce edge
- Wider bands (2.5x) reduce both frequency and edge
- Faster period (10) maintains edge with lower frequency
- Slower period (50) increases frequency but reduces edge

### 3. Trade Duration Analysis

The filtered strategies showed much longer average durations because the original filter `"signal != 0 and rsi(14) < 70"` was blocking ALL exit signals. After fixing with `"signal == 0 or rsi(14) < 70"`, durations normalized.

### 4. Volatility and VWAP Filters

- **Volatility percentile > 60%**: Too restrictive, generated only 1 signal
- **VWAP distance < 0.2%**: No filtering effect (identical to baseline)
- These filters need parameter tuning for effectiveness

### 5. Configuration Examples

#### Best performing filter combination:
```yaml
- name: kb_rsi_volume
  type: keltner_bands
  params:
    period: 20
    multiplier: 2.0
  filter: "signal == 0 or (rsi(14) < 70 and volume_ratio(20) > 1.1)"
```

#### Directional RSI filter:
```yaml
- name: kb_directional
  type: keltner_bands
  params:
    period: 20
    multiplier: 2.0
  filter: "signal == 0 or (signal > 0 and rsi(14) < 40) or (signal < 0 and rsi(14) > 60)"
```

## Recommendations

1. **Stop Loss Implementation**: The earlier analysis showed 0.3% stop loss improved edge significantly (0.05 → 0.17 bps). This needs to be implemented in the risk module.

2. **5-Minute Timeframe**: Previous analysis showed 2.18 bps on 5-minute data. Need to properly test with `timeframe: "5m"`.

3. **Filter Optimization**:
   - Test RSI thresholds: 30, 40, 50, 60
   - Test volume ratios: 1.0, 1.2, 1.5, 2.0
   - Test volatility percentiles: 0.3, 0.5, 0.7, 0.9

4. **Exit Timing**: Consider time-based exits or profit targets to reduce average duration while maintaining edge.

5. **Market Regime Filters**: Add trend filters (e.g., price above/below 200 SMA) to trade only in favorable conditions.

## Parameter Spaces for Optimization

### Keltner Bands Core Parameters
- **Period**: [10, 15, 20, 30, 50]
- **Multiplier**: [1.0, 1.5, 2.0, 2.5, 3.0]

### Filter Parameters
- **RSI Threshold**: [30, 40, 50, 60, 70]
- **Volume Ratio**: [1.0, 1.1, 1.2, 1.5, 2.0]
- **Volatility Percentile**: [0.3, 0.5, 0.7, 0.9]
- **VWAP Distance**: [0.001, 0.002, 0.003, 0.005]

### Directional RSI Parameters
- **Long Entry RSI**: [30, 35, 40, 45]
- **Short Entry RSI**: [55, 60, 65, 70]

### Combined Filter Logic
- RSI < X AND Volume > Y
- RSI < X AND Volatility > Y
- (Long: RSI < X) OR (Short: RSI > Y)
- Trend filters: Price > SMA(200) for longs, Price < SMA(200) for shorts

## Configuration Files Created

1. **test_keltner_comprehensive.yaml** - Tests multiple filter types on 1-minute data
2. **test_keltner_5min_proper.yaml** - Proper 5-minute configuration with `timeframe: "5m"`
3. **test_keltner_parameter_sweep.yaml** - Template for parameter optimization (requires manual expansion)
4. **optimize_keltner_with_filters.yaml** - Optimization config with proper parameter_space syntax
5. **optimize_keltner_5m.yaml** - 5-minute optimization configuration

## Optimization Usage

Run optimization with the `--optimize` flag:
```bash
# 1-minute optimization with filters
python3 main.py --config config/optimize_keltner_with_filters.yaml --signal-generation --optimize

# 5-minute optimization
python3 main.py --config config/optimize_keltner_5m.yaml --signal-generation --optimize
```

The system will automatically:
- Expand all parameter combinations from `parameter_space`
- Infer required features from filter expressions
- Generate unique strategy names for each combination
- Save results to a timestamped workspace

### Important: Dataset Configuration

When using `dataset: train` or `dataset: test`, you MUST also specify `split_ratio`:
```yaml
dataset: train
split_ratio: 0.8  # 80% for train, 20% for test
```

Without `split_ratio`, the data handler will receive `None` and fail to load data properly.

## Next Steps

1. Run the 5-minute configuration: `python3 main.py --config config/test_keltner_5min_proper.yaml --signal-generation --dataset train`
2. Implement stop loss logic in risk module
3. Create scripts to systematically test parameter combinations
4. Test combination strategies with multiple indicators
5. Analyze performance across different market conditions (bull/bear/sideways)

## Technical Improvements Made

1. **ConfigSignalFilter Enhancement** (src/strategy/components/config_filter.py):
   - Added `volume_ratio()`, `volatility_percentile()`, `vwap_distance()` accessor functions
   - Updated allowed functions list in AST validation

2. **Feature Configuration Fix** (src/core/coordinator/topology.py:966-982):
   - Changed from overwriting to merging manual features with discovered features
   - Preserves user-defined features needed for filters

3. **Filter Expression Pattern**:
   - Always use `signal == 0 or ...` to allow exit signals
   - This prevents positions from being stuck open indefinitely