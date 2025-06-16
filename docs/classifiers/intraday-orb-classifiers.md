# Intraday ORB Classifiers Documentation

## Overview

The intraday ORB (Opening Range Breakout) classifiers are specialized market regime detection components designed for 1-minute timeframe trading. They detect various intraday patterns including opening range breakouts, session effects, and microstructure patterns.

## Classifiers

### 1. Intraday ORB Classifier

**Name**: `intraday_orb_classifier`

**Regime Types**:
- `orb_breakout_up`: Price breaks above opening range with volume confirmation
- `orb_breakout_down`: Price breaks below opening range with volume confirmation
- `orb_range_bound`: Price trading within opening range
- `session_open_vol`: Opening session high volatility period
- `midday_drift`: Low volatility midday consolidation
- `close_volatility`: End-of-session volatility spike

**Required Features**:
- `high`, `low`, `close`, `volume`
- `sma_5`: 5-period simple moving average
- `atr_10`: 10-period average true range

**Parameters**:
- `orb_minutes`: Opening range period (default: 30)
- `open_session_minutes`: High volatility period after open (default: 90)
- `close_session_minutes`: High volatility period before close (default: 60)
- `orb_breakout_threshold`: Breakout threshold percentage (default: 0.002)
- `volume_surge_threshold`: Volume surge multiplier (default: 1.5)
- `volatility_threshold`: Volatility threshold multiplier (default: 1.5)

### 2. Microstructure Momentum Classifier

**Name**: `microstructure_momentum_classifier`

**Regime Types**:
- `momentum_acceleration`: Price moving with increasing momentum
- `momentum_deceleration`: Price momentum slowing down
- `volume_breakout`: Sudden volume spike with price movement
- `liquidity_void`: Low volume, potential for quick moves
- `normal_flow`: Regular trading flow

**Required Features**:
- `close`, `volume`
- `sma_3`: 3-period simple moving average
- `sma_5`: 5-period simple moving average
- `atr_5`: 5-period average true range

**Parameters**:
- `momentum_accel_threshold`: Momentum acceleration threshold (default: 0.001)
- `volume_spike_threshold`: Volume spike multiplier (default: 2.0)
- `liquidity_threshold`: Low volume threshold (default: 0.5)

### 3. Session Pattern Classifier

**Name**: `session_pattern_classifier`

**Regime Types**:
- `gap_up`: Significant gap up from previous close
- `gap_down`: Significant gap down from previous close
- `opening_auction`: High volume opening phase
- `trending_session`: Sustained directional movement
- `consolidation_session`: Range-bound session

**Required Features**:
- `open`, `high`, `low`, `close`, `volume`

**Parameters**:
- `gap_threshold`: Gap threshold percentage (default: 0.005)
- `trend_session_threshold`: Trending session threshold (default: 0.01)
- `opening_volume_threshold`: Opening volume multiplier (default: 2.0)

## Usage Example

```yaml
classifiers:
  - name: orb_detector
    type: intraday_orb_classifier
    params:
      orb_minutes: 30
      orb_breakout_threshold: 0.0025  # 0.25%
      volume_surge_threshold: 2.0
    
  - name: momentum_detector
    type: microstructure_momentum_classifier
    params:
      momentum_accel_threshold: 0.0015  # 0.15%
      volume_spike_threshold: 2.5
    
  - name: session_detector
    type: session_pattern_classifier
    params:
      gap_threshold: 0.0075  # 0.75%
      trend_session_threshold: 0.015  # 1.5%
```

## Implementation Notes

1. **Time Handling**: The classifiers use timestamp information from the bar data to determine session timing. Market open/close times may need adjustment based on your market.

2. **ORB Levels**: The ORB levels (opening range high/low) should be calculated from the first N minutes of the trading session and passed as features.

3. **Volume Averages**: Volume comparison requires historical average volume data to be meaningful.

4. **Feature Requirements**: All classifiers follow the standard interface and use the centralized FeatureHub for feature computation.

## Integration

These classifiers integrate seamlessly with the ADMF-PC architecture:

1. **Discovery**: Decorated with `@classifier` for automatic discovery
2. **Feature Config**: Declares required features for automatic inference
3. **Stateless**: Pure functions that receive features and return regime classifications
4. **Standard Output**: Returns regime type, confidence, and metadata

## Performance Considerations

- Designed for 1-minute timeframes but can adapt to other intraday timeframes
- Lightweight calculations suitable for real-time processing
- O(1) complexity for all regime determinations
- No state management required (handled by FeatureHub)