# Classifier Regime State Distribution Analysis

## Overview
Analyzed regime state distributions for all classifiers in `/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_7fca9dff/`

**Total bars analyzed:** 102,236 (SPY 1-minute data)

## Summary of Findings

### 1. All Classifiers Are Heavily Skewed

Every classifier shows extreme skewness in regime state distributions:

| Classifier Type | Dominant State | Dominant % | Balance Score | Changes/1000 bars |
|-----------------|----------------|------------|---------------|-------------------|
| **Multi Timeframe Trend Grid** | sideways | 99.9% | 43.42 | 0.19 |
| **Microstructure Grid** | consolidation | 97.7% | 37.14 | 13.98 |
| **Hidden Markov Grid** | uncertainty | 89.7% | 34.92 | 27.47 |
| **Volatility Momentum Grid** | low_vol_bearish | 55.1% | 23.31 | 81.67 |
| **Market Regime Grid** | bear_ranging | 46.6% | 17.03 | 64.62 |

### 2. Classifier-Specific Observations

#### Hidden Markov Grid (4 states)
- **States:** uncertainty, accumulation, markup, markdown
- **Distribution:** ~90% uncertainty, ~7% accumulation, ~1.5% each for markup/markdown
- **Activity:** Moderate (27 changes per 1000 bars)

#### Market Regime Grid (3 states)
- **States:** bull_ranging, bear_ranging, neutral
- **Distribution:** 46.6% bear_ranging, 32.5% bull_ranging, 20.8% neutral
- **Activity:** High (65 changes per 1000 bars)
- **Note:** Most balanced of all classifiers

#### Microstructure Grid (5 states)
- **States:** consolidation, reversal_up, reversal_down, breakout_up, breakout_down
- **Distribution:** 97.7% consolidation, ~1% each for reversals, <0.1% for breakouts
- **Activity:** Low (14 changes per 1000 bars)

#### Multi Timeframe Trend Grid (4 states)
- **States:** sideways, weak_uptrend, weak_downtrend, strong_downtrend
- **Distribution:** 99.9% sideways, <0.1% for all other states
- **Activity:** Extremely low (0.19 changes per 1000 bars)
- **Note:** Essentially non-functional - almost never leaves sideways state

#### Volatility Momentum Grid (3 states)
- **States:** low_vol_bearish, low_vol_bullish, neutral
- **Distribution:** 55.1% low_vol_bearish, 40.2% low_vol_bullish, 4.7% neutral
- **Activity:** Very high (82 changes per 1000 bars)

### 3. Key Insights

1. **Extreme Skewness:** All classifiers spend the vast majority of time in a single "default" state
2. **Limited Functionality:** Some classifiers (especially multi_timeframe_trend) are effectively non-functional
3. **Sparse Storage Efficiency:** The sparse storage format is appropriate given low regime change frequency
4. **Retuning Needed:** Classifiers likely need parameter adjustment for this specific dataset/timeframe
5. **Most Active:** Volatility momentum grid (82 changes/1000 bars)
6. **Least Active:** Multi timeframe trend grid (0.19 changes/1000 bars)

### 4. Recommendations

1. **Parameter Tuning:** All classifiers need significant parameter adjustment to achieve better state balance
2. **Threshold Review:** Many classifiers may have thresholds set too high for 1-minute data
3. **State Definition:** Some states (like "strong trends" in 1-minute data) may be too rare to be useful
4. **Consider Removal:** Multi timeframe trend grid provides almost no value in current configuration

## Technical Details

- **Sparse Storage Format:** Classifiers only store regime changes, not full state for every bar
- **Reconstruction Method:** Full states reconstructed by applying changes sequentially from initial state
- **Data Structure:** Each entry contains: bar index, timestamp, symbol, regime value, strategy ID, price