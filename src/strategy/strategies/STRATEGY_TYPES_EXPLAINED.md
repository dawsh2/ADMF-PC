# Strategy Types: Momentum vs Trend Following vs Breakout

## Core Definitions

### 1. **Momentum Strategies**
**Philosophy**: "What has been going up will continue going up (in the short term)"
- **Timeframe**: Short to medium-term (minutes to days)
- **Entry**: When price/returns show strong directional movement
- **Exit**: When momentum weakens or reverses
- **Key Indicators**: Rate of Change (ROC), RSI momentum, price acceleration

**Examples**:
```python
# Price momentum: Recent returns predict near-term returns
if price_change_5_bars > threshold:
    signal = 1  # Momentum is positive
    
# Dual momentum: Compare two momentum measures
if short_term_momentum > long_term_momentum:
    signal = 1  # Relative momentum is positive
```

### 2. **Trend Following Strategies**
**Philosophy**: "The trend is your friend until it ends"
- **Timeframe**: Medium to long-term (days to months)
- **Entry**: When a trend is established and confirmed
- **Exit**: When trend shows signs of reversal
- **Key Indicators**: Moving averages, ADX, trend lines

**Examples**:
```python
# Classic trend following: Price above multiple MAs
if price > ma_20 and ma_20 > ma_50 and ma_50 > ma_200:
    signal = 1  # Uptrend confirmed across timeframes
    
# ADX trend strength: Only trade strong trends
if ma_fast > ma_slow and ADX > 25:
    signal = 1  # Trend + strength confirmation
```

### 3. **Breakout Strategies**
**Philosophy**: "New highs/lows signal continuation"
- **Timeframe**: Variable (depends on lookback period)
- **Entry**: When price breaks key levels
- **Exit**: When price fails to continue or reverses
- **Key Indicators**: Donchian channels, support/resistance, volume

**Examples**:
```python
# Channel breakout: Price exceeds recent range
if price > highest_high_20_bars:
    signal = 1  # Breaking to new highs
    
# Volume-confirmed breakout
if price > resistance and volume > volume_ma * 1.5:
    signal = 1  # Breakout with volume confirmation
```

## Key Differences

### 1. **Time Horizon**
- **Momentum**: Shortest (exploits short-term price continuation)
- **Breakout**: Event-driven (can be any timeframe)
- **Trend Following**: Longest (rides established trends)

### 2. **Entry Timing**
- **Momentum**: Enters during acceleration
- **Breakout**: Enters at specific price levels
- **Trend Following**: Enters after trend confirmation

### 3. **Risk Profile**
- **Momentum**: Higher turnover, more whipsaws
- **Breakout**: False breakout risk
- **Trend Following**: Larger drawdowns, but bigger wins

### 4. **Market Conditions**
- **Momentum**: Works in volatile, directional markets
- **Breakout**: Works when ranges resolve into trends
- **Trend Following**: Works in sustained trending markets

## Overlap Analysis

### Momentum vs Trend Following
```
Momentum ──────────┐
                   ├─── Both use directional price movement
Trend Following ───┘
                   
Differences:
- Momentum: "Price is moving fast NOW"
- Trend: "Price has BEEN moving consistently"
```

### Breakout vs Trend Following
```
Breakout ──────────┐
                   ├─── Both can initiate trend trades
Trend Following ───┘

Differences:
- Breakout: "Price just exceeded a key level"
- Trend: "Multiple confirmations of direction"
```

### All Three Can Overlap
A single move can trigger all three:
1. Price breaks resistance (Breakout)
2. The break shows strong momentum (Momentum)
3. This establishes a new trend (Trend Following)

## Practical Implementation

### Pure Examples (No Overlap)

**Pure Momentum**:
```python
# RSI momentum oscillator
if RSI < 30:  # Oversold momentum
    signal = 1
elif RSI > 70:  # Overbought momentum
    signal = -1
```

**Pure Trend Following**:
```python
# Multiple MA confirmation
if ma_10 > ma_20 > ma_50 > ma_200:
    signal = 1  # All timeframes aligned
```

**Pure Breakout**:
```python
# Simple range breakout
if price > last_week_high:
    signal = 1
elif price < last_week_low:
    signal = -1
```

### Hybrid Examples (With Overlap)

**Momentum + Trend**:
```python
# Moving average crossover (Rules 1-9)
if fast_ma > slow_ma:  # Trend direction
    signal = 1
# This is BOTH momentum (fast MA represents recent momentum)
# AND trend following (crossover confirms trend change)
```

**Breakout + Trend**:
```python
# Donchian channel with trend filter
if price > donchian_upper and ma_50 > ma_200:
    signal = 1
# Breakout signal with trend confirmation
```

## Regarding Rules vs Indicators Organization

You're right about the overlap! Here's how to think about it:

### Current Organization Issues:
1. **Rules 1-9 (Crossovers)**: These are actually trend/momentum hybrids
2. **Rules 14-16 (Channels)**: These are breakout strategies
3. **Indicator-based strategies**: May duplicate rule implementations

### Better Conceptual Organization:

```
strategies/
├── momentum/
│   ├── price_momentum.py      # Pure price rate-of-change
│   ├── oscillator_momentum.py # RSI, Stochastic momentum
│   └── relative_momentum.py   # Dual momentum, relative strength
│
├── trend_following/
│   ├── moving_averages.py     # MA-based trend strategies
│   ├── multi_timeframe.py     # Multiple timeframe alignment
│   └── adaptive_trend.py      # ADX, trend strength filters
│
├── breakout/
│   ├── channel_breakout.py    # Donchian, Keltner breaks
│   ├── volatility_breakout.py # Bollinger, ATR-based
│   └── volume_breakout.py     # Volume-confirmed breaks
│
├── mean_reversion/
│   ├── band_reversion.py      # Bollinger band snaps
│   ├── oscillator_reversion.py # RSI/CCI extremes
│   └── statistical_reversion.py # Z-score, cointegration
│
└── hybrid/
    ├── trend_momentum.py      # MA crossovers (trend + momentum)
    ├── breakout_momentum.py   # Breakouts with momentum confirm
    └── adaptive_strategies.py # Regime-switching strategies
```

### Key Insight:
Many strategies are naturally hybrids. Instead of forcing them into one category:
1. **Acknowledge the overlap** in documentation
2. **Place in primary category** based on main signal logic
3. **Use metadata** to tag secondary characteristics

Example metadata:
```python
@strategy(
    name='ma_crossover',
    primary_type='trend_following',
    secondary_types=['momentum'],
    timeframe='medium',
    feature_config={...}
)
```

This would make the codebase more honest about what strategies actually do while maintaining clean organization.