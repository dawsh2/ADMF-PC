# Composable Strategy Organization

## Core Concept: Strategies as Building Blocks

Each indicator-based strategy is an **element** that can be combined into composite strategies.

```
Composite Strategy
    ├── Crossover Signal (e.g., Rule 1: MA crossover)
    ├── Oscillator Filter (e.g., Rule 10: RSI threshold)
    └── Volatility Confirmation (e.g., Rule 16: Bollinger Bands)
```

## Proposed Organization

```
strategies/
├── __init__.py
├── README.md
│
├── indicators/                  # Atomic building blocks
│   ├── crossovers/             # Signal from crossing relationships
│   │   ├── ma_crossovers.py   # Rules 1-3: SMA, EMA crossovers
│   │   ├── advanced_ma.py     # Rules 4-6: DEMA, TEMA crossovers
│   │   ├── stochastic.py      # Rule 7: Stochastic %K/%D
│   │   ├── vortex.py          # Rule 8: Vortex VI+/VI-
│   │   └── ichimoku.py        # Rule 9: Cloud crossovers
│   │
│   ├── oscillators/            # Signal from bounded indicators
│   │   ├── rsi_signals.py     # Rules 10, 12: RSI threshold/bands
│   │   ├── cci_signals.py     # Rules 11, 13: CCI threshold/bands
│   │   ├── stochastic_levels.py # Stochastic overbought/oversold
│   │   └── williams_r.py      # Williams %R signals
│   │
│   └── volatility/             # Signal from volatility measures
│       ├── bollinger_bands.py # Rule 16: Band penetration
│       ├── keltner_channel.py # Rule 14: Channel breaks
│       ├── donchian_channel.py # Rule 15: High/low breaks
│       └── atr_based.py       # ATR expansion/contraction
│
├── ensemble/                    # Strategies combining multiple indicators
│   ├── trend_momentum.py       # Combines crossovers + oscillators
│   ├── filtered_breakout.py   # Breakout + volume + oscillator
│   ├── mean_reversion_suite.py # Bands + RSI + volume
│   └── adaptive_ensemble.py    # Regime-based strategy selection
│
├── core/                        # Standalone strategies (not indicator-based)
│   ├── arbitrage.py            # Statistical arbitrage
│   ├── market_making.py        # Bid-ask spread capture
│   └── null.py                 # Testing strategy
│
└── experimental/                # New ideas and testing
    └── ml_strategies.py        # ML-based approaches
```

## Example: Ensemble Strategy Implementation

```python
# ensemble/trend_momentum.py

@strategy(
    name='trend_momentum_ensemble',
    feature_config={
        'sma': {'params': ['fast_period', 'slow_period']},
        'rsi': {'params': ['rsi_period']},
        'atr': {'params': ['atr_period']}
    }
)
def trend_momentum_ensemble(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ensemble strategy combining:
    1. MA crossover for trend direction (Rule 1)
    2. RSI for momentum confirmation (Rule 10)
    3. ATR for volatility filter
    """
    # Get individual signals
    ma_signal = evaluate_ma_crossover(features, params)      # -1, 0, 1
    rsi_signal = evaluate_rsi_threshold(features, params)    # -1, 0, 1
    volatility_ok = evaluate_volatility_filter(features, params)  # True/False
    
    # Combine signals with logic
    if ma_signal == 0 or not volatility_ok:
        return 0  # No signal
    
    # Require agreement between trend and momentum
    if ma_signal == rsi_signal:
        signal_value = ma_signal  # Strong agreement
    elif ma_signal == 1 and rsi_signal == 0:
        signal_value = 1  # Trend without extreme RSI is OK
    elif ma_signal == -1 and rsi_signal == 0:
        signal_value = -1  # Trend without extreme RSI is OK  
    else:
        signal_value = 0  # Disagreement = no trade
    
    # Return combined signal
    return create_signal(
        symbol_timeframe=f"{bar['symbol']}_{bar['timeframe']}",
        signal_value=signal_value,
        metadata={'components': {
            'ma_crossover': ma_signal,
            'rsi_momentum': rsi_signal,
            'volatility_filter': volatility_ok
        }}
    )
```

## Benefits of This Organization

1. **Clear Building Blocks**: Each indicator strategy is atomic and reusable
2. **Natural Grouping**: Crossovers, oscillators, and volatility indicators have different signal characteristics
3. **Explicit Composition**: Composite strategies show exactly which elements they combine
4. **Easy Testing**: Can test each indicator strategy independently
5. **Flexible Combinations**: New composites can mix and match any indicators

## Signal Combination Patterns

### 1. **Confirmation Pattern**
All indicators must agree:
```python
if all(signal == 1 for signal in [ma_signal, rsi_signal, volume_signal]):
    return 1  # Strong buy
```

### 2. **Voting Pattern**
Majority rules:
```python
signals = [ma_signal, rsi_signal, stoch_signal]
signal_sum = sum(signals)
if signal_sum >= 2:
    return 1
elif signal_sum <= -2:
    return -1
else:
    return 0
```

### 3. **Primary + Filter Pattern**
One primary signal with filters:
```python
if ma_signal != 0:  # Primary signal
    if rsi_not_extreme and volatility_normal:  # Filters
        return ma_signal
return 0
```

### 4. **Weighted Pattern**
Different weights for different indicators:
```python
weighted_signal = (
    0.5 * ma_signal +      # Trend is most important
    0.3 * momentum_signal + # Momentum secondary
    0.2 * volume_signal     # Volume confirmation
)
return 1 if weighted_signal > 0.5 else (-1 if weighted_signal < -0.5 else 0)
```

## Implementation Notes

1. **Keep Indicators Pure**: Each indicator strategy should return only its own signal
2. **No Cross-Dependencies**: Indicator strategies shouldn't depend on each other
3. **Metadata Rich**: Include component signals in metadata for analysis
4. **Testable**: Each level (indicator, composite) should be independently testable

This organization treats your 16 rules as the essential building blocks they are, making it easy to create sophisticated strategies by combining them in different ways.