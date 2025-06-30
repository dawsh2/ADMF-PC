# Fixed: FLAT Signals Now Published to Event Bus 🎯

## Issue
Zero signals (FLAT) were being filtered out and not published to the event bus, which prevented proper position management since these signals indicate when to close positions.

## Root Cause
In `src/strategy/state.py`, line 1277 had:
```python
if should_publish and direction != SignalDirection.FLAT:
```

This explicitly filtered out FLAT signals from being published.

## Fix Applied
Changed to:
```python
if should_publish:
```

Now ALL signals (LONG, SHORT, and FLAT) are published to the event bus.

## Verification

### RSI Bands Strategy
```bash
python3 main.py --config config/indicators/oscillator/test_rsi_bands.yaml --signal-generation --bars 300
```

Results:
- ✅ Generates FLAT signals when RSI is between 30-70
- ✅ Generates LONG signals when RSI < 30 (oversold)
- ✅ Generates SHORT signals when RSI > 70 (overbought)

Example output:
```
📡 SIGNAL: SPY_compiled_strategy_0 → FLAT @ 2024-03-26 13:48:00
📡 SIGNAL: SPY_compiled_strategy_0 → FLAT @ 2024-03-26 13:49:00
...
📡 SIGNAL: SPY_compiled_strategy_0 → LONG @ 2024-03-26 14:31:00  # RSI < 30
📡 SIGNAL: SPY_compiled_strategy_0 → FLAT @ 2024-03-26 14:32:00  # RSI back above 30
```

### SMA Crossover Strategy
```bash
python3 main.py --config config/indicators/crossover/test_sma_crossover.yaml --signal-generation --bars 100
```

Results:
- ✅ Generates sustained LONG signals when fast SMA > slow SMA
- ✅ Generates sustained SHORT signals when fast SMA < slow SMA
- ✅ Changes signal at crossover points

## Why This Matters

1. **Position Management**: Execution engines use FLAT signals to close positions
2. **Complete Signal Flow**: All market states are now represented
3. **Strategy Accuracy**: Mean reversion strategies properly signal neutral zones
4. **Event Bus Completeness**: Downstream components receive full signal stream

## Signal Types Summary

- **LONG (1)**: Buy signal, open/maintain long position
- **SHORT (-1)**: Sell signal, open/maintain short position  
- **FLAT (0)**: Neutral signal, close any open position

All three signal types are now properly published to the event bus for complete strategy execution.