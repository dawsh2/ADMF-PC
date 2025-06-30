# Exit Memory Implementation

## Overview

Exit memory is a risk management feature that prevents immediate re-entry into positions after risk-based exits (stop loss, trailing stop, take profit) when the strategy signal remains unchanged.

## Problem Solved

Previously, when a position was closed due to risk management (e.g., stop loss), the system would immediately open a new position on the next bar if the strategy signal persisted. This led to:

- **Whipsawing**: Repeatedly hitting stop losses in volatile markets
- **Increased transaction costs**: Multiple entries/exits for the same signal
- **Poor risk-adjusted returns**: Re-entering losing positions without market conditions changing

## How It Works

### 1. Signal State Tracking

When a risk exit occurs, the portfolio stores the current signal state:

```python
# Store current signal state in exit memory if this is a risk exit
if self._exit_memory_enabled and exit_signal.exit_type in self._exit_memory_types:
    current_signal_value = 1.0 if position.quantity > 0 else -1.0
    memory_key = (symbol, strategy_id)
    self._exit_memory[memory_key] = current_signal_value
```

### 2. Re-entry Prevention

When new signals arrive, the portfolio checks if the signal has changed:

```python
# Check exit memory - prevent re-entry if signal hasn't changed since risk exit
if self._exit_memory_enabled:
    memory_key = (symbol, base_strategy_id)
    if memory_key in self._exit_memory:
        exit_signal_value = self._exit_memory[memory_key]
        if abs(direction_value - exit_signal_value) < 0.01:  # Same signal
            return  # Block order creation
        else:
            # Signal has changed, clear exit memory
            del self._exit_memory[memory_key]
```

### 3. Signal Change Detection

The system maps signal directions to numeric values:
- **LONG**: 1.0
- **SHORT**: -1.0
- **FLAT**: 0.0

A signal is considered "changed" when its numeric value differs from the stored exit value.

## Configuration

### Enable/Disable Exit Memory

```python
# Enable exit memory (default)
portfolio.configure_exit_memory(enabled=True)

# Disable exit memory
portfolio.configure_exit_memory(enabled=False)
```

### Configure Exit Types

```python
# Default: triggers on stop_loss, trailing_stop, take_profit
portfolio.configure_exit_memory(
    enabled=True,
    exit_types={'stop_loss', 'trailing_stop', 'take_profit'}
)

# Only trigger on stop losses
portfolio.configure_exit_memory(
    enabled=True,
    exit_types={'stop_loss'}
)
```

## Example Scenarios

### Scenario 1: Stop Loss with Persistent Signal

```
Bar 100: Strategy signal = LONG (1.0)
         → Position opened at $450.00

Bar 150: Price drops to $445.00, stop loss triggered
         → Position closed with loss
         → Exit memory stores: (SPY, strategy_0) → 1.0

Bar 151: Strategy signal still = LONG (1.0)
         → Exit memory blocks new position (signal unchanged)

Bar 152: Strategy signal = FLAT (0.0)
         → Exit memory cleared (signal changed)
         → No position opened (flat signal)

Bar 153: Strategy signal = LONG (1.0)
         → New position opened (exit memory was cleared)
```

### Scenario 2: Signal Reversal After Exit

```
Bar 200: Strategy signal = SHORT (-1.0)
         → Short position opened

Bar 220: Trailing stop triggered
         → Position closed
         → Exit memory stores: (SPY, strategy_0) → -1.0

Bar 221: Strategy signal = LONG (1.0)
         → Exit memory cleared (signal changed from -1.0 to 1.0)
         → New long position opened immediately
```

## Benefits

1. **Reduced Whipsawing**: Avoids repeated entries in choppy markets
2. **Lower Transaction Costs**: Fewer trades when conditions haven't changed
3. **Better Risk Management**: Forces signal confirmation before re-entry
4. **Improved Returns**: Prevents compounding losses from repeated stops

## Trade Analysis

The enhanced notebook template (`trade_analysis_simple.ipynb`) now includes:

1. **Trade Reconstruction**: Builds complete trades from position events
2. **Performance Metrics**: 
   - Win rate, average returns
   - Sharpe ratio, maximum drawdown
   - Profit factor
3. **Exit Analysis**: Performance breakdown by exit type
4. **Re-entry Detection**: Identifies immediate re-entries after exits

## Future Enhancements

1. **Configurable Cooldown Period**: Time-based prevention (e.g., 5 bars)
2. **Signal Strength Threshold**: Require stronger signals after exits
3. **Volatility-Based Rules**: Adjust behavior based on market conditions
4. **Per-Strategy Configuration**: Different rules for different strategy types

## Usage Notes

- Exit memory is enabled by default for all new portfolio instances
- The feature only affects risk-based exits, not signal-based exits
- Memory is cleared when the signal changes or reverses
- Each (symbol, strategy) pair has independent memory