# Threshold Behavior Examples

## Valid Threshold Examples

### Example 1: Intraday Only
```yaml
strategy: [
  {
    bollinger_bands: {
      period: 20,
      std_dev: 2.0
    },
    threshold: "intraday"
  }
]
```
- **During market hours (9:30 AM - 4:00 PM)**: `intraday = true`, signals pass through
- **After hours**: `intraday = false`, all signals become 0

### Example 2: Volume Filter
```yaml
strategy: [
  {
    rsi_extreme: {
      period: 14,
      oversold: 30
    },
    threshold: "volume > sma(volume, 20) * 1.5"
  }
]
```
- **High volume**: Threshold is true, RSI signals pass through
- **Low volume**: Threshold is false, signals become 0

### Example 3: Combined Conditions
```yaml
strategy: [
  {
    keltner_bands: {
      period: 20,
      multiplier: 2.0
    },
    threshold: |
      intraday and 
      atr(14) > atr(50) and
      volume > 1000000
  }
]
```
- **All conditions met**: Signals pass through
- **Any condition false**: Signals become 0

## Invalid Threshold Examples (System Halts)

### Example 1: Undefined Variable
```yaml
threshold: undefined_variable > 100  # ERROR: undefined_variable not found
```

### Example 2: Syntax Error
```yaml
threshold: volume >>> 1000  # ERROR: Invalid syntax
```

### Example 3: Type Error
```yaml
threshold: volume + "string"  # ERROR: Cannot add number and string
```

## Key Points

1. **Thresholds are boolean expressions** - they evaluate to true or false
2. **False doesn't mean failure** - it means "condition not met"
3. **Actual errors halt the system** - syntax errors, undefined variables, etc.
4. **Signal becomes 0 when threshold is false** - this closes positions

This design ensures:
- No silent failures
- Clear position management
- Fail-fast on configuration errors