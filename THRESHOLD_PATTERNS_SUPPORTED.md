# Threshold Patterns - Implementation Status

## ✅ Per-Strategy Thresholds (SUPPORTED)

Each strategy can have its own threshold conditions:

```yaml
strategy: [
  {
    bollinger_bands: {
      period: 20,
      std_dev: 2.0
    },
    threshold: "intraday and volume > sma(volume, 20) * 1.2"
  },
  {
    rsi_strategy: {
      period: 14,
      oversold: 30,
      overbought: 70
    },
    threshold: "intraday and atr(14) > atr(50)"
  }
]
```

## ✅ Global Thresholds (SUPPORTED)

A single threshold can be applied to all strategies in the ensemble:

```yaml
strategy: [
  {bollinger_bands: {period: 20, std_dev: 2.5}},
  {rsi_strategy: {period: 14, oversold: 30, overbought: 70}},
  {
    threshold: |
      intraday and (
        volume > sma(volume, 20) * 1.3 or
        volatility_percentile(50) > 0.4
      )
  }
]
```

## How It Works

1. **Global Threshold Detection**: The compiler identifies threshold-only objects in the strategy list
2. **Automatic Application**: Global thresholds are applied to all strategies that don't have their own threshold
3. **Priority**: If a strategy has its own threshold, it takes precedence over the global threshold
4. **Threshold Evaluation**: When threshold evaluates to false, the signal becomes 0 (forcing position closure)

## Mixed Example

You can combine both patterns - some strategies with their own thresholds, plus a global threshold for others:

```yaml
strategy: [
  # This strategy has its own threshold
  {
    bollinger_bands: {
      period: 20,
      std_dev: 2.0
    },
    threshold: "volume > 2000000"  # High volume only
  },
  # These strategies will use the global threshold
  {macd: {fast: 12, slow: 26, signal: 9}},
  {momentum: {period: 14}},
  # Global threshold for strategies without their own
  {
    threshold: "intraday and volatility_percentile(20) > 0.3"
  }
]
```

In this example:
- Bollinger Bands uses its own volume threshold
- MACD and Momentum use the global intraday + volatility threshold

## Benefits

1. **Flexibility**: Apply different conditions to different strategies
2. **DRY Principle**: Define common conditions once with global threshold
3. **EOD Closure**: Use `intraday` keyword for automatic position closure
4. **Complex Logic**: Combine multiple conditions with AND/OR operators