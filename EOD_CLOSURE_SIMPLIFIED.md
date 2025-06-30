# Simplified EOD Closure with Threshold

## The Problem with the Original Implementation

The threshold approach `time < 1550` didn't work for EOD closure because:

1. When threshold failed, the filter returned `None` (no signal)
2. This prevented new signals but didn't actively close existing positions
3. A position opened at 3:45 PM would remain open indefinitely

## The Solution

We fixed the filtered strategy wrapper to return a flat signal (0) when threshold fails, instead of returning `None`. This ensures positions are actively closed when conditions aren't met.

## Simple EOD Closure Examples

### Using 'intraday' keyword (simplest):
```yaml
strategy: [
  {
    bollinger_bands: {
      period: 20,
      std_dev: 2.0
    },
    threshold: "intraday"  # Only trade during market hours
  }
]
```

### With additional conditions:
```yaml
strategy: [
  {bollinger_bands: {period: 20, std_dev: 2.5}},
  {rsi_extreme: {period: 14, oversold: 30}},
  {
    threshold: |
      intraday and (
        volume > sma(volume, 20) * 1.3 or
        volatility_percentile(50) > 0.4
      )
  }
]
```

### Custom market hours:
```yaml
strategy: [
  {
    keltner_bands: {
      period: 20,
      multiplier: 2.0
    },
    threshold: "time >= 930 and time < 1550"  # 9:30 AM to 3:50 PM
  }
]
```

## How It Works

1. **During market hours**: `intraday` is true, signals pass through normally
2. **After hours**: `intraday` is false, threshold fails, returns signal_value=0
3. **Result**: Any open positions are automatically closed when market closes

## Available Time Variables

- `intraday`: Boolean, true during regular market hours (9:30 AM - 4:00 PM ET)
- `time`: Current time in HHMM format (e.g., 1530 = 3:30 PM)
- `hour`: Current hour (0-23)
- `minute`: Current minute (0-59)

## Benefits

- No complex forced closure logic needed
- Works with any strategy automatically
- Can combine with other conditions (volume, volatility, etc.)
- Clear and intuitive configuration