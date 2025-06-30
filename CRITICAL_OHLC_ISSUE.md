# CRITICAL ISSUE: Missing OHLC Data in Signals

## The Problem

The Bollinger Bands strategy (and likely all strategies) only pass the **close price** in the signal metadata:

```python
'metadata': {
    'price': price,  # This is just bar.get('close', 0)
    'upper_band': upper_band,
    'lower_band': lower_band,
    # ... but NO high/low prices!
}
```

## Why This Breaks Stop Loss / Take Profit

When the risk manager checks if stop loss or take profit should be triggered, it needs:
- **LOW price** of the bar to check if stop loss was hit
- **HIGH price** of the bar to check if take profit was hit

But it only has the **CLOSE price**, so it can't accurately determine intrabar exits!

## The Consequence

This explains:
1. Why you're seeing exits at wrong prices (0.075% gain instead of loss)
2. Why you have 453 trades instead of 416
3. Why performance doesn't match the notebook

## The Solution

We need to modify the strategy to pass OHLC data:

```python
'metadata': {
    'price': price,  # Keep for compatibility
    'open': bar.get('open', 0),
    'high': bar.get('high', 0),
    'low': bar.get('low', 0),
    'close': bar.get('close', 0),
    # ... other metadata
}
```

Then the risk manager can use:
- `low` price to check if stop loss was hit during the bar
- `high` price to check if take profit was hit during the bar

## Quick Fix

1. Exit Python completely
2. Clear cache: `./clear_python_cache.sh`
3. Apply this fix to the Bollinger strategy
4. Run the backtest again

This should resolve all the issues!