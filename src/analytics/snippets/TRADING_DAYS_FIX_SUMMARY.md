# Trading Days Calculation Fix

## Problem
Analysis scripts were incorrectly calculating trading days using the entire market data file:
```python
trading_days = len(market_data['timestamp'].dt.date.unique())
```

This caused incorrect metrics when the actual trading period was shorter than the data file:
- Test data: ~2 months (~50 days)
- Calculated: 256 days (full year in data file)
- Result: Trade frequency showed 1.6/day instead of 8.3/day

## Solution
Added `get_actual_trading_days()` function to `helpers.py` that:

1. **First tries**: Get actual period from signal dates in performance_df
2. **Then tries**: Read a trace file to get the date range
3. **Then tries**: Check signal files in run directory
4. **Last resort**: Use market data range (with warning)

## Files Updated
- `/src/analytics/snippets/helpers.py` - Added `get_actual_trading_days()` function
- `/src/analytics/snippets/complete_stop_target_analysis.py` - Updated to use new function
- `/src/analytics/snippets/complete_stop_target_analysis_enhanced.py` - Updated to use new function

## Usage
In any analysis script, replace:
```python
trading_days = len(market_data['timestamp'].dt.date.unique())
```

With:
```python
from helpers import get_actual_trading_days
trading_days = get_actual_trading_days(performance_df, market_data)
```

## Impact
- Correct trades/day calculations
- Correct Sharpe ratio annualization
- More accurate performance metrics
- Works for both train and test data