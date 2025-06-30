# Stop Loss Bug for Short Positions

## The Issue

Your stop losses ARE working correctly in terms of execution, but the RETURN CALCULATION is wrong for short positions:

- **LONG positions**: Correctly show -0.075% returns on stop loss
- **SHORT positions**: Incorrectly show +0.075% returns on stop loss (should be -0.075%)

## Why This Happens

When calculating returns, the system uses:
```
return = (exit_price - entry_price) / entry_price
```

This works for long positions but is inverted for shorts:
- SHORT at $100, stop loss at $100.075
- Simple calc: (100.075 - 100) / 100 = +0.075% ❌
- But you actually LOST money!

## The Correct Formula

For SHORT positions, the return should be:
```
return = (entry_price - exit_price) / entry_price
```

Or equivalently:
```
return = -1 * (exit_price - entry_price) / entry_price
```

## Where to Fix

The return calculation happens in analysis/reporting, not in the trading logic. Your stops are executing at the correct prices, but the performance metrics are wrong.

## Impact on Results

With this fix:
- All 110 short stop losses would show -0.075% instead of +0.075%
- Average return per trade would decrease
- Win rate would drop (those "winning" shorts are actually losses)
- Total return would be lower
- This explains why your results don't match the notebook!

## Summary

1. **Stop losses are executing correctly** at the right prices
2. **Take profit is set to 0.15%** (your experiment)
3. **Return calculation is inverted for shorts**, making losses look like gains
4. This significantly inflates your performance metrics