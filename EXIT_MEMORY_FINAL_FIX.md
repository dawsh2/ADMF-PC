# Exit Memory Final Fix

## The Problem
When a risk exit (stop loss or take profit) occurred while the signal was FLAT (0), the system would store 0 in exit memory. Since Bollinger signals are FLAT 50% of the time, this would block all future entries whenever the signal went to FLAT.

## The Solution
Only store exit memory for directional signals (LONG=1 or SHORT=-1). Never store exit memory when the signal is FLAT (0).

## What This Means
- If stop loss triggers while signal is -1 (SHORT), exit memory stores -1
- Future SHORT signals (-1) are blocked until signal changes
- If take profit triggers while signal is 0 (FLAT), NO exit memory is stored
- Future entries are not blocked by FLAT signals

## Expected Behavior
1. Position opens on directional signal (1 or -1)
2. If risk exit occurs:
   - While signal is still directional: Store in exit memory, block same signal
   - While signal is FLAT: Don't store in exit memory, allow future entries
3. Exit memory only clears when signal actually changes (not just goes to FLAT)

## To Test
1. Clear Python cache: `rm -rf __pycache__ src/**/__pycache__`
2. Run backtest: `python main.py config/bollinger/test.yaml`
3. Should see more trades than 7 (but less than 463 due to proper exit memory)

The fix balances between:
- Preventing immediate re-entry after stop loss (the original goal)
- Not blocking all future trades when exits happen during FLAT signals