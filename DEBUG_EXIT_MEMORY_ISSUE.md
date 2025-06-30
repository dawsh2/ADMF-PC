# Exit Memory Debug

## The Pattern

Looking at the trades:
1. Trade at bar 16629 (entry)
2. Trade at bar 16639 (entry) 
3. Trade at bar 16646 (entry)
4. Trade at bar 16649 (entry) - This is 1 bar after stop loss at 16648
5. Trade at bar 16664 (entry)
6. Trade at bar 16726 (entry)
7. Trade at bar 16728 (entry) - This is 1 bar after stop loss at 16727

After bar 16728, NO MORE TRADES despite 400+ entry signals!

## The Issue

Looking at exits:
- Stop loss at bar 16648: signal was -1, changed to 0 at bar 16650
- Stop loss at bar 16666: signal was 1, changed to 0 at bar 16667
- Stop loss at bar 16727: signal was -1, changed to 0 at bar 16729
- Take profit at bar 16729: signal was 0

The problem might be:
1. Take profit at bar 16729 when signal was already 0 (FLAT)
2. Exit memory stores 0
3. All future FLAT signals (0) match the stored value (0)
4. System thinks we're still in the same signal that caused the exit
5. No more trades allowed!

## The Root Cause

When a take profit occurs while the signal is FLAT (0), the exit memory stores 0. Then every time the signal goes to FLAT (which happens 50% of the time with Bollinger), it blocks entry because it matches the stored exit memory value.

## The Fix

We should NOT store exit memory when the signal is FLAT (0) at the time of risk exit. Exit memory should only be stored for directional signals (1 or -1).