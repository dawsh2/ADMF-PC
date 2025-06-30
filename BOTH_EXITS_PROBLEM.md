# The Both-Exits-In-Same-Bar Problem

## The Issue

When a 5-minute bar has a range > 0.175% (0.075% stop loss + 0.1% take profit), both exits could be hit within the same bar. We don't know which happened first!

Example:
- Entry: $100.00 at bar open
- Stop Loss: $99.925 (-0.075%)
- Take Profit: $100.10 (+0.1%)
- Bar: Open=$100, High=$100.20, Low=$99.90

Both exits were hit, but which came first?

## Common Solutions

### 1. Conservative Approach (Most Common)
**Always assume stop loss hits first** when both are possible.
- Rationale: Risk management takes priority
- This is what most backtesting systems do
- Slightly pessimistic but safer

### 2. Optimistic Approach
**Always assume take profit hits first** when both are possible.
- Less common, overly optimistic
- Can lead to unrealistic backtest results

### 3. Statistical Approach
Use the bar's close price as a hint:
- If close is nearer to high → assume TP hit first
- If close is nearer to low → assume SL hit first
- Still just a guess, but uses available information

### 4. The "50/50" Approach
Randomly choose which exit hit first (not recommended for reproducibility)

### 5. Tick Data Solution (Most Accurate)
Use tick-by-tick data instead of bars to know exact order.
- Most accurate but requires much more data
- Not available in your current dataset

## What the Notebook Likely Does

Given the high win rate (91.1%), the notebook probably:
1. Uses the conservative approach (SL first)
2. OR has logic to prevent both from triggering
3. OR the data rarely has bars wide enough for both

## Recommended Fix

Implement the conservative approach - when both exits are possible, only trigger the stop loss:

```python
# Pseudocode
if bar.low <= stop_loss_price and bar.high >= take_profit_price:
    # Both possible - conservative approach
    return stop_loss_exit
elif bar.low <= stop_loss_price:
    return stop_loss_exit
elif bar.high >= take_profit_price:
    return take_profit_exit
```

This matches what most professional backtesting systems do and provides consistent, conservative results.