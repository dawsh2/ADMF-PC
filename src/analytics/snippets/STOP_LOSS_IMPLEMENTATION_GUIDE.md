# Stop Loss Implementation Guide

## ⚠️ CRITICAL: Correct vs Wrong Implementation

### ❌ WRONG Implementation (DO NOT USE)

```python
# This is WRONG - it only caps losses retrospectively
if trade['raw_return'] < -sl_decimal:
    trades_with_sl.loc[idx, 'raw_return'] = -sl_decimal
```

**Why this is wrong:**
1. It looks at the final return AFTER the trade is complete
2. It pretends losses were smaller than they actually were
3. It doesn't check if the stop price was actually hit during the trade
4. It makes results look artificially better
5. It doesn't stop out trades that would have been winners

### ✅ CORRECT Implementation

```python
# Check each bar between entry and exit
for _, bar in trade_prices.iterrows():
    if direction == 1:  # Long
        if bar['low'] <= stop_price:  # Check if low hit stop
            exit_price = stop_price
            exit_type = 'stop'
            break
    else:  # Short
        if bar['high'] >= stop_price:  # Check if high hit stop
            exit_price = stop_price
            exit_type = 'stop'
            break
```

**Why this is correct:**
1. Checks actual intraday price movements (high/low)
2. Exits immediately when stop is triggered
3. Calculates the actual return from the stop price
4. Stops out trades that would have eventually been winners
5. Accurately simulates real trading conditions

## Key Principles

### 1. Always Use Intraday Data
- Stop losses trigger on intraday movements, not closing prices
- Must check the low price for long positions
- Must check the high price for short positions

### 2. Exit Order Matters
- Check stops before profit targets (more conservative)
- Exit at the first condition met
- Track which exit type was triggered

### 3. Track Stopped Winners
- Many trades that get stopped out would have been winners
- This is a critical metric for stop loss analysis
- Shows the true cost of using stops

### 4. Include All Trades
- Don't just apply stops to losing trades
- Apply to all trades uniformly
- This reveals the full impact

## Available Functions

### `calculate_stop_loss_impact()`
Located in: `stop_loss_analysis_correct.py`
- Properly simulates stop losses using intraday data
- Tracks stopped winners
- Returns comprehensive metrics

### `apply_stop_target()`
Located in: `stop_target_analysis_comprehensive.py`
- Handles both stops and profit targets
- Checks exits in correct order
- Works for long and short positions

### `analyze_stop_target_combinations()`
Located in: `stop_target_analysis_comprehensive.py`
- Tests all combinations of stops and targets
- Finds optimal configurations
- Creates comprehensive visualizations

## Example Usage

```python
# Load your data
trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps=1.0)

# Analyze stop losses correctly
from stop_loss_analysis_correct import calculate_stop_loss_impact, visualize_stop_loss_impact

sl_results = calculate_stop_loss_impact(
    trades, 
    stop_loss_levels=[0.1, 0.2, 0.3, 0.5, 1.0],
    market_data=market_data  # REQUIRED!
)

optimal = visualize_stop_loss_impact(sl_results, "My Strategy")

# Analyze stop/target combinations
from stop_target_analysis_comprehensive import create_comprehensive_report

results, optimal = create_comprehensive_report(trades, market_data, "My Strategy")
```

## Common Pitfalls to Avoid

1. **Don't use closing prices** - Use high/low for accurate simulation
2. **Don't forget execution costs** - Apply them after calculating returns
3. **Don't assume tight stops are better** - They often stop out winners
4. **Don't ignore stopped winners** - This metric reveals the true cost
5. **Don't use the retrospective method** - It's fundamentally flawed

## Testing Your Implementation

To verify your implementation is correct:

1. Check that some winning trades get stopped out
2. Verify stop exit rates increase as stop levels tighten
3. Compare with the flawed method - correct method should show worse results
4. Look at individual trades to verify stop prices are calculated correctly

## Remember

**A proper stop loss implementation will often show WORSE results than no stops**, especially with tight stops. This is realistic - stops have a cost. The flawed implementation makes stops look artificially beneficial by only capping losses without simulating the actual exit mechanics.