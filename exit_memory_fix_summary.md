# Exit Memory Fix Summary

## Problem Identified

The system was experiencing immediate re-entries after risk-based exits (stop loss, take profit, trailing stop), causing:
- 212 extra trades (453 total vs 241 expected)
- Degraded performance (-4.12% vs 10.27% in notebook)
- Exit memory feature was not preventing re-entries as designed

## Root Cause

The `strategy_id` was not being properly set when POSITION_OPEN events were published. The sequence was:

1. Fill event arrives with metadata containing `strategy_id`
2. `update_position()` is called, creating a new position
3. POSITION_OPEN event is published with `strategy_id: None`
4. THEN the strategy_id is set in the position metadata (too late)

This meant that when exit memory tried to match `(symbol, strategy_id)`, it couldn't find the stored exit memory because the strategy_id was different.

## Fix Applied

Modified `src/portfolio/state.py`:

1. Added optional `metadata` parameter to `update_position()` method
2. Pass strategy_id metadata when calling `update_position()`
3. Merge metadata into position before publishing POSITION_OPEN event

### Key Changes:

```python
# Before: strategy_id set after event published
position = self.update_position(symbol, quantity, fill.price, fill.executed_at)
if position and position.quantity != 0 and strategy_id:
    position.metadata['strategy_id'] = strategy_id

# After: strategy_id passed to update_position
position_metadata = {}
if strategy_id:
    position_metadata['strategy_id'] = strategy_id
position = self.update_position(symbol, quantity, fill.price, fill.executed_at, metadata=position_metadata)
```

## Expected Results

With this fix:
- Strategy_id will be correctly included in POSITION_OPEN events
- Exit memory will properly match and prevent immediate re-entries
- Trade count should reduce from 453 to ~241
- Performance should improve to match notebook results

## Next Steps

1. Run the backtest again with the fix
2. Verify that immediate re-entries are prevented
3. Check that performance matches the notebook expectations
4. Consider adding tests to prevent regression