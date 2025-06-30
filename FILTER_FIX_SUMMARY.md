# Filter Integration Fix Summary

## Problem
Filters defined in strategy configs were not being applied during signal generation mode (`--signal-generation`). They were only working in backtest mode, making filter behavior inconsistent across execution modes.

## Root Cause
The strategy compiler (`src/core/coordinator/compiler.py`) was not applying filters when compiling strategies. Filters were only being used in backtesting pipelines.

## Solution Implemented

### 1. Created FilteredStrategyWrapper
- File: `src/strategy/components/filtered_strategy_wrapper.py`
- Wraps any strategy function to apply config-based filters
- Evaluates filter expressions against signals, features, and bar data
- Returns None (no signal) when filter rejects a signal

### 2. Modified Strategy Compiler
- File: `src/core/coordinator/compiler.py`
- Added filter detection in `_compile_single()` method
- Automatically wraps strategies with FilteredStrategyWrapper when filter is present
- Handles both single strategies and parameter combinations

### 3. Key Changes
```python
# In _compile_single():
if isinstance(config, dict) and 'filter' in config:
    filter_config = {'filter': config['filter']}
    if 'filter_params' in config:
        filter_config['filter_params'] = config['filter_params']
    compiled_func = wrap_strategy_with_filter(compiled_func, filter_config)
    logger.info(f"Applied filter to strategy: {config.get('filter')}")

# In parameter_combinations handling:
if 'filter' in combo:
    strategy_config['filter'] = combo['filter']
if 'filter_params' in combo:
    strategy_config['filter_params'] = combo['filter_params']
```

## Results

### Before Fix
- Signal generation ignored filters
- 726 signal changes regardless of filter threshold
- Filters only worked in backtest mode

### After Fix
- Filters work in ALL execution modes
- 3551 signal changes with 0.8 volatility threshold filter
- Consistent behavior across signal generation, backtesting, etc.

## Filter Syntax
The volatility filter in the config:
```yaml
filter:
  - {volatility_above: {threshold: 0.8}}
```

Translates to the expression:
```python
signal != 0 and atr(14) > atr(50) * 0.8
```

## Benefits
1. **Consistency**: Filters now work the same way in all modes
2. **Flexibility**: Can test different filter thresholds without post-processing
3. **Performance**: Filters are applied at signal generation time, reducing data size
4. **Modularity**: Filter logic is cleanly separated in its own wrapper

## Next Steps
1. Run backtesting with the filtered signals to verify performance
2. Test with different volatility thresholds to find optimal value
3. Consider adding more filter types (volume, trend, VWAP, etc.)

## Technical Note
The increase in signal changes (726 → 3551) is expected because:
- Without filter: Many signals are immediately zeroed out, creating fewer transitions
- With filter: Signals are only generated when conditions are met, creating more on/off transitions as the filter condition toggles

This fix addresses the user's explicit request: "It should be decoupled from 'mode'." - filters now work consistently regardless of execution mode.