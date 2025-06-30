# Filter Feature Discovery Fix

## Problem
When using filters in strategy configurations, features referenced in the filter expressions (like `volume_sma_20`, `vwap`, etc.) were not being discovered and added to the FeatureHub. This caused errors like:
- `name 'volume_sma_20' is not defined`
- `name 'vwap' is not defined`
- `name 'bar_of_day' is not defined`

## Root Cause
In the StrategyCompiler's `extract_features` method, when processing parameter combinations, the filter expressions were not being included in the strategy configuration passed to feature extraction.

## Solution
Fixed in `src/core/coordinator/compiler.py` by including filter and filter_params in the strategy config during feature extraction:

```python
# In extract_features method, around line 95:
if 'filter' in combo:
    strategy_config['filter'] = combo['filter']
if 'filter_params' in combo:
    strategy_config['filter_params'] = combo['filter_params']
```

## Additional Fixes
1. **Feature Discovery** (`src/core/coordinator/feature_discovery.py`):
   - Added special handling for `vwap` as a computed feature
   - Skip raw data fields (open, high, low, close, volume) 
   - Skip `bar_of_day` as it's computed in the filter context
   - Parse `volume_sma_20` correctly from filter expressions

2. **Filter Context** (`src/strategy/components/config_filter.py`):
   - Added `bar_of_day` calculation based on minutes since market open
   - Ensure `vwap` is available as both a value and function

## Testing
The fix was verified with:
1. Direct feature extraction test - ✅ PASSED
2. Signal generation with filters - ✅ PASSED
3. Features are now properly discovered and computed by FeatureHub

## Usage
No changes needed in config files. Filters will now work without manually defining features:

```yaml
strategy:
  - keltner_bands:
      period: 20
      multiplier: 2.0
      filter: "signal == 0 or volume > volume_sma_20 * 1.2"
# No need to manually define volume_sma_20 in feature_configs!
```