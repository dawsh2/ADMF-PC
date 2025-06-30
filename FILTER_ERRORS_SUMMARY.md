# Filter Error Summary and Solutions

## Fixed Issues

### 1. volume_sma_20
✅ **FIXED** - Now properly discovered from filter expressions and added to FeatureHub

### 2. vwap  
✅ **FIXED** - Treated as a computed feature and properly discovered

### 3. bar_of_day calculation
✅ **FIXED** - Computed in filter context from timestamp

## Remaining Issues

### 1. atr_sma_50 (Composite Features)
**Issue**: Features that are computations of other features (like SMA of ATR) are not automatically supported.

**Current Status**: 
- Added `atr_sma()` function to filter context
- Feature discovery warns about composite features

**Workaround**: Manually define in config:
```yaml
feature_configs:
  atr_14:
    type: atr
    period: 14
  atr_sma_50:
    type: sma
    source: atr_14  # If supported by your SMA implementation
    period: 50
```

### 2. bar_of_day "not defined" errors
**Likely Cause**: Bar data missing timestamp field

**Solution**: Ensure bar data includes timestamp:
```python
bar = {
    'timestamp': '2024-01-15 10:00:00',  # Required for bar_of_day
    'open': 100.0,
    'high': 101.0,
    'low': 99.0,
    'close': 100.5,
    'volume': 1000000
}
```

## How to Debug Filter Errors

1. **Check what features are being discovered**:
   ```python
   from src.core.coordinator.compiler import StrategyCompiler
   compiler = StrategyCompiler()
   features = compiler.extract_features(your_config)
   print([f.canonical_name for f in features])
   ```

2. **Check what's in the filter context**:
   ```python
   from src.strategy.components.config_filter import ConfigSignalFilter
   filter_obj = ConfigSignalFilter()
   context = filter_obj._build_context(signal, features, bar)
   print(context.keys())
   ```

3. **Verify bar data has required fields**:
   - `timestamp` - Required for bar_of_day
   - `close` - Required for price comparisons
   - `volume` - Required for volume filters

## Feature Discovery Improvements Made

1. **Compiler Enhancement** (`src/core/coordinator/compiler.py`):
   - Now includes filter expressions when extracting features from parameter combinations

2. **Feature Discovery** (`src/core/coordinator/feature_discovery.py`):
   - Special handling for vwap as computed feature
   - Proper parsing of volume_sma_N patterns
   - Skip raw data fields that don't need feature computation

3. **Filter Context** (`src/strategy/components/config_filter.py`):
   - Added atr_sma() function
   - Improved bar_of_day calculation with timestamp handling