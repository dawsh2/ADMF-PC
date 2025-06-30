# Composite Features Status and Solution

## Current State

### What Works Now
1. **Simple features** (e.g., `sma_20`, `rsi_14`) - ✅ Fully supported
2. **volume_sma_20** - ✅ Properly discovered and computed
3. **vwap** - ✅ Treated as computed feature
4. **bar_of_day** - ✅ Computed in filter context
5. **Basic error prevention** - ✅ Features default to safe values instead of crashing

### What Doesn't Work (Yet)
1. **True composite features** (e.g., `atr_sma_50` as actual SMA of ATR values)
2. **Historical feature access** (needed for moving averages of other indicators)
3. **Feature dependency resolution** (automatic ordering of dependent features)

## Temporary Solution (Implemented)

### Filter Error Prevention
The filter system now:
1. Detects composite feature patterns (`{source}_{ma_type}_{period}`)
2. Ensures base features are discovered (e.g., `atr` for `atr_sma_50`)
3. Provides fallback values to prevent "not defined" errors
4. Logs warnings about limitations

### Current Behavior
When you use `atr_sma_50` in a filter:
- The system discovers and computes `atr_14` 
- In the filter, `atr_sma_50` gets the current ATR value (not true SMA)
- A warning is logged about the limitation
- The filter doesn't crash

## Proper Solution (Documented)

See `/Users/daws/ADMF-PC/docs/FEATURE_DEPENDENCIES_STANDARD.md` for the full architectural solution.

### Key Components Needed
1. **Enhanced FeatureSpec** with dependencies field
2. **Composite feature types** in FEATURE_REGISTRY  
3. **Dependency graph resolution** in FeatureHub
4. **Historical feature storage** for computing MAs of indicators
5. **New feature classes** (SMAOf, EMAOf, etc.)

## Recommended Usage Until Full Implementation

### Option 1: Use Function Syntax
```yaml
# Instead of: atr_sma_50 > 0.5
# Use: atr_sma(50) > 0.5
filter: "signal == 0 or atr_sma(50) > ${threshold}"
```

### Option 2: Use Simpler Filters
```yaml
# Instead of: Complex SMA of ATR comparison
# Use: Direct ATR comparison
filter: "signal == 0 or atr(14) > ${atr_threshold}"
```

### Option 3: Manual Feature Configuration (Last Resort)
```yaml
# If you absolutely need true SMA of ATR
feature_configs:
  atr_14:
    type: atr
    period: 14
  # Note: This won't work with current system
  # atr_sma_50:
  #   type: sma
  #   source: atr_14  # Not supported yet
  #   period: 50
```

## Why This Matters

The current architecture treats filters as first-class citizens for:
- **Simple features** ✅
- **Standard indicators** ✅  
- **Basic computations** ✅

But not yet for:
- **Composite features** ❌
- **Cross-indicator computations** ❌
- **Historical transformations** ❌

This violates the principle of "automatic feature inference" for advanced use cases.

## Timeline

1. **Immediate** (Done): Error prevention and warnings
2. **Short term**: Basic composite feature support for common cases
3. **Medium term**: Full dependency resolution system
4. **Long term**: Advanced composite features with optimization

## For Production Use

Until full composite feature support is implemented:
1. Stick to simple features in filters
2. Use parameter-based thresholds instead of complex comparisons
3. Monitor warning logs for composite feature usage
4. Consider whether the complexity is necessary for your strategy

The system now handles these cases gracefully without crashing, but true composite feature computation requires architectural enhancements.