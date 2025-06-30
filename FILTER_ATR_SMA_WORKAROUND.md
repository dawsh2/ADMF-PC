# ATR SMA Filter Workaround

## Issue
Filters that use `atr_sma_50` or similar composite features (SMA of another indicator) are not automatically supported by the feature discovery system.

## Current Status
- `atr_sma(50)` function is now available in filter expressions
- However, the actual `atr_sma_50` feature needs to be manually defined

## Workaround
For now, manually define the composite feature in your config:

```yaml
# Add this to your config
feature_configs:
  atr_14:
    type: atr
    period: 14
  atr_sma_50:
    type: sma
    input_feature: atr_14  # This assumes the SMA feature type supports input_feature
    period: 50
```

Or if the above doesn't work, you can compute it differently in the filter:

```yaml
# Instead of: atr(14) > atr_sma(50) * 1.2
# Use a different approach or manually define the feature
```

## Future Enhancement
The feature discovery system should be enhanced to:
1. Detect composite features like `atr_sma_N`
2. Automatically create both the base feature (ATR) and the derived feature (SMA of ATR)
3. Support chained feature computations in FeatureHub

## Note on bar_of_day
The `bar_of_day` error should now be fixed - it's computed directly in the filter context based on the timestamp.