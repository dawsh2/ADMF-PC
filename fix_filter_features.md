# Filter Feature Resolution Issue

## Problem
When filters reference features like `volume_sma_20` or use `bar_of_day`, these features need to be available in the evaluation context. Currently getting errors like:
- `name 'volume_sma_20' is not defined`
- `name 'bar_of_day' is not defined`

## Root Cause
1. Filter expressions are evaluated in a context that includes:
   - Bar data (open, high, low, close, volume)
   - Features computed by FeatureHub
   - Special values like bar_of_day

2. The issue is that features referenced in filters need to be:
   - Discovered from the filter expression
   - Added to the feature configuration
   - Computed by FeatureHub
   - Available in the filter evaluation context

3. Currently, feature discovery happens AFTER FeatureHub is initialized, so filter-required features aren't computed.

## Solutions

### Solution 1: Manual Feature Configuration (Immediate Fix)
Add required features to the config file:

```yaml
# Example: config/test_keltner_5min_proper.yaml
feature_configs:
  volume_sma_20:
    type: volume_sma
    period: 20
  vwap:
    type: vwap

strategy:
  - keltner_bands:
      period: [30]
      multiplier: [1.5]
      filter: "signal == 0 or volume > volume_sma_20 * 1.2"
```

### Solution 2: Fix Feature Discovery Order (Proper Fix)
The proper fix would be to:
1. Parse strategy configurations first
2. Extract all filter expressions
3. Discover required features from filters
4. Merge with strategy-required features
5. Initialize FeatureHub with complete feature set

This requires modifying the topology building process in `src/core/coordinator/topology.py`.

### Solution 3: Lazy Feature Evaluation
Another approach is to make feature access lazy in filters, similar to how `ma()`, `ema()`, `rsi()` functions work.

## bar_of_day Fix
For `bar_of_day`, I've already added the calculation to the filter context in `config_filter.py`. It calculates the bar index since market open (9:30 AM).

## Recommended Action
For now, use Solution 1 - manually add required features to configs that use filters. This ensures the features are computed and available when filters are evaluated.