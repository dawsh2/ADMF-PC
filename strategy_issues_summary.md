# Strategy Signal Generation Issues Summary

## Issues Found:

### 1. Feature Naming Mismatch
- **Stochastic Strategy** expects: `stochastic_{k_period}_{d_period}_k` and `stochastic_{k_period}_{d_period}_d`
- **Available features**: `stochastic_{k_period}_{d_period}` (without k/d suffix)

### 2. Only 18 out of 537 strategies became ready
- This is because most strategies couldn't find their required features due to naming mismatches

### 3. Even the 18 "ready" strategies generated 0 signals
- This needs further investigation - they should generate signals even if the value is 0

## Examples of Feature Mismatches:

1. **Stochastic Crossover**:
   - Expects: `stochastic_5_3_k`, `stochastic_5_3_d`
   - Available: `stochastic_5_3`

2. **MACD Crossover** (need to check):
   - Expects: `macd_value_5_20_7`, `macd_signal_5_20_7`
   - Available: `macd_5_20_7` (need to verify format)

3. **Vortex** (need to check):
   - Expects: `vortex_vi_plus_11`, `vortex_vi_minus_11`
   - Available: `vortex_11`

## Root Cause:
The feature naming convention used by the strategies doesn't match the feature naming convention used by the feature computation system.

## Solution Options:
1. Fix the strategies to use the correct feature names
2. Fix the feature computation to generate features with the expected names
3. Add a feature name mapping layer
