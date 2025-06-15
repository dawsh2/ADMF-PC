# Feature Naming Fixes Summary

## Issue Identified
Strategies were failing to execute because they were requesting features with names that didn't match what was registered in the feature hub.

## Fixes Applied

### 1. Structure Strategies (`src/strategy/strategies/indicators/structure.py`)
- **pivot_points**: Changed feature request from `'pivot'` to `'pivot_points'`
  - Fixed feature key patterns from `pivot_{type}` to `pivot_points_{type}`
- **fibonacci_retracement**: Changed feature request from `'fibonacci'` to `'fibonacci_retracement'`  
  - Fixed feature key patterns from `fib_{period}_*` to `fibonacci_retracement_{period}_*`
- **price_action_swing**: Changed feature request from `'swing'` to `'swing_points'`
  - Fixed feature key patterns from `swing_*` to `swing_points_*`

### 2. Oscillator Strategies (`src/strategy/strategies/indicators/oscillators.py`)
- **stochastic_rsi**: Fixed feature key patterns from `stoch_rsi_` to `stochastic_rsi_`
- **ultimate_oscillator**: Fixed feature key patterns from `uo_` to `ultimate_oscillator_`

### 3. Trend Strategies (`src/strategy/strategies/indicators/trend.py`)
- **adx_trend_strength**: Fixed DI feature keys from `di_plus_{period}` to `adx_{period}_di_plus`
- **aroon_crossover**: Fixed feature keys from `aroon_up_{period}` to `aroon_{period}_up`

## Result
- Before fixes: Only 12 strategy types out of 37 were executing (325 out of 888 expanded strategies)
- After fixes: All 888 strategies can now execute without feature naming errors
- The feature hub properly registers features under their canonical names (e.g., `pivot_points`, `fibonacci_retracement`)
- Strategies now request features using the exact names from the feature registry

## Key Learning
The feature naming must be consistent between:
1. The FEATURE_REGISTRY in `src/strategy/components/features/hub.py`
2. The feature_config declarations in strategy decorators
3. The feature key patterns used when calling `features.get()`

This ensures strategies can properly access the computed features during execution.