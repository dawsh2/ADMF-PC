# Strategy Comparison: expansive_grid_search.yaml vs Codebase

## Summary

After checking the strategies mentioned in `expansive_grid_search.yaml` against the actual implementations in the codebase, here's what I found:

### ✅ Strategies that EXIST in the codebase:

#### Crossover Strategies (from indicators/crossovers.py):
- ✅ `sma_crossover`
- ✅ `ema_crossover` 
- ✅ `ema_sma_crossover`
- ✅ `dema_crossover`
- ✅ `dema_sma_crossover`
- ✅ `tema_sma_crossover`
- ✅ `stochastic_crossover`
- ✅ `vortex_crossover`
- ✅ `ichimoku_cloud_position`
- ✅ `macd_crossover` (defined in crossovers.py, uses params: fast_ema, slow_ema, signal_ema)

#### Oscillator Strategies (from indicators/oscillators.py):
- ✅ `rsi_threshold`
- ✅ `rsi_bands`
- ✅ `cci_threshold`
- ✅ `cci_bands`
- ✅ `stochastic_rsi`
- ✅ `williams_r`
- ✅ `roc_threshold`
- ✅ `ultimate_oscillator`

#### Volatility Strategies (from indicators/volatility.py):
- ✅ `keltner_breakout`
- ✅ `donchian_breakout`
- ✅ `bollinger_breakout`

#### Trend Strategies (from indicators/trend.py):
- ✅ `adx_trend_strength`
- ✅ `parabolic_sar`
- ✅ `aroon_crossover`
- ✅ `supertrend`
- ✅ `linear_regression_slope`

#### Volume Strategies (from indicators/volume.py):
- ✅ `obv_trend`
- ✅ `mfi_bands`
- ✅ `vwap_deviation`
- ✅ `chaikin_money_flow`
- ✅ `accumulation_distribution`

#### Market Structure Strategies (from indicators/structure.py):
- ✅ `pivot_points`
- ✅ `fibonacci_retracement`
- ✅ `support_resistance_breakout`
- ✅ `atr_channel_breakout`
- ✅ `price_action_swing`

### ⚠️ Parameter Naming Issues:

1. **MACD Crossover**: There are TWO implementations:
   - `crossovers.py`: Uses `fast_ema`, `slow_ema`, `signal_ema` (matches YAML)
   - `momentum.py`: Uses `fast_period`, `slow_period`, `signal_period` (different from YAML)
   - The YAML uses the crossovers.py parameter names

### 📊 Total Strategy Count:
- **YAML declares**: 37 strategy types
- **Actually found**: All 37 strategies exist! ✅

### 🔍 Additional Strategies Found in Codebase:
From indicators/structure.py (not in YAML):
- `pivot_channel_breaks`
- `pivot_channel_bounces`
- `trendline_breaks`
- `trendline_bounces`

From indicators/momentum.py (not imported in __init__.py):
- `macd_crossover_strategy` (duplicate of crossovers.py version)
- `momentum_breakout_strategy`
- `roc_trend_strategy`
- `adx_trend_strength_strategy`
- `aroon_oscillator_strategy`
- `vortex_trend_strategy`
- `momentum_composite_strategy`

## Conclusion

All strategies referenced in the `expansive_grid_search.yaml` file exist in the codebase. The main issue is that there are duplicate implementations of some strategies (like MACD crossover) in different files, which could cause confusion. The YAML file appears to be using the correct parameter names for the version in `crossovers.py`.