# Strategy Migration Checklist: Complex to Simplified Features

## Overview
Migrate all indicator strategies from complex parameter-based feature declarations to simplified base feature lists. This eliminates naming mismatches and enables shared computation.

## Migration Pattern

### BEFORE (Complex/Broken)
```python
@strategy(
    name='bollinger_breakout',
    feature_config={
        'bollinger': {  # Wrong name (should be 'bollinger_bands')
            'params': ['period', 'std_dev'],
            'defaults': {'period': 20, 'std_dev': 2.0}
        }
    }
)
def bollinger_breakout(features, bar, params):
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    upper = features.get(f'bollinger_{period}_{std_dev}_upper')  # Wrong name!
```

### AFTER (Simplified/Working)
```python
@strategy(
    name='bollinger_breakout',
    feature_config=['bollinger_bands']  # Matches FeatureHub registry
)
def bollinger_breakout(features, bar, params):
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    upper = features.get(f'bollinger_bands_{period}_{std_dev}_upper')  # Correct!
    
    return {
        'signal_value': signal_value,
        'metadata': {
            'period': period,        # Parameters for sparse storage
            'std_dev': std_dev,
            'upper_band': upper,     # Values for analysis
            'lower_band': lower
        }
    }
```

## Step-by-Step Migration Process

### Step 1: Identify Feature Registry Names
For each strategy file, check what the FeatureHub registry actually calls the features:

**From `src/strategy/components/features/hub.py` FEATURE_REGISTRY:**
```python
# Volatility features
"bollinger_bands": bollinger_bands_feature,  # NOT "bollinger"
"keltner_channel": keltner_channel_feature,  # NOT "keltner" 
"donchian_channel": donchian_channel_feature, # NOT "donchian"

# Oscillator features  
"williams_r": williams_r_feature,            # NOT "williams"
"ultimate_oscillator": ultimate_oscillator_feature, # NOT "ultimate"
"stochastic_rsi": stochastic_rsi_feature,    # Different from "stochastic"

# Others
"chaikin_money_flow": cmf_feature,           # NOT "cmf"
"accumulation_distribution": ad_feature,     # NOT "ad"
```

### Step 2: Update Strategy Decorator
Replace complex feature_config with simple list:
```python
# OLD
feature_config={
    'bollinger': {
        'params': ['period', 'std_dev'],
        'defaults': {'period': 20, 'std_dev': 2.0}
    }
}

# NEW  
feature_config=['bollinger_bands']
```

### Step 3: Update Feature Access Code
Fix the features.get() calls to use correct names:
```python
# OLD (wrong name)
upper = features.get(f'bollinger_{period}_{std_dev}_upper')

# NEW (correct name)
upper = features.get(f'bollinger_bands_{period}_{std_dev}_upper')
```

### Step 4: Add Parameters to Metadata
Ensure signal metadata includes parameters for sparse storage separation:
```python
'metadata': {
    # Parameters (for separation)
    'period': period,
    'std_dev': std_dev,
    
    # Values (for analysis)
    'upper_band': upper,
    'lower_band': lower,
    'price': price
}
```

## Files to Migrate

### ✅ COMPLETED
- [x] `crossovers.py` - ALL strategies migrated (10 strategies)
  - [x] sma_crossover, ema_sma_crossover  
  - [x] ema_crossover, dema_crossover, dema_sma_crossover, tema_sma_crossover
  - [x] stochastic_crossover, vortex_crossover, macd_crossover, ichimoku_cloud_position

- [x] `oscillators.py` - ALL strategies migrated (8 strategies)
  - [x] rsi_threshold, rsi_bands, cci_threshold, cci_bands
  - [x] stochastic_rsi, williams_r, roc_threshold, ultimate_oscillator

- [x] `trend.py` - ALL strategies migrated (5 strategies)
  - [x] adx_trend_strength, parabolic_sar, aroon_crossover
  - [x] supertrend, linear_regression_slope

- [x] `volume.py` - ALL strategies migrated (5 strategies)
  - [x] obv_trend, mfi_bands, vwap_deviation
  - [x] chaikin_money_flow, accumulation_distribution

- [x] `structure.py` - ALL strategies migrated (5 strategies)
  - [x] pivot_points, fibonacci_retracement, support_resistance_breakout
  - [x] atr_channel_breakout, price_action_swing

- [x] `volatility.py` - Previously migrated (3 strategies)
  - [x] bollinger_breakout, keltner_breakout, donchian_breakout

### 🎉 MIGRATION COMPLETE
**Total: 36 strategies successfully migrated across 5 indicator files**

## Feature Name Mapping Reference

| Strategy Decorator | FeatureHub Registry | Example Access |
|-------------------|-------------------|----------------|
| `'sma'` | `'sma'` | `sma_20` |
| `'ema'` | `'ema'` | `ema_12` |
| `'rsi'` | `'rsi'` | `rsi_14` |
| `'bollinger'` ❌ | `'bollinger_bands'` ✅ | `bollinger_bands_20_2.0_upper` |
| `'keltner'` ❌ | `'keltner_channel'` ✅ | `keltner_channel_20_2.0_upper` |
| `'donchian'` ❌ | `'donchian_channel'` ✅ | `donchian_channel_20_upper` |
| `'williams'` ❌ | `'williams_r'` ✅ | `williams_r_14` |
| `'ultimate'` ❌ | `'ultimate_oscillator'` ✅ | `ultimate_oscillator_7_14_28` |
| `'stochastic_rsi'` | `'stochastic_rsi'` | `stochastic_rsi_14_14_k` |
| `'macd'` | `'macd'` | `macd_12_26_9_macd` |
| `'adx'` | `'adx'` | `adx_14` |
| `'vwap'` | `'vwap'` | `vwap` |
| `'obv'` | `'obv'` | `obv` |
| `'cmf'` ❌ | `'chaikin_money_flow'` ✅ | `chaikin_money_flow_20` |
| `'ad'` ❌ | `'accumulation_distribution'` ✅ | `accumulation_distribution` |

## Common Multi-Value Feature Patterns

### Bollinger Bands
```python
# Returns: {"upper": ..., "lower": ..., "middle": ...}
upper = features.get(f'bollinger_bands_{period}_{std_dev}_upper')
lower = features.get(f'bollinger_bands_{period}_{std_dev}_lower')
middle = features.get(f'bollinger_bands_{period}_{std_dev}_middle')
```

### MACD
```python
# Returns: {"macd": ..., "signal": ..., "histogram": ...}
macd_line = features.get(f'macd_{fast}_{slow}_{signal}_macd')
signal_line = features.get(f'macd_{fast}_{slow}_{signal}_signal')
histogram = features.get(f'macd_{fast}_{slow}_{signal}_histogram')
```

### Stochastic
```python
# Returns: {"k": ..., "d": ...}
stoch_k = features.get(f'stochastic_{k_period}_{d_period}_k')
stoch_d = features.get(f'stochastic_{k_period}_{d_period}_d')
```

### ADX with DI
```python
# Returns: adx value + di_plus, di_minus
adx_value = features.get(f'adx_{period}')
di_plus = features.get(f'adx_{period}_di_plus')
di_minus = features.get(f'adx_{period}_di_minus')
```

## Testing Checklist

After migrating each strategy:

### ✅ Unit Test
```python
def test_strategy_with_features():
    feature_hub = FeatureHub(['TEST'])
    feature_hub.configure_features({
        'bollinger_bands_20_2.0': {'type': 'bollinger_bands', 'period': 20, 'std_dev': 2.0}
    })
    
    # Feed data and test
    result = strategy(features, bar, params)
    assert result is not None
    assert 'metadata' in result
    assert 'period' in result['metadata']  # Parameters present
```

### ✅ Integration Test  
```python
# Test with topology builder feature inference
topology_builder.build_topology({
    'mode': 'signal_generation',
    'config': {'strategies': [{'type': 'bollinger_breakout', 'params': {...}}]}
})
# Should auto-configure bollinger_bands features
```

## Common Issues & Solutions

### Issue 1: "Feature always None"
**Cause**: Wrong feature name in features.get()
**Fix**: Check FEATURE_REGISTRY mapping, use correct name

### Issue 2: "Strategy not discovered"  
**Cause**: Decorator has list but topology builder doesn't handle it
**Fix**: Ensure topology builder supports both old dict and new list formats

### Issue 3: "Parameters missing from metadata"
**Cause**: Forgot to add parameters to signal metadata
**Fix**: Always include strategy parameters in metadata for sparse storage

### Issue 4: "Multi-value feature access wrong"
**Cause**: Using single name for multi-output feature
**Fix**: Use suffix pattern (e.g., `_upper`, `_lower`, `_k`, `_d`)

## Validation

After completing migration:

1. **Run topology builder** on expansive_grid_search.yaml
2. **Check feature inference logs** - should show simplified feature collection
3. **Test signal generation** - all 882 strategies should generate signals
4. **Verify metadata** - signal metadata should contain parameters
5. **Check sparse storage** - signals should be properly separated by parameters

## Success Metrics

- ✅ All 36 base strategy types generate signals (not just 14)
- ✅ All 882 strategy instances work (not just 404)  
- ✅ No "features always None" errors
- ✅ Signal metadata contains strategy parameters
- ✅ FeatureHub computes shared features efficiently
- ✅ Topology builder logs show successful feature inference