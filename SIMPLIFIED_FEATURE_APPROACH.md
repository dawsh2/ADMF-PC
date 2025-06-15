# Simplified Feature Approach - Fix for Naming Issues

## Problem We're Solving
Complex parameter-based feature naming creates mismatches between:
- Strategy decorators (`'donchian'`)
- FeatureHub registry (`'donchian_channel'`) 
- Strategy expectations (`'donchian_20_upper'`)

## Solution: Base Features + Shared Computation

### New Strategy Pattern
```python
@strategy(
    name='sma_crossover',
    feature_config=['sma']  # Just declare base feature types needed
)
def sma_crossover(features, bar, params):
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    # Simple, clear feature names
    sma_fast = features.get(f'sma_{fast_period}')  # sma_10
    sma_slow = features.get(f'sma_{slow_period}')  # sma_20
    
    if sma_fast is None or sma_slow is None:
        return None
        
    signal_value = 1 if sma_fast > sma_slow else -1
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'sma_crossover',
        'metadata': {
            'fast_period': fast_period,     # Parameters for analysis
            'slow_period': slow_period,
            'sma_fast': sma_fast,          # Actual values for debugging
            'sma_slow': sma_slow
        }
    }
```

### Discovery Process
1. **Scan strategies** → Find base features needed: `['sma', 'rsi', 'bollinger_bands']`
2. **Collect parameters** → Find all periods used: `[10, 14, 20, 30]`
3. **Generate feature list** → `['sma_10', 'sma_14', 'sma_20', 'rsi_14', 'bollinger_bands_20_2.0']`
4. **Configure FeatureHub** → Compute shared features once

### Benefits
- ✅ **No naming mismatches** - Direct mapping to FeatureHub registry
- ✅ **Shared computation** - `sma_20` computed once, used by many strategies  
- ✅ **Simple discovery** - No complex parameter mapping
- ✅ **Metadata-driven analysis** - Sparse storage uses signal metadata
- ✅ **Backward compatible** - Existing `features.get()` calls still work

### Multi-Value Features (Bollinger, etc.)
```python
@strategy(feature_config=['bollinger_bands'])
def bollinger_breakout(features, bar, params):
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    
    # Clear, predictable naming
    upper = features.get(f'bollinger_bands_{period}_{std_dev}_upper')
    lower = features.get(f'bollinger_bands_{period}_{std_dev}_lower')
    
    return {
        'signal_value': signal_value,
        'metadata': {
            'period': period,
            'std_dev': std_dev,
            'upper_band': upper,
            'lower_band': lower
        }
    }
```

## Implementation Steps
1. **Simplify decorators** - Use lists instead of complex configs
2. **Update discovery** - Collect base types + parameters separately  
3. **Fix naming** - Ensure FeatureHub names match strategy expectations
4. **Test with existing signals** - Verify sparse storage still works