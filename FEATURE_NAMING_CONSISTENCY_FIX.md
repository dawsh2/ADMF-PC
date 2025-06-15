# Feature Naming Consistency Fix

## Problem Statement

There's a systematic naming mismatch between:
- **Strategy decorators** (`@strategy` feature_config)
- **FeatureHub registry** (FEATURE_REGISTRY keys)  
- **Strategy implementations** (features.get() calls)

This causes strategies to fail silently because they look for features that don't exist.

## Root Cause Examples

### Volatility Strategies
```python
# ❌ BEFORE (Broken)
@strategy(feature_config={'donchian': {...}})           # Decorator says 'donchian'
def donchian_breakout(features, bar, params):
    upper = features.get(f'donchian_{period}_upper')    # Code expects 'donchian_*'
    # But FeatureHub has: "donchian_channel": donchian_channel_feature
    # Which creates: donchian_channel_{period}_upper

# ✅ AFTER (Fixed)  
@strategy(feature_config={'donchian_channel': {...}})   # Matches registry
def donchian_breakout(features, bar, params):
    upper = features.get(f'donchian_channel_{period}_upper')  # Matches output
```

### Other Examples
- `bollinger` vs `bollinger_bands` 
- `keltner` vs `keltner_channel`
- `stochastic` vs `stochastic_rsi` (different features!)
- `ultimate` vs `ultimate_oscillator`

## The Fix: Three-Step Alignment

### Step 1: Create Feature Name Mapping
```python
# src/strategy/components/features/naming.py
FEATURE_NAME_MAPPING = {
    # Strategy decorator name -> FeatureHub registry key
    'donchian': 'donchian_channel',
    'keltner': 'keltner_channel', 
    'bollinger': 'bollinger_bands',
    'ultimate': 'ultimate_oscillator',
    'williams': 'williams_r',
    # Add more as needed
}
```

### Step 2: Update Topology Builder
Auto-map feature names during inference:
```python
def _infer_features_from_strategies(self, strategies):
    # ...existing code...
    for feature_name, feature_meta in feature_config.items():
        # Map to correct registry name
        registry_name = FEATURE_NAME_MAPPING.get(feature_name, feature_name)
        # Generate features using registry name
        required_features.add(f'{registry_name}_{param_str}')
```

### Step 3: Validation Tool
Automatically detect mismatches:
```python
def validate_feature_consistency():
    """Check all strategies for feature naming consistency."""
    # Scan all @strategy decorators
    # Check if declared features exist in FEATURE_REGISTRY
    # Verify strategy code uses correct feature names
```

## Implementation Plan

1. **Fix existing strategies** (volatility.py, others)
2. **Create mapping dictionary** 
3. **Update topology builder** to use mapping
4. **Add validation** to catch future issues
5. **Update STRATEGY_DEVELOPMENT_GUIDE.md**

## Prevention: Standard Naming Rules

### Rule 1: Decorator Names Must Match Registry
```python
# FeatureHub registry key
"bollinger_bands": bollinger_bands_feature

# Strategy decorator must use SAME key
@strategy(feature_config={'bollinger_bands': {...}})
```

### Rule 2: Multi-value Features Use Suffixes
```python
# Feature function returns: {"upper": ..., "lower": ..., "middle": ...}
# FeatureHub creates: bollinger_bands_20_2.0_upper, bollinger_bands_20_2.0_lower, etc.
# Strategy must use: features.get(f'bollinger_bands_{period}_{std_dev}_upper')
```

### Rule 3: Validation in CI/CD
Add pre-commit hook:
```bash
python validate_feature_consistency.py
# Fails if any strategy has naming mismatches
```

## Files to Update

1. **Strategy files** - Fix decorator/code mismatches
2. **Topology builder** - Add auto-mapping  
3. **Feature naming module** - Centralized mapping
4. **Strategy guide** - Updated examples
5. **Validation tool** - Automated checking

This will eliminate the recurrent naming issues and prevent future ones.