# Strategy System Interface Analysis

## Executive Summary

The ADMF-PC strategy system follows a clean two-tier architecture that separates stateless strategy logic from stateful feature computation. This review examines the cohesion between strategies, features, feature configs, feature hub, and automatic discovery/inference by the topology builder.

## Architecture Overview

### Two-Tier Design
- **Tier 1 (Stateless)**: Strategies and classifiers - pure functions with no state
- **Tier 2 (Stateful)**: FeatureHub - centralized incremental feature computation engine

### Key Components

1. **Feature Hub** (`src/strategy/components/features/hub.py`)
   - Centralized feature computation with O(1) incremental updates
   - Feature registry mapping names to computation functions
   - Support for both simple (`sma_20`) and compound (`bollinger_bands_20_2.0`) features

2. **Discovery System** (`src/core/components/discovery.py`)
   - Decorator-based registration (`@strategy`, `@classifier`)
   - Automatic component discovery and metadata storage
   - Foundation for feature inference

3. **Topology Builder** (`src/core/coordinator/topology.py`)
   - Automatic feature inference from strategy configurations
   - Feature config generation from strategy parameters
   - Context injection for container creation

4. **Component State** (`src/strategy/state.py`)
   - Generic execution engine for stateless components
   - FeatureHub integration for feature access
   - Event publishing for signals and classifications

## Integration Flow

```
User Config → Discovery → Feature Inference → FeatureHub → Strategy Execution
```

1. User defines strategies with parameters in YAML
2. Discovery system registers strategies with metadata
3. Topology builder infers required features from strategy parameters
4. FeatureHub computes features incrementally
5. ComponentState executes strategies with computed features

## Strengths

### 1. Clean Separation of Concerns
- Stateless strategies remain pure functions
- Stateful computation isolated in FeatureHub
- Clear protocol-based interfaces

### 2. Automatic Feature Inference
- Users specify only strategy parameters
- System determines required features automatically
- Reduces configuration complexity

### 3. Flexible Feature Naming
- Consistent naming patterns: `feature_param1_param2`
- Support for compound features
- Predictable feature resolution

### 4. Performance Optimization
- O(1) incremental feature updates
- Component readiness caching after warmup
- Efficient for streaming data

## Issues and Recommendations

### 1. Feature Config Format Inconsistency

**Issue**: Multiple formats exist (legacy dict vs new list format)

**Current State**:
```python
# Legacy complex format
@strategy(feature_config={
    'sma': {'params': ['fast_period', 'slow_period'], 'defaults': {'fast_period': 10}}
})

# New simplified format
@strategy(feature_config=['sma', 'rsi', 'bollinger_bands'])
```

**Recommendation**: Standardize on simplified list format
```python
@strategy(
    name='sma_crossover',
    feature_config=['sma']  # Simple list of required features
)
```

### 2. Feature Inference Complexity

**Issue**: Feature inference logic in topology.py contains hardcoded parameter mappings

**Current State**:
```python
# In topology.py - hardcoded mappings
feature_param_mapping = {
    'sma': ['period', 'sma_period', 'fast_period', 'slow_period'],
    # ... many more hardcoded mappings
}
```

**Recommendation**: Move mappings to strategy metadata
```python
@strategy(
    name='sma_crossover',
    feature_config=['sma'],
    param_feature_mapping={
        'fast_period': 'sma_{value}',
        'slow_period': 'sma_{value}'
    }
)
```

### 3. Discovery Registration Timing

**Issue**: Modules must be imported for decorators to execute

**Current State**:
- Manual module imports in topology builder
- Risk of missing strategies if modules not imported

**Recommendation**: Explicit registration module
```python
# src/strategy/strategies/__init__.py
from .indicators import *  # Force decorator registration
from .ensemble import *
from .core import *

# Export all registered strategies
__all__ = ['get_all_strategies', 'get_strategy_by_name']
```

### 4. FeatureHub Connection Complexity

**Issue**: Deferred connection logic in ComponentState is complex

**Current State**:
```python
# Complex deferred connection handling
if not parent:
    self._deferred_feature_hub_name = feature_hub_name
# Later...
if self._deferred_feature_hub_name:
    self.complete_deferred_connections()
```

**Recommendation**: Establish connection during container creation
```python
# In container factory
if 'feature_hub_name' in config:
    feature_hub = self.resolve_feature_hub(config['feature_hub_name'])
    component_state.set_feature_hub(feature_hub)
```

### 5. Feature Registry Duplication

**Issue**: Feature computation functions registered separately from inference logic

**Current State**:
- Feature functions in `hub.py`
- Parameter inference logic duplicated in `topology.py`

**Recommendation**: Centralize in feature registry
```python
FEATURE_REGISTRY = {
    'sma': {
        'function': sma_feature,
        'params': ['period'],
        'param_names': ['period', 'sma_period'],  # Accepted param names
        'default_period': 20,
        'output_format': 'sma_{period}'
    }
}
```

### 6. Strategy Parameter Validation

**Issue**: Validation logic in topology.py disconnected from strategy definitions

**Current State**:
```python
# In topology.py
def _validate_strategy_parameters(self, strategy_type: str, params: Dict):
    if strategy_type == 'ma_crossover':
        if params['fast_period'] >= params['slow_period']:
            return False
```

**Recommendation**: Add validation to strategy decorator
```python
@strategy(
    name='ma_crossover',
    feature_config=['sma'],
    validators={
        'params': {
            'fast_period': lambda v: v > 0,
            'slow_period': lambda v: v > 0,
        },
        'cross': lambda p: p['fast_period'] < p['slow_period']
    }
)
```

## Priority Improvements

1. **High Priority**: Standardize feature config format across all strategies
2. **High Priority**: Move feature inference logic closer to strategy definitions  
3. **Medium Priority**: Improve discovery registration reliability
4. **Medium Priority**: Simplify FeatureHub connection mechanism
5. **Low Priority**: Add comprehensive parameter validation

## Best Practices Going Forward

### Strategy Definition
```python
@strategy(
    name='momentum_breakout',
    feature_config=['sma', 'rsi', 'atr'],  # Simple list format
    param_validators={
        'sma_period': lambda v: 10 <= v <= 200,
        'rsi_threshold': lambda v: 0 <= v <= 100
    }
)
def momentum_breakout(features, bar, params):
    # Pure function - no state management
    # Access features by name: features['sma_20']
    # Return signal dict with standard format
    pass
```

### Feature Naming Convention
- Simple features: `{feature}_{period}` (e.g., `sma_20`)
- Compound features: `{feature}_{param1}_{param2}` (e.g., `bollinger_bands_20_2.0`)
- Multi-output features: `{feature}_{params}_{output}` (e.g., `bollinger_bands_20_2.0_upper`)

### Configuration Flow
1. User specifies strategy with parameters
2. System infers features from parameters
3. FeatureHub computes required features
4. Strategy receives features and generates signals

## Conclusion

The strategy system demonstrates solid architectural design with clear separation between stateless logic and stateful computation. The automatic feature inference reduces configuration burden on users. Key improvements should focus on standardizing formats, consolidating feature metadata, and simplifying connection mechanisms.