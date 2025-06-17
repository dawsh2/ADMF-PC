# Feature System Disruptive Redesign

## Executive Summary

Complete replacement of the current feature inference system with a **validation-first, deterministic approach**. This design eliminates all guesswork, provides strict validation, and purges legacy patterns to prevent new developers from falling into bad habits.

## Core Principles

### 1. **Deterministic Feature Naming** - No Guessing
```python
# WRONG: Try different patterns until something works
def old_approach(features):
    for pattern in ['stochastic_14_3_k', 'stoch_14_3_k', 'stochastic_k_14_3']:
        if pattern in features:
            return features[pattern]
    return None  # Silent failure

# RIGHT: Exact specification with validation
def new_approach(features):
    required_feature = 'stochastic_14_3_k'  # Deterministic naming
    if required_feature not in features:
        raise FeatureValidationError(f"Required feature '{required_feature}' not found")
    return features[required_feature]
```

### 2. **Strict Validation** - Fail Fast
```python
# Strategy declares exactly what it needs
@strategy(
    name='stochastic_crossover',
    required_features=[
        FeatureSpec('stochastic', k_period=14, d_period=3, output='k'),
        FeatureSpec('stochastic', k_period=14, d_period=3, output='d')
    ]
)
def stochastic_crossover(features: ValidatedFeatures, bar, params):
    # Features are guaranteed to exist and be correctly named
    stoch_k = features['stochastic_14_3_k']
    stoch_d = features['stochastic_14_3_d']
    # No error handling needed - validation ensures they exist
```

### 3. **Complete Legacy Purge** - No Backward Compatibility
- Delete all old feature inference code
- Delete parameter mapping systems  
- Delete runtime discovery patterns
- Force all strategies to use new system
- Remove confusing naming variations

## New Architecture

### Layer 1: Standardized Feature Specifications

```python
@dataclass(frozen=True)
class FeatureSpec:
    """Immutable feature specification with deterministic naming"""
    feature_type: str
    params: Dict[str, Any]
    output_component: Optional[str] = None
    
    def __post_init__(self):
        # Validate parameters at creation time
        self._validate_params()
    
    @property 
    def canonical_name(self) -> str:
        """Generate deterministic feature name"""
        # Standard format: {type}_{param1}_{param2}_{component}
        param_str = '_'.join(str(v) for v in self.params.values())
        base_name = f"{self.feature_type}_{param_str}"
        
        if self.output_component:
            return f"{base_name}_{self.output_component}"
        return base_name
    
    def _validate_params(self):
        """Strict parameter validation"""
        registry = FEATURE_REGISTRY[self.feature_type]
        
        # Check all required parameters present
        missing = set(registry.required_params) - set(self.params.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {self.feature_type}: {missing}")
        
        # Check no extra parameters
        extra = set(self.params.keys()) - set(registry.all_params)
        if extra:
            raise ValueError(f"Unknown parameters for {self.feature_type}: {extra}")
        
        # Validate parameter values
        for param, value in self.params.items():
            validator = registry.param_validators.get(param)
            if validator and not validator(value):
                raise ValueError(f"Invalid value {value} for parameter {param}")

# Usage - Exact and explicit
stochastic_k = FeatureSpec('stochastic', {'k_period': 14, 'd_period': 3}, 'k')
print(stochastic_k.canonical_name)  # Always: stochastic_14_3_k

macd_signal = FeatureSpec('macd', {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, 'signal')  
print(macd_signal.canonical_name)  # Always: macd_12_26_9_signal
```

### Layer 2: Feature Registry with Strict Schemas

```python
@dataclass
class FeatureRegistryEntry:
    """Complete specification for a feature type"""
    name: str
    required_params: List[str]
    optional_params: Dict[str, Any]  # param_name -> default_value
    param_validators: Dict[str, Callable[[Any], bool]]
    output_components: List[str]  # For multi-output features
    computation_func: Callable
    description: str
    
    @property
    def all_params(self) -> Set[str]:
        return set(self.required_params) | set(self.optional_params.keys())

# Centralized registry with strict validation
FEATURE_REGISTRY = {
    'stochastic': FeatureRegistryEntry(
        name='stochastic',
        required_params=['k_period', 'd_period'],
        optional_params={},
        param_validators={
            'k_period': lambda x: isinstance(x, int) and 1 <= x <= 100,
            'd_period': lambda x: isinstance(x, int) and 1 <= x <= 20
        },
        output_components=['k', 'd'],
        computation_func=compute_stochastic,
        description='Stochastic oscillator with %K and %D components'
    ),
    
    'macd': FeatureRegistryEntry(
        name='macd', 
        required_params=['fast_period', 'slow_period', 'signal_period'],
        optional_params={},
        param_validators={
            'fast_period': lambda x: isinstance(x, int) and 1 <= x <= 50,
            'slow_period': lambda x: isinstance(x, int) and 10 <= x <= 200,
            'signal_period': lambda x: isinstance(x, int) and 1 <= x <= 50
        },
        output_components=['macd', 'signal', 'histogram'],
        computation_func=compute_macd,
        description='MACD with signal line and histogram'
    ),
    
    'sma': FeatureRegistryEntry(
        name='sma',
        required_params=['period'],
        optional_params={},
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 500
        },
        output_components=[],  # Single output
        computation_func=compute_sma,
        description='Simple moving average'
    )
}
```

### Layer 3: Validated Feature Container

```python
class ValidatedFeatures:
    """Container that guarantees all requested features exist"""
    
    def __init__(self, raw_features: Dict[str, Any], required_specs: List[FeatureSpec]):
        self.features = raw_features
        self.required_specs = required_specs
        self._validate_all_features()
    
    def _validate_all_features(self):
        """Strict validation - fail if ANY required feature missing"""
        missing_features = []
        
        for spec in self.required_specs:
            feature_name = spec.canonical_name
            if feature_name not in self.features:
                missing_features.append(feature_name)
        
        if missing_features:
            available = list(self.features.keys())
            raise FeatureValidationError(
                f"Missing required features: {missing_features}. "
                f"Available features: {available[:10]}..."  # Show first 10 for debugging
            )
    
    def __getitem__(self, feature_name: str) -> Any:
        """Dictionary-style access with guarantee feature exists"""
        return self.features[feature_name]
    
    def get(self, feature_name: str, default=None) -> Any:
        """Optional access for non-required features"""
        return self.features.get(feature_name, default)

class FeatureValidationError(Exception):
    """Clear error when feature validation fails"""
    pass
```

### Layer 4: Strict Strategy Interface

```python
def strategy(name: str, required_features: List[FeatureSpec]):
    """Strategy decorator with mandatory feature specification"""
    def decorator(func):
        # Validate that all feature specs are valid
        for spec in required_features:
            if spec.feature_type not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown feature type: {spec.feature_type}")
        
        # Register strategy with explicit feature requirements
        metadata = {
            'name': name,
            'required_features': required_features,
            'factory': func
        }
        
        # Wrapper function that validates features before calling strategy
        def validated_strategy(raw_features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]):
            # Create validated feature container
            validated_features = ValidatedFeatures(raw_features, required_features)
            
            # Call original strategy with validated features
            return func(validated_features, bar, params)
        
        register_strategy(name, metadata, validated_strategy)
        return validated_strategy
    
    return decorator

# Strategy usage - Explicit and validated
@strategy(
    name='stochastic_crossover',
    required_features=[
        FeatureSpec('stochastic', {'k_period': 14, 'd_period': 3}, 'k'),
        FeatureSpec('stochastic', {'k_period': 14, 'd_period': 3}, 'd')
    ]
)
def stochastic_crossover(features: ValidatedFeatures, bar, params):
    # Guaranteed to work - features validated before function called
    stoch_k = features['stochastic_14_3_k']
    stoch_d = features['stochastic_14_3_d']
    
    if stoch_k > stoch_d:
        return {'signal_value': 1}
    elif stoch_k < stoch_d: 
        return {'signal_value': -1}
    else:
        return {'signal_value': 0}
```

### Layer 5: Ensemble Feature Aggregation

```python
class EnsembleFeatureCollector:
    """Collects and validates features from nested strategies"""
    
    def collect_all_features(self, ensemble_config: Dict[str, Any]) -> List[FeatureSpec]:
        """Collect all features needed by ensemble and sub-strategies"""
        all_features = []
        
        # Collect from ensemble's own declaration
        if 'required_features' in ensemble_config:
            all_features.extend(ensemble_config['required_features'])
        
        # Collect from sub-strategies
        for strategy_group in ['baseline_strategies', 'regime_boosters']:
            if strategy_group in ensemble_config:
                sub_features = self._collect_from_strategy_group(ensemble_config[strategy_group])
                all_features.extend(sub_features)
        
        # Remove duplicates while preserving order
        return self._deduplicate_features(all_features)
    
    def _collect_from_strategy_group(self, strategy_group) -> List[FeatureSpec]:
        """Collect features from a group of strategies"""
        features = []
        
        if isinstance(strategy_group, dict):
            # Handle regime_boosters: {regime: [strategies]}
            for regime_strategies in strategy_group.values():
                for strategy_config in regime_strategies:
                    features.extend(self._get_strategy_features(strategy_config))
        elif isinstance(strategy_group, list):
            # Handle baseline_strategies: [strategies]
            for strategy_config in strategy_group:
                features.extend(self._get_strategy_features(strategy_config))
        
        return features
    
    def _get_strategy_features(self, strategy_config: Dict[str, Any]) -> List[FeatureSpec]:
        """Get validated feature requirements for a single strategy"""
        strategy_name = strategy_config['name']
        strategy_params = strategy_config.get('params', {})
        
        # Look up strategy in registry
        strategy_metadata = get_strategy_metadata(strategy_name)
        if not strategy_metadata:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Get strategy's required features and parameterize them
        required_features = strategy_metadata['required_features']
        parameterized_features = []
        
        for feature_spec in required_features:
            # Replace feature spec parameters with actual strategy parameters
            updated_params = self._resolve_feature_params(feature_spec, strategy_params)
            parameterized_spec = FeatureSpec(
                feature_spec.feature_type,
                updated_params,
                feature_spec.output_component
            )
            parameterized_features.append(parameterized_spec)
        
        return parameterized_features
    
    def _resolve_feature_params(self, feature_spec: FeatureSpec, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve feature parameters from strategy parameters"""
        resolved = {}
        
        for param_name, param_value in feature_spec.params.items():
            if isinstance(param_value, str) and param_value.startswith('${'):
                # Template parameter: ${fast_period} -> strategy_params['fast_period']
                param_key = param_value[2:-1]  # Remove ${ and }
                if param_key not in strategy_params:
                    raise ValueError(f"Strategy parameter '{param_key}' not found in {strategy_params}")
                resolved[param_name] = strategy_params[param_key]
            else:
                # Literal parameter
                resolved[param_name] = param_value
        
        return resolved

# Ensemble strategy with automatic feature collection
@strategy(
    name='two_layer_ensemble',
    required_features=[]  # Features collected automatically from sub-strategies
)
def two_layer_ensemble(features: ValidatedFeatures, bar, params):
    # All sub-strategy features guaranteed to be available
    # No manual feature declaration needed
    pass
```

## Implementation Plan - Disruptive Approach

### Week 1: Core Infrastructure + Legacy Purge
**Monday-Tuesday: Delete Legacy Code**
- Delete current feature inference system in topology.py
- Delete parameter mapping systems
- Delete runtime discovery code in state.py
- Delete feature config dict formats

**Wednesday-Friday: Implement New Core**
- Implement FeatureSpec with strict validation
- Implement FeatureRegistry with complete schemas
- Implement ValidatedFeatures container
- Create comprehensive test suite

### Week 2: Strategy Migration + Validation
**Monday-Wednesday: Migrate All Strategies**
- Convert all 36+ strategy types to new system
- Update all parameter names to standard format
- Add strict feature specifications to each strategy
- Fix any configuration mismatches

**Thursday-Friday: End-to-End Validation**
- Test complete grid search with new system
- Validate all 880+ strategy instances work
- Fix any remaining integration issues
- Performance testing and optimization

## Configuration Changes

### Old Format (DELETED)
```yaml
# OLD - Will be completely removed
strategies:
  - type: macd_crossover
    name: macd_crossover_grid
    params:
      fast_ema: [5, 12]    # ❌ Non-standard parameter names
      slow_ema: [20, 26]   # ❌ Inconsistent with feature system
      signal_ema: [7, 9]   # ❌ Will cause validation errors
```

### New Format (REQUIRED)
```yaml
# NEW - Only format supported
strategies:
  - type: macd_crossover
    name: macd_crossover
    params:
      fast_period: [5, 12]     # ✅ Standard parameter names
      slow_period: [20, 26]    # ✅ Matches feature registry
      signal_period: [7, 9]    # ✅ Validated at strategy load
```

## Strategy Migration Examples

### Simple Strategy Migration
```python
# BEFORE: Fragile and unclear
@strategy(
    name='sma_crossover',
    feature_config=['sma'],
    param_feature_mapping={
        'fast_period': 'sma_{fast_period}',
        'slow_period': 'sma_{slow_period}'
    }
)
def sma_crossover(features, bar, params):
    fast_sma = features.get(f"sma_{params['fast_period']}")  # Hope it exists
    slow_sma = features.get(f"sma_{params['slow_period']}")  # Silent failure if not
    if fast_sma is None or slow_sma is None:
        return None

# AFTER: Explicit and validated
@strategy(
    name='sma_crossover',
    required_features=[
        FeatureSpec('sma', {'period': '${fast_period}'}),
        FeatureSpec('sma', {'period': '${slow_period}'})
    ]
)
def sma_crossover(features: ValidatedFeatures, bar, params):
    # Template parameters resolved automatically
    fast_sma = features[f"sma_{params['fast_period']}"]  # Guaranteed to exist
    slow_sma = features[f"sma_{params['slow_period']}"]  # Validation ensures it
    # No error handling needed
```

### Complex Strategy Migration
```python
# BEFORE: Multi-output feature guesswork
@strategy(
    name='macd_crossover',
    feature_config=['macd'],
    param_feature_mapping={'fast_ema': 'macd_{fast_ema}_{slow_ema}_{signal_ema}'}
)
def macd_crossover(features, bar, params):
    macd_key = f"macd_{params['fast_ema']}_{params['slow_ema']}_{params['signal_ema']}"
    macd_data = features.get(macd_key)  # Hope the key format is right
    if macd_data is None:
        return None
    macd_line = macd_data.get('macd')  # Hope the sub-key exists

# AFTER: Explicit component specification
@strategy(
    name='macd_crossover', 
    required_features=[
        FeatureSpec('macd', {
            'fast_period': '${fast_period}',
            'slow_period': '${slow_period}', 
            'signal_period': '${signal_period}'
        }, 'macd'),
        FeatureSpec('macd', {
            'fast_period': '${fast_period}',
            'slow_period': '${slow_period}',
            'signal_period': '${signal_period}'
        }, 'signal')
    ]
)
def macd_crossover(features: ValidatedFeatures, bar, params):
    # Exact feature names guaranteed
    macd_line = features[f"macd_{params['fast_period']}_{params['slow_period']}_{params['signal_period']}_macd"]
    signal_line = features[f"macd_{params['fast_period']}_{params['slow_period']}_{params['signal_period']}_signal"]
    # No guesswork, no error handling needed
```

## Error Handling and Debugging

### Clear Validation Errors
```python
# Feature validation error
FeatureValidationError: Missing required features: ['stochastic_14_3_k', 'stochastic_14_3_d']. 
Available features: ['sma_10', 'sma_20', 'rsi_14', 'macd_12_26_9_macd', ...]

# Parameter validation error  
ValueError: Missing required parameters for stochastic: ['d_period']

# Invalid parameter error
ValueError: Invalid value 0 for parameter k_period (must be 1 <= k_period <= 100)
```

### Development Time Debugging
```python
# Feature spec validation at strategy definition time
@strategy(
    name='bad_strategy',
    required_features=[
        FeatureSpec('unknown_feature', {'period': 10})  # ❌ Fails immediately
    ]
)
def bad_strategy(features, bar, params):
    pass

# Error: ValueError: Unknown feature type: unknown_feature
```

## Benefits of Disruptive Approach

### 1. **Guaranteed Correctness**
- No more silent failures from missing features
- No more parameter name mismatches
- No more guessing feature formats

### 2. **Clear Mental Model**
- One way to declare features
- One way to access features
- One way to handle parameters

### 3. **Excellent Developer Experience**
```python
# Everything is explicit and validated
@strategy(
    name='my_strategy',
    required_features=[
        FeatureSpec('sma', {'period': 20}),           # I need SMA(20)
        FeatureSpec('rsi', {'period': 14}),           # I need RSI(14)  
        FeatureSpec('macd', {                         # I need MACD signal line
            'fast_period': 12, 
            'slow_period': 26,
            'signal_period': 9
        }, 'signal')
    ]
)
def my_strategy(features: ValidatedFeatures, bar, params):
    sma = features['sma_20']                          # Guaranteed to exist
    rsi = features['rsi_14']                          # Guaranteed to exist
    macd_signal = features['macd_12_26_9_signal']     # Guaranteed to exist
    # Write strategy logic without any error handling
```

### 4. **Future-Proof Architecture**
- New developers can't use old patterns (code deleted)
- Clear validation prevents configuration errors
- Ensemble strategies work automatically
- Easy to add new feature types

## Migration Risk Mitigation

### 1. **Comprehensive Testing Before Cutover**
- Test new system with copy of production data
- Validate all 880+ strategy instances
- Performance benchmark vs current system

### 2. **Feature Parity Validation**
- Automated comparison: old system output vs new system output
- Bit-for-bit identical results required before cutover

### 3. **Rollback Plan**
- Keep complete backup of old system
- Can revert entire codebase in 1 hour if critical issues found

This disruptive approach eliminates all legacy patterns and creates a bulletproof foundation for future development.