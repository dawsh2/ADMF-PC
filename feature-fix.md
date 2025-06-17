# Feature System Comprehensive Redesign

## Executive Summary

The current feature inference system suffers from fundamental architectural problems that cause constant development friction. This document outlines a complete redesign that eliminates all major pain points while maintaining backward compatibility and improving robustness.

## Current System Problems

### 1. **Build-time vs Runtime Disconnect**
- Feature inference happens at build-time using different logic than runtime access
- Ensemble strategies can't communicate sub-strategy feature needs to build-time inference
- Parameter mapping templates work differently in different contexts

### 2. **Parameter Name Hell**
```python
# Same feature, 4 different naming conventions:
MACD class:     fast_period, slow_period, signal_period
Config YAML:    fast_ema, slow_ema, signal_ema  
Topology code:  fast, slow, signal
Strategy code:  fast_period, slow_period, signal_period
```

### 3. **Feature Decomposition Complexity**
```python
# Feature system generates: stochastic_14_3
# But strategies need: stochastic_14_3_k and stochastic_14_3_d
# No clear mapping between base feature and decomposed components
```

### 4. **Nested Strategy Blindness**
```python
# Ensemble strategy parameters invisible to feature inference:
ensemble_params = {
    'baseline_strategies': [
        {'name': 'sma_crossover', 'params': {'fast_period': 10}}  # ← Build-time can't see this
    ]
}
```

### 5. **Multiple Sources of Truth**
- Parameter mapping in strategy decorators
- Hardcoded mappings in topology.py
- Feature computation logic in hub.py
- Runtime inference in state.py

## Proposed Solution: Four-Layer Architecture

### Layer 1: Feature Request System (Runtime Discovery)
**Eliminates build-time inference complexity**

```python
class FeatureRequest:
    """Smart feature request that handles discovery and fallbacks"""
    
    def __init__(self, features_dict: Dict[str, Any]):
        self.features = features_dict
        self._cache = {}
    
    def get(self, feature_type: str, **params) -> Any:
        """Request a feature with parameters, handles discovery automatically"""
        # Try exact match first
        exact_name = self._build_feature_name(feature_type, params)
        if exact_name in self.features:
            return self.features[exact_name]
        
        # Try common naming variations
        for variant in self._get_naming_variants(feature_type, params):
            if variant in self.features:
                return self.features[variant]
        
        # Try decomposed features (for multi-output indicators)
        for component in self._get_decomposed_variants(feature_type, params):
            if component in self.features:
                return self.features[component]
        
        return None
    
    def require(self, feature_type: str, **params) -> Any:
        """Like get() but raises clear error if feature not found"""
        result = self.get(feature_type, **params)
        if result is None:
            available = [k for k in self.features.keys() if feature_type in k]
            raise FeatureNotFoundError(
                f"Feature '{feature_type}' with params {params} not found. "
                f"Available {feature_type} features: {available}"
            )
        return result

# Strategy usage becomes bulletproof:
@strategy(name='stochastic_crossover')
def stochastic_crossover(features_dict: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]):
    features = FeatureRequest(features_dict)
    
    # Automatically handles stochastic_14_3_k, stoch_14_3_k, stochastic_k_14_3, etc.
    stoch_k = features.get('stochastic', k_period=14, d_period=3, component='k')
    stoch_d = features.get('stochastic', k_period=14, d_period=3, component='d')
    
    if stoch_k is None or stoch_d is None:
        return None
```

### Layer 2: Centralized Feature Registry
**Single source of truth for all feature metadata**

```python
@dataclass
class FeatureSpec:
    """Complete specification for a feature type"""
    name: str
    computation_func: Callable
    parameter_names: List[str]
    parameter_aliases: Dict[str, str]  # 'fast_ema' -> 'fast_period'
    default_params: Dict[str, Any]
    output_format: str  # Template for feature naming
    output_components: List[str]  # For multi-output features
    description: str
    
    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter aliases to canonical names"""
        normalized = {}
        for key, value in params.items():
            canonical_key = self.parameter_aliases.get(key, key)
            normalized[canonical_key] = value
        return normalized
    
    def generate_feature_name(self, params: Dict[str, Any]) -> str:
        """Generate standardized feature name"""
        norm_params = self.normalize_params(params)
        return self.output_format.format(**norm_params)
    
    def get_component_names(self, params: Dict[str, Any]) -> List[str]:
        """Get all component names for multi-output features"""
        base_name = self.generate_feature_name(params)
        if self.output_components:
            return [f"{base_name}_{comp}" for comp in self.output_components]
        return [base_name]

# Centralized registry
FEATURE_REGISTRY = {
    'stochastic': FeatureSpec(
        name='stochastic',
        computation_func=stochastic_feature,
        parameter_names=['k_period', 'd_period'],
        parameter_aliases={
            'stochastic_k_period': 'k_period',
            'stoch_k_period': 'k_period',
            'k': 'k_period',
            'stochastic_d_period': 'd_period', 
            'stoch_d_period': 'd_period',
            'd': 'd_period'
        },
        default_params={'k_period': 14, 'd_period': 3},
        output_format='stochastic_{k_period}_{d_period}',
        output_components=['k', 'd'],
        description='Stochastic oscillator with %K and %D lines'
    ),
    
    'macd': FeatureSpec(
        name='macd',
        computation_func=macd_feature,
        parameter_names=['fast_period', 'slow_period', 'signal_period'],
        parameter_aliases={
            'fast_ema': 'fast_period',
            'slow_ema': 'slow_period', 
            'signal_ema': 'signal_period',
            'fast': 'fast_period',
            'slow': 'slow_period',
            'signal': 'signal_period'
        },
        default_params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        output_format='macd_{fast_period}_{slow_period}_{signal_period}',
        output_components=['macd', 'signal', 'histogram'],
        description='MACD with signal line and histogram'
    ),
    
    'sma': FeatureSpec(
        name='sma',
        computation_func=sma_feature,
        parameter_names=['period'],
        parameter_aliases={
            'sma_period': 'period',
            'fast_period': 'period',  # Context-dependent
            'slow_period': 'period',  # Context-dependent
            'length': 'period'
        },
        default_params={'period': 20},
        output_format='sma_{period}',
        output_components=[],
        description='Simple moving average'
    )
}
```

### Layer 3: Ensemble Strategy Feature Forwarding
**Allows ensemble strategies to communicate sub-strategy feature needs**

```python
class EnsembleFeatureCollector:
    """Collects feature requirements from nested strategies"""
    
    def __init__(self, registry):
        self.registry = registry
        
    def collect_from_ensemble(self, ensemble_config: Dict[str, Any]) -> Set[str]:
        """Extract all features needed by ensemble and sub-strategies"""
        required_features = set()
        
        # Get ensemble's own features
        if 'feature_config' in ensemble_config:
            required_features.update(ensemble_config['feature_config'])
        
        # Recursively collect from sub-strategies
        for strategy_group in ['baseline_strategies', 'regime_boosters']:
            if strategy_group in ensemble_config:
                sub_features = self._collect_from_strategy_group(ensemble_config[strategy_group])
                required_features.update(sub_features)
                
        return required_features
    
    def _collect_from_strategy_group(self, strategy_group) -> Set[str]:
        """Collect features from a group of strategies"""
        features = set()
        
        if isinstance(strategy_group, dict):
            # Handle regime_boosters: {regime: [strategies]}
            for regime_strategies in strategy_group.values():
                for strategy_config in regime_strategies:
                    features.update(self._get_strategy_features(strategy_config))
        elif isinstance(strategy_group, list):
            # Handle baseline_strategies: [strategies]
            for strategy_config in strategy_group:
                features.update(self._get_strategy_features(strategy_config))
                
        return features
    
    def _get_strategy_features(self, strategy_config: Dict[str, Any]) -> Set[str]:
        """Get feature requirements for a single strategy"""
        strategy_name = strategy_config['name']
        strategy_params = strategy_config.get('params', {})
        
        # Look up strategy in registry to get its feature requirements
        strategy_info = self.registry.get_strategy(strategy_name)
        if not strategy_info:
            return set()
        
        # Get base feature types from strategy
        base_features = strategy_info.get('feature_config', [])
        param_mapping = strategy_info.get('param_feature_mapping', {})
        
        # Generate parameterized features
        parameterized_features = set()
        for param_name, feature_template in param_mapping.items():
            if param_name in strategy_params:
                # Replace template variables with actual values
                feature_name = feature_template.format(**strategy_params)
                parameterized_features.add(feature_name)
        
        return set(base_features) | parameterized_features

# Enhanced strategy decorator
def strategy(name: str, feature_config: List[str] = None, **kwargs):
    """Enhanced strategy decorator with ensemble support"""
    def decorator(func):
        # Register strategy with feature requirements
        metadata = {
            'name': name,
            'feature_config': feature_config or [],
            'param_feature_mapping': kwargs.get('param_feature_mapping', {}),
            'factory': func,
            'is_ensemble': 'ensemble' in name.lower()
        }
        
        # For ensemble strategies, collect sub-strategy features
        if metadata['is_ensemble']:
            metadata['feature_collector'] = EnsembleFeatureCollector.collect_from_ensemble
        
        register_strategy(metadata)
        return func
    return decorator
```

### Layer 4: Unified Parameter Standardization
**Eliminates parameter name inconsistencies**

```python
class ParameterStandardizer:
    """Converts between different parameter naming conventions"""
    
    STANDARD_PATTERNS = {
        # Time-based parameters
        'period': ['period', 'length', 'window', 'lookback'],
        'fast_period': ['fast_period', 'fast_ema', 'fast', 'short_period', 'short'],
        'slow_period': ['slow_period', 'slow_ema', 'slow', 'long_period', 'long'],
        'signal_period': ['signal_period', 'signal_ema', 'signal', 'smooth_period'],
        
        # Threshold parameters  
        'threshold': ['threshold', 'level', 'limit'],
        'upper_threshold': ['upper_threshold', 'overbought', 'upper_limit'],
        'lower_threshold': ['lower_threshold', 'oversold', 'lower_limit'],
        
        # Multiplier parameters
        'multiplier': ['multiplier', 'factor', 'mult', 'std_dev'],
        'atr_multiplier': ['atr_multiplier', 'atr_factor', 'volatility_factor']
    }
    
    @classmethod
    def standardize_params(cls, params: Dict[str, Any], target_strategy: str = None) -> Dict[str, Any]:
        """Convert parameter names to standard conventions"""
        standardized = {}
        
        for param_name, param_value in params.items():
            standard_name = cls._find_standard_name(param_name)
            standardized[standard_name] = param_value
        
        return standardized
    
    @classmethod
    def _find_standard_name(cls, param_name: str) -> str:
        """Find the standard name for a parameter"""
        for standard, variants in cls.STANDARD_PATTERNS.items():
            if param_name in variants:
                return standard
        return param_name  # Return original if no standard found

# Usage in strategies
@strategy(name='macd_crossover', feature_config=['macd'])
def macd_crossover(features_dict: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]):
    # Automatically handles fast_ema->fast_period, slow_ema->slow_period, etc.
    std_params = ParameterStandardizer.standardize_params(params)
    
    features = FeatureRequest(features_dict)
    macd_data = features.require('macd', **std_params)
    
    # Rest of strategy logic...
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. **Create FeatureRequest class** with smart discovery
2. **Build centralized FeatureRegistry** with all current features
3. **Implement ParameterStandardizer** for name normalization
4. **Create comprehensive test suite** for new components

### Phase 2: Strategy Migration (Week 2)  
1. **Migrate 5 simple strategies** to new system (sma_crossover, rsi_crossover, etc.)
2. **Migrate 5 complex strategies** with multi-output features (macd_crossover, bollinger_bands, etc.)
3. **Validate all migrated strategies** work with existing configs
4. **Create migration guide** for remaining strategies

### Phase 3: Ensemble Support (Week 3)
1. **Implement EnsembleFeatureCollector** 
2. **Migrate ensemble strategies** to use feature forwarding
3. **Test nested strategy feature discovery**
4. **Validate ensemble strategies** work end-to-end

### Phase 4: System Integration (Week 4)
1. **Integrate new system** with topology builder
2. **Update ComponentState** to use FeatureRequest
3. **Deprecate old inference system** (keep for backward compatibility)
4. **Performance optimization** and caching

### Phase 5: Cleanup and Documentation (Week 5)
1. **Complete documentation** with examples
2. **Migration scripts** for remaining strategies  
3. **Remove deprecated code** after validation
4. **Performance benchmarking** vs old system

## Backward Compatibility Strategy

### Gradual Migration Approach
```python
# New system supports old-style strategies during transition
class LegacyFeatureAdapter:
    """Adapter to make old strategies work with new FeatureRequest"""
    
    def __init__(self, new_features: FeatureRequest):
        self.new_features = new_features
        
    def get(self, feature_name: str):
        """Legacy dict-style access"""
        # Try direct lookup first
        if feature_name in self.new_features.features:
            return self.new_features.features[feature_name]
        
        # Try to parse legacy feature names
        parsed = self._parse_legacy_name(feature_name)
        if parsed:
            return self.new_features.get(**parsed)
        
        return None

# Strategies can be migrated incrementally
@strategy(name='sma_crossover', feature_config=['sma'])
def sma_crossover(features_dict, bar, params):
    # NEW: Use FeatureRequest directly
    if isinstance(features_dict, dict):
        features = FeatureRequest(features_dict)
        fast_sma = features.get('sma', period=params['fast_period'])
        slow_sma = features.get('sma', period=params['slow_period'])
    else:
        # OLD: Legacy dict access (during transition)
        fast_sma = features_dict.get(f"sma_{params['fast_period']}")
        slow_sma = features_dict.get(f"sma_{params['slow_period']}")
```

## Expected Benefits

### 1. **Eliminate Feature Inference Pain**
- No more build-time vs runtime mismatches
- No more missing features for ensemble strategies
- No more parameter name conflicts

### 2. **Bulletproof Strategy Development**
```python
# Old way: Fragile and error-prone
def old_strategy(features, bar, params):
    # Guess the exact feature name
    macd = features.get('macd_12_26_9_macd')  # Hope this exists
    if macd is None:
        return None  # Silent failure

# New way: Robust and self-documenting  
def new_strategy(features_dict, bar, params):
    features = FeatureRequest(features_dict)
    macd = features.require('macd', fast_period=12, slow_period=26, signal_period=9, component='macd')
    # Clear error if feature not found, automatic parameter handling
```

### 3. **Ensemble Strategy Support**
```python
# Old way: Ensemble strategies can't communicate feature needs
@strategy(feature_config=['sma'])  # Can't see sub-strategy needs
def ensemble_strategy(features, bar, params):
    # Sub-strategies fail because their features weren't inferred
    
# New way: Automatic feature forwarding
@strategy(name='two_layer_ensemble', feature_config=['sma'])
def ensemble_strategy(features_dict, bar, params):
    # System automatically discovers all sub-strategy feature needs
    # All required features are guaranteed to be available
```

### 4. **Self-Healing System**
```python
# Automatic feature discovery with fallbacks
features = FeatureRequest(features_dict)

# Tries: stochastic_14_3_k, stoch_14_3_k, stochastic_k_14_3, etc.
stoch_k = features.get('stochastic', k_period=14, d_period=3, component='k')

# Clear error messages when features genuinely missing
try:
    macd = features.require('macd', fast_period=12, slow_period=26, signal_period=9)
except FeatureNotFoundError as e:
    print(f"Strategy macd_crossover failed: {e}")
    # Error shows exactly what was requested and what's available
```

## Risk Mitigation

### 1. **Comprehensive Testing**
- Unit tests for every component
- Integration tests with real market data
- Performance benchmarks vs current system
- Backward compatibility validation

### 2. **Gradual Rollout**
- Feature flag to enable/disable new system
- Side-by-side validation during transition
- Rollback plan if issues discovered

### 3. **Performance Monitoring**
- Benchmark feature discovery performance
- Cache frequently accessed features
- Monitor memory usage vs current system

## Success Metrics

### Immediate (1 month)
- [ ] Zero feature inference errors in development
- [ ] All ensemble strategies work without manual feature declaration
- [ ] 50% reduction in strategy development time

### Medium-term (3 months)  
- [ ] All strategies migrated to new system
- [ ] Zero parameter name conflicts
- [ ] Documentation shows clear, simple examples

### Long-term (6 months)
- [ ] New developers can create strategies without feature system knowledge
- [ ] System supports complex ensemble strategies effortlessly
- [ ] Feature system is source of pride, not pain

## Implementation Details

### Directory Structure
```
src/strategy/features/
├── __init__.py
├── registry.py          # FeatureRegistry with all feature specs
├── request.py           # FeatureRequest class
├── standardizer.py      # ParameterStandardizer
├── collector.py         # EnsembleFeatureCollector
├── specs/              # Individual feature specifications
│   ├── technical.py    # SMA, EMA, RSI, etc.
│   ├── oscillators.py  # Stochastic, MACD, etc.
│   └── volatility.py   # ATR, Bollinger Bands, etc.
└── migration/          # Migration utilities and adapters
    ├── adapter.py      # LegacyFeatureAdapter
    └── validator.py    # Migration validation tools
```

### Configuration Examples

#### Simple Strategy (No Changes Required)
```yaml
# YAML stays exactly the same
strategies:
  - type: sma_crossover
    name: sma_fast
    params:
      fast_period: 10
      slow_period: 20
```

#### Complex Ensemble Strategy (Automatic Feature Discovery)
```yaml
# System automatically discovers all nested feature needs
strategies:
  - type: two_layer_ensemble  
    name: adaptive_ensemble
    params:
      baseline_strategies:
        - name: sma_crossover
          params: {fast_period: 10, slow_period: 20}  # SMA features auto-discovered
        - name: rsi_crossover  
          params: {rsi_period: 14}                     # RSI features auto-discovered
      regime_boosters:
        bull_trending:
          - name: macd_crossover
            params: {fast_ema: 12, slow_ema: 26}       # MACD features auto-discovered
```

This comprehensive redesign eliminates every major pain point while providing a foundation for robust, maintainable strategy development going forward.