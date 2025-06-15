# Coordinator Module

The coordinator module orchestrates the ADMF-PC system, building execution topologies from declarative patterns and managing workflow execution.

## Core Components

### Topology Builder (`topology.py`)

The topology builder constructs execution topologies from YAML patterns, automatically inferring requirements and wiring components together.

#### Key Features

1. **Automatic Feature Inference**: Infers required features from strategy and classifier configurations
2. **Component Discovery**: Uses the discovery registry to find strategies and classifiers
3. **Container Creation**: Builds container hierarchies based on patterns
4. **Event Wiring**: Sets up event subscriptions between components

#### Integration Flow

```
Configuration → Parameter Expansion → Feature Inference → Container Creation → Event Wiring
```

### Feature Inference System

The topology builder automatically determines what features are needed based on strategy and classifier parameters:

```python
# User configuration (simple)
strategies:
  - type: sma_crossover
    params:
      fast_period: 10
      slow_period: 50

classifiers:
  - type: trend_classifier
    params:
      sma_period: 20
      lookback: 50

# Topology builder infers
required_features: ['sma_10', 'sma_20', 'sma_50']

# And generates feature configs
feature_configs:
  sma_10: {feature: 'sma', period: 10}
  sma_20: {feature: 'sma', period: 20}
  sma_50: {feature: 'sma', period: 50}
```

#### How Feature Inference Works

1. **Component Registration**: Strategies use `@strategy` decorator and classifiers use `@classifier` decorator with feature metadata
2. **Parameter Analysis**: Builder examines component parameters
3. **Feature Generation**: Creates feature configs based on parameter values
4. **Context Injection**: Adds features to context for container creation

### Best Practices for Feature Inference

#### 1. Standard Parameter Naming

Use consistent parameter names for automatic inference:

```python
# Good - standard naming patterns
params:
  sma_period: 20        # → infers sma_20
  fast_period: 10       # → infers sma_10 (for SMA strategies)
  rsi_period: 14        # → infers rsi_14
  bollinger_period: 20  # → infers bollinger_bands_20_2.0_*
  atr_period: 14        # → infers atr_14

# Avoid - non-standard names
params:
  my_average: 20       # Won't be inferred correctly
  window_size: 10      # Not recognized
```

#### 2. Strategy Feature Declaration

Strategies should use the simplified list format:

```python
@strategy(
    name='trend_following',
    feature_config=['sma', 'atr', 'adx']  # Simple list
)
def trend_following(features, bar, params):
    # Topology builder will infer specific features from params
    pass
```

#### 3. Classifier Feature Declaration

Classifiers follow the same pattern:

```python
@classifier(
    name='volatility_regime',
    feature_config=['atr', 'bollinger_bands', 'volatility']
)
def volatility_regime(features, params):
    # Topology builder will infer specific features from params
    pass
```

#### 4. Compound Feature Handling

The builder handles compound features correctly:

```python
# Configuration
params:
  bollinger_period: 20
  bollinger_std_dev: 2.0

# Inferred features
- bollinger_bands_20_2.0_upper
- bollinger_bands_20_2.0_middle
- bollinger_bands_20_2.0_lower
```

### Pattern-Based Topology Building

Topologies are built from declarative patterns in YAML:

```yaml
# Pattern definition
signal_generation:
  containers:
    - name: root
      type: coordinator
      config:
        role: orchestrator
        
    - name: data_manager
      type: data_streaming
      parent: root
      
    - name: feature_hub_replay
      type: feature_hub_replay
      parent: root
      
    - name: strategy_executor
      type: strategy_execution
      parent: root
      config:
        feature_hub_name: feature_hub_replay
```

### Component Discovery Integration

The topology builder integrates with the discovery system:

```python
# 1. Import component modules to trigger registration
for strategy_config in strategies:
    strategy_type = strategy_config.get('type')
    # Import attempts trigger @strategy decorators
    
for classifier_config in classifiers:
    classifier_type = classifier_config.get('type')
    # Import attempts trigger @classifier decorators
    
# 2. Query registry for component info
strategy_info = registry.get_component(strategy_type)
classifier_info = registry.get_component(classifier_type)

# 3. Use metadata for inference
strategy_features = infer_from_metadata(strategy_info.metadata, params)
classifier_features = infer_from_metadata(classifier_info.metadata, params)
```

### Parameter Expansion

The builder supports parameter list expansion for grid search:

```python
# Input configuration
strategies:
  - type: sma_crossover
    params:
      fast_period: [5, 10, 20]    # List of values
      slow_period: [50, 100, 200]

classifiers:
  - type: trend_classifier
    params:
      lookback: [20, 50, 100]
      threshold: [0.6, 0.7, 0.8]

# Expanded to multiple configurations
strategies: 9 combinations (3x3)
classifiers: 9 combinations (3x3)
```

### Event Subscription Setup

The topology builder sets up event flows based on topology mode:

```python
# Signal generation mode
- Strategy containers → SIGNAL events → Root bus
- Classifier containers → CLASSIFICATION events → Root bus
- Root bus → MultiStrategyTracer (for storage)
- Root bus → Portfolio containers

# Backtest mode  
- Strategy containers → SIGNAL events → Portfolio containers
- Classifier containers → CLASSIFICATION events → Strategy containers
- Portfolio containers → ORDER events → Execution
- Execution → FILL events → Portfolio containers
```

## Configuration Best Practices

### 1. Keep Configurations Simple

Let the topology builder infer as much as possible:

```yaml
# Good - minimal configuration
strategies:
  - type: momentum_breakout
    params:
      sma_period: 20
      rsi_threshold: 70

classifiers:
  - type: market_regime
    params:
      lookback_period: 50

# Avoid - over-specification
strategies:
  - type: momentum_breakout
    params:
      sma_period: 20
      rsi_threshold: 70
    features:  # Let builder infer this!
      - sma_20
      - rsi_14
```

### 2. Use Standard Topology Patterns

Leverage predefined patterns instead of custom topologies:

```python
# Good - use standard pattern
topology_def = {
    'mode': 'signal_generation',
    'config': config,
    'tracing_config': {...}
}

# Avoid - custom topology building
# (unless you have specific requirements)
```

### 3. Enable Tracing for Debugging

Use tracing to understand topology behavior:

```yaml
execution:
  enable_event_tracing: true
  trace_settings:
    use_sparse_storage: true
    trace_dir: ./traces
```

## Troubleshooting

### Features Not Being Inferred

1. **Check Component Registration**:
   ```python
   # Verify strategy has @strategy decorator
   # Verify classifier has @classifier decorator
   # Check that component modules are imported
   ```

2. **Verify Parameter Names**:
   ```python
   # Use standard names: period, fast_period, slow_period
   # Not: window, length, span
   ```

3. **Enable Debug Logging**:
   ```python
   import logging
   logging.getLogger('src.core.coordinator').setLevel(logging.DEBUG)
   ```

### Components Not Found

1. **Ensure Module Import**:
   ```python
   # The topology builder must import the module
   # Check module mappings in topology.py
   ```

2. **Verify Decorator**:
   ```python
   @strategy(name='my_strategy', feature_config=[...])
   def my_strategy(...):
   
   @classifier(name='my_classifier', feature_config=[...])
   def my_classifier(...):
   ```

### Feature Mismatch Errors

1. **Check Feature Naming**:
   ```python
   # Feature access must match generated names
   sma = features.get(f'sma_{period}')  # Correct
   sma = features.get('sma')  # Wrong
   ```

2. **Verify Feature Generation**:
   ```python
   # Log inferred features
   logger.info(f"Inferred features: {required_features}")
   ```

## Advanced Topics

### Custom Feature Inference

Override default inference for special cases:

```python
def _infer_custom_features(self, component_type, params):
    if component_type == 'my_complex_strategy':
        # Custom inference logic for strategies
        return ['custom_feature_1', 'custom_feature_2']
    elif component_type == 'my_complex_classifier':
        # Custom inference logic for classifiers
        return ['regime_feature_1', 'regime_feature_2']
    return None
```

### Dynamic Module Loading

Add new component modules without modifying topology.py:

```python
# In your strategy module's __init__.py
from .my_strategies import *  # Force registration

# In your classifier module's __init__.py
from .my_classifiers import *  # Force registration

# Components will be discovered automatically
```

### Multi-Phase Workflows

The coordinator supports complex multi-phase workflows:

```python
workflow = {
    'phases': [
        {'name': 'optimization', 'topology': 'parameter_optimization'},
        {'name': 'validation', 'topology': 'walk_forward_validation'},
        {'name': 'production', 'topology': 'signal_generation'}
    ]
}
```

## Integration Points

- **Discovery System**: Finds strategies and classifiers via decorators
- **Container Factory**: Creates containers from patterns
- **Event System**: Wires up event flows
- **Feature Hub**: Provides feature computation
- **Storage System**: Captures signals and events

## Future Enhancements

1. **Improved Feature Inference**: Move parameter mappings to component metadata
2. **Dynamic Pattern Loading**: Load patterns from external sources
3. **Validation Framework**: Validate topologies before execution
4. **Visual Topology Builder**: GUI for topology construction
5. **Performance Optimization**: Parallel container creation