# Feature Dependencies and Composite Features Standard

## Problem Statement

The current feature system doesn't support composite features (features computed from other features) as first-class citizens. This leads to:
- Manual workarounds in filter expressions
- Features like `atr_sma_50` (SMA of ATR) not being automatically discovered
- Violation of the "automatic feature inference" principle

## Proposed Standard

### 1. Feature Dependency Declaration

Features should declare their dependencies in the FeatureSpec:

```python
# In feature_spec.py
@dataclass
class FeatureSpec:
    feature_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    output_component: Optional[str] = None
    # NEW: Dependencies on other features
    dependencies: List['FeatureSpec'] = field(default_factory=list)
```

### 2. Composite Feature Types

Define standard composite feature types in the registry:

```python
FEATURE_REGISTRY = {
    # ... existing features ...
    
    # Composite features
    'sma_of': {
        'class': SMAOf,
        'required_params': ['source', 'period'],
        'output_type': 'single',
        'composite': True  # Mark as composite
    },
    'ema_of': {
        'class': EMAOf,
        'required_params': ['source', 'period'],
        'output_type': 'single',
        'composite': True
    }
}
```

### 3. Feature Discovery Enhancement

Enhance feature discovery to handle composite patterns:

```python
def _create_feature_spec_from_name(self, feature_name: str) -> Optional[FeatureSpec]:
    """
    Create FeatureSpec from feature name, including composite features.
    
    Examples:
    - "atr_sma_50" -> SMA of ATR with period 50
    - "rsi_ema_20" -> EMA of RSI with period 20
    - "volume_sma_20" -> SMA of volume with period 20
    """
    parts = feature_name.split('_')
    
    # Handle composite pattern: {source}_{ma_type}_{period}
    if len(parts) == 3 and parts[1] in ['sma', 'ema', 'wma']:
        source = parts[0]
        ma_type = parts[1]
        period = int(parts[2]) if parts[2].isdigit() else None
        
        if period:
            # Create composite feature spec with dependency
            if source in ['atr', 'rsi', 'obv', 'adx']:  # Known indicators
                return FeatureSpec(
                    feature_type=f'{ma_type}_of',
                    params={'source': source, 'period': period},
                    dependencies=[
                        FeatureSpec(feature_type=source, params={'period': 14})  # Default period
                    ]
                )
```

### 4. FeatureHub Enhancement

Enhance FeatureHub to handle feature dependencies:

```python
class FeatureHub:
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]]) -> None:
        """Configure features with dependency resolution."""
        # First pass: identify all dependencies
        dependency_graph = self._build_dependency_graph(feature_configs)
        
        # Topological sort to determine computation order
        computation_order = self._topological_sort(dependency_graph)
        
        # Configure features in dependency order
        for feature_name in computation_order:
            config = feature_configs[feature_name]
            self._configure_single_feature(feature_name, config)
    
    def _configure_single_feature(self, name: str, config: Dict[str, Any]) -> None:
        """Configure a single feature, potentially composite."""
        feature_type = config['type']
        
        if self._is_composite_feature(feature_type):
            # Handle composite feature
            source = config.get('source')
            if source in self.features[symbol]:
                # Source feature exists, create composite
                feature = self._create_composite_feature(name, config)
            else:
                raise ValueError(f"Dependency {source} not configured for composite feature {name}")
        else:
            # Regular feature
            feature = self._create_feature(name, config)
```

### 5. Composite Feature Implementation

Standard implementation for composite features:

```python
class SMAOf:
    """SMA of another feature (composite feature)."""
    
    def __init__(self, source: str, period: int = 20, name: str = None):
        self._state = FeatureState(name or f"{source}_sma_{period}")
        self.source = source
        self.period = period
        self._buffer = deque(maxlen=period)
        self._sum = 0.0
    
    @property
    def dependencies(self) -> List[str]:
        """List of feature dependencies."""
        return [self.source]
    
    def update(self, features: Dict[str, Any], **kwargs) -> Optional[float]:
        """Update using source feature value."""
        source_value = features.get(self.source)
        if source_value is None:
            return None
            
        # Standard SMA logic on source feature
        if len(self._buffer) == self.period:
            self._sum -= self._buffer[0]
        
        self._buffer.append(source_value)
        self._sum += source_value
        
        if len(self._buffer) == self.period:
            self._state.set_value(self._sum / self.period)
        
        return self._state.value
```

### 6. Filter Expression Enhancement

Allow both direct access and function syntax:

```yaml
# Both should work after implementation
filter: "atr_sma_50 > 0.5"  # Direct access (auto-discovered)
filter: "sma_of('atr', 50) > 0.5"  # Function syntax
```

### 7. Configuration Examples

```yaml
# Automatic discovery from filter
strategy:
  - keltner_bands:
      period: 20
      multiplier: 2.0
      filter: "atr_sma_50 > 0.5"  # Automatically creates atr_14 and atr_sma_50

# Resulting feature configs (auto-generated):
feature_configs:
  atr_14:
    type: atr
    period: 14
  atr_sma_50:
    type: sma_of
    source: atr_14
    period: 50
```

## Implementation Plan

### Phase 1: Core Infrastructure (Priority: High)
1. Add `dependencies` field to FeatureSpec
2. Implement dependency graph builder in FeatureHub
3. Add topological sort for computation order
4. Update feature discovery to handle composite patterns

### Phase 2: Composite Features (Priority: High)
1. Implement base composite feature classes (SMAOf, EMAOf)
2. Update FEATURE_REGISTRY with composite types
3. Enhance FeatureHub to handle composite feature updates
4. Add composite feature tests

### Phase 3: Filter Integration (Priority: Medium)
1. Update filter expression parser to support function syntax
2. Add composite feature functions to filter context
3. Ensure automatic discovery from filters
4. Update documentation

### Phase 4: Extended Composites (Priority: Low)
1. Add more composite types (StdDevOf, RatioOf, DiffOf)
2. Support multi-level composites (SMA of SMA of ATR)
3. Add performance optimizations for composite chains

## Benefits

1. **No Manual Configuration**: Filters can use `atr_sma_50` without manual feature configs
2. **Consistent Naming**: `{source}_{operation}_{params}` pattern
3. **Automatic Discovery**: System infers all dependencies
4. **Type Safety**: Dependencies declared in FeatureSpec
5. **Performance**: Dependency graph ensures optimal computation order

## Migration Strategy

1. **Backward Compatibility**: Existing features continue to work
2. **Gradual Adoption**: New composite features can be added incrementally
3. **Clear Errors**: Helpful error messages for missing dependencies
4. **Documentation**: Update strategy-interface.md with composite feature patterns

## Example Use Cases

```python
# Various composite features that would be supported
features = {
    'atr_sma_50': SMAOf('atr', 50),           # SMA of ATR
    'volume_ema_20': EMAOf('volume', 20),     # EMA of Volume  
    'rsi_sma_10': SMAOf('rsi', 10),          # SMA of RSI
    'macd_signal_sma_5': SMAOf('macd_signal', 5),  # SMA of MACD signal line
}
```

This standard ensures that composite features are first-class citizens in the feature system, maintaining the principle of automatic feature inference while supporting complex technical analysis patterns.