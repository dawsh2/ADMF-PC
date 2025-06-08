# Signal Generation and Replay Approaches

## Overview

This document outlines different approaches for handling signal generation and replay in the ADMF-PC system, particularly focusing on regime-aware strategies and efficient data flow patterns.

## Event Flow Architecture

### Proposed Routing Pattern

```
BARS → BAR + FEATURES → (Filtered Features per Classifier + subscribed strategies) → BAR + Filtered Features + Classifier → Filter Classifier and Features → Strategy
```

### Detailed Flow Breakdown

1. **BAR Events**
   - Raw market data events
   - Contains: timestamp, OHLCV, volume

2. **Feature Enrichment**
   - BAR events trigger feature calculation
   - Features attached to create composite events

3. **Classifier-Specific Filtering**
   - Each classifier subscribes to specific features
   - Strategies subscribe to specific classifiers
   - Filtering prevents unnecessary data flow

4. **Classification Layer**
   - Classifiers add their output to the event
   - Creates: BAR + Filtered Features + Classification

5. **Strategy-Specific Filtering**
   - Final filtering before strategy receives event
   - Strategy only gets relevant classifiers and features

### Implementation with ADMF-PC FilterRoute

The existing `src/core/routing/filter.py` provides a sophisticated filtering infrastructure that can be leveraged for this pattern:

#### 1. Feature Filtering Route
```python
# Configuration for feature filtering to classifiers
feature_filter_config = {
    'name': 'feature_to_classifier_filter',
    'type': 'filter',
    'source': 'feature_calculator',  # Container calculating features
    'filter_field': 'payload.features',
    'event_types': ['FEATURES', 'BAR_WITH_FEATURES']
}

# Create filter route
feature_filter = FilterRoute('feature_filter', feature_filter_config)

# Register classifier requirements
feature_filter.register_requirements(
    target_id='trend_classifier',
    required_keys=['sma_20', 'sma_50', 'volume_ratio'],
    conditions=[lambda e: e.payload.get('volume', 0) > 100000]  # Min volume
)

feature_filter.register_requirements(
    target_id='volatility_classifier',
    required_keys=['atr_14', 'bb_width', 'volume'],
    conditions=[lambda e: e.timestamp.hour >= 9 and e.timestamp.hour < 16]  # Market hours
)

feature_filter.register_requirements(
    target_id='momentum_classifier',
    required_keys=['rsi_14', 'macd', 'volume_trend'],
    transform=lambda e: transform_for_momentum(e)  # Custom transform
)
```

#### 2. Classifier Subscription Pattern
```python
class ClassifierContainer:
    """Container that receives filtered features and adds classification."""
    
    def __init__(self, container_id: str, classifier_type: str):
        self.container_id = container_id
        self.classifier_type = classifier_type
        self.event_bus = EventBus()
        
        # Subscribe to filtered features
        self.event_bus.subscribe('BAR_WITH_FEATURES', self.on_features)
    
    def on_features(self, event: Event):
        """Process filtered features and emit classified event."""
        # Features are already filtered to what we need
        features = event.payload.get('features', {})
        
        # Classify based on features
        classification = self._classify(features)
        
        # Create enriched event with classification
        classified_event = Event(
            event_type='CLASSIFIED_BAR',
            payload={
                **event.payload,  # All original data
                'classification': {
                    self.classifier_type: classification,
                    'confidence': self.confidence,
                    'features_used': list(features.keys())
                }
            },
            correlation_id=event.correlation_id,
            metadata={**event.metadata, 'classifier': self.classifier_type}
        )
        
        self.event_bus.publish(classified_event)
```

#### 3. Strategy-Level Filtering
```python
# Configuration for routing classified events to strategies
strategy_filter_config = {
    'name': 'classifier_to_strategy_filter',
    'type': 'filter',
    'source': 'event_aggregator',  # Aggregates all classifier outputs
    'filter_field': 'payload.classification',
    'event_types': ['CLASSIFIED_BAR']
}

strategy_filter = FilterRoute('strategy_filter', strategy_filter_config)

# Register strategy requirements
strategy_filter.register_requirements(
    target_id='trend_following_strategy',
    required_keys=['trend_classifier'],  # Only need trend classification
    conditions=[
        lambda e: e.payload['classification'].get('trend_classifier') is not None,
        lambda e: e.payload['classification']['trend_classifier']['confidence'] > 0.7
    ]
)

strategy_filter.register_requirements(
    target_id='mean_reversion_strategy',
    required_keys=['volatility_classifier', 'momentum_classifier'],
    conditions=[
        lambda e: all(
            clf in e.payload['classification'] 
            for clf in ['volatility_classifier', 'momentum_classifier']
        )
    ]
)

strategy_filter.register_requirements(
    target_id='regime_aware_strategy',
    # This strategy needs all classifiers
    required_keys=['trend_classifier', 'volatility_classifier', 'momentum_classifier'],
    transform=lambda e: create_regime_context(e)
)
```

#### 4. Event Aggregation Pattern
```python
class EventAggregator:
    """
    Aggregates multiple classifier outputs into single event.
    
    Waits for all required classifiers before emitting composite event.
    """
    
    def __init__(self, required_classifiers: List[str], timeout_ms: int = 100):
        self.required_classifiers = set(required_classifiers)
        self.timeout_ms = timeout_ms
        self.pending_events = {}  # correlation_id -> {classifier -> event}
        
    def on_classified_event(self, event: Event):
        """Aggregate classifier outputs."""
        correlation_id = event.correlation_id
        classifier = event.metadata.get('classifier')
        
        if correlation_id not in self.pending_events:
            self.pending_events[correlation_id] = {}
            # Set timeout for this correlation
            self._set_timeout(correlation_id)
        
        # Store this classifier's output
        self.pending_events[correlation_id][classifier] = event
        
        # Check if we have all required classifiers
        if set(self.pending_events[correlation_id].keys()) >= self.required_classifiers:
            self._emit_aggregated_event(correlation_id)
    
    def _emit_aggregated_event(self, correlation_id: str):
        """Emit event with all classifier outputs."""
        events = self.pending_events.pop(correlation_id)
        
        # Merge all classifications
        aggregated_classification = {}
        aggregated_features = {}
        
        for classifier, event in events.items():
            classification = event.payload.get('classification', {})
            aggregated_classification.update(classification)
            
            # Also merge features if present
            features = event.payload.get('features', {})
            aggregated_features.update(features)
        
        # Use first event as base (they should have same bar data)
        base_event = next(iter(events.values()))
        
        aggregated_event = Event(
            event_type='AGGREGATED_CLASSIFICATION',
            payload={
                **base_event.payload,
                'classification': aggregated_classification,
                'features': aggregated_features,
                'classifiers_included': list(events.keys())
            },
            correlation_id=correlation_id,
            metadata={'aggregated': True, 'classifier_count': len(events)}
        )
        
        self.event_bus.publish(aggregated_event)
```

#### 5. Configuration-Driven Routing Setup
```yaml
# routing_config.yaml
routes:
  # Feature calculation and initial filtering
  - name: feature_calculator_route
    type: pipeline
    containers: [data_source, feature_calculator]
    
  # Filter features to classifiers
  - name: feature_to_classifier_filter
    type: filter
    source: feature_calculator
    filter_field: payload.features
    event_types: [FEATURES]
    targets:
      trend_classifier:
        required_features: [sma_20, sma_50, volume_ratio]
        conditions:
          min_volume: 100000
      volatility_classifier:
        required_features: [atr_14, bb_width]
        conditions:
          market_hours_only: true
      momentum_classifier:
        required_features: [rsi_14, macd, volume_trend]
        
  # Aggregate classifier outputs
  - name: classifier_aggregation
    type: aggregator
    sources: [trend_classifier, volatility_classifier, momentum_classifier]
    target: event_aggregator
    timeout_ms: 100
    
  # Filter aggregated events to strategies
  - name: strategy_routing_filter
    type: filter
    source: event_aggregator
    filter_field: payload
    event_types: [AGGREGATED_CLASSIFICATION]
    targets:
      trend_following_strategy:
        required_classifiers: [trend_classifier]
        min_confidence: 0.7
      mean_reversion_strategy:
        required_classifiers: [volatility_classifier, momentum_classifier]
      regime_aware_strategy:
        required_classifiers: all  # Gets everything
```

#### 6. Runtime Filter Registration
```python
class DynamicFilterManager:
    """Manages dynamic filter registration based on strategy needs."""
    
    def __init__(self, filter_route: FilterRoute):
        self.filter_route = filter_route
        self.registered_strategies = {}
        
    def register_strategy(self, strategy_id: str, requirements: Dict[str, Any]):
        """
        Dynamically register strategy requirements.
        
        Args:
            strategy_id: Strategy identifier
            requirements: Dict with:
                - features: List of required features
                - classifiers: List of required classifiers
                - conditions: Optional conditions
        """
        # Build required keys from features and classifiers
        required_keys = []
        
        if 'features' in requirements:
            required_keys.extend(requirements['features'])
            
        if 'classifiers' in requirements:
            required_keys.extend([
                f"classification.{clf}" for clf in requirements['classifiers']
            ])
        
        # Build condition functions
        conditions = []
        
        if requirements.get('min_confidence'):
            min_conf = requirements['min_confidence']
            conditions.append(
                lambda e: all(
                    e.payload.get('classification', {}).get(clf, {}).get('confidence', 0) >= min_conf
                    for clf in requirements.get('classifiers', [])
                )
            )
        
        if requirements.get('market_hours_only'):
            conditions.append(
                lambda e: 9 <= e.timestamp.hour < 16
            )
        
        # Register with filter
        self.filter_route.register_requirements(
            target_id=strategy_id,
            required_keys=required_keys,
            conditions=conditions
        )
        
        self.registered_strategies[strategy_id] = requirements
        
    def unregister_strategy(self, strategy_id: str):
        """Remove strategy from filtering."""
        if strategy_id in self.filter_route.requirements:
            del self.filter_route.requirements[strategy_id]
            del self.registered_strategies[strategy_id]
```

### Benefits of This Architecture

1. **Efficient Data Flow**
   - Only required features flow to each classifier
   - Only relevant classifications flow to each strategy
   - Reduces memory and processing overhead

2. **Dynamic Configuration**
   - Strategies can register/unregister at runtime
   - Filter requirements can be updated without restarts
   - New classifiers can be added without touching existing code

3. **Clear Separation of Concerns**
   - Feature calculation is separate from classification
   - Classification is separate from strategy logic
   - Routing logic is externalized to configuration

4. **Composable and Testable**
   - Each component can be tested in isolation
   - Mock classifiers can be injected for testing
   - Filter logic can be unit tested

5. **Performance Optimization**
   - Conditional filtering reduces unnecessary computation
   - Aggregation with timeout prevents hanging on missing data
   - Parallel processing of independent classifiers

### Shared State Abstraction

To optimize memory and computation, shared state components like lookback windows should be abstracted into generic, reusable components:

#### 1. Generic Lookback Window Component
```python
from collections import deque
from typing import TypeVar, Generic, List, Optional, Callable
import numpy as np

T = TypeVar('T')

class LookbackWindow(Generic[T]):
    """
    Generic lookback window that can be shared across features.
    
    This component maintains a sliding window of data that multiple
    features can reference without duplication.
    """
    
    def __init__(self, window_size: int, dtype: type = float):
        self.window_size = window_size
        self.dtype = dtype
        self.data = deque(maxlen=window_size)
        self._array_cache: Optional[np.ndarray] = None
        self._cache_valid = False
        
    def append(self, value: T) -> None:
        """Add new value and invalidate cache."""
        self.data.append(value)
        self._cache_valid = False
        
    def as_array(self) -> np.ndarray:
        """Get data as numpy array with caching."""
        if not self._cache_valid:
            self._array_cache = np.array(self.data, dtype=self.dtype)
            self._cache_valid = True
        return self._array_cache
    
    def is_ready(self) -> bool:
        """Check if window has enough data."""
        return len(self.data) == self.window_size
    
    def apply(self, func: Callable[[np.ndarray], Any]) -> Any:
        """Apply function to window data."""
        if not self.is_ready():
            return None
        return func(self.as_array())
```

#### 2. Shared State Manager
```python
class SharedStateManager:
    """
    Manages shared state components across features.
    
    This prevents duplicate computation and memory usage for common
    data structures like price/volume windows.
    """
    
    def __init__(self):
        self.lookback_windows: Dict[str, LookbackWindow] = {}
        self.computed_values: Dict[str, Any] = {}
        self.computation_graph: Dict[str, List[str]] = {}  # Dependencies
        
    def register_lookback(self, name: str, window_size: int, 
                         source_field: str = 'close') -> None:
        """Register a shared lookback window."""
        key = f"{name}_{window_size}_{source_field}"
        if key not in self.lookback_windows:
            self.lookback_windows[key] = LookbackWindow(window_size)
            logger.debug(f"Registered shared lookback: {key}")
    
    def update_lookbacks(self, bar_data: Dict[str, float]) -> None:
        """Update all lookback windows with new bar data."""
        for key, window in self.lookback_windows.items():
            # Extract source field from key
            _, _, source_field = key.rsplit('_', 2)
            if source_field in bar_data:
                window.append(bar_data[source_field])
        
        # Invalidate computed values
        self.computed_values.clear()
    
    def get_lookback(self, name: str, window_size: int, 
                    source_field: str = 'close') -> Optional[LookbackWindow]:
        """Get a shared lookback window."""
        key = f"{name}_{window_size}_{source_field}"
        return self.lookback_windows.get(key)
    
    def compute_once(self, key: str, compute_func: Callable[[], Any]) -> Any:
        """
        Compute a value only once per bar.
        
        This ensures expensive calculations (like covariance matrices)
        are only done once even if multiple features need them.
        """
        if key not in self.computed_values:
            self.computed_values[key] = compute_func()
        return self.computed_values[key]
```

#### 3. Feature Calculator with Shared State
```python
class FeatureCalculator:
    """
    Feature calculator that uses shared state components.
    
    This prevents duplicate computation and memory usage across features.
    """
    
    def __init__(self, shared_state: SharedStateManager):
        self.shared_state = shared_state
        self.feature_registry = {}
        
        # Register common lookback windows
        self._register_common_lookbacks()
        
    def _register_common_lookbacks(self):
        """Register commonly used lookback windows."""
        # Price windows for moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            self.shared_state.register_lookback('price', period, 'close')
            
        # Volume windows
        for period in [5, 10, 20]:
            self.shared_state.register_lookback('volume', period, 'volume')
            
        # High/Low windows for volatility
        for period in [14, 20]:
            self.shared_state.register_lookback('high', period, 'high')
            self.shared_state.register_lookback('low', period, 'low')
    
    def on_bar(self, bar_event: Event) -> None:
        """Process new bar and calculate features."""
        bar_data = bar_event.payload
        
        # Update all shared lookback windows
        self.shared_state.update_lookbacks(bar_data)
        
        # Calculate features using shared state
        features = {}
        
        # Simple moving averages using shared windows
        for period in [20, 50]:
            window = self.shared_state.get_lookback('price', period)
            if window and window.is_ready():
                features[f'sma_{period}'] = window.apply(np.mean)
        
        # Exponential moving average (still uses shared window)
        window_20 = self.shared_state.get_lookback('price', 20)
        if window_20 and window_20.is_ready():
            features['ema_20'] = self._calculate_ema(window_20.as_array(), 20)
        
        # RSI using shared window
        window_14 = self.shared_state.get_lookback('price', 14)
        if window_14 and window_14.is_ready():
            features['rsi_14'] = self._calculate_rsi(window_14.as_array())
        
        # ATR using shared high/low windows
        high_14 = self.shared_state.get_lookback('high', 14)
        low_14 = self.shared_state.get_lookback('low', 14)
        if high_14 and high_14.is_ready() and low_14 and low_14.is_ready():
            features['atr_14'] = self._calculate_atr(
                high_14.as_array(), 
                low_14.as_array(),
                window_14.as_array()
            )
        
        # Volume-based features
        vol_window = self.shared_state.get_lookback('volume', 20)
        if vol_window and vol_window.is_ready():
            features['volume_ratio'] = bar_data['volume'] / vol_window.apply(np.mean)
            features['volume_std'] = vol_window.apply(np.std)
        
        # Correlation matrix (expensive, computed once)
        if all(features.get(f'sma_{p}') for p in [20, 50]):
            corr_matrix = self.shared_state.compute_once(
                'correlation_matrix_20_50',
                lambda: self._calculate_correlation_matrix()
            )
            features['price_correlation'] = corr_matrix[0, 1]
        
        # Emit feature event
        feature_event = Event(
            event_type='FEATURES',
            payload={
                **bar_data,  # Include original bar data
                'features': features
            },
            correlation_id=bar_event.correlation_id,
            metadata={'feature_count': len(features)}
        )
        
        self.event_bus.publish(feature_event)
```

#### 4. Memory-Efficient Feature Configuration
```yaml
# feature_config.yaml
shared_state:
  lookback_windows:
    # Define shared windows that multiple features can use
    price_windows:
      - name: price
        periods: [5, 10, 14, 20, 50, 100, 200]
        source: close
    
    volume_windows:
      - name: volume  
        periods: [5, 10, 20]
        source: volume
        
    volatility_windows:
      - name: high
        periods: [14, 20]
        source: high
      - name: low
        periods: [14, 20]
        source: low
        
  # Computed values that should be cached
  cached_computations:
    - correlation_matrices
    - covariance_matrices
    - pca_components

features:
  # Features grouped by their shared dependencies
  moving_averages:
    shared_lookback: price_windows
    calculations:
      - sma_5
      - sma_20
      - sma_50
      - ema_20
      
  momentum:
    shared_lookback: price_windows
    calculations:
      - rsi_14
      - macd_12_26
      - stochastic_14_3
      
  volatility:
    shared_lookback: [price_windows, volatility_windows]
    calculations:
      - atr_14
      - bollinger_bands_20_2
      - keltner_channels_20_1.5
```

#### 5. Lazy Evaluation Pattern
```python
class LazyFeature:
    """
    Lazy evaluation wrapper for features.
    
    Features are only calculated when accessed, preventing
    unnecessary computation for filtered-out features.
    """
    
    def __init__(self, name: str, compute_func: Callable[[], float],
                 dependencies: List[str] = None):
        self.name = name
        self.compute_func = compute_func
        self.dependencies = dependencies or []
        self._value = None
        self._computed = False
        
    def get(self) -> Optional[float]:
        """Get feature value, computing if necessary."""
        if not self._computed:
            self._value = self.compute_func()
            self._computed = True
        return self._value
    
    def reset(self):
        """Reset for next bar."""
        self._computed = False
        self._value = None


class LazyFeatureCalculator:
    """Calculator that uses lazy evaluation for efficiency."""
    
    def __init__(self, shared_state: SharedStateManager):
        self.shared_state = shared_state
        self.lazy_features: Dict[str, LazyFeature] = {}
        
    def register_feature(self, name: str, compute_func: Callable,
                        dependencies: List[str] = None):
        """Register a lazy feature."""
        self.lazy_features[name] = LazyFeature(name, compute_func, dependencies)
        
    def on_bar(self, bar_event: Event):
        """Update state but don't compute features yet."""
        # Update shared state
        self.shared_state.update_lookbacks(bar_event.payload)
        
        # Reset lazy features
        for feature in self.lazy_features.values():
            feature.reset()
        
        # Store bar reference for lazy computation
        self.current_bar = bar_event
        
    def get_features(self, required_features: List[str]) -> Dict[str, float]:
        """
        Get only required features, computing on demand.
        
        This is called by the filter route to get only needed features.
        """
        features = {}
        
        for feature_name in required_features:
            if feature_name in self.lazy_features:
                value = self.lazy_features[feature_name].get()
                if value is not None:
                    features[feature_name] = value
                    
        return features
```

### Benefits of Shared State Abstraction

1. **Memory Efficiency**
   - Single copy of lookback windows shared across features
   - No duplicate price/volume history storage
   - Cached expensive computations

2. **Computation Efficiency**
   - Correlation matrices computed once per bar
   - Lazy evaluation prevents unnecessary calculations
   - Shared intermediate results

3. **Maintainability**
   - Centralized lookback window management
   - Clear dependency tracking
   - Easy to add new shared components

4. **Scalability**
   - Memory usage scales with unique windows, not feature count
   - Can handle hundreds of features efficiently
   - Supports dynamic feature addition/removal

## Approach 1: Composite Events (Recommended)

### Overview
Events are progressively enriched as they flow through the system, creating composite events that contain all necessary data.

### Implementation

```python
# Classifier enriches bar events
class RegimeClassifier:
    def on_bar(self, bar_event: Event):
        # Classify current regime
        regime = self._classify(bar_event)
        
        # Create enriched event
        enriched_event = Event(
            event_type=EventType.CLASSIFIED_BAR,
            payload={
                **bar_event.payload,  # All bar data
                'regime': regime,
                'regime_confidence': self.confidence,
                'regime_features': self.last_features
            },
            correlation_id=bar_event.correlation_id,
            causation_id=bar_event.event_id
        )
        self.event_bus.publish(enriched_event)

# Strategy receives enriched events
class RegimeAwareStrategy:
    def on_event(self, event: Event):
        if event.event_type == EventType.CLASSIFIED_BAR:
            regime = event.payload['regime']
            bar_data = event.payload  # Has everything
            
            # Generate signal based on regime
            signal = self._generate_signal(bar_data, regime)
            
            # Further enrich (Classification + Bar + Signal)
            signal_event = Event(
                event_type=EventType.REGIME_SIGNAL,
                payload={
                    **event.payload,  # Classification + Bar
                    'signal': signal,
                    'strategy_id': self.strategy_id
                }
            )
```

### Pros
- Clean event flow with natural causation chain
- Easy replay (single stream of enriched events)
- Natural boundary handling for regime transitions
- No synchronization issues between different event types

### Cons
- Data duplication (bar data repeated in each enriched event)
- Larger storage footprint
- Potential coupling between components

### Storage Format

```python
# Parquet schema for composite events
signal_schema = {
    'timestamp': 'datetime64[ns]',
    'symbol': 'string',
    'open': 'float64',
    'high': 'float64', 
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64',
    'regime': 'string',
    'regime_confidence': 'float64',
    'signal': 'float64',  # -1.0 to 1.0
    'strategy_id': 'string',
    'features': 'json'  # Serialized features used
}
```

## Approach 2: Reference-Based Events

### Overview
Store different event types separately with references between them, minimizing data duplication.

### Implementation

```python
# Store regimes separately, reference in events
class RegimeTracker:
    def __init__(self):
        self.regime_history = []  # [(timestamp, regime, confidence)]
        self.current_regime = None
        
    def on_bar(self, bar_event: Event):
        regime = self._classify(bar_event)
        
        if regime != self.current_regime:
            # Only store changes
            self.regime_history.append({
                'timestamp': bar_event.timestamp,
                'regime': regime,
                'confidence': self.confidence
            })
            self.current_regime = regime
            
            # Broadcast lightweight change event
            self.event_bus.publish(Event(
                event_type=EventType.REGIME_CHANGE,
                payload={
                    'regime': regime,
                    'prev_regime': self.current_regime,
                    'timestamp': bar_event.timestamp
                }
            ))

# Strategy maintains regime state
class RegimeAwareStrategy:
    def __init__(self):
        self.current_regime = 'unknown'
        
    def on_event(self, event: Event):
        if event.event_type == EventType.REGIME_CHANGE:
            self.current_regime = event.payload['regime']
            
        elif event.event_type == EventType.BAR:
            # Use current regime state
            signal = self._generate_signal(event, self.current_regime)
```

### Pros
- Minimal storage (only changes stored)
- Clean separation of concerns
- Efficient for long stable regimes

### Cons
- State synchronization complexity
- Replay requires careful event ordering
- Each component must maintain regime state

## Approach 3: Metadata Stream Approach

### Overview
Separate metadata streams that can be joined during replay.

### Implementation

```python
# Separate metadata stream that can be joined
class MetadataEnricher:
    """Enriches events with metadata without duplication."""
    
    def __init__(self):
        self.metadata_streams = {
            'regime': RegimeClassifier(),
            'volatility': VolatilityClassifier(),
            'liquidity': LiquidityClassifier()
        }
        
    def enrich_event(self, event: Event) -> Event:
        # Add metadata references, not data
        metadata = {}
        for name, classifier in self.metadata_streams.items():
            metadata[name] = classifier.get_current_state()
        
        event.metadata['market_state'] = metadata
        return event
```

### Pros
- Flexible metadata attachment
- Efficient storage with deduplication
- Can add/remove metadata streams easily

### Cons
- More complex implementation
- Requires join logic for replay
- Potential performance overhead during replay

## Regime-Aware Signal Generation

### Phase 1: Regime-Specific Optimization

```python
class RegimeFilteredSignalGeneration:
    """Generate signals for specific regime periods only."""
    
    def __init__(self, regime_type: str, strategies: List[StrategyBase]):
        self.regime_type = regime_type
        self.strategies = strategies
        self.in_position = False
        self.current_regime = None
        
    def on_event(self, event: Event):
        if event.event_type == EventType.REGIME_CHANGE:
            self.current_regime = event.payload['regime']
            
        elif event.event_type == EventType.BAR:
            # Only process bars during our target regime
            if self.current_regime == self.regime_type:
                # Generate signals from our strategies
                for strategy in self.strategies:
                    strategy.process_bar(event)
                    
            elif self.in_position and self.current_regime != self.regime_type:
                # Handle boundary: we're in a position but regime changed
                # Need to close position gracefully
                self._handle_regime_exit(event)
```

### Boundary Condition Handling

```python
class RegimeBoundaryHandler:
    """
    Handle smooth transitions between regimes.
    """
    
    def __init__(self, transition_rules: Dict[str, Any]):
        """
        transition_rules = {
            'close_on_regime_change': False,  # Force close positions
            'grace_period_bars': 5,  # Continue old signals for N bars
            'neutral_zone_bars': 10,  # No new positions for N bars after change
            'require_confirmation': True  # Need N bars of new regime
        }
        """
        self.rules = transition_rules
        self.transition_state = None
        
    def handle_transition(self, old_regime: str, new_regime: str, 
                         timestamp: datetime, active_positions: List[str]) -> str:
        """
        Determine how to handle regime transition.
        
        Returns: 'close_all', 'grace_period', 'neutral_zone', or 'immediate'
        """
        if active_positions and self.rules['close_on_regime_change']:
            return 'close_all'
            
        elif active_positions and self.rules['grace_period_bars'] > 0:
            # Continue using old regime signals for grace period
            self.transition_state = {
                'type': 'grace_period',
                'old_regime': old_regime,
                'new_regime': new_regime,
                'start_time': timestamp,
                'bars_remaining': self.rules['grace_period_bars'],
                'positions_to_monitor': active_positions.copy()
            }
            return 'grace_period'
```

## Signal Replay Architecture

### Phase 3: Regime-Aware Signal Replay

```python
class SignalReplayContainer:
    """Container that replays stored signals synchronized with market data."""
    
    def __init__(self, container_id: str, signal_files: List[str]):
        self.container_id = container_id
        self.event_bus = EventBus()
        
        # Load all signals into memory (or stream if too large)
        self.signal_streams = {}
        for file in signal_files:
            strategy_id = self._extract_strategy_id(file)
            self.signal_streams[strategy_id] = pd.read_parquet(file)
        
        # Index by timestamp for efficient lookup
        self.current_index = {sid: 0 for sid in self.signal_streams}
    
    def on_bar(self, bar_event: Event):
        """When we receive a bar, emit corresponding signals."""
        timestamp = bar_event.timestamp
        symbol = bar_event.payload['symbol']
        
        # Emit bar first (ensemble strategy needs it)
        self.event_bus.publish(bar_event)
        
        # Then emit all signals for this timestamp
        for strategy_id, signal_df in self.signal_streams.items():
            # Find signals matching this timestamp
            while (self.current_index[strategy_id] < len(signal_df) and 
                   signal_df.iloc[self.current_index[strategy_id]]['timestamp'] <= timestamp):
                
                signal_row = signal_df.iloc[self.current_index[strategy_id]]
                
                if signal_row['timestamp'] == timestamp and signal_row['symbol'] == symbol:
                    # Recreate signal event
                    signal_event = Event(
                        event_type=EventType.SIGNAL,
                        payload={
                            'symbol': symbol,
                            'strategy_id': strategy_id,
                            'signal': signal_row['signal'],
                            'strength': abs(signal_row['signal']),
                            'direction': 'BUY' if signal_row['signal'] > 0 else 'SELL'
                        },
                        timestamp=timestamp,
                        source_id=f"replay_{strategy_id}"
                    )
                    self.event_bus.publish(signal_event)
                
                self.current_index[strategy_id] += 1
```

## Ensemble Strategy Integration

### Basic Ensemble

```python
class EnsembleStrategy:
    """
    Ensemble that receives both bars and signals from atomic strategies.
    """
    
    def __init__(self, strategy_id: str, atomic_weights: Dict[str, float]):
        self.strategy_id = strategy_id
        self.atomic_weights = atomic_weights  # e.g., {'MA_5_20': 0.3, 'RSI_30_70': 0.7}
        self.received_signals = {}  # Buffer signals until we have all
        self.event_bus = EventBus()
        
    def on_event(self, event: Event):
        """Process both bars and signals."""
        if event.event_type == EventType.BAR:
            # Clear signal buffer for new bar
            self.received_signals.clear()
            
        elif event.event_type == EventType.SIGNAL:
            # Collect signals from atomic strategies
            strategy_id = event.payload['strategy_id']
            if strategy_id in self.atomic_weights:
                self.received_signals[strategy_id] = event.payload['signal']
                
                # Check if we have all signals
                if set(self.received_signals.keys()) == set(self.atomic_weights.keys()):
                    self._generate_ensemble_signal(event.timestamp, event.payload['symbol'])
```

### Regime-Aware Ensemble

```python
class RegimeAwareSignalReplay:
    """
    Replay regime-specific signals, handling boundaries carefully.
    """
    
    def __init__(self, regime_signal_map: Dict[str, List[str]]):
        """
        regime_signal_map = {
            'bull': ['MA_5_20_bull_signals.parquet', 'RSI_30_70_bull_signals.parquet'],
            'bear': ['MA_10_30_bear_signals.parquet', 'RSI_14_30_bear_signals.parquet'],
            'sideways': ['MA_20_50_sideways_signals.parquet', 'RSI_21_50_sideways_signals.parquet']
        }
        """
        self.regime_signal_map = regime_signal_map
        self.current_regime = None
        self.active_positions = {}  # Track open positions
        
        # Load all signal streams
        self.signal_streams = {}
        for regime, files in regime_signal_map.items():
            self.signal_streams[regime] = self._load_regime_signals(files)
```

## Configuration Examples

### Phase 1: Signal Generation
```yaml
phase1_config:
  mode: signal_generation
  strategies:
    - type: adaptive_ensemble
      extract: atomic  # Extract atomic strategies for grid search
      
  grid_search:
    MA_*:  # Pattern matching for MA strategies
      parameters:
        fast: [5, 10, 15, 20]
        slow: [20, 30, 50, 100]
    
    RSI_*:  # Pattern matching for RSI strategies
      parameters:
        period: [14, 21, 30]
        overbought: [70, 75, 80]
  
  signal_storage:
    output_dir: ./signals/phase1/
    format: parquet
    include_metadata: true
```

### Phase 3: Signal Replay
```yaml
phase3_config:
  mode: signal_replay
  signal_sources:
    - ./signals/phase1/MA_5_20_signals.parquet
    - ./signals/phase1/RSI_30_70_signals.parquet
  
  ensemble_optimization:
    type: regime_aware_ensemble
    regimes: [bull, bear, sideways]
    weight_constraints:
      min: 0.0
      max: 1.0
      sum_to: 1.0
      
  boundary_rules:
    close_on_regime_change: false
    grace_period_bars: 10
    neutral_zone_bars: 5
```

## Recommendations

1. **Use Composite Events** for simplicity and natural event flow
2. **Store complete events** during signal generation for easy replay
3. **Handle regime boundaries** with grace periods and position tracking
4. **Filter aggressively** to reduce unnecessary data flow
5. **Index by timestamp** for efficient signal replay synchronization

The composite event approach with proper filtering provides the best balance of simplicity, performance, and maintainability.