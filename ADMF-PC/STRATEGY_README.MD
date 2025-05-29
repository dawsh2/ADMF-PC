# ADMF-PC Strategy Module

The Strategy Module implements a pure Protocol + Composition (PC) architecture for algorithmic trading strategies, with ZERO inheritance patterns. This module provides strategies, indicators, classifiers, and optimization capabilities that work seamlessly with the container isolation system.

## Architecture Overview

### Core Principles

1. **Protocol-Based Design**: All components implement protocols, not base classes
2. **No Inheritance**: Strategies are simple classes with NO inheritance
3. **Capability Enhancement**: Features added through composition, not inheritance
4. **Container Awareness**: Full isolation for parallel execution
5. **Event-Driven**: Communication through container-scoped event buses

### Module Structure

```
src/strategy/
├── protocols.py           # Core protocols (Strategy, Indicator, Classifier)
├── capabilities.py        # Strategy-specific capabilities
├── strategies/           # Strategy implementations
│   ├── momentum.py       # Momentum trading strategy
│   ├── mean_reversion.py # Mean reversion strategy
│   ├── trend_following.py # Trend following strategy
│   ├── arbitrage.py      # Arbitrage strategy
│   └── market_making.py  # Market making strategy
├── components/           # Reusable components
│   ├── indicators.py     # Technical indicators
│   ├── classifiers.py    # Market regime classifiers
│   └── signal_replay.py  # Signal capture and replay
├── optimization/         # Optimization framework
│   ├── protocols.py      # Optimization protocols
│   ├── optimizers.py     # Optimization algorithms
│   ├── objectives.py     # Objective functions
│   ├── constraints.py    # Parameter constraints
│   ├── containers.py     # Optimization containers
│   └── workflows.py      # Multi-phase workflows
└── classifiers/          # Advanced classifier system
    ├── classifier.py     # Classifier implementations
    └── classifier_container.py # Container integration

```

## Key Protocols

### Strategy Protocol

```python
@runtime_checkable
class Strategy(Protocol):
    """Core trading strategy protocol."""
    
    @property
    def name(self) -> str:
        """Strategy identifier."""
        ...
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal from market data."""
        ...
```

### Indicator Protocol

```python
@runtime_checkable
class Indicator(Protocol):
    """Technical indicator protocol."""
    
    def calculate(self, data: List[float]) -> float:
        """Calculate indicator value."""
        ...
    
    def update(self, value: float) -> None:
        """Update with new data point."""
        ...
```

### Classifier Protocol

```python
@runtime_checkable
class Classifier(Protocol):
    """Market regime classifier protocol."""
    
    def classify(self, features: Dict[str, Any]) -> str:
        """Classify current market regime."""
        ...
    
    def get_confidence(self) -> float:
        """Get classification confidence."""
        ...
```

## Strategy Implementation

### Example: Momentum Strategy (NO Inheritance!)

```python
class MomentumStrategy:
    """Momentum-based trading strategy - pure class, no inheritance."""
    
    def __init__(self, lookback_period: int = 20, 
                 momentum_threshold: float = 0.02):
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.price_history = []
    
    @property
    def name(self) -> str:
        return "momentum_strategy"
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        price = market_data.get('close')
        self.price_history.append(price)
        
        if len(self.price_history) < self.lookback_period:
            return None
        
        momentum = self._calculate_momentum()
        
        if momentum > self.momentum_threshold:
            return {
                'symbol': market_data['symbol'],
                'direction': SignalDirection.BUY,
                'strength': min(momentum / (self.momentum_threshold * 2), 1.0),
                'timestamp': market_data['timestamp']
            }
        
        return None
```

## Capability Enhancement

### Adding Capabilities to Strategies

```python
from src.core.components import ComponentFactory

# Create strategy with capabilities
strategy = ComponentFactory().create_component({
    'class': 'MomentumStrategy',
    'params': {
        'lookback_period': 20,
        'momentum_threshold': 0.02
    },
    'capabilities': [
        'strategy',      # Core strategy capability
        'lifecycle',     # Start/stop lifecycle
        'events',        # Event publishing
        'optimization',  # Parameter optimization
        'monitoring'     # Performance monitoring
    ]
})

# Now strategy has enhanced features
strategy.start()  # From lifecycle capability
strategy.publish_event('signal_generated', signal)  # From events
params = strategy.get_parameters()  # From optimization
```

## Container Integration

### Strategy Execution in Containers

```python
from src.core.containers import UniversalScopedContainer

# Create strategy container
container = UniversalScopedContainer(
    scope_id="strategy_momentum_001",
    parent_container=coordinator_container
)

# Register strategy
container.register_component('strategy', strategy, {
    'capabilities': ['strategy', 'events']
})

# Execute in isolation
with container.create_scope():
    signal = strategy.generate_signal(market_data)
```

## Optimization Integration

### Making Strategies Optimizable

```python
from src.strategy.optimization import OptimizationCapability

# Apply optimization capability
opt_capability = OptimizationCapability()
optimizable_strategy = opt_capability.apply(strategy, {
    'parameter_space': {
        'lookback_period': [10, 20, 30, 40],
        'momentum_threshold': [0.01, 0.02, 0.03, 0.04]
    },
    'constraints': [
        RangeConstraint('lookback_period', min_value=5, max_value=100)
    ]
})

# Run optimization
optimizer = GridOptimizer()
best_params = optimizer.optimize(
    lambda params: evaluate_strategy(optimizable_strategy, params),
    parameter_space=optimizable_strategy.get_parameter_space()
)
```

## Regime-Aware Strategies

### Adaptive Strategy Behavior

```python
class RegimeAdaptiveStrategy:
    """Strategy that adapts to market regimes."""
    
    def __init__(self, regime_params: Dict[str, Dict[str, Any]]):
        self.regime_params = regime_params
        self.current_regime = None
    
    def set_regime(self, regime: str) -> None:
        """Switch to regime-specific parameters."""
        self.current_regime = regime
        if regime in self.regime_params:
            self._apply_params(self.regime_params[regime])
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Adapt behavior based on current regime
        if self.current_regime == 'HIGH_VOLATILITY':
            # More conservative in volatile markets
            return self._generate_conservative_signal(market_data)
        else:
            # Normal signal generation
            return self._generate_normal_signal(market_data)
```

## Event Integration

### Publishing Strategy Events

```python
# Strategies with event capability can publish
strategy.publish_event('signal.generated', {
    'strategy': strategy.name,
    'signal': signal,
    'confidence': 0.85,
    'timestamp': datetime.now()
})

# Other components can subscribe
event_bus.subscribe('signal.generated', lambda event: 
    logger.info(f"Signal from {event.data['strategy']}")
)
```

## Performance Considerations

### 1. Efficient Data Handling

```python
class EfficientStrategy:
    """Memory-efficient strategy implementation."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._price_buffer = deque(maxlen=max_history)
        self._indicator_cache = {}
```

### 2. Signal Caching

```python
class CachedStrategy:
    """Strategy with signal caching."""
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_key = self._generate_cache_key(market_data)
        
        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]
        
        signal = self._calculate_signal(market_data)
        self._signal_cache[cache_key] = signal
        return signal
```

## Testing Strategies

### Unit Testing

```python
def test_momentum_strategy():
    strategy = MomentumStrategy(lookback_period=10)
    
    # Feed historical data
    for price in [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]:
        signal = strategy.generate_signal({'close': price, 'symbol': 'TEST'})
    
    # Should generate buy signal on strong momentum
    assert signal is not None
    assert signal['direction'] == SignalDirection.BUY
```

### Integration Testing

```python
def test_strategy_in_container():
    container = UniversalScopedContainer("test_strategy")
    strategy = ComponentFactory().create_component({
        'class': 'MomentumStrategy',
        'capabilities': ['strategy', 'events']
    })
    
    container.register_component('strategy', strategy)
    
    # Test execution in isolation
    with container.create_scope():
        signal = strategy.generate_signal(test_data)
        assert signal is not None
```

## Best Practices

1. **No Inheritance**: Keep strategies as simple classes
2. **Protocol Compliance**: Ensure strategies implement required protocol methods
3. **Capability Composition**: Add features through capabilities, not inheritance
4. **Container Isolation**: Always run strategies in containers for production
5. **Event Publishing**: Use events for loose coupling
6. **Parameter Validation**: Validate parameters in __init__
7. **Efficient State Management**: Limit historical data storage
8. **Clear Signal Format**: Use consistent signal structure
9. **Regime Awareness**: Consider market conditions
10. **Comprehensive Testing**: Test both logic and integration

## Common Patterns

### 1. Multi-Timeframe Strategy

```python
class MultiTimeframeStrategy:
    """Strategy using multiple timeframes."""
    
    def __init__(self, timeframes: List[str]):
        self.indicators = {tf: {} for tf in timeframes}
    
    def update_timeframe(self, timeframe: str, data: Dict[str, Any]):
        # Update indicators for specific timeframe
        self.indicators[timeframe]['sma'] = calculate_sma(data)
```

### 2. Ensemble Strategy

```python
class EnsembleStrategy:
    """Combines multiple sub-strategies."""
    
    def __init__(self, strategies: List[Any], weights: List[float]):
        self.strategies = strategies
        self.weights = weights
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        signals = [s.generate_signal(market_data) for s in self.strategies]
        return self._combine_signals(signals, self.weights)
```

### 3. Risk-Aware Strategy

```python
class RiskAwareStrategy:
    """Strategy with integrated risk management."""
    
    def __init__(self, base_strategy: Any, max_position_size: float):
        self.base_strategy = base_strategy
        self.max_position_size = max_position_size
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        signal = self.base_strategy.generate_signal(market_data)
        
        if signal:
            # Adjust signal strength based on risk
            signal['strength'] = min(
                signal['strength'],
                self._calculate_risk_adjusted_size(market_data)
            )
        
        return signal
```

## Migration Guide

### Converting Inherited Strategies to PC

Before (with inheritance):
```python
class MyStrategy(BaseStrategy):  # BAD - uses inheritance
    def __init__(self):
        super().__init__()
        self.param = 10
    
    def generate_signal(self, data):
        return super().generate_signal(data)
```

After (Protocol + Composition):
```python
class MyStrategy:  # GOOD - no inheritance
    def __init__(self):
        self.param = 10
        self.name = "my_strategy"
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Direct implementation
        if self._should_trade(market_data):
            return {
                'symbol': market_data['symbol'],
                'direction': SignalDirection.BUY,
                'strength': 0.8,
                'timestamp': market_data['timestamp']
            }
        return None
```

## Conclusion

The Strategy Module demonstrates that complex trading strategies can be implemented without ANY inheritance, using only protocols and composition. This approach provides:

- Maximum flexibility and reusability
- Clean separation of concerns
- Easy testing and mocking
- Seamless integration with the container system
- Natural parallelization through isolation

Every strategy is a simple class that gains power through capability composition, not inheritance hierarchies.