# Strategy Module - Protocol + Composition Architecture

## Overview

The Strategy module implements trading strategies using pure protocol-based design with **zero inheritance**. All functionality is added through capabilities and composition.

## Key Design Principles

1. **No Base Classes** - Strategies are simple classes, not subclasses
2. **Protocol-Based** - Components implement protocols without inheritance
3. **Capability Composition** - Add only the functionality you need
4. **Container Isolation** - Each strategy runs in its own isolated environment
5. **Event-Driven** - Integrates with the core event system

## Architecture

```
src/strategy/
├── protocols.py           # All protocol definitions
├── capabilities.py        # Strategy-specific capabilities
├── strategies/           # Concrete strategy implementations
│   ├── momentum.py       # Example: Plain class, no inheritance
│   ├── mean_reversion.py # Example: Plain class, no inheritance
│   └── ...
├── optimization/         # Container-aware optimization
├── components/          # Reusable components (indicators, rules, etc)
└── classifiers/         # Market regime classifiers
```

## Quick Start

### 1. Basic Strategy (No Capabilities)

```python
from src.strategy import MomentumStrategy

# Just a plain class - no inheritance!
strategy = MomentumStrategy(
    lookback_period=20,
    momentum_threshold=0.02
)

# Has basic strategy methods
market_data = {'symbol': 'AAPL', 'close': 150.00}
signal = strategy.generate_signal(market_data)
```

### 2. Strategy with Capabilities

```python
from src.core.components import ComponentFactory

# Add capabilities through composition
strategy = ComponentFactory().create_component({
    'class': 'MomentumStrategy',
    'params': {
        'lookback_period': 20,
        'momentum_threshold': 0.02
    },
    'capabilities': [
        'strategy',        # Signal tracking, event handling
        'optimization',    # Parameter optimization
        'indicators',      # Indicator management  
        'regime_adaptive'  # Regime adaptation
    ],
    'parameter_space': {
        'lookback_period': [10, 20, 30],
        'momentum_threshold': [0.01, 0.02, 0.03]
    }
})

# Now has methods added by capabilities:
strategy.get_parameter_space()  # From OptimizationCapability
strategy.register_indicator()   # From IndicatorCapability
strategy.on_regime_change()     # From RegimeAdaptiveCapability
```

### 3. Container-Based Execution

```python
from src.core.containers import UniversalScopedContainer

# Each strategy runs in isolation
container = UniversalScopedContainer("strategy_001")

strategy = container.create_component({
    'name': 'my_momentum',
    'class': 'MomentumStrategy',
    'params': {'lookback_period': 20},
    'capabilities': ['strategy', 'events']
})

# Container provides:
# - Complete isolation
# - Scoped event bus
# - Resource management
# - Lifecycle control
```

### 4. Optimization with Containers

```python
from src.strategy.optimization import OptimizationContainer

# Optimization runs each trial in isolation
opt_container = OptimizationContainer(
    container_id="opt_001",
    base_config={
        'class': 'MomentumStrategy',
        'capabilities': ['strategy', 'optimization']
    }
)

# Each parameter trial is completely isolated
results = opt_container.run_trial(
    parameters={'lookback_period': 30},
    evaluator=backtest_function
)
```

## Creating a New Strategy

### Step 1: Create a Plain Class

```python
class MyStrategy:
    """My custom strategy - no inheritance!"""
    
    def __init__(self, param1: float = 0.5):
        self.param1 = param1
        self.state = {}  # Track any state needed
    
    @property
    def name(self) -> str:
        return "my_strategy"
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Implement the Strategy protocol."""
        # Your logic here
        if some_condition:
            return {
                'symbol': market_data['symbol'],
                'direction': SignalDirection.BUY,
                'strength': 0.8,
                'timestamp': datetime.now()
            }
        return None
```

### Step 2: Use with Capabilities

```python
# Add capabilities as needed
strategy = ComponentFactory().create_component({
    'class': 'MyStrategy',
    'params': {'param1': 0.7},
    'capabilities': ['strategy', 'optimization'],
    'parameter_space': {
        'param1': [0.3, 0.5, 0.7, 0.9]
    }
})
```

## Available Capabilities

### StrategyCapability
- Signal generation tracking
- Event-based market data handling  
- Signal emission through event bus
- Metadata management

### OptimizationCapability
- Parameter space management
- Parameter validation and application
- Optimization history tracking
- Constraint handling

### IndicatorCapability
- Indicator registration and tracking
- Bulk indicator updates
- Indicator value access
- Automatic indicator creation from spec

### RegimeAdaptiveCapability
- Regime change handling
- Parameter switching based on regime
- Regime-specific performance tracking
- Smooth parameter transitions

### RuleManagementCapability
- Rule registration and management
- Rule evaluation with aggregation
- Weighted voting mechanisms
- Rule state tracking

## Protocol Reference

### Core Protocols

```python
@runtime_checkable
class Strategy(Protocol):
    """Core strategy protocol."""
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal from market data."""
        ...
    
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        ...
```

### Supporting Protocols

- `Indicator` - Technical indicators
- `Feature` - Market feature extraction
- `Rule` - Trading rules
- `Classifier` - Market regime classification
- `Optimizable` - Components that can be optimized
- `RegimeAdaptive` - Regime-aware components

## Best Practices

1. **Never Use Inheritance** - Create plain classes that implement protocols
2. **Add Only Needed Capabilities** - Don't add optimization if you won't optimize
3. **Use Containers for Isolation** - Run strategies in containers for production
4. **Leverage Event System** - Use events for loose coupling
5. **Keep Strategies Focused** - Single responsibility per strategy

## Examples

See `examples/` directory for:
- `pc_architecture_demo.py` - Complete architecture demonstration
- `optimization_example.py` - Optimization workflow examples
- More coming soon...

## Migration from Inheritance-Based Code

If you have old inheritance-based strategies:

```python
# OLD (Don't do this!)
class OldStrategy(StrategyBase):
    def __init__(self):
        super().__init__()
        # ...

# NEW (Do this!)
class NewStrategy:
    def __init__(self):
        # Just initialize what you need
        self.state = {}
    
    def generate_signal(self, market_data):
        # Implement protocol method
        pass
```

Then add capabilities through ComponentFactory instead of inheriting functionality.

## Testing

Strategies are easy to test because they're just plain classes:

```python
# Direct testing - no complex setup
strategy = MomentumStrategy(lookback_period=10)
assert strategy.name == "momentum_strategy"

# Test with mock data
market_data = {'symbol': 'TEST', 'close': 100.0}
signal = strategy.generate_signal(market_data)
assert signal is None  # Not enough history yet
```

## Performance Considerations

1. **Zero Overhead** - No inheritance chains or super() calls
2. **Isolated Execution** - Strategies can run in parallel safely
3. **Efficient Indicators** - Shared computation through IndicatorHub
4. **Container Reuse** - Containers can be pooled for efficiency

## Future Enhancements

- [ ] More built-in strategies
- [ ] Advanced classifiers
- [ ] GPU acceleration support
- [ ] Distributed optimization
- [ ] Real-time performance monitoring

---

*Remember: In Protocol + Composition architecture, inheritance is the enemy of flexibility. Compose, don't inherit!*