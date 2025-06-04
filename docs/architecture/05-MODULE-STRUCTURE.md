# Module Structure and Organization

## Overview

ADMF-PC's module structure reflects its core architectural principles: protocol-based interfaces, compositional design, and clear separation of concerns. This document details the organization, responsibilities, and communication patterns of each module.

## Module Hierarchy

```
src/
├── core/           # Foundation - containers, events, protocols
├── data/           # Data acquisition and streaming
├── strategy/       # Trading logic and optimization
├── risk/           # Risk management and limits
├── portfolio/      # Portfolio allocation and tracking
└── execution/      # Order execution and simulation
```

## Core Module

The foundation that all other modules build upon.

### Structure

```
core/
├── coordinator/    # Workflow orchestration
├── containers/     # Container architecture
├── components/     # Component registry and factory
├── events/         # Event bus and isolation
├── config/         # Configuration management
├── infrastructure/ # Cross-cutting concerns
└── logging/        # Structured logging
```

### Key Responsibilities

1. **Container Management**
   - Container lifecycle (create, initialize, start, stop)
   - Resource isolation and limits
   - Hierarchical container relationships

2. **Event Infrastructure**
   - Isolated event buses per container
   - Event routing and subscription
   - Event type definitions

3. **Component Registry**
   - Dynamic component discovery
   - Factory pattern for creation
   - Protocol validation

4. **Workflow Coordination**
   - YAML interpretation
   - Phase orchestration
   - Resource allocation

### Core Interfaces

```python
# Minimal protocols that everything else builds on
class Component(Protocol):
    """Base component protocol"""
    @property
    def component_id(self) -> str: ...

class Container(Protocol):
    """Container protocol"""
    def initialize(self, context: Dict[str, Any]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

class EventBus(Protocol):
    """Event bus protocol"""
    def publish(self, event: Event) -> None: ...
    def subscribe(self, event_type: str, handler: Callable) -> None: ...
```

## Data Module

Handles all data acquisition, storage, and streaming.

### Structure

```
data/
├── protocols.py    # Data provider interfaces
├── handlers.py     # Concrete implementations
├── loaders.py      # File and database loaders
├── streamers.py    # Real-time data streaming
└── models.py       # Data models (Bar, Tick, etc.)
```

### Key Responsibilities

1. **Data Acquisition**
   - CSV file loading
   - Database connections
   - API integrations
   - Real-time feeds

2. **Data Streaming**
   - Historical data replay
   - Real-time data handling
   - Multi-timeframe support
   - Data synchronization

3. **Data Models**
   - Standardized bar format
   - Tick data handling
   - Alternative data structures

### Data Flow

```python
class DataFlow:
    """
    CSV/DB/API → Loader → Validator → Streamer → EVENT_BUS
                                                     ↓
                                            (BAR_DATA events)
    """
```

### Example Implementation

```python
class DataStreamer:
    """Streams data as events"""
    
    def __init__(self, data_source: Any, event_bus: EventBus):
        self.data_source = data_source
        self.event_bus = event_bus
    
    def stream(self):
        """Stream data as events"""
        for bar in self.data_source:
            self.event_bus.publish(Event(
                type="BAR_DATA",
                data=bar,
                timestamp=bar.timestamp
            ))
```

## Strategy Module

Contains trading logic, indicators, and optimization.

### Structure

```
strategy/
├── protocols.py       # Strategy interfaces
├── strategies/        # Concrete strategy implementations
│   ├── momentum.py
│   ├── mean_reversion.py
│   └── market_making.py
├── classifiers/       # Market regime detection
│   ├── hmm_classifier.py
│   └── pattern_classifier.py
├── optimization/      # Parameter optimization
│   ├── optimizers.py
│   └── walk_forward.py
└── components/        # Reusable components
    ├── indicators.py
    └── signal_replay.py
```

### Key Responsibilities

1. **Signal Generation**
   - Strategy implementation
   - Indicator calculation
   - Signal combination

2. **Regime Detection**
   - Market classification
   - Regime-aware strategies
   - Adaptive behavior

3. **Optimization**
   - Parameter search
   - Walk-forward analysis
   - Ensemble optimization

### Strategy Protocol

```python
class SignalGenerator(Protocol):
    """Anything that generates trading signals"""
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns: {
            'action': 'BUY' | 'SELL' | 'HOLD',
            'strength': float,  # 0.0 to 1.0
            'metadata': {}      # Optional context
        }
        """
        ...
```

### Component Interaction

```
BAR_DATA → FeatureHub → FEATURE events
                ↓
         StrategyContainer → SIGNAL events
                ↓
         ClassifierContainer → REGIME events
```

## Risk Module

Manages portfolio state, position sizing, and risk limits.

### Structure

```
risk/
├── protocols.py         # Risk management interfaces
├── portfolio_state.py   # Portfolio tracking
├── position_sizing.py   # Size calculation algorithms
├── risk_limits.py       # Risk constraint enforcement
├── signal_flow.py       # Signal to order conversion
└── capabilities.py      # Risk capabilities
```

### Key Responsibilities

1. **Portfolio Management**
   - Position tracking
   - P&L calculation
   - Exposure monitoring

2. **Risk Control**
   - Position limits
   - Drawdown control
   - Risk budgets
   - Circuit breakers

3. **Signal Processing**
   - Signal validation
   - Position sizing
   - Order generation

### Risk Flow

```python
class RiskFlow:
    """
    SIGNAL event → Risk validation → Position sizing → ORDER event
                         ↓                                ↓
                    (may VETO)                    (or no order)
    
    FILL event → Portfolio update → Risk metrics → State tracking
    """
```

### Risk Protocol

```python
class RiskManager(Protocol):
    """Risk management protocol"""
    
    def process_signal(self, signal: Signal) -> Optional[Order]:
        """Convert signal to order with risk checks"""
        ...
    
    def update_portfolio(self, fill: Fill) -> None:
        """Update portfolio after execution"""
        ...
```

## Execution Module

Handles order execution, market simulation, and broker integration.

### Structure

```
execution/
├── protocols.py           # Execution interfaces
├── backtest_engine.py     # Historical simulation
├── market_simulation.py   # Realistic execution
├── order_manager.py       # Order lifecycle
├── execution_engine.py    # Core execution logic
└── modes.py              # Execution modes
```

### Key Responsibilities

1. **Order Management**
   - Order validation
   - Order routing
   - Fill simulation
   - Partial fills

2. **Market Simulation**
   - Slippage modeling
   - Market impact
   - Liquidity constraints
   - Transaction costs

3. **Execution Modes**
   - Backtest mode
   - Paper trading
   - Live trading

### Execution Protocol

```python
class ExecutionEngine(Protocol):
    """Execution protocol"""
    
    def execute_order(self, order: Order) -> Fill:
        """Execute order and return fill"""
        ...
    
    def get_results(self) -> BacktestResults:
        """Get execution results"""
        ...
```

## Inter-Module Communication

### Event-Based Only

Modules communicate exclusively through events:

```python
# Data → Strategy
event_bus.publish(Event(type="BAR_DATA", data=bar))

# Strategy → Risk  
event_bus.publish(Event(type="SIGNAL", data=signal))

# Risk → Execution
event_bus.publish(Event(type="ORDER", data=order))

# Execution → Risk
event_bus.publish(Event(type="FILL", data=fill))
```

### No Direct Dependencies

```python
# WRONG: Direct module coupling
class Strategy:
    def __init__(self):
        self.risk_manager = RiskManager()  # NO!
        self.executor = ExecutionEngine()   # NO!

# RIGHT: Event-based communication
class Strategy:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    def generate_signal(self, data):
        signal = self.calculate_signal(data)
        self.event_bus.publish(Event(type="SIGNAL", data=signal))
```

## Module Boundaries

### Clear Interfaces

Each module exposes only protocol-based interfaces:

```python
# data/__init__.py
from .protocols import DataProvider, DataStreamer
from .handlers import CSVHandler, DatabaseHandler

__all__ = ['DataProvider', 'DataStreamer', 'CSVHandler', 'DatabaseHandler']
```

### Dependency Rules

1. **Core** → No dependencies (foundation)
2. **Data** → Depends only on Core
3. **Strategy** → Depends only on Core
4. **Risk** → Depends only on Core
5. **Execution** → Depends only on Core

### Testing Boundaries

Each module can be tested in complete isolation:

```python
# Test strategy without risk or execution
def test_strategy_signals():
    mock_bus = MockEventBus()
    strategy = MomentumStrategy(mock_bus)
    
    strategy.process_data(test_data)
    
    assert mock_bus.published_events[0].type == "SIGNAL"
    assert mock_bus.published_events[0].data['action'] == "BUY"
```

## Configuration Management

### Module Configuration

Each module accepts configuration through dictionaries:

```yaml
# config.yaml
data:
  source: "csv"
  path: "data/SPY.csv"
  
strategy:
  type: "momentum"
  fast_period: 10
  slow_period: 30
  
risk:
  position_size_pct: 2.0
  max_drawdown_pct: 15.0
  
execution:
  slippage_pct: 0.1
  commission: 0.001
```

### Configuration Flow

```python
class ModuleConfiguration:
    """
    YAML → Coordinator → Module configs → Container initialization
                              ↓
                    Module-specific validation
    """
```

## Performance Considerations

### Module-Level Optimization

1. **Data Module**
   - Lazy loading for large datasets
   - Streaming instead of loading all data
   - Efficient data structures (numpy arrays)

2. **Strategy Module**
   - Vectorized calculations
   - Feature caching
   - Minimal state

3. **Risk Module**
   - Incremental portfolio updates
   - Pre-calculated risk metrics
   - Efficient position tracking

4. **Execution Module**
   - Order book simulation
   - Batch fill processing
   - Memory-mapped trade logs

### Cross-Module Performance

```python
# Event batching for performance
class BatchEventBus(EventBus):
    """Batch events for efficiency"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.pending_events = []
    
    def publish(self, event: Event):
        self.pending_events.append(event)
        if len(self.pending_events) >= self.batch_size:
            self._flush_events()
```

## Module Extension

### Adding New Modules

1. Create module directory under `src/`
2. Define protocols in `protocols.py`
3. Implement concrete classes
4. Register with component factory
5. Add configuration schema

### Example: Adding Alternative Data Module

```python
# src/alternative_data/protocols.py
class AlternativeDataProvider(Protocol):
    """Protocol for alternative data sources"""
    
    def get_sentiment(self, symbol: str, timestamp: datetime) -> float:
        ...

# src/alternative_data/providers.py
class TwitterSentimentProvider:
    """Twitter sentiment analysis"""
    
    def get_sentiment(self, symbol: str, timestamp: datetime) -> float:
        # Implementation
        return sentiment_score
```

## Module Guidelines

### Do's

1. **Define clear protocols** for all interfaces
2. **Use events** for all inter-module communication
3. **Keep modules independent** - no cross-dependencies
4. **Test in isolation** using mocks
5. **Document module boundaries** clearly

### Don'ts

1. **Don't import** from other modules (except Core)
2. **Don't share state** between modules
3. **Don't use global variables**
4. **Don't bypass the event system**
5. **Don't create circular dependencies**

## Summary

ADMF-PC's module structure provides:

- **Clear separation** of concerns
- **Protocol-based** interfaces
- **Event-driven** communication
- **Independent** testing
- **Flexible** composition

Each module focuses on its domain while the Core module provides the infrastructure that ties everything together through containers and events.