# ADMF-PC Core Module Documentation

## Overview

The `src/core` module is the foundation of the Adaptive Dynamic Multi-Factor Protocol - Polymorphic Composition (ADMF-PC) system. It provides a sophisticated component-based architecture that enables parallel execution of trading strategies, backtests, and optimizations with complete state isolation.

## Architecture Principles

### 1. Protocol-Based Design
Components implement protocols (interfaces) rather than inheriting from base classes, enabling flexible composition and reducing coupling.

```python
@runtime_checkable
class SignalGenerator(Protocol):
    def generate_signal(self, data: Any) -> Optional[Dict[str, Any]]: ...
```

### 2. Capability Enhancement
Components receive only the capabilities they need through composition, following the "pay for what you use" principle.

```python
# Component gets logging capability only if needed
component = factory.create(MyComponent, capabilities=['logging', 'monitoring'])
```

### 3. Container Isolation
Each execution context (backtest, optimization trial, etc.) runs in its own isolated container with separate state and event spaces.

```python
container = UniversalScopedContainer(
    container_id="backtest_001",
    container_type="backtest"
)
```

### 4. Event-Driven Communication
Components communicate through isolated event buses, preventing cross-contamination between parallel executions.

## Module Structure

```
src/core/
├── components/           # Component framework
│   ├── protocols.py     # Core protocols and capabilities
│   ├── registry.py      # Component registration
│   └── factory.py       # Component creation with enhancement
│
├── containers/          # Container system
│   ├── container.py     # Universal container implementation
│   ├── protocols.py     # Container protocols
│   ├── naming.py        # Container naming conventions
│   └── factory.py       # Container creation and patterns
│
├── coordinator/         # Workflow orchestration
│   ├── coordinator.py   # Main coordinator
│   ├── managers.py      # Workflow-specific managers
│   ├── types.py         # Type definitions
│   ├── protocols.py     # Coordinator protocols
│   └── infrastructure.py # Resource management
│
├── dependencies/        # Dependency injection
│   ├── container.py     # DI container
│   └── graph.py         # Dependency graph and resolution
│
├── events/             # Event system
│   ├── event_bus.py    # Container-isolated event bus
│   ├── isolation.py    # Event isolation management
│   ├── subscription_manager.py # Subscription lifecycle
│   └── types.py        # Event types and protocols
│
├── infrastructure/     # Cross-cutting concerns
│   ├── capabilities.py # Infrastructure capabilities
│   ├── monitoring.py   # Metrics and health checks
│   ├── error_handling.py # Error policies and recovery
│   ├── validation.py   # Validation framework
│   └── protocols.py    # Infrastructure protocols
│
└── logging/            # Structured logging
    └── structured.py   # JSON-based structured logging
```

## Key Components

### 1. Component System (`components/`)

The component system provides the foundation for building modular, reusable components:

- **Protocols**: Define interfaces that components can implement
- **Registry**: Central registration of components
- **Factory**: Creates components with automatic capability enhancement

Example:
```python
from core.components import Component, register_component, get_registry

@register_component(tags=["strategy", "momentum"])
class MomentumStrategy:
    @property
    def component_id(self) -> str:
        return "momentum_strategy_v1"
    
    def generate_signal(self, data: Any) -> Dict[str, Any]:
        # Strategy logic here
        pass

# Discovery and usage
registry = get_registry()
strategy_class = registry.get_class("MomentumStrategy")
```

### 2. Container System (`containers/`)

Provides complete isolation for parallel execution:

- **UniversalScopedContainer**: Core container with state isolation
- **ContainerLifecycleManager**: Manages multiple containers
- **Bootstrap**: Initialize containers from configuration
- **Factory**: Create specialized containers (backtest, optimization, etc.)

Example:
```python
from core.containers import UniversalScopedContainer, create_backtest_container

# Create isolated backtest container
container = create_backtest_container(
    strategy_spec={
        'class': 'MomentumStrategy',
        'parameters': {'lookback': 20}
    },
    container_id="backtest_001"
)

# Initialize and start
container.initialize_scope()
container.start()

# Access components
strategy = container.get_component('Strategy')
```

### 3. Coordinator (`coordinator/`)

Orchestrates complex workflows with proper lifecycle management:

- **Coordinator**: Main entry point for all workflows
- **WorkflowManagers**: Specialized managers for different workflow types
- **Phase-based execution**: Structured workflow phases
- **Resource management**: Shared infrastructure and services

Example:
```python
from core.coordinator import Coordinator, WorkflowConfig, WorkflowType

coordinator = Coordinator(shared_services={'data_provider': data_service})

# Execute optimization workflow
config = WorkflowConfig(
    workflow_type=WorkflowType.OPTIMIZATION,
    optimization_config={
        'algorithm': 'grid_search',
        'objective': 'sharpe_ratio',
        'parameters': {...}
    },
    data_config={...}
)

result = await coordinator.execute_workflow(config)
```

### 4. Dependency Injection (`dependencies/`)

Automatic dependency resolution with cycle detection:

- **DependencyContainer**: Manages component instances and dependencies
- **DependencyGraph**: Tracks dependencies and determines resolution order

Example:
```python
from core.dependencies import DependencyContainer

container = DependencyContainer()
container.register_type("DataService", DataService)
container.register_type("Strategy", MomentumStrategy, dependencies=["DataService"])

# Automatic resolution
strategy = container.resolve("Strategy")  # DataService injected automatically
```

### 5. Event System (`events/`)

Container-isolated event-driven communication:

- **EventBus**: Per-container event routing
- **EventIsolationManager**: Ensures events don't leak between containers
- **SubscriptionManager**: Manages component subscriptions

Example:
```python
from core.events import Event, EventType

# Within a container
event_bus = container.event_bus

# Subscribe to events
def handle_signal(event: Event):
    print(f"Signal received: {event.payload}")

event_bus.subscribe(EventType.SIGNAL, handle_signal)

# Publish events
event = Event(
    event_type=EventType.SIGNAL,
    payload={'symbol': 'AAPL', 'action': 'BUY'},
    container_id=container.container_id
)
event_bus.publish(event)
```

### 6. Infrastructure Capabilities (`infrastructure/`)

Cross-cutting concerns as pluggable capabilities:

- **Logging**: Structured JSON logging with correlation tracking
- **Monitoring**: Metrics collection and health checks
- **Error Handling**: Retry policies, circuit breakers, error boundaries
- **Validation**: Rule-based validation framework

Example:
```python
# Component with infrastructure capabilities
class TradingStrategy:
    def __init__(self):
        self.component_id = "trading_strategy_v1"
    
    # Capabilities added by factory
    # self.logger - structured logging
    # self.metrics_collector - performance metrics
    # self.error_policy - error handling
```

### 7. Structured Logging (`logging/`)

Production-ready logging with correlation tracking:

```python
from core.logging import StructuredLogger, ContainerLogger

# Container-aware logging
logger = ContainerLogger(
    "MyComponent",
    container_id="backtest_001",
    component_id="strategy_01"
)

# Structured output with context
logger.info("Trade executed", 
    symbol="AAPL",
    quantity=100,
    price=150.25,
    correlation_id="req_123"
)
```

## Usage Patterns

### 1. Creating a Strategy Component

```python
from core.components import Component, Lifecycle, EventCapable

class MyStrategy:
    @property
    def component_id(self) -> str:
        return "my_strategy_v1"
    
    # Lifecycle management (optional)
    def initialize(self, context: Dict[str, Any]) -> None:
        self.data_service = context['data_service']
    
    def start(self) -> None:
        self.logger.info("Strategy started")
    
    # Event handling (optional)
    def initialize_events(self) -> None:
        self.event_bus.subscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event) -> None:
        # Process market data
        signal = self.generate_signal(event.payload)
        if signal:
            self.event_bus.publish(Event(
                event_type=EventType.SIGNAL,
                payload=signal
            ))
```

### 2. Running a Backtest

```python
async def run_backtest():
    # Create coordinator
    coordinator = Coordinator(
        shared_services={
            'data_provider': HistoricalDataProvider(),
            'feature_hub': FeatureHub()
        }
    )
    
    # Configure backtest
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        backtest_config={
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'strategy': {
                'class': 'MyStrategy',
                'parameters': {'risk_limit': 0.02}
            }
        },
        data_config={
            'symbols': ['AAPL', 'GOOGL'],
            'frequency': '1D'
        }
    )
    
    # Execute and get results
    result = await coordinator.execute_workflow(config)
    print(f"Backtest complete: {result.final_results}")
```

### 3. Running Parallel Optimizations

```python
async def optimize_strategy():
    coordinator = Coordinator()
    
    # Define parameter space
    config = WorkflowConfig(
        workflow_type=WorkflowType.OPTIMIZATION,
        optimization_config={
            'algorithm': 'grid_search',
            'parameter_space': {
                'lookback': [10, 20, 30],
                'threshold': [0.01, 0.02, 0.03]
            },
            'objective': 'sharpe_ratio'
        }
    )
    
    # Runs multiple trials in isolated containers
    result = await coordinator.execute_workflow(config)
    
    # Best parameters
    print(f"Optimal parameters: {result.final_results['best_params']}")
```

## Best Practices

### 1. Component Design
- Keep components focused on a single responsibility
- Implement only the protocols you need
- Use dependency injection for external dependencies
- Avoid storing state that should be container-scoped

### 2. Container Usage
- Always use containers for execution isolation
- Share only read-only services between containers
- Properly initialize and teardown containers
- Use the lifecycle manager for multiple containers

### 3. Event Handling
- Use events for loose coupling between components
- Always include container_id in events
- Clean up subscriptions in teardown_events
- Use typed events for better maintainability

### 4. Error Handling
- Define error policies at the component level
- Use error boundaries for critical operations
- Implement retry strategies for transient failures
- Log errors with proper context

### 5. Performance
- Use container pooling for repeated executions
- Cache expensive computations at container level
- Monitor performance with built-in metrics
- Profile using the debugging capability

## Integration Points

The core module is designed to integrate with:

1. **Data Systems**: Through DataProvider protocol
2. **Strategy Modules**: Through SignalGenerator protocol
3. **Risk Management**: Through RiskManager protocol
4. **Execution Systems**: Through OrderExecutor protocol
5. **Analytics**: Through event system and metrics

## Testing

Each module includes comprehensive tests:
- Unit tests for individual components
- Integration tests for module interactions
- Container isolation tests
- Workflow execution tests

Run tests with:
```bash
pytest src/core/components/test_components.py
pytest src/core/containers/test_containers.py
pytest test_coordinator_integration.py
```

## Performance Considerations

1. **Container Overhead**: Minimal - containers are lightweight wrappers
2. **Event System**: Optimized with handler caching
3. **Dependency Resolution**: Cached after first resolution
4. **Parallel Execution**: Thread-safe, suitable for concurrent workflows

## Security Considerations

1. **Isolation**: Containers prevent cross-contamination
2. **Validation**: Input validation at component boundaries
3. **Error Handling**: Prevents information leakage
4. **Logging**: Structured logging prevents injection

## Future Enhancements

1. **Distributed Execution**: Extend containers for remote execution
2. **Persistence**: Add state persistence for containers
3. **Hot Reloading**: Dynamic component updates
4. **Advanced Monitoring**: Real-time performance dashboards
5. **GPU Support**: Infrastructure for GPU-accelerated components

## Contributing

When adding new components or capabilities:
1. Follow the protocol-based design pattern
2. Ensure thread-safety for concurrent use
3. Add comprehensive tests
4. Document with clear docstrings
5. Consider backward compatibility

## License

This module is part of the ADMF-PC system. See the main project license for details.