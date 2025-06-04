# Container Hierarchy and Design Philosophy

## Overview

The ADMF-PC container architecture provides complete isolation and lifecycle management for trading system components. This document explains the container design philosophy, nesting patterns, and resource management strategies.

## Container Design Philosophy

### Core Principles

1. **Complete Isolation**: Each container has its own event bus, state, and resources
2. **Hierarchical Organization**: Containers can nest to form execution trees
3. **Lifecycle Management**: Standardized initialization, execution, and cleanup phases
4. **Resource Boundaries**: Memory, CPU, and I/O resources are container-scoped

### Container Types

```
Universal Container (Base)
├── Backtest Container
│   ├── Strategy Container
│   ├── Risk Container
│   └── Execution Container
├── Signal Generation Container
│   ├── Strategy Container
│   └── Analysis Container
└── Signal Replay Container
    ├── Ensemble Container
    └── Risk Container
```

## Container Nesting Patterns

### Parent-Child Relationships

```python
class ContainerHierarchy:
    """
    Parent containers:
    - Create and manage child containers
    - Subscribe to child events
    - Control child lifecycle
    
    Child containers:
    - Are unaware of parent existence
    - Emit events to their local bus
    - Have independent lifecycle
    """
```

### Example: Backtest Container Hierarchy

```
BacktestContainer (ID: backtest_001)
├── FeatureHub (ID: backtest_001.features)
├── StrategyContainer (ID: backtest_001.strategy_momentum)
│   └── MomentumStrategy
├── RiskContainer (ID: backtest_001.risk)
│   ├── RiskLimits
│   └── PortfolioContainer (ID: backtest_001.portfolio)
│       ├── PositionSizer
│       └── PortfolioState
└── ExecutionEngine (ID: backtest_001.execution)
    ├── OrderManager
    └── MarketSimulator
```

### Event Flow in Hierarchy

```
1. Data → FeatureHub
   - Computes features
   - Emits FEATURE events

2. FEATURE → StrategyContainer
   - Generates signals
   - Emits SIGNAL events

3. SIGNAL → RiskContainer
   - Applies risk rules
   - Emits ORDER events

4. ORDER → ExecutionEngine
   - Simulates execution
   - Emits FILL events

5. FILL → RiskContainer
   - Updates positions
   - Tracks portfolio state
```

## Lifecycle Management

### Container Lifecycle Phases

```python
class ContainerLifecycle:
    """
    1. CREATED: Container instantiated but not initialized
    2. INITIALIZING: Setting up resources and dependencies
    3. READY: Initialized and ready to process
    4. RUNNING: Actively processing events
    5. STOPPING: Graceful shutdown initiated
    6. STOPPED: All resources released
    """
```

### Lifecycle Methods

```python
class UniversalContainer:
    def initialize(self, context: Dict[str, Any]) -> None:
        """Set up container resources"""
        self._setup_event_bus()
        self._initialize_components()
        self._validate_configuration()
    
    def start(self) -> None:
        """Begin processing"""
        self._start_event_processing()
        self._notify_ready()
    
    def stop(self) -> None:
        """Graceful shutdown"""
        self._stop_event_processing()
        self._cleanup_resources()
        self._notify_stopped()
```

### Resource Cleanup

```python
class ResourceManager:
    def cleanup_container(self, container_id: str) -> None:
        """Ensure complete resource cleanup"""
        # 1. Stop event processing
        self.event_buses[container_id].stop()
        
        # 2. Clear subscriptions
        self.subscriptions[container_id].clear()
        
        # 3. Release memory
        self.cached_data[container_id] = None
        
        # 4. Close connections
        self.connections[container_id].close()
        
        # 5. Remove from registry
        del self.containers[container_id]
```

## Resource Management

### Memory Management

```python
class ContainerMemoryManager:
    """Tracks and limits container memory usage"""
    
    def __init__(self, container_id: str, memory_limit_mb: int = 500):
        self.container_id = container_id
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.allocations = {}
    
    def allocate(self, size: int, purpose: str) -> bool:
        """Allocate memory with tracking"""
        current_usage = sum(self.allocations.values())
        if current_usage + size > self.memory_limit:
            raise MemoryError(f"Container {self.container_id} memory limit exceeded")
        self.allocations[purpose] = size
        return True
```

### CPU Resource Management

```python
class ContainerCPUManager:
    """Manages CPU allocation for containers"""
    
    def __init__(self, container_id: str, cpu_shares: int = 1024):
        self.container_id = container_id
        self.cpu_shares = cpu_shares
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    def submit_task(self, task: Callable) -> Future:
        """Submit task with resource constraints"""
        return self.thread_pool.submit(task)
```

## Inter-Container Communication

### Event-Based Communication Only

```python
class ContainerCommunication:
    """
    Containers communicate ONLY through events:
    - No direct method calls
    - No shared state
    - No global variables
    """
    
    def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish to local event bus only"""
        self.local_event_bus.publish(Event(
            type=event_type,
            source=self.container_id,
            data=data,
            timestamp=time.time()
        ))
```

### Parent-Child Event Bridging

```python
class ParentContainer:
    def create_child(self, child_id: str) -> Container:
        """Create child with event bridge"""
        child = Container(child_id)
        
        # Parent subscribes to child events
        child.event_bus.subscribe("*", self._handle_child_event)
        
        # Child never knows about parent
        return child
    
    def _handle_child_event(self, event: Event) -> None:
        """Process events from children"""
        # Parent logic here
        pass
```

## Container Patterns

### 1. Isolated Execution Pattern

```python
def run_isolated_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """Each backtest runs in complete isolation"""
    
    # Create isolated container
    container = BacktestContainer(f"backtest_{uuid.uuid4()}")
    
    try:
        # Initialize with config
        container.initialize(config)
        
        # Run backtest
        container.start()
        results = container.run()
        
        return results
        
    finally:
        # Ensure cleanup
        container.stop()
```

### 2. Parallel Container Pattern

```python
def run_parallel_optimizations(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run multiple containers in parallel"""
    
    containers = []
    futures = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        for config in configs:
            container = BacktestContainer(f"opt_{uuid.uuid4()}")
            containers.append(container)
            
            future = executor.submit(run_isolated_backtest, config)
            futures.append(future)
        
        # Collect results
        results = [f.result() for f in futures]
    
    return results
```

### 3. Hierarchical Processing Pattern

```python
class HierarchicalProcessor:
    def process_multi_strategy(self, strategies: List[Strategy]) -> None:
        """Process multiple strategies in hierarchy"""
        
        # Root container
        root = Container("root")
        
        # Create child for each strategy
        for i, strategy in enumerate(strategies):
            child = StrategyContainer(f"strategy_{i}", strategy)
            root.add_child(child)
        
        # Process events hierarchically
        root.start()
```

## Best Practices

### 1. Container Creation

```python
# Good: Explicit container creation with ID
container = BacktestContainer("backtest_2024_01_15_001")

# Bad: No container ID or random generation
container = BacktestContainer()
```

### 2. Resource Limits

```python
# Good: Set explicit resource limits
container = BacktestContainer(
    container_id="backtest_001",
    memory_limit_mb=500,
    cpu_shares=1024
)

# Bad: Unlimited resources
container = BacktestContainer("backtest_001")
```

### 3. Lifecycle Management

```python
# Good: Proper lifecycle management
try:
    container.initialize(config)
    container.start()
    results = container.run()
finally:
    container.stop()

# Bad: No cleanup
container.run()  # Leaks resources
```

### 4. Event Isolation

```python
# Good: Events scoped to container
self.event_bus.publish("SIGNAL", signal_data)

# Bad: Global event bus
global_event_bus.publish("SIGNAL", signal_data)
```

## Testing Container Hierarchies

### Unit Testing

```python
def test_container_isolation():
    """Test containers are truly isolated"""
    
    container1 = Container("test1")
    container2 = Container("test2")
    
    # Verify separate event buses
    assert container1.event_bus != container2.event_bus
    
    # Verify no shared state
    container1.state["key"] = "value1"
    assert "key" not in container2.state
```

### Integration Testing

```python
def test_parent_child_communication():
    """Test parent-child event flow"""
    
    parent = Container("parent")
    child = parent.create_child("child")
    
    events_received = []
    parent.event_bus.subscribe("CHILD_EVENT", events_received.append)
    
    # Child emits event
    child.event_bus.publish("CHILD_EVENT", {"data": "test"})
    
    # Parent receives it
    assert len(events_received) == 1
    assert events_received[0].source == "child"
```

## Performance Considerations

### Container Creation Overhead

- Target: < 10ms per container
- Includes: Event bus setup, component initialization
- Optimization: Reuse containers when possible

### Memory Overhead

- Base container: ~5MB
- With components: 20-50MB typical
- Large datasets: Monitor and limit

### Event Processing Overhead

- Event dispatch: < 0.1ms
- Event serialization: Avoid for performance
- Batch processing: Group events when possible

## Common Pitfalls

### 1. Shared State Between Containers

```python
# Wrong: Shared state
GLOBAL_CACHE = {}

class BadContainer:
    def process(self):
        GLOBAL_CACHE[self.id] = self.data  # Breaks isolation!
```

### 2. Direct Container References

```python
# Wrong: Direct reference
class BadParent:
    def create_child(self):
        self.child = Container("child")
        self.child.parent = self  # Child knows about parent!
```

### 3. Resource Leaks

```python
# Wrong: No cleanup
def run_backtest():
    container = Container("test")
    return container.run()  # Container never stopped!
```

## Summary

The container hierarchy provides:
- Complete isolation between components
- Hierarchical organization with clear boundaries
- Standardized lifecycle management
- Resource control and limits
- Event-based communication only

This architecture enables massive parallelization, reproducible results, and clean testing boundaries while maintaining system flexibility.