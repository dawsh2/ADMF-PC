# Container Lifecycle Management

## Overview

The container lifecycle follows a deterministic pattern that ensures consistent resource management and proper cleanup. Each container progresses through defined states: initialization, component registration, event bus wiring, execution, and disposal. This lifecycle is managed by the container factory system, which enforces identical creation patterns regardless of the complexity of the enclosed components.

## Why Lifecycle Management Matters

The lifecycle management addresses a critical challenge in distributed systems: ensuring that resource allocation and cleanup happen predictably. Traditional trading frameworks often suffer from resource leaks or inconsistent initialization ordering that can affect backtest results. By enforcing a standardized lifecycle, ADMF-PC eliminates these sources of non-determinism while enabling reliable resource tracking and debugging capabilities.

## Container Lifecycle States

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Container Lifecycle States                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CREATED ──► INITIALIZING ──► READY ──► RUNNING ──► DISPOSING      │
│     │             │             │          │            │           │
│     │             │             │          │            │           │
│     └─────────────┴─────────────┴──────────┴────────────┴──► ERROR │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### State Definitions

1. **CREATED**: Container instance exists but no resources allocated
2. **INITIALIZING**: Allocating resources, creating components
3. **READY**: All components initialized, event bus wired
4. **RUNNING**: Actively processing events
5. **DISPOSING**: Cleaning up resources
6. **ERROR**: Failed state requiring cleanup

## Detailed Lifecycle Phases

### Phase 1: Creation
```python
def create_container(container_type: str, config: Dict) -> Container:
    """Factory creates container in CREATED state"""
    container = Container(
        container_id=generate_id(),
        container_type=container_type,
        state=ContainerState.CREATED
    )
    container.config = config
    return container
```

### Phase 2: Initialization
```python
def initialize(self) -> None:
    """Transition from CREATED to INITIALIZING to READY"""
    try:
        self.state = ContainerState.INITIALIZING
        
        # 1. Create isolated event bus
        self.event_bus = EventBus(container_id=self.container_id)
        
        # 2. Allocate resources
        self._allocate_resources()
        
        # 3. Create components
        self._create_components()
        
        # 4. Wire event connections
        self._wire_events()
        
        # 5. Validate configuration
        self._validate_setup()
        
        self.state = ContainerState.READY
        
    except Exception as e:
        self.state = ContainerState.ERROR
        self._emergency_cleanup()
        raise ContainerInitializationError(f"Failed to initialize: {e}")
```

### Phase 3: Execution
```python
def run(self) -> None:
    """Transition from READY to RUNNING"""
    if self.state != ContainerState.READY:
        raise InvalidStateError(f"Cannot run from state {self.state}")
        
    self.state = ContainerState.RUNNING
    
    try:
        # Start event processing
        self.event_bus.start()
        
        # Execute container logic
        self._execute()
        
    except Exception as e:
        self.state = ContainerState.ERROR
        raise ContainerExecutionError(f"Execution failed: {e}")
```

### Phase 4: Disposal
```python
def dispose(self) -> None:
    """Clean shutdown from any state"""
    if self.state == ContainerState.DISPOSING:
        return  # Already disposing
        
    previous_state = self.state
    self.state = ContainerState.DISPOSING
    
    try:
        # 1. Stop event processing
        if self.event_bus:
            self.event_bus.stop()
        
        # 2. Dispose components in reverse order
        for component in reversed(self.components):
            component.dispose()
        
        # 3. Release resources
        self._release_resources()
        
        # 4. Clear references
        self.components.clear()
        self.event_bus = None
        
    except Exception as e:
        logger.error(f"Error during disposal from {previous_state}: {e}")
        # Continue cleanup despite errors
```

## Resource Management Patterns

### Memory Allocation
```python
class ResourceManager:
    def allocate_for_container(self, container_id: str, config: Dict) -> Resources:
        """Allocate resources with limits"""
        resources = Resources()
        
        # Set memory limits
        resources.memory_limit = config.get('memory_limit', '1GB')
        
        # Allocate data buffers
        resources.data_buffer = self._allocate_buffer(
            size=config.get('buffer_size', '100MB')
        )
        
        # Track allocation
        self.allocations[container_id] = resources
        
        return resources
```

### Component Registration Order
```python
def _create_components(self) -> None:
    """Create components in dependency order"""
    # 1. Data handlers (no dependencies)
    self.data_handler = self._create_data_handler()
    
    # 2. Features (depend on data)
    self.feature_hub = self._create_feature_hub()
    
    # 3. Strategies (depend on features)
    self.strategies = self._create_strategies()
    
    # 4. Risk manager (depends on strategies)
    self.risk_manager = self._create_risk_manager()
    
    # 5. Execution engine (depends on risk)
    self.execution_engine = self._create_execution_engine()
```

### Event Bus Wiring
```python
def _wire_events(self) -> None:
    """Wire event connections in proper order"""
    # Data → Features
    self.event_bus.subscribe(
        EventType.BAR,
        self.feature_hub.on_bar,
        source=self.data_handler
    )
    
    # Features → Strategies
    self.event_bus.subscribe(
        EventType.FEATURE,
        self.strategy.on_feature,
        source=self.feature_hub
    )
    
    # Strategies → Risk
    self.event_bus.subscribe(
        EventType.SIGNAL,
        self.risk_manager.on_signal,
        source=self.strategy
    )
    
    # Risk → Execution
    self.event_bus.subscribe(
        EventType.ORDER,
        self.execution_engine.on_order,
        source=self.risk_manager
    )
```

## Initialization Ordering Guarantees

The container system provides strict guarantees about initialization order:

1. **Parent Before Child**: Parent containers fully initialize before children
2. **Dependencies First**: Components initialize in dependency order
3. **Events After Components**: Event wiring happens after all components exist
4. **Validation Last**: Configuration validation after everything is wired

### Example: Nested Container Initialization

```python
def initialize_nested_containers(self) -> None:
    """Initialize container hierarchy"""
    # 1. Initialize parent container
    parent = BacktestContainer()
    parent.initialize()  # CREATED → READY
    
    # 2. Create child containers
    for classifier_config in self.config.classifiers:
        child = ClassifierContainer(
            parent=parent,
            config=classifier_config
        )
        # 3. Initialize child
        child.initialize()  # CREATED → READY
        
        # 4. Connect parent-child events
        parent.event_bus.bridge_child(child.event_bus)
        
        parent.children.append(child)
```

## Error Handling and Recovery

### Graceful Degradation
```python
def handle_component_error(self, component: Component, error: Exception) -> None:
    """Handle component failure without killing container"""
    logger.error(f"Component {component} failed: {error}")
    
    if component.is_critical:
        # Critical component - must stop container
        self.state = ContainerState.ERROR
        self.dispose()
    else:
        # Non-critical - disable and continue
        component.disable()
        self.event_bus.unsubscribe_all(component)
        logger.warning(f"Disabled non-critical component {component}")
```

### Cleanup Guarantees
```python
class Container:
    def __init__(self):
        # Register cleanup handler
        atexit.register(self._emergency_cleanup)
        
    def _emergency_cleanup(self) -> None:
        """Ensure cleanup even on unexpected exit"""
        if self.state not in [ContainerState.DISPOSING, ContainerState.CREATED]:
            logger.warning(f"Emergency cleanup from state {self.state}")
            try:
                self.dispose()
            except:
                pass  # Best effort cleanup
```

## Testing Lifecycle Management

### Lifecycle State Verification
```python
def test_container_lifecycle():
    """Test complete lifecycle transitions"""
    container = Container()
    
    # Initial state
    assert container.state == ContainerState.CREATED
    
    # Initialization
    container.initialize()
    assert container.state == ContainerState.READY
    assert container.event_bus is not None
    assert len(container.components) > 0
    
    # Execution
    container.run()
    assert container.state == ContainerState.RUNNING
    
    # Disposal
    container.dispose()
    assert container.state == ContainerState.DISPOSING
    assert container.event_bus is None
    assert len(container.components) == 0
```

### Resource Leak Detection
```python
def test_no_resource_leaks():
    """Ensure proper resource cleanup"""
    initial_memory = get_memory_usage()
    
    # Create and destroy many containers
    for _ in range(100):
        container = create_large_container()
        container.initialize()
        container.run()
        container.dispose()
    
    gc.collect()
    final_memory = get_memory_usage()
    
    # Memory should not grow significantly
    assert final_memory - initial_memory < 10 * MB
```

## Best Practices

### DO:
- Always check state before transitions
- Clean up resources in reverse order of creation
- Handle errors at each lifecycle phase
- Use context managers for automatic cleanup
- Log state transitions for debugging

### DON'T:
- Skip disposal on errors
- Assume components clean themselves up
- Create circular references between containers
- Mix lifecycle management with business logic
- Ignore resource limits

## Advanced Patterns

### Container Pooling
```python
class ContainerPool:
    """Reuse containers for performance"""
    def __init__(self, container_type: Type[Container], pool_size: int = 10):
        self.container_type = container_type
        self.available = []
        self.in_use = set()
        
        # Pre-create containers
        for _ in range(pool_size):
            container = container_type()
            container.initialize()
            self.available.append(container)
    
    def acquire(self) -> Container:
        """Get container from pool"""
        if self.available:
            container = self.available.pop()
        else:
            container = self.container_type()
            container.initialize()
            
        self.in_use.add(container)
        container.reset()  # Clear previous state
        return container
    
    def release(self, container: Container) -> None:
        """Return container to pool"""
        self.in_use.remove(container)
        if len(self.available) < self.pool_size:
            self.available.append(container)
        else:
            container.dispose()
```

### Lifecycle Hooks
```python
class Container:
    """Container with lifecycle hooks"""
    def __init__(self):
        self.lifecycle_hooks = {
            'pre_init': [],
            'post_init': [],
            'pre_run': [],
            'post_run': [],
            'pre_dispose': [],
            'post_dispose': []
        }
    
    def add_lifecycle_hook(self, phase: str, hook: Callable) -> None:
        """Add hook to lifecycle phase"""
        self.lifecycle_hooks[phase].append(hook)
    
    def initialize(self) -> None:
        """Initialize with hooks"""
        for hook in self.lifecycle_hooks['pre_init']:
            hook(self)
            
        # Normal initialization
        self._do_initialization()
        
        for hook in self.lifecycle_hooks['post_init']:
            hook(self)
```

## Summary

Container lifecycle management is critical for:

1. **Deterministic Behavior**: Same initialization order every time
2. **Resource Safety**: No leaks or dangling references
3. **Error Recovery**: Graceful handling of failures
4. **Performance**: Efficient resource allocation and reuse
5. **Debugging**: Clear state transitions and logging

The standardized lifecycle ensures that complex multi-container systems behave predictably and can be tested reliably.