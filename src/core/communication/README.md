# Protocol-Based Communication Adapters

This package implements ADMF-PC's protocol-based communication adapter system, connecting isolated container event buses without inheritance.

## 🏗️ Architecture Principles

### 1. No Inheritance
Adapters implement protocols, not inherit from base classes. This provides maximum flexibility and avoids the fragile base class problem.

```python
# ❌ Old way - Inheritance
class PipelineAdapter(CommunicationAdapter):
    def __init__(self, config):
        super().__init__(config)  # Tight coupling!

# ✅ New way - Protocol implementation
class PipelineAdapter:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config  # No super(), no inheritance!
```

### 2. Composition Over Inheritance
Common functionality is provided through helper functions, not base class methods:

```python
# Helper functions provide reusable behavior
adapter = create_adapter_with_logging(PipelineAdapter, name, config)
handle_event_with_metrics(adapter, event, source)
```

### 3. Protocol Compliance
Containers and adapters must implement specific protocols to work together:

```python
@runtime_checkable
class Container(Protocol):
    """What makes something a container"""
    @property
    def name(self) -> str: ...
    @property
    def event_bus(self) -> EventBusProtocol: ...
    def receive_event(self, event: Event) -> None: ...
    def publish_event(self, event: Event) -> None: ...
```

## 📦 Available Adapters

### Pipeline Adapter
Routes events sequentially through containers:
```python
config = {
    'type': 'pipeline',
    'containers': ['data', 'strategy', 'risk', 'execution']
}
```

### Broadcast Adapter
Sends events from one source to multiple targets:
```python
config = {
    'type': 'broadcast',
    'source': 'market_data',
    'targets': ['strategy1', 'strategy2', 'strategy3']
}
```

### Hierarchical Adapter
Tree-structured event routing with up/down propagation:
```python
config = {
    'type': 'hierarchical',
    'root': 'supervisor',
    'hierarchy': {
        'risk_manager': {
            'portfolio1': {},
            'portfolio2': {}
        }
    }
}
```

### Selective Adapter
Content-based routing with rules:
```python
config = {
    'type': 'selective',
    'source': 'signal_generator',
    'routing_rules': [{
        'target': 'aggressive_strategy',
        'conditions': [
            {'field': 'payload.confidence', 'operator': 'greater_than', 'value': 0.8}
        ]
    }]
}
```

## 🚀 Usage Examples

### Basic Setup
```python
from src.core.communication import AdapterFactory

# Create factory
factory = AdapterFactory()

# Create adapter from config
adapter = factory.create_adapter('main_pipeline', {
    'type': 'pipeline',
    'containers': ['data', 'strategy', 'risk']
})

# Setup with containers
adapter.setup(containers)
adapter.start()
```

### Advanced Patterns

#### Conditional Pipeline
```python
adapter = factory.create_adapter('conditional_flow', {
    'type': 'conditional_pipeline',
    'containers': ['input', 'validator', 'processor', 'output'],
    'conditions': [{
        'stage': 'validator',
        'skip_if': 'event.payload.get("validated") == True'
    }]
})
```

#### Load-Balanced Router
```python
adapter = factory.create_adapter('load_balancer', {
    'type': 'load_balanced_router',
    'source': 'gateway',
    'strategy': 'round_robin',  # or 'random', 'least_used', 'weighted'
    'routing_rules': [...]
})
```

#### Fan-Out with Transformations
```python
adapter = factory.create_adapter('transformer', {
    'type': 'fan_out',
    'source': 'raw_data',
    'targets': [{
        'name': 'ml_strategy',
        'transform': lambda e: transform_to_ml_format(e)
    }, {
        'name': 'rule_strategy',
        'transform': lambda e: transform_to_rule_format(e)
    }]
})
```

## 🔧 Creating Custom Adapters

To create a custom adapter, just implement the protocol:

```python
class MyCustomAdapter:
    """Custom adapter - no inheritance needed!"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.containers = {}
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure with containers"""
        self.containers = containers
        
    def start(self) -> None:
        """Begin operation"""
        # Set up subscriptions
        pass
        
    def stop(self) -> None:
        """Shutdown"""
        # Clean up subscriptions
        pass
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Process events"""
        # Your custom routing logic
        pass

# Register with factory
factory.register_adapter_type('my_custom', 
    lambda n, c: create_adapter_with_logging(MyCustomAdapter, n, c))
```

## 📊 Metrics and Monitoring

All adapters support metrics through composition:

```python
# Metrics are attached via helper
adapter = create_adapter_with_logging(PipelineAdapter, name, config)

# Access metrics
print(f"Events processed: {adapter.metrics.success_count}")
print(f"Average latency: {adapter.metrics.get_average_latency()}")
```

## 🧪 Testing Adapters

Test adapters using mock containers:

```python
@dataclass
class MockContainer:
    """Mock container for testing"""
    name: str
    event_bus: MockEventBus = field(default_factory=MockEventBus)
    received_events: List[Event] = field(default_factory=list)
    
    def receive_event(self, event: Event) -> None:
        self.received_events.append(event)

# Test adapter
containers = {'source': MockContainer('source'), ...}
adapter.setup(containers)
adapter.start()

# Verify routing
adapter.handle_event(test_event, containers['source'])
assert len(containers['target'].received_events) == 1
```

## 🎯 Best Practices

1. **Use Type Hints**: All adapters should use type hints for clarity
2. **Validate Config**: Use `validate_adapter_config()` helper
3. **Handle Errors**: Use try/except in routing logic
4. **Log Events**: Use structured logging for debugging
5. **Test Thoroughly**: Test all routing paths and edge cases

## 🚫 Common Pitfalls

1. **Don't inherit**: Resist the urge to create base classes
2. **Don't share state**: Each adapter should be independent
3. **Don't assume order**: Event delivery order isn't guaranteed
4. **Don't block**: Keep event handling fast and async-friendly

## 📚 Further Reading

- [Event-Driven Architecture](../../docs/architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)
- [Container Organization](../../docs/architecture/container-organization-patterns_v2.md)
- [System Architecture](../../docs/SYSTEM_ARCHITECTURE_v5.MD)