# Migration Guide: Inheritance to Protocol-Based Adapters

This guide helps you migrate from the old inheritance-based adapter system to the new protocol-based design.

## 🔄 Key Changes

### 1. No More Base Class Inheritance

**Before (Inheritance-based):**
```python
from .base_adapter import CommunicationAdapter, AdapterConfig

class MyAdapter(CommunicationAdapter):
    def __init__(self, config: AdapterConfig):
        super().__init__(config)  # Required!
        self.my_state = {}
        
    async def connect(self) -> bool:
        # Must implement abstract method
        return True
        
    async def disconnect(self) -> None:
        # Must implement abstract method
        pass
```

**After (Protocol-based):**
```python
from typing import Dict, Any
from .protocols import Container

class MyAdapter:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config  # No super() call!
        self.my_state = {}
        
    def setup(self, containers: Dict[str, Container]) -> None:
        # Configure connections
        pass
        
    def start(self) -> None:
        # Begin operation
        pass
```

### 2. Metrics and Logging via Composition

**Before:**
```python
class MyAdapter(CommunicationAdapter):
    def process_event(self, event):
        # Metrics automatically tracked by base class
        self.metrics.events_sent += 1
        self.logger.info("Processing event")
```

**After:**
```python
from .helpers import create_adapter_with_logging, handle_event_with_metrics

# Create adapter with infrastructure attached
adapter = create_adapter_with_logging(MyAdapter, name, config)

# Use helper for metrics
def handle_event(self, event: Event, source: Container):
    handle_event_with_metrics(self, event, source)
```

### 3. Event Routing Methods

**Before:**
```python
class MyAdapter(CommunicationAdapter):
    async def send_raw(self, data: bytes, correlation_id: Optional[str]) -> bool:
        # Send bytes implementation
        pass
        
    async def receive_raw(self) -> Optional[bytes]:
        # Receive bytes implementation
        pass
```

**After:**
```python
class MyAdapter:
    def route_event(self, event: Event, source: Container) -> None:
        # Direct event routing - no byte conversion
        target = self.select_target(event)
        target.receive_event(event)
```

## 📋 Step-by-Step Migration

### Step 1: Remove Inheritance
```python
# Remove base class and imports
- from .base_adapter import CommunicationAdapter, AdapterConfig
- class MyAdapter(CommunicationAdapter):
+ class MyAdapter:

# Update constructor
- def __init__(self, config: AdapterConfig):
-     super().__init__(config)
+ def __init__(self, name: str, config: Dict[str, Any]):
+     self.name = name
+     self.config = config
```

### Step 2: Replace Abstract Methods
```python
# Remove async abstract methods
- async def connect(self) -> bool:
- async def disconnect(self) -> None:
- async def send_raw(self, data: bytes) -> bool:
- async def receive_raw(self) -> Optional[bytes]:

# Add protocol methods
+ def setup(self, containers: Dict[str, Container]) -> None:
+ def start(self) -> None:
+ def stop(self) -> None:
+ def handle_event(self, event: Event, source: Container) -> None:
```

### Step 3: Use Helper Functions
```python
# For creating adapters with infrastructure
adapter = create_adapter_with_logging(MyAdapter, "my_adapter", config)

# For event handling with metrics
def handle_event(self, event: Event, source: Container):
    handle_event_with_metrics(self, event, source)
    
def route_event(self, event: Event, source: Container):
    # Your routing logic here
    pass
```

### Step 4: Update Configuration
```python
# Before - AdapterConfig dataclass
config = AdapterConfig(
    name="my_adapter",
    adapter_type="custom",
    retry_attempts=3,
    custom_settings={...}
)

# After - Plain dictionary
config = {
    'type': 'custom',
    'retry_attempts': 3,
    'custom_settings': {...}
}
```

## 🔧 Common Patterns

### Pattern 1: Simple Event Forwarding
```python
class ForwardingAdapter:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.source = None
        self.target = None
        
    def setup(self, containers: Dict[str, Container]):
        self.source = containers[self.config['source']]
        self.target = containers[self.config['target']]
        
    def start(self):
        # Subscribe to source events
        handler = lambda e: self.handle_event(e, self.source)
        self.source.event_bus.subscribe_all(handler)
        
    def handle_event(self, event: Event, source: Container):
        # Forward to target
        self.target.receive_event(event)
```

### Pattern 2: Conditional Routing
```python
class ConditionalAdapter:
    def route_event(self, event: Event, source: Container):
        # Check conditions
        if self.matches_condition(event):
            self.primary_target.receive_event(event)
        else:
            self.fallback_target.receive_event(event)
            
    def matches_condition(self, event: Event) -> bool:
        # Your condition logic
        return event.payload.get('priority') == 'high'
```

### Pattern 3: Multi-Target Distribution
```python
class DistributionAdapter:
    def route_event(self, event: Event, source: Container):
        # Distribute to multiple targets
        for target in self.targets:
            if self.should_route_to(event, target):
                target.receive_event(event)
```

## ⚠️ Important Notes

1. **No Async/Await**: The new system uses synchronous methods. If you need async behavior, implement it within your adapter.

2. **Direct Event Routing**: No more byte serialization - events are routed directly between containers.

3. **Container References**: Store container references during `setup()` for use in routing.

4. **Subscription Management**: You're responsible for subscribing to event sources in `start()`.

## 🚀 Using the Factory

Register your migrated adapter:
```python
from src.core.communication import AdapterFactory

factory = AdapterFactory()

# Register custom adapter type
factory.register_adapter_type('my_custom', 
    lambda n, c: create_adapter_with_logging(MyAdapter, n, c))

# Create instance
adapter = factory.create_adapter('instance1', {
    'type': 'my_custom',
    'source': 'container1',
    'target': 'container2'
})
```

## 📝 Checklist

- [ ] Remove base class inheritance
- [ ] Replace constructor to accept name and config dict
- [ ] Remove `super().__init__()` call
- [ ] Replace abstract methods with protocol methods
- [ ] Update to use helper functions for metrics/logging
- [ ] Convert async methods to sync
- [ ] Update configuration from dataclass to dict
- [ ] Test adapter with mock containers
- [ ] Register with factory if needed