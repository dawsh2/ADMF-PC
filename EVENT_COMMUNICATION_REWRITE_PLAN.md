# Event Communication System - Complete Rewrite Plan

## Overview

Complete ground-up rewrite of the event communication system using protocol-based design with zero inheritance.

## Core Principles

1. **No Inheritance** - Only protocols and composition
2. **No Base Classes** - Events and adapters are just data and functions
3. **Maximum Flexibility** - Any object with the right shape can participate
4. **Performance First** - Design for speed from the ground up
5. **Simple Over Complex** - Start minimal, add features only when needed

## Phase 1: Core Protocols (Week 1)

### Event Protocol
```python
from typing import Protocol, runtime_checkable, Optional, Dict, Any
from datetime import datetime

@runtime_checkable
class Event(Protocol):
    """Minimal event protocol - any object with these fields is an event"""
    event_id: str
    timestamp: datetime
    
# That's it! No base class, no inheritance

# Events are just dataclasses:
@dataclass
class MarketData:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    price: float = 0.0
    volume: int = 0
    
# Or plain dicts:
market_data = {
    "event_id": "123",
    "timestamp": datetime.utcnow(),
    "symbol": "AAPL",
    "price": 150.0,
    "volume": 1000000
}

# Or any object:
class MyCustomEvent:
    def __init__(self):
        self.event_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.data = "whatever"
```

### Container Protocol
```python
@runtime_checkable
class Container(Protocol):
    """Any object that can receive events"""
    def receive(self, event: Any) -> None: ...

# Containers can be anything:
class SimpleContainer:
    def receive(self, event):
        print(f"Got event: {event}")

# Or just functions:
def my_container(event):
    print(f"Function got: {event}")

# Or lambdas:
container = lambda e: print(f"Lambda got: {e}")
```

### Adapter Protocol
```python
@runtime_checkable
class Adapter(Protocol):
    """Any object that routes events"""
    def route(self, event: Any, source: str) -> None: ...

# Adapters are just routing logic:
class PipelineAdapter:
    def __init__(self, steps: list):
        self.steps = steps
    
    def route(self, event, source):
        for step in self.steps:
            step.receive(event)
```

## Phase 2: Core Implementations (Week 1-2)

### 1. Simple Event Types (No Inheritance!)
```python
# Just dataclasses with the right fields
from dataclasses import dataclass, field
from typing import Optional
import uuid

def make_id() -> str:
    return str(uuid.uuid4())

@dataclass
class MarketData:
    # Required by Event protocol
    event_id: str = field(default_factory=make_id)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Domain specific
    symbol: str = ""
    price: float = 0.0
    volume: int = 0
    
@dataclass
class Signal:
    # Required by Event protocol
    event_id: str = field(default_factory=make_id)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Domain specific
    symbol: str = ""
    action: str = "HOLD"  # BUY, SELL, HOLD
    strength: float = 0.0
    
@dataclass
class Order:
    # Required by Event protocol
    event_id: str = field(default_factory=make_id)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Domain specific
    symbol: str = ""
    side: str = "BUY"
    quantity: int = 0
    order_type: str = "MARKET"
```

### 2. Simple Adapters (No Base Class!)
```python
# Pipeline - just a function
def pipeline(*steps):
    def route(event, source):
        current = event
        for step in steps:
            if hasattr(step, 'receive'):
                step.receive(current)
            elif callable(step):
                current = step(current) or current
    return route

# Broadcast - just a function
def broadcast(*targets):
    def route(event, source):
        for target in targets:
            if hasattr(target, 'receive'):
                target.receive(event)
            elif callable(target):
                target(event)
    return route

# Filter - just a function
def filter_adapter(condition, target):
    def route(event, source):
        if condition(event):
            if hasattr(target, 'receive'):
                target.receive(event)
            elif callable(target):
                target(event)
    return route

# Compose them!
my_flow = pipeline(
    filter_adapter(lambda e: e.symbol == "AAPL", 
        broadcast(logger, analyzer)
    ),
    transformer,
    executor
)
```

### 3. Performance Tiers (Simple Functions)
```python
from collections import deque
import asyncio

# Fast tier - batching
def fast_tier(batch_size=1000, max_wait_ms=1):
    batch = deque()
    
    def route(event, source):
        batch.append((event, source))
        if len(batch) >= batch_size:
            flush_batch(batch)
    
    # Timer to flush periodically
    asyncio.create_task(periodic_flush(batch, max_wait_ms))
    return route

# Standard tier - async
def standard_tier():
    queue = asyncio.Queue()
    
    def route(event, source):
        queue.put_nowait((event, source))
    
    asyncio.create_task(process_queue(queue))
    return route

# Reliable tier - persistence
def reliable_tier(retry_count=3):
    def route(event, source):
        for attempt in range(retry_count):
            try:
                deliver(event, source)
                break
            except Exception as e:
                if attempt == retry_count - 1:
                    send_to_dead_letter(event, e)
                else:
                    time.sleep(2 ** attempt)
    return route
```

## Phase 3: Router Implementation (Week 2)

### Simple Router (No Inheritance!)
```python
class Router:
    """Simple event router - no inheritance needed"""
    
    def __init__(self):
        self.routes = {}  # {source: {event_type: [adapters]}}
        self.tiers = {
            'fast': fast_tier(),
            'standard': standard_tier(),
            'reliable': reliable_tier()
        }
    
    def register(self, source: str, event_type: type, adapter: Any, tier: str = 'standard'):
        """Register routing rule"""
        if source not in self.routes:
            self.routes[source] = {}
        if event_type not in self.routes[source]:
            self.routes[source][event_type] = []
        
        # Wrap adapter in appropriate tier
        tiered_adapter = self.tiers[tier]
        wrapped = lambda e, s: tiered_adapter(lambda: adapter.route(e, s))
        
        self.routes[source][event_type].append(wrapped)
    
    def route(self, event: Any, source: str):
        """Route event based on source and type"""
        if source in self.routes:
            event_type = type(event)
            if event_type in self.routes[source]:
                for adapter in self.routes[source][event_type]:
                    adapter(event, source)

# Usage
router = Router()
router.register("data_source", MarketData, pipeline_adapter, tier='fast')
router.register("strategy", Signal, risk_adapter, tier='standard')
router.register("risk", Order, execution_adapter, tier='reliable')
```

## Phase 4: Configuration System (Week 2-3)

### YAML Configuration
```yaml
# Simple, clean configuration
routing:
  # Source -> Event Type -> Routing
  data_source:
    MarketData:
      tier: fast
      routes:
        - type: pipeline
          steps: [indicators, strategies]
        
  strategy:
    Signal:
      tier: standard
      routes:
        - type: filter
          condition: "strength > 0.7"
          target: aggressive_risk
        - type: filter
          condition: "strength <= 0.7"
          target: conservative_risk
          
  risk:
    Order:
      tier: reliable
      routes:
        - type: pipeline
          steps: [validation, execution]
```

### Configuration Loader
```python
def load_routing(config: dict) -> Router:
    """Load routing from config - simple function"""
    router = Router()
    
    for source, event_configs in config['routing'].items():
        for event_type_name, config in event_configs.items():
            event_type = globals()[event_type_name]  # Or use a registry
            tier = config.get('tier', 'standard')
            
            for route_config in config['routes']:
                adapter = create_adapter(route_config)
                router.register(source, event_type, adapter, tier)
    
    return router

def create_adapter(config: dict) -> Any:
    """Create adapter from config - simple factory"""
    adapter_type = config['type']
    
    if adapter_type == 'pipeline':
        steps = [resolve_component(s) for s in config['steps']]
        return pipeline(*steps)
    
    elif adapter_type == 'filter':
        condition = eval(f"lambda e: e.{config['condition']}")
        target = resolve_component(config['target'])
        return filter_adapter(condition, target)
    
    elif adapter_type == 'broadcast':
        targets = [resolve_component(t) for t in config['targets']]
        return broadcast(*targets)
    
    # Add more as needed
```

## Phase 5: Integration (Week 3)

### Container Integration
```python
# Containers just need receive method
class DataContainer:
    def __init__(self, router):
        self.router = router
    
    def emit(self, event):
        self.router.route(event, "data_source")
    
    def process_data(self, data):
        event = MarketData(
            symbol=data['symbol'],
            price=data['price'],
            volume=data['volume']
        )
        self.emit(event)

class StrategyContainer:
    def __init__(self, router):
        self.router = router
    
    def receive(self, event):
        if isinstance(event, MarketData):
            signal = self.process_market_data(event)
            if signal:
                self.router.route(signal, "strategy")
    
    def process_market_data(self, data):
        # Strategy logic
        return Signal(
            symbol=data.symbol,
            action="BUY",
            strength=0.8
        )
```

## Phase 6: Advanced Features (Week 3-4)

### 1. Event Transformation
```python
# Simple transformer functions
def market_to_signal(event: MarketData) -> Signal:
    return Signal(
        symbol=event.symbol,
        action="BUY" if event.price > 100 else "SELL",
        strength=0.5
    )

def signal_to_order(event: Signal) -> Order:
    return Order(
        symbol=event.symbol,
        side=event.action,
        quantity=100,
        order_type="MARKET"
    )

# Use in pipeline
transform_pipeline = pipeline(
    market_to_signal,
    signal_to_order,
    executor
)
```

### 2. Monitoring
```python
# Simple metrics collection
class Metrics:
    def __init__(self):
        self.counts = {}
        self.latencies = []
    
    def record(self, event_type, latency):
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        self.latencies.append(latency)
    
    @property
    def stats(self):
        return {
            'total': sum(self.counts.values()),
            'by_type': self.counts,
            'avg_latency': sum(self.latencies) / len(self.latencies) if self.latencies else 0
        }

# Wrap any adapter with metrics
def with_metrics(adapter, metrics):
    def route(event, source):
        start = time.time()
        adapter.route(event, source)
        latency = time.time() - start
        metrics.record(type(event).__name__, latency)
    return route
```

### 3. Error Handling
```python
# Simple error handling wrapper
def with_error_handling(adapter, on_error=None):
    def route(event, source):
        try:
            adapter.route(event, source)
        except Exception as e:
            if on_error:
                on_error(event, source, e)
            else:
                logger.error(f"Error routing {event}: {e}")
    return route

# Circuit breaker
def circuit_breaker(adapter, failure_threshold=5, reset_timeout=60):
    failures = 0
    last_failure_time = None
    
    def route(event, source):
        nonlocal failures, last_failure_time
        
        # Check if circuit should reset
        if last_failure_time and time.time() - last_failure_time > reset_timeout:
            failures = 0
        
        # Check if circuit is open
        if failures >= failure_threshold:
            raise CircuitOpenError("Circuit breaker is open")
        
        try:
            adapter.route(event, source)
        except Exception as e:
            failures += 1
            last_failure_time = time.time()
            raise
    
    return route
```

## Implementation Checklist

### Week 1: Core
- [ ] Define Event, Container, and Adapter protocols
- [ ] Create basic event types (MarketData, Signal, Order)
- [ ] Implement simple adapters (pipeline, broadcast, filter)
- [ ] Build performance tier functions

### Week 2: Router & Config
- [ ] Implement Router class
- [ ] Create configuration schema
- [ ] Build configuration loader
- [ ] Add adapter factory

### Week 3: Integration
- [ ] Integrate with containers
- [ ] Add event transformation
- [ ] Implement monitoring
- [ ] Add error handling

### Week 4: Testing & Polish
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Documentation

## Key Benefits of This Approach

1. **Zero Inheritance**: Everything is protocols and functions
2. **Maximum Flexibility**: Any object with the right shape works
3. **Simple to Understand**: No complex class hierarchies
4. **Easy to Test**: Everything is just functions
5. **Performance**: Designed for speed from the start
6. **Composable**: Build complex flows from simple pieces

## Migration Strategy

Since we're rewriting from scratch:

1. **Build in parallel**: New system alongside old
2. **Test thoroughly**: Ensure feature parity
3. **Cut over gradually**: Route by route
4. **Keep it simple**: Don't add features until needed

The beauty of this approach is its simplicity. No inheritance, no complex base classes, just simple protocols and functions that compose together.