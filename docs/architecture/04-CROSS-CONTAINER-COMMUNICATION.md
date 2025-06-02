# Cross-Container Communication Architecture

This document defines the standard mechanism for communication between containers while maintaining isolation and architectural integrity.

## 🎯 Design Principles

1. **Isolation First** - Each container maintains its own event bus for internal operations
2. **Explicit Subscriptions** - Containers must explicitly declare interest in external events
3. **No Direct References** - Containers never directly access each other
4. **Discoverable Sources** - Containers advertise what events they publish
5. **Type-Safe Routing** - Strong typing for event routes and contracts
6. **Configuration-Driven** - Routing topology defined in YAML, not code

## 🏗️ Architecture Overview

### Event Router Pattern

```
┌─────────────────────────────────────────────────────┐
│                  Event Router                        │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │          Subscription Registry               │   │
│  │  ┌────────────────────────────────────────┐ │   │
│  │  │ indicator_container.INDICATOR →         │ │   │
│  │  │   [strategy_container_a, b, c]         │ │   │
│  │  │ data_container.BAR →                   │ │   │
│  │  │   [indicator_container, strategy_*]    │ │   │
│  │  │ strategy_container.SIGNAL →            │ │   │
│  │  │   [risk_container]                     │ │   │
│  │  └────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │           Route Management                   │   │
│  │  • Register publishers                      │   │
│  │  • Register subscribers                     │   │
│  │  • Validate connections                     │   │
│  │  • Handle lifecycle                         │   │
│  │  • Prevent cycles                           │   │
│  │  • Performance optimization                 │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Container Event Interface

Every container must implement the standard interface:

```python
from typing import Set, Dict, Protocol
from src.core.events import Event, EventType

class ContainerEventInterface(Protocol):
    """Standard interface for cross-container event communication"""
    
    # Publishing declarations
    publishes: Set[EventType] = set()
    
    # Subscription declarations  
    subscribes_to: Dict[EventType, Set[str]] = {}
    
    def register_with_router(self, router: 'EventRouter') -> None:
        """Register this container's pub/sub needs with router"""
        ...
    
    def handle_routed_event(self, event: Event, source: str) -> None:
        """Handle event routed from another container"""
        ...
```

## 📡 Event Flow Patterns

### 1. Standard Flow

```
Container A              Event Router              Container B
    │                         │                         │
    │──publish(EVENT)─────────►│                         │
    │                         │                         │
    │                         │──handle_routed_event───►│
    │                         │   (event, "container_a") │
    │                         │                         │
    │                         │◄──publish(ACK)─────────│
    │◄────handle_routed_event─│                         │
    │   (ack, "container_b")  │                         │
```

### 2. Broadcast Flow

```
Data Container           Event Router         Multiple Subscribers
    │                         │                         │
    │──publish(BAR_EVENT)─────►│                         │
    │                         ├──handle_routed_event───►│ Indicator Container
    │                         ├──handle_routed_event───►│ Strategy Container A  
    │                         ├──handle_routed_event───►│ Strategy Container B
    │                         └──handle_routed_event───►│ Analytics Container
```

### 3. Filtered Flow

```yaml
# Only route events matching filters
subscription:
  source: "data_container"
  events: ["BAR"]
  filters:
    symbol: "SPY"
    timeframe: "1min"
```

## 🔧 Implementation

### Core Event Router

```python
# src/core/events/event_router.py
from typing import Dict, Set, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class DeliveryMode(Enum):
    ASYNC = "async"
    SYNC = "sync"
    BATCH = "batch"

@dataclass
class Subscription:
    """Event subscription configuration"""
    subscriber_id: str
    source_container: str
    event_types: Set[EventType]
    filters: Dict[str, any]
    delivery_mode: DeliveryMode
    callback: Callable[[Event, str], None]

class EventRouter:
    """Central event router for cross-container communication"""
    
    def __init__(self):
        self.publishers: Dict[str, Set[EventType]] = {}
        self.subscriptions: List[Subscription] = []
        self.subscription_graph = SubscriptionGraph()
        
    def register_publisher(self, container_id: str, event_types: Set[EventType]):
        """Register a container as publisher of event types"""
        self.publishers[container_id] = event_types
        self.subscription_graph.add_publisher(container_id, event_types)
        
    def register_subscriber(self, 
                          subscriber_id: str,
                          source_container: str, 
                          event_types: Set[EventType],
                          filters: Dict[str, any] = None,
                          delivery_mode: DeliveryMode = DeliveryMode.ASYNC,
                          callback: Callable[[Event, str], None] = None):
        """Register a subscription"""
        
        # Validate subscription
        if not self._validate_subscription(subscriber_id, source_container, event_types):
            raise ValueError(f"Invalid subscription: {subscriber_id} -> {source_container}")
            
        subscription = Subscription(
            subscriber_id=subscriber_id,
            source_container=source_container,
            event_types=event_types,
            filters=filters or {},
            delivery_mode=delivery_mode,
            callback=callback
        )
        
        self.subscriptions.append(subscription)
        self.subscription_graph.add_subscription(subscription)
        
    def route_event(self, event: Event, source_container: str):
        """Route event to interested subscribers"""
        matching_subscriptions = self._find_matching_subscriptions(
            event, source_container
        )
        
        for subscription in matching_subscriptions:
            if subscription.delivery_mode == DeliveryMode.SYNC:
                self._deliver_sync(event, source_container, subscription)
            else:
                self._deliver_async(event, source_container, subscription)
                
    def _validate_subscription(self, subscriber: str, source: str, events: Set[EventType]) -> bool:
        """Validate subscription doesn't create cycles or invalid routes"""
        # Check if source publishes these events
        if source not in self.publishers:
            return False
            
        if not events.issubset(self.publishers[source]):
            return False
            
        # Check for cycles
        if self.subscription_graph.would_create_cycle(subscriber, source):
            return False
            
        return True
```

### Container Base Implementation

```python
# src/core/containers/event_capable_container.py
class EventCapableContainer:
    """Base container with cross-container event capabilities"""
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.local_event_bus = EventBus()
        self.router: Optional[EventRouter] = None
        
        # Must be defined by subclasses
        self.publishes: Set[EventType] = set()
        self.subscribes_to: Dict[EventType, Set[str]] = {}
        
    def register_with_router(self, router: EventRouter):
        """Register with event router"""
        self.router = router
        
        # Register as publisher
        router.register_publisher(self.container_id, self.publishes)
        
        # Register subscriptions
        for event_type, sources in self.subscribes_to.items():
            for source in sources:
                router.register_subscriber(
                    subscriber_id=self.container_id,
                    source_container=source,
                    event_types={event_type},
                    callback=self.handle_routed_event
                )
                
    def publish_external(self, event: Event):
        """Publish event for external routing"""
        if self.router and event.type in self.publishes:
            self.router.route_event(event, self.container_id)
            
    def handle_routed_event(self, event: Event, source: str):
        """Handle events routed from other containers"""
        # Default implementation - subclasses should override
        self.local_event_bus.publish(event)
```

## ⚙️ Configuration Schema

### Container Routing Configuration

```yaml
# Container routing configuration
containers:
  data_container:
    publishes:
      - events: ["BAR", "TICK"]
        visibility: "global"
        
  indicator_container:
    subscribes_to:
      - source: "data_container"
        events: ["BAR"]
        delivery: "sync"
        filters:
          timeframe: ["1min", "5min"]
          
    publishes:
      - events: ["INDICATOR"]
        visibility: "subscribers"
        
  strategy_container:
    subscribes_to:
      - source: "data_container"
        events: ["BAR"]
        delivery: "async"
        
      - source: "indicator_container"
        events: ["INDICATOR"]
        delivery: "async"
        filters:
          symbol: "SPY"
          
    publishes:
      - events: ["SIGNAL"]
        visibility: "parent"
        
  risk_container:
    subscribes_to:
      - source: "strategy_container"
        events: ["SIGNAL"]
        delivery: "sync"  # Critical path
        
    publishes:
      - events: ["ORDER", "RISK_ALERT"]
        visibility: "global"
```

### Advanced Routing Features

```yaml
# Advanced routing configuration
routing:
  # Batch delivery for high-frequency events
  batching:
    enabled: true
    batch_size: 100
    max_delay_ms: 10
    
  # Event filtering
  filters:
    - name: "symbol_filter"
      expression: "event.data.symbol in ['SPY', 'QQQ']"
      
    - name: "priority_filter"
      expression: "event.priority >= 'HIGH'"
      
  # Dead letter queue
  dead_letter_queue:
    enabled: true
    max_retries: 3
    retry_delay_ms: 1000
    
  # Performance monitoring
  monitoring:
    track_latency: true
    track_throughput: true
    alert_on_backlog: 1000
```

## 🔍 Debugging and Visualization

### Event Flow Inspection

```python
# Debugging tools
def visualize_event_flow(router: EventRouter):
    """Generate visual representation of event flows"""
    graph = router.subscription_graph.to_networkx()
    
    # Generate graph visualization
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    
def inspect_subscriptions(router: EventRouter, container_id: str):
    """Inspect all subscriptions for a container"""
    subs = router.get_subscriptions_for_container(container_id)
    
    for sub in subs:
        print(f"{sub.subscriber_id} ← {sub.source_container}.{sub.event_types}")
        
def trace_event_path(router: EventRouter, event: Event, source: str):
    """Trace the path of an event through the system"""
    path = router.trace_event_delivery(event, source)
    
    for step in path:
        print(f"{step.source} → {step.destination} ({step.latency_ms}ms)")
```

### Performance Monitoring

```python
# Performance monitoring integration
class RouterMetrics:
    def __init__(self):
        self.event_counts: Dict[str, int] = {}
        self.latency_stats: Dict[str, List[float]] = {}
        self.subscription_health: Dict[str, bool] = {}
        
    def record_event_delivery(self, route: str, latency_ms: float):
        """Record event delivery metrics"""
        self.event_counts[route] = self.event_counts.get(route, 0) + 1
        
        if route not in self.latency_stats:
            self.latency_stats[route] = []
        self.latency_stats[route].append(latency_ms)
        
    def get_health_summary(self) -> Dict:
        """Get overall router health"""
        return {
            "total_events": sum(self.event_counts.values()),
            "avg_latency_ms": self._calculate_avg_latency(),
            "failed_subscriptions": len([s for s in self.subscription_health.values() if not s]),
            "active_routes": len(self.event_counts)
        }
```

## 🛡️ Safety and Constraints

### Cycle Prevention

```python
class SubscriptionGraph:
    """Manages subscription graph and prevents cycles"""
    
    def would_create_cycle(self, subscriber: str, source: str) -> bool:
        """Check if adding subscription would create cycle"""
        # Use topological sort to detect cycles
        temp_graph = self.graph.copy()
        temp_graph.add_edge(source, subscriber)
        
        try:
            list(nx.topological_sort(temp_graph))
            return False
        except nx.NetworkXError:
            return True
```

### Type Safety

```python
from typing import TypeVar, Generic

EventDataT = TypeVar('EventDataT')

class TypedEvent(Generic[EventDataT]):
    """Type-safe event with payload validation"""
    
    def __init__(self, event_type: EventType, data: EventDataT):
        self.type = event_type
        self.data = data
        self._validate_data()
        
    def _validate_data(self):
        """Validate event data matches expected schema"""
        # Schema validation logic
        pass
```

## 📋 Best Practices

### 1. Subscription Design

```python
# GOOD: Explicit, minimal subscriptions
subscribes_to = {
    "INDICATOR": {"indicator_container"},
    "BAR": {"data_container"}
}

# BAD: Overly broad subscriptions
subscribes_to = {
    "*": {"*"}  # Too broad, performance issues
}
```

### 2. Event Design

```python
# GOOD: Well-structured events
@dataclass
class SignalEvent:
    symbol: str
    action: str
    strength: float
    metadata: Dict[str, any]

# BAD: Unstructured events
event_data = {"stuff": "random", "data": 123}
```

### 3. Error Handling

```python
def handle_routed_event(self, event: Event, source: str):
    """Handle routed events with proper error handling"""
    try:
        if event.type == "INDICATOR":
            self._process_indicator(event)
        elif event.type == "BAR":
            self._process_bar(event)
        else:
            self.logger.warning(f"Unknown event type: {event.type}")
            
    except Exception as e:
        self.logger.error(
            f"Failed to process routed event",
            event_type=event.type,
            source=source,
            exc_info=True
        )
        # Don't let failures in one container break others
```

## 🎯 Migration Guide

### From Direct Communication

```python
# OLD: Direct container access (WRONG)
class StrategyContainer:
    def __init__(self, indicator_container):
        self.indicator_container = indicator_container  # NO!
        
    def get_indicators(self):
        return self.indicator_container.get_latest()  # NO!

# NEW: Event-driven communication (CORRECT)
class StrategyContainer(EventCapableContainer):
    publishes = {"SIGNAL"}
    subscribes_to = {"INDICATOR": {"indicator_container"}}
    
    def handle_routed_event(self, event: Event, source: str):
        if event.type == "INDICATOR":
            self._process_indicators(event.data)
```

## 📚 References

- [Event-Driven Architecture](01-EVENT-DRIVEN-ARCHITECTURE.md) - Core event patterns
- [Container Hierarchy](02-CONTAINER-HIERARCHY.md) - Container design principles
- [Protocol Composition](03-PROTOCOL-COMPOSITION.md) - Protocol-based design
- [Style Guide](../standards/STYLE-GUIDE.md) - Implementation standards

---

*This architecture ensures containers remain isolated while enabling flexible, type-safe communication through a central routing system.*