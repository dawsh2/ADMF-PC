# Event Adapters

Complete reference for ADMF-PC's event communication system including event scopes, routing patterns, and cross-container communication based on the actual implementation.

## 🔄 Event System Overview

ADMF-PC implements a sophisticated event-driven architecture where containers communicate exclusively through events. The Event Router manages cross-container communication while maintaining complete isolation between containers.

### Core Event Concepts

**From actual implementation**:
- **Event Bus per Container**: Each container has its own isolated event bus
- **Cross-Container Routing**: HybridContainerInterface bridges internal and external events
- **Scoped Communication**: Events routed based on container hierarchy and scope
- **Type-Safe Events**: All events are strongly typed with validation
- **Performance Tiers**: Different delivery guarantees based on event importance

## 📡 Event Scopes

The Event Router uses scopes to control how events flow between containers:

### Available Event Scopes

**From actual `EventScope` enum**:

#### LOCAL Scope
```python
scope = EventScope.LOCAL
```
**Description**: Events stay within the same container
**Use cases**: Internal component communication within a container
**Example**: Strategy component communicating with its internal risk manager

#### PARENT Scope
```python
scope = EventScope.PARENT
```
**Description**: Events sent to the parent container
**Use cases**: Child reporting to parent, escalating events up hierarchy
**Example**: Strategy container sending signals to parent Risk container

#### CHILDREN Scope
```python
scope = EventScope.CHILDREN
```
**Description**: Events sent to all direct child containers
**Use cases**: Parent broadcasting to immediate children
**Example**: Data container broadcasting market data to indicator children

#### SIBLINGS Scope
```python
scope = EventScope.SIBLINGS
```
**Description**: Events sent to sibling containers at same hierarchy level
**Use cases**: Peer-to-peer communication between related containers
**Example**: Multiple strategy containers sharing regime information

#### UPWARD Scope
```python
scope = EventScope.UPWARD
```
**Description**: Events propagated up the container hierarchy
**Use cases**: Escalating events to higher-level containers
**Example**: Trading signals flowing from Strategy → Risk → Portfolio → Execution

#### DOWNWARD Scope
```python
scope = EventScope.DOWNWARD
```
**Description**: Events propagated down the container hierarchy
**Use cases**: Broadcasting information to all descendants
**Example**: Market data flowing from Data down to all strategy components

#### GLOBAL Scope
```python
scope = EventScope.GLOBAL
```
**Description**: Events sent to all containers in the system
**Use cases**: System-wide notifications, critical alerts
**Example**: Emergency stop signals, system health alerts

## 🔌 Event Router Architecture

### HybridContainerInterface

The actual implementation uses a `HybridContainerInterface` that bridges internal container events with external routing:

```python
class HybridContainerInterface:
    """Bridges internal container events with external event routing."""
    
    def __init__(self, container_id: str, internal_bus: EventBus):
        self.container_id = container_id
        self.internal_bus = internal_bus
        self.external_router = None
        
    def publish_external(
        self,
        event: Event,
        scope: EventScope,
        target_roles: Optional[List[ContainerRole]] = None
    ) -> None:
        """Publish event externally via router."""
        if self.external_router:
            self.external_router.route_event(
                source_container=self.container_id,
                event=event,
                scope=scope,
                target_roles=target_roles
            )
    
    def subscribe_external(
        self,
        event_types: List[Type[Event]],
        scope: EventScope,
        handler: Callable
    ) -> None:
        """Subscribe to external events."""
        if self.external_router:
            self.external_router.subscribe(
                container_id=self.container_id,
                event_types=event_types,
                scope=scope,
                handler=handler
            )
```

### Event Router Configuration

```yaml
# Event routing configuration in YAML
event_routing:
  # Router settings
  router_config:
    buffer_size: 10000
    max_latency_ms: 100
    delivery_guarantee: "at_least_once"
    
  # Default routing patterns
  default_patterns:
    market_data:
      scope: "DOWNWARD"
      event_types: ["BarEvent", "TickEvent", "QuoteEvent"]
      target_roles: ["INDICATOR", "STRATEGY"]
      
    trading_signals:
      scope: "UPWARD"
      event_types: ["TradingSignal"]
      target_roles: ["RISK", "PORTFOLIO", "EXECUTION"]
      
    portfolio_updates:
      scope: "GLOBAL"
      event_types: ["PortfolioUpdate", "PositionUpdate"]
      
    system_events:
      scope: "GLOBAL"
      event_types: ["SystemAlert", "HealthCheck", "PerformanceMetric"]
      
  # Custom routing rules
  custom_rules:
    # Strategy to Risk direct communication
    - name: "strategy_risk_direct"
      source_role: "STRATEGY"
      target_role: "RISK"
      event_types: ["TradingSignal", "PositionRequest"]
      scope: "PARENT"
      priority: "high"
      
    # Classifier to Strategies broadcast
    - name: "regime_broadcast"
      source_role: "CLASSIFIER"
      target_role: "STRATEGY"
      event_types: ["RegimeChange", "ClassificationUpdate"]
      scope: "GLOBAL"
      filter_criteria:
        regime_aware: true
        
  # Event filtering
  event_filters:
    # Symbol-based filtering
    symbol_filter:
      enabled: true
      filter_field: "symbol"
      allowed_symbols: ["SPY", "QQQ", "IWM"]
      
    # Time-based filtering
    time_filter:
      enabled: true
      market_hours_only: true
      timezone: "US/Eastern"
      
    # Performance filtering
    performance_filter:
      enabled: true
      min_signal_strength: 0.1
      min_confidence: 0.3
```

## 📊 Event Types and Routing Patterns

### Market Data Events

```python
# Market data events flow DOWNWARD through hierarchy
@dataclass
class BarEvent(Event):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
# Default routing
routing_pattern = {
    "scope": EventScope.DOWNWARD,
    "source_roles": ["DATA"],
    "target_roles": ["INDICATOR", "STRATEGY", "CLASSIFIER"],
    "delivery_guarantee": "at_least_once"
}
```

### Trading Signal Events

```python
# Trading signals flow UPWARD through hierarchy
@dataclass
class TradingSignal(Event):
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float
    confidence: float
    strategy_id: str
    timestamp: datetime
    
# Default routing
routing_pattern = {
    "scope": EventScope.UPWARD,
    "source_roles": ["STRATEGY"],
    "target_roles": ["RISK", "PORTFOLIO", "EXECUTION"],
    "delivery_guarantee": "exactly_once"
}
```

### Portfolio Events

```python
# Portfolio updates broadcast GLOBALLY
@dataclass
class PortfolioUpdate(Event):
    total_value: float
    cash: float
    positions: Dict[str, int]
    unrealized_pnl: float
    timestamp: datetime
    
# Default routing
routing_pattern = {
    "scope": EventScope.GLOBAL,
    "source_roles": ["PORTFOLIO"],
    "target_roles": ["ALL"],
    "delivery_guarantee": "at_least_once"
}
```

### Regime Classification Events

```python
# Regime changes broadcast to relevant strategies
@dataclass
class RegimeChange(Event):
    previous_regime: str
    new_regime: str
    confidence: float
    transition_probability: float
    timestamp: datetime
    
# Routing with filtering
routing_pattern = {
    "scope": EventScope.GLOBAL,
    "source_roles": ["CLASSIFIER"],
    "target_roles": ["STRATEGY"],
    "filter_criteria": {"regime_aware": True},
    "delivery_guarantee": "exactly_once"
}
```

## 🎯 Communication Patterns

### Hierarchical Communication

**Pattern**: Parent-Child communication through container hierarchy

```yaml
# Example: Data → Indicators → Strategy flow
communication_patterns:
  hierarchical:
    name: "data_processing_pipeline"
    
    # Data flows downward
    data_flow:
      - source: "DATA"
        target: "INDICATOR"
        scope: "CHILDREN"
        events: ["BarEvent", "TickEvent"]
        
      - source: "INDICATOR"
        target: "STRATEGY"
        scope: "CHILDREN"
        events: ["IndicatorEvent"]
        
    # Signals flow upward
    signal_flow:
      - source: "STRATEGY"
        target: "RISK"
        scope: "PARENT"
        events: ["TradingSignal"]
        
      - source: "RISK"
        target: "PORTFOLIO"
        scope: "PARENT"
        events: ["RiskAssessment", "PositionSize"]
```

### Broadcast Communication

**Pattern**: One-to-many broadcasting with filtering

```yaml
communication_patterns:
  broadcast:
    name: "market_data_distribution"
    
    # Source configuration
    source:
      role: "DATA"
      container_id: "market_data_provider"
      
    # Target configuration with filtering
    targets:
      - role: "STRATEGY"
        filter:
          symbols: ["SPY", "QQQ"]
          event_types: ["BarEvent"]
          
      - role: "INDICATOR"
        filter:
          symbols: ["SPY", "QQQ", "IWM"]
          event_types: ["BarEvent", "TickEvent"]
          
      - role: "CLASSIFIER"
        filter:
          event_types: ["BarEvent"]
          min_volume: 100000
          
    # Performance configuration
    performance:
      delivery_guarantee: "at_least_once"
      max_latency_ms: 50
      batch_size: 100
```

### Peer-to-Peer Communication

**Pattern**: Direct communication between sibling containers

```yaml
communication_patterns:
  peer_to_peer:
    name: "strategy_coordination"
    
    # Peer groups
    peer_groups:
      - name: "momentum_strategies"
        members: ["momentum_fast", "momentum_slow", "momentum_adaptive"]
        shared_events: ["RegimeChange", "VolatilityUpdate"]
        
      - name: "mean_reversion_strategies"
        members: ["rsi_strategy", "bollinger_strategy", "stat_arb"]
        shared_events: ["OverboughtOversold", "MeanReversionSignal"]
        
    # Coordination rules
    coordination_rules:
      - trigger: "RegimeChange"
        action: "broadcast_to_group"
        target_group: "all"
        
      - trigger: "HighVolatility"
        action: "reduce_position_sizes"
        target_group: "momentum_strategies"
```

## 🔧 Event Adapter Implementation

### Pipeline Adapter

**Description**: Sequential processing through ordered containers

```python
class PipelineAdapter:
    """Sequential event processing through ordered containers."""
    
    def __init__(self, containers: List[str], config: Dict[str, Any]):
        self.containers = containers
        self.config = config
        self.current_stage = 0
        
    async def route_event(self, event: Event, source: str) -> None:
        """Route event through pipeline stages."""
        current_stage = self.containers.index(source)
        
        if current_stage < len(self.containers) - 1:
            next_container = self.containers[current_stage + 1]
            await self.send_to_container(event, next_container)
            
    def configure_pipeline(self) -> Dict[str, Any]:
        """Configure pipeline routing."""
        return {
            "type": "pipeline",
            "containers": self.containers,
            "synchronous": True,
            "buffer_size": self.config.get("buffer_size", 1000),
            "timeout_seconds": self.config.get("timeout", 30)
        }
```

### Hub-and-Spoke Adapter

**Description**: Central hub distributing to multiple spokes

```python
class HubSpokeAdapter:
    """Hub-and-spoke event distribution pattern."""
    
    def __init__(self, hub: str, spokes: List[str], config: Dict[str, Any]):
        self.hub = hub
        self.spokes = spokes
        self.config = config
        self.filters = config.get("filters", {})
        
    async def route_event(self, event: Event, source: str) -> None:
        """Route event from hub to spokes or aggregate from spokes."""
        if source == self.hub:
            # Distribute to spokes
            for spoke in self.spokes:
                if self.should_route_to_spoke(event, spoke):
                    await self.send_to_container(event, spoke)
        else:
            # Aggregate from spoke to hub
            await self.send_to_container(event, self.hub)
            
    def should_route_to_spoke(self, event: Event, spoke: str) -> bool:
        """Check if event should be routed to specific spoke."""
        spoke_filter = self.filters.get(spoke, {})
        
        # Check event type filter
        if "event_types" in spoke_filter:
            if event.__class__.__name__ not in spoke_filter["event_types"]:
                return False
                
        # Check symbol filter
        if "symbols" in spoke_filter and hasattr(event, "symbol"):
            if event.symbol not in spoke_filter["symbols"]:
                return False
                
        return True
```

## 📊 Performance and Monitoring

### Event Performance Tiers

```yaml
# Performance tier configuration
event_performance:
  # High-frequency market data
  fast_tier:
    events: ["BarEvent", "TickEvent", "QuoteEvent"]
    delivery_guarantee: "at_most_once"
    max_latency_ms: 10
    buffer_size: 10000
    batch_processing: true
    
  # Trading signals and portfolio updates
  standard_tier:
    events: ["TradingSignal", "IndicatorEvent", "PortfolioUpdate"]
    delivery_guarantee: "at_least_once"
    max_latency_ms: 100
    buffer_size: 1000
    acknowledgments: true
    
  # Critical system events
  reliable_tier:
    events: ["SystemAlert", "EmergencyStop", "RegimeChange"]
    delivery_guarantee: "exactly_once"
    max_latency_ms: 1000
    buffer_size: 100
    persistence: true
    retry_attempts: 5
```

### Event Monitoring

```yaml
# Event system monitoring
event_monitoring:
  # Performance metrics
  performance_metrics:
    enabled: true
    collection_interval_seconds: 10
    
    metrics:
      - "events_per_second"
      - "average_latency_ms"
      - "queue_depth"
      - "delivery_success_rate"
      - "error_rate"
      
  # Flow analysis
  flow_analysis:
    enabled: true
    trace_event_flows: true
    detect_bottlenecks: true
    
    bottleneck_detection:
      queue_depth_threshold: 5000
      latency_threshold_ms: 500
      error_rate_threshold: 0.05
      
  # Alerting
  alerting:
    enabled: true
    
    alert_conditions:
      high_latency:
        threshold_ms: 1000
        duration_seconds: 30
        
      high_error_rate:
        threshold_percent: 5.0
        duration_seconds: 60
        
      queue_overflow:
        threshold_depth: 8000
        immediate: true
```

## 🔍 Event Debugging and Tracing

### Event Correlation

```python
# Event correlation for debugging
@dataclass
class EventMetadata:
    correlation_id: str
    causation_id: Optional[str]
    trace_id: str
    timestamp: datetime
    source_container: str
    target_containers: List[str]
    routing_path: List[str]

# Example event with correlation
trading_signal = TradingSignal(
    symbol="SPY",
    action="BUY",
    strength=0.8,
    confidence=0.75,
    strategy_id="momentum_001",
    timestamp=datetime.now(),
    metadata=EventMetadata(
        correlation_id="trade_session_20231201_001",
        causation_id="bar_spy_20231201_093000",
        trace_id="trace_001",
        source_container="strategy_momentum",
        target_containers=["risk_manager", "portfolio_tracker"],
        routing_path=["strategy", "risk", "portfolio", "execution"]
    )
)
```

### Event Flow Visualization

```yaml
# Event flow tracing
event_tracing:
  enabled: true
  
  # Trace collection
  trace_collection:
    sample_rate: 0.1  # Sample 10% of events
    max_trace_length: 100
    trace_timeout_seconds: 300
    
  # Flow visualization
  flow_visualization:
    enabled: true
    real_time_dashboard: true
    
    visualization_config:
      container_graph: true
      event_flow_arrows: true
      performance_heatmap: true
      bottleneck_highlighting: true
      
  # Export configuration
  trace_export:
    format: "json"
    output_path: "traces/"
    compression: true
    retention_days: 30
```

## 🛠️ Custom Event Adapter Development

### Creating Custom Adapters

```python
from src.core.events import EventAdapter, Event, EventScope

class CustomEventAdapter(EventAdapter):
    """Custom event adapter implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_config = config.get("custom_settings", {})
        
    async def route_event(
        self,
        event: Event,
        source_container: str,
        scope: EventScope
    ) -> None:
        """Custom event routing logic."""
        
        # Custom routing logic here
        target_containers = self.determine_targets(event, source_container, scope)
        
        for target in target_containers:
            await self.send_event_to_container(event, target)
            
    def determine_targets(
        self,
        event: Event,
        source: str,
        scope: EventScope
    ) -> List[str]:
        """Custom target determination logic."""
        
        # Implement custom logic
        if isinstance(event, TradingSignal):
            return self.get_signal_targets(event, source)
        elif isinstance(event, BarEvent):
            return self.get_market_data_targets(event, source)
        else:
            return self.get_default_targets(event, source, scope)
```

### Registering Custom Adapters

```python
# Register custom adapter
from src.core.events import EventRouterRegistry

registry = EventRouterRegistry()
registry.register_adapter_type(
    adapter_name="custom_adapter",
    adapter_class=CustomEventAdapter,
    config_schema={
        "type": "object",
        "properties": {
            "custom_settings": {"type": "object"},
            "routing_rules": {"type": "array"}
        }
    }
)
```

## 🤔 Common Questions

**Q: How do containers discover each other for communication?**
A: The Event Router maintains a registry of all containers and their roles. Containers register with the router on initialization.

**Q: What happens if a target container is offline?**
A: Depends on delivery guarantee. "at_most_once" drops the event, "at_least_once" and "exactly_once" queue for retry.

**Q: Can I create custom event types?**
A: Yes, inherit from the base Event class and register with the event system. Ensure proper serialization support.

**Q: How does event filtering work?**
A: Events can be filtered by type, content, source/target roles, or custom criteria before routing.

**Q: What's the performance overhead of the event system?**
A: Minimal - fast tier events have <10ms latency. The system is optimized for high-frequency trading data.

---

Continue to [Workflow Blocks](workflow-blocks.md) for building block specifications →