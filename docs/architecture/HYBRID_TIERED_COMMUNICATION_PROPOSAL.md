# Hybrid Tiered Communication Architecture Proposal

## Executive Summary

This proposal combines the best concepts from both the **Standardized Communication Plan** and the **Tiered Event Router** approach to create a unified communication architecture that provides:

1. **Performance-optimized Event Router** for cross-container communication
2. **Simple direct communication** for sub-container coordination
3. **Clear architectural boundaries** that align with container composition patterns
4. **Configuration-driven organization** that supports elegant container inheritance

## Core Principles

### 1. **Container Boundary = Communication Pattern Boundary**

```
┌─────────────────────────────────────────────────────────┐
│                Container Boundary                        │
│                                                         │
│  Internal Communication: Direct Event Bus               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ Sub-Container│───►│ Sub-Container│───►│ Sub-Container│ │
│  │      A      │    │      B      │    │      C      │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
│           │                 │                 │        │
│           └─────────────────┼─────────────────┘        │
│                             │                          │
└─────────────────────────────┼──────────────────────────┘
                              │
              External Communication: Tiered Event Router
                              │
┌─────────────────────────────▼──────────────────────────┐
│              Other Container                           │
└────────────────────────────────────────────────────────┘
```

### 2. **Performance Tiers for Event Router**

External communication uses a **Tiered Event Router** with three performance levels:

- **Fast Tier**: Market data (BAR, TICK) - < 1ms latency
- **Standard Tier**: Business logic (SIGNAL, INDICATOR) - < 10ms latency  
- **Reliable Tier**: Critical events (ORDER, FILL) - 100% delivery guarantee

### 3. **Composable Container Organization**

Containers are organized by **configuration inheritance patterns**:

```yaml
market_regime_container:
  classifier: "hmm_bull_bear"
  
  risk_containers:
    - name: "conservative_risk"
      max_position_pct: 2.0
      max_exposure_pct: 10.0
      
      portfolio_containers:
        - name: "tech_portfolio"
          symbols: ["AAPL", "GOOGL", "MSFT"]
          
          strategies:
            - type: "momentum"
              allocation: 0.6
            - type: "mean_reversion"
              allocation: 0.4
```

## Architecture Design

### Communication Classification

All communication is classified into two categories with different optimization strategies:

#### **External Communication** (Cross-Container)
- **Pattern**: Tiered Event Router
- **Use Cases**: 
  - Data distribution (Data → IndicatorHub, StrategyContainers)
  - Signal flow (Strategy → Portfolio → Risk)
  - Order execution (Risk ↔ Execution)
- **Benefits**: Selective subscriptions, filtering, observability, performance optimization

#### **Internal Communication** (Sub-Container)
- **Pattern**: Direct Event Bus  
- **Use Cases**:
  - Sub-container coordination within same logical boundary
  - Parent ↔ Child communication
  - Sibling coordination within container
- **Benefits**: High performance, simple patterns, easy debugging

### Hybrid Communication Interface

```python
class HybridContainerInterface:
    """Container interface supporting both communication patterns"""
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        
        # External communication via Tiered Event Router
        self.external_router: Optional[TieredEventRouter] = None
        self._external_publications: List[EventPublication] = []
        self._external_subscriptions: List[EventSubscription] = []
        
        # Internal communication via Direct Event Bus
        self.internal_bus = EventBus()
        self.children: List['HybridContainerInterface'] = []
        self.parent: Optional['HybridContainerInterface'] = None
    
    # === External Communication (Cross-Container) ===
    
    def register_with_router(self, router: TieredEventRouter) -> None:
        """Register for cross-container communication"""
        self.external_router = router
        
        # Register publications and subscriptions
        if self._external_publications:
            router.register_publisher(self.container_id, self._external_publications)
        
        if self._external_subscriptions:
            router.register_subscriber(
                self.container_id,
                self._external_subscriptions,
                self.handle_external_event
            )
        
        # CASCADE: Register all children for external communication
        for child in self.children:
            child.register_with_router(router)
    
    def publish_external(self, event: Event, tier: str = "standard") -> None:
        """Publish event to other containers via Event Router"""
        if not self.external_router:
            raise RuntimeError(f"Container {self.container_id} not registered with router")
        
        self.external_router.route_event(event, self.container_id, tier)
        logger.debug(f"📡 {self.container_id} published {event.event_type} via {tier} tier")
    
    def handle_external_event(self, event: Event, source: str) -> None:
        """Handle events from other containers"""
        # Add source metadata
        event.metadata['source_container'] = source
        
        # Process through normal event handling
        if hasattr(self, 'process_event'):
            if asyncio.iscoroutinefunction(self.process_event):
                asyncio.create_task(self.process_event(event))
            else:
                self.process_event(event)
    
    # === Internal Communication (Sub-Container) ===
    
    def add_child_container(self, child: 'HybridContainerInterface') -> None:
        """Add child with automatic communication setup"""
        self.children.append(child)
        child.parent = self
        
        # Setup internal event bridging
        child.internal_bus.subscribe_all(self._forward_child_event)
        self.internal_bus.subscribe_all(child._handle_parent_event)
        
        # Register child with external router if available
        if self.external_router:
            child.register_with_router(self.external_router)
        
        logger.info(f"Added child {child.container_id} to {self.container_id}")
    
    def publish_internal(self, event: Event, scope: str = "children") -> None:
        """Publish event within container boundary"""
        if scope == "children":
            for child in self.children:
                child.internal_bus.publish(event)
                logger.debug(f"📨 {self.container_id} → {child.container_id} (internal)")
        
        elif scope == "parent":
            if self.parent:
                self.parent.internal_bus.publish(event)
                logger.debug(f"📨 {self.container_id} → {self.parent.container_id} (internal)")
        
        elif scope == "siblings":
            if self.parent:
                for sibling in self.parent.children:
                    if sibling != self:
                        sibling.internal_bus.publish(event)
                        logger.debug(f"📨 {self.container_id} → {sibling.container_id} (internal)")
    
    def _forward_child_event(self, event: Event) -> None:
        """Forward child events that need external routing"""
        # Only forward events declared for external publication
        if event.event_type in [pub.events for pub in self._external_publications]:
            self.publish_external(event)
    
    def _handle_parent_event(self, event: Event) -> None:
        """Handle events from parent container"""
        if hasattr(self, 'process_event'):
            if asyncio.iscoroutinefunction(self.process_event):
                asyncio.create_task(self.process_event(event))
            else:
                self.process_event(event)
    
    # === Configuration Support ===
    
    def configure_external_communication(self, config: Dict[str, Any]) -> None:
        """Configure external Event Router communication"""
        if 'external_events' not in config:
            return
        
        ext_config = config['external_events']
        
        # Configure publications
        if 'publishes' in ext_config:
            publications = []
            for pub_config in ext_config['publishes']:
                pub = EventPublication(
                    events=set(pub_config['events']),
                    scope=EventScope[pub_config.get('scope', 'GLOBAL').upper()],
                    tier=pub_config.get('tier', 'standard'),
                    qos=pub_config.get('qos', 'best_effort')
                )
                publications.append(pub)
            self._external_publications = publications
        
        # Configure subscriptions
        if 'subscribes' in ext_config:
            subscriptions = []
            for sub_config in ext_config['subscribes']:
                sub = EventSubscription(
                    source=sub_config['source'],
                    events=set(sub_config['events']),
                    filters=sub_config.get('filters', {}),
                    tier=sub_config.get('tier', 'standard')
                )
                subscriptions.append(sub)
            self._external_subscriptions = subscriptions
```

### Tiered Event Router Implementation

```python
class TieredEventRouter:
    """Event Router with performance tiers for different event types"""
    
    def __init__(self):
        self.fast_tier = FastTierRouter()      # BAR, TICK events
        self.standard_tier = StandardTierRouter()  # SIGNAL, INDICATOR events
        self.reliable_tier = ReliableTierRouter()  # ORDER, FILL events
        
        self.tier_mapping = {
            'fast': self.fast_tier,
            'standard': self.standard_tier,
            'reliable': self.reliable_tier
        }
        
        # Event type to tier mapping
        self.event_tier_map = {
            EventType.BAR: 'fast',
            EventType.TICK: 'fast',
            EventType.QUOTE: 'fast',
            EventType.SIGNAL: 'standard',
            EventType.INDICATOR: 'standard',
            EventType.PORTFOLIO_UPDATE: 'standard',
            EventType.ORDER: 'reliable',
            EventType.FILL: 'reliable',
            EventType.SYSTEM: 'reliable'
        }
    
    def route_event(self, event: Event, source: str, tier: str = None) -> None:
        """Route event through appropriate tier"""
        # Determine tier
        if tier is None:
            tier = self.event_tier_map.get(event.event_type, 'standard')
        
        # Route through appropriate tier
        router = self.tier_mapping[tier]
        router.route_event(event, source)
        
        # Performance monitoring
        self._record_routing_metrics(event, source, tier)
    
    def register_publisher(self, container_id: str, publications: List[EventPublication]) -> None:
        """Register publisher across all relevant tiers"""
        for publication in publications:
            tier = getattr(publication, 'tier', 'standard')
            router = self.tier_mapping[tier]
            router.register_publisher(container_id, [publication])
    
    def register_subscriber(self, container_id: str, subscriptions: List[EventSubscription], callback: Callable) -> None:
        """Register subscriber across all relevant tiers"""
        for subscription in subscriptions:
            tier = getattr(subscription, 'tier', 'standard')
            router = self.tier_mapping[tier]
            router.register_subscriber(container_id, [subscription], callback)


class FastTierRouter:
    """Optimized router for high-frequency data events"""
    
    def __init__(self):
        self.routing_cache: Dict[EventType, List[str]] = {}
        self.subscribers: Dict[str, Callable] = {}
        self.batch_buffer: List[Tuple[Event, str]] = []
        self.batch_size = 1000
        self.max_latency_ms = 1
        
    def route_event(self, event: Event, source: str) -> None:
        """Ultra-fast routing for data events"""
        # Add to batch buffer
        self.batch_buffer.append((event, source))
        
        # Flush if batch is full or latency threshold reached
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()
    
    def _flush_batch(self) -> None:
        """Flush batched events for delivery"""
        for event, source in self.batch_buffer:
            subscribers = self.routing_cache.get(event.event_type, [])
            for subscriber_id in subscribers:
                callback = self.subscribers.get(subscriber_id)
                if callback:
                    # Direct callback for speed
                    callback(event, source)
        
        self.batch_buffer.clear()


class StandardTierRouter:
    """Standard router for business logic events"""
    
    def __init__(self):
        self.subscriptions: Dict[str, List[EventSubscription]] = {}
        self.async_queue = asyncio.Queue()
        
    async def route_event(self, event: Event, source: str) -> None:
        """Async routing for business logic"""
        await self.async_queue.put((event, source))
        
    async def _process_queue(self) -> None:
        """Process queued events asynchronously"""
        while True:
            event, source = await self.async_queue.get()
            await self._deliver_event(event, source)


class ReliableTierRouter:
    """Reliable router for critical events with guarantees"""
    
    def __init__(self):
        self.persistent_queue = PersistentQueue()
        self.delivery_confirmations: Dict[str, bool] = {}
        self.retry_attempts = 3
        
    async def route_event(self, event: Event, source: str) -> None:
        """Reliable routing with retry and confirmation"""
        delivery_id = str(uuid.uuid4())
        
        # Persist event
        await self.persistent_queue.enqueue(event, source, delivery_id)
        
        # Attempt delivery with retries
        for attempt in range(self.retry_attempts):
            try:
                await self._deliver_with_confirmation(event, source, delivery_id)
                break
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    await self._send_to_dead_letter_queue(event, source, e)
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Configuration Examples

### Container Definition with Hybrid Communication

```yaml
# Multi-strategy container with hybrid communication
strategy_ensemble_container:
  container_type: "strategy_ensemble"
  container_id: "momentum_ensemble"
  
  # External communication via Event Router
  external_events:
    publishes:
      - events: ["SIGNAL"]
        scope: "PARENT"
        tier: "standard"
    
    subscribes:
      - source: "data_container"
        events: ["BAR"]
        tier: "fast"
        filters:
          symbols: ["AAPL", "GOOGL", "MSFT"]
          timeframe: "1min"
      
      - source: "indicator_container"  
        events: ["INDICATOR"]
        tier: "standard"
        filters:
          subscriber: "momentum_ensemble"
  
  # Internal sub-containers (direct event bus)
  sub_containers:
    - container_type: "strategy"
      container_id: "momentum_fast"
      strategy:
        type: "momentum"
        fast_period: 10
        slow_period: 20
        
    - container_type: "strategy"
      container_id: "momentum_slow"  
      strategy:
        type: "momentum"
        fast_period: 20
        slow_period: 50
  
  # Internal aggregation (direct event bus)
  aggregation:
    method: "weighted_voting"
    weights:
      momentum_fast: 0.6
      momentum_slow: 0.4
```

### Tiered Event Router Configuration

```yaml
# Event Router tier configuration
event_router:
  tiers:
    fast:
      events: ["BAR", "TICK", "QUOTE", "BOOK_UPDATE"]
      optimizations:
        batch_size: 1000
        max_latency_ms: 1
        zero_copy_delivery: true
        in_memory_only: true
        
    standard:
      events: ["SIGNAL", "INDICATOR", "PORTFOLIO_UPDATE", "REGIME"]
      optimizations:
        batch_size: 100
        max_latency_ms: 10
        async_delivery: true
        intelligent_batching: true
        
    reliable:
      events: ["ORDER", "FILL", "SYSTEM", "ERROR", "RISK_ALERT"]
      optimizations:
        persistent_queue: true
        retry_attempts: 3
        delivery_confirmation: true
        dead_letter_queue: true
        max_latency_ms: 100
        
  monitoring:
    track_latency: true
    track_throughput: true
    alert_on_backlog: 1000
    performance_metrics: true
```

## Usage Patterns

### Pattern 1: Data Distribution (External)

```python
class DataContainer(HybridContainerInterface):
    def __init__(self, config):
        super().__init__("data_container")
        
        # Configure external publications
        self.configure_external_communication({
            'external_events': {
                'publishes': [
                    {
                        'events': ['BAR'],
                        'scope': 'GLOBAL',
                        'tier': 'fast'
                    }
                ]
            }
        })
    
    async def stream_data(self):
        """Stream market data to all subscribers"""
        for bar in self.data_stream:
            bar_event = Event(
                event_type=EventType.BAR,
                payload={'bar': bar, 'symbol': bar.symbol},
                timestamp=bar.timestamp
            )
            
            # External: Broadcast to all containers via Fast Tier
            self.publish_external(bar_event, tier='fast')
```

### Pattern 2: Strategy Ensemble (Internal)

```python
class StrategyEnsembleContainer(HybridContainerInterface):
    def __init__(self, config):
        super().__init__("strategy_ensemble")
        
        # Create sub-strategies
        for strategy_config in config['strategies']:
            strategy_container = StrategyContainer(strategy_config)
            self.add_child_container(strategy_container)  # Automatic internal wiring
        
        # Configure external communication
        self.configure_external_communication(config)
    
    def aggregate_signals(self, signals: List[Signal]) -> Signal:
        """Aggregate signals from sub-strategies (internal communication)"""
        # This happens via internal event bus automatically
        aggregated = self.signal_aggregator.aggregate(signals)
        
        # Forward aggregated signal externally
        signal_event = Event(
            event_type=EventType.SIGNAL,
            payload={'signals': [aggregated]},
            timestamp=datetime.now()
        )
        
        # External: Send to Portfolio/Risk containers
        self.publish_external(signal_event, tier='standard')
```

### Pattern 3: Risk Management (Cross-Container)

```python
class RiskContainer(HybridContainerInterface):
    def __init__(self, config):
        super().__init__("risk_container")
        
        # Configure external communication
        self.configure_external_communication({
            'external_events': {
                'subscribes': [
                    {
                        'source': 'strategy_ensemble',
                        'events': ['SIGNAL'],
                        'tier': 'standard'
                    }
                ],
                'publishes': [
                    {
                        'events': ['ORDER'],
                        'scope': 'SIBLINGS',
                        'tier': 'reliable'
                    }
                ]
            }
        })
    
    def handle_external_event(self, event: Event, source: str):
        """Handle signals from strategy containers"""
        if event.event_type == EventType.SIGNAL:
            signals = event.payload.get('signals', [])
            orders = self.risk_manager.process_signals(signals)
            
            if orders:
                order_event = Event(
                    event_type=EventType.ORDER,
                    payload={'orders': orders},
                    timestamp=datetime.now()
                )
                
                # External: Send to ExecutionContainer via Reliable Tier
                self.publish_external(order_event, tier='reliable')
```

## Benefits of Hybrid Tiered Approach

### 1. **Performance Optimization**
- **Fast Tier**: < 1ms latency for market data distribution
- **Internal Communication**: Direct calls with minimal overhead
- **Reliable Tier**: Guaranteed delivery for critical events

### 2. **Architectural Clarity**
- **Clear Boundaries**: Container boundary = communication pattern boundary
- **Predictable Patterns**: External = Event Router, Internal = Direct Bus
- **Easy Debugging**: Separate tooling for each communication type

### 3. **Composable Flexibility**
```python
# Easy to reorganize containers
config = {
    'deployment_mode': 'distributed',  # vs 'embedded'
    'container_organization': 'by_risk_profile',  # vs 'by_strategy_type'
    'communication_optimization': 'latency'  # vs 'throughput'
}
```

### 4. **Configuration Inheritance**
```yaml
# Natural inheritance patterns
bull_market_setup:
  classifier: "hmm_bull_bear"
  risk_profile: "aggressive"
  
  strategies:
    - inherits_from: "bull_market_setup"
      type: "momentum"
      symbols: ["AAPL", "GOOGL"]
    
    - inherits_from: "bull_market_setup"  
      type: "breakout"
      symbols: ["MSFT", "NVDA"]
```

### 5. **Selective Subscriptions**
```yaml
# Components subscribe only to what they need
tech_strategies:
  subscribes:
    - source: "data_container"
      events: ["BAR"]
      filters:
        symbols: ["AAPL", "GOOGL", "MSFT", "NVDA"]
        timeframe: "1min"

crypto_strategies:
  subscribes:
    - source: "crypto_data_container"
      events: ["BAR"]
      filters:
        symbols: ["BTC", "ETH"]
        timeframe: "5min"
```

## Implementation Roadmap

### Phase 1: Foundation (Immediate)
1. **Implement HybridContainerInterface** base class
2. **Create TieredEventRouter** with three performance tiers
3. **Update existing containers** to use hybrid interface
4. **Configure event tier mappings** in YAML

### Phase 2: Migration (Next Sprint)  
1. **Update DataContainer** for Fast Tier broadcasting
2. **Migrate StrategyContainer** to hybrid pattern for sub-strategies
3. **Update RiskContainer** for Reliable Tier ORDER/FILL events
4. **Add performance monitoring** for all tiers

### Phase 3: Advanced Features (Future)
1. **Dynamic tier assignment** based on load
2. **Container composition optimization** (embedded vs distributed)
3. **Advanced filtering and routing** capabilities
4. **Multi-process/multi-machine** distribution support

## Success Metrics

### Performance Targets
- **Fast Tier**: < 1ms latency for BAR events
- **Standard Tier**: < 10ms latency for SIGNAL events  
- **Reliable Tier**: 100% delivery for ORDER/FILL events
- **Internal Communication**: < 0.1ms for sub-container events

### Architecture Goals
- **Communication Consistency**: Clear patterns for each use case
- **Complete Observability**: Full visibility into all event flows
- **Configuration Flexibility**: Easy container reorganization
- **Development Simplicity**: Intuitive patterns for each scenario

## Conclusion

The **Hybrid Tiered Communication Architecture** combines the performance benefits of the Tiered Event Router with the simplicity of direct communication for sub-containers. This provides:

- ✅ **Best Performance**: Optimized patterns for each use case
- ✅ **Architectural Elegance**: Clear boundaries and consistent patterns  
- ✅ **Composable Flexibility**: Easy container reorganization
- ✅ **Development Productivity**: Simple patterns for common scenarios
- ✅ **Configuration Inheritance**: Natural organizational hierarchies
- ✅ **Selective Subscriptions**: Components get only what they need

This architecture enables the sophisticated composable container organization you envision while maintaining high performance and clear architectural boundaries.