# Event-Driven Design

Event-driven design is the communication backbone of ADMF-PC, enabling loose coupling, natural parallelism, and production consistency. This document explains how events flow through the system and why this approach is superior to traditional method calls.

## 🎯 Core Concept

Instead of components calling each other directly, everything in ADMF-PC communicates through **events**:

### Traditional Approach (Tight Coupling)
```python
class Strategy:
    def __init__(self, risk_manager, execution_engine):
        self.risk_manager = risk_manager      # Direct dependency
        self.execution_engine = execution_engine  # Direct dependency
    
    def on_signal(self, signal):
        # Direct method calls create tight coupling
        if self.risk_manager.check_risk(signal):
            order = self.risk_manager.size_position(signal)
            self.execution_engine.submit_order(order)
```

### ADMF-PC Approach (Loose Coupling)
```python
class Strategy:
    def __init__(self, event_bus):
        self.event_bus = event_bus  # Only dependency is event bus
    
    def on_signal(self, signal):
        # Emit event - no knowledge of who handles it
        self.event_bus.emit(TradingSignal(
            symbol=signal.symbol,
            action=signal.action,
            strength=signal.strength
        ))
```

## 📡 Event Flow Architecture

### The Standard Signal Flow

ADMF-PC follows a standardized event flow for all trading operations:

```
Market Data → Indicators → Strategies → Risk → Execution → Portfolio
     ↓           ↓           ↓          ↓        ↓           ↓
   [BAR]    [INDICATOR]  [STRATEGY]  [SIGNAL] [ORDER]    [FILL]
                                                            ↓
                                                   [PORTFOLIO_UPDATE]
```

Each arrow represents an event flowing between components.

### Event Types and Schema

#### 1. Market Data Events
```python
@dataclass
class BarEvent:
    """Market data bar event"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Event metadata
    event_id: str = field(default_factory=uuid4)
    correlation_id: str = ""
```

#### 2. Indicator Events
```python
@dataclass
class IndicatorEvent:
    """Technical indicator calculation result"""
    indicator_name: str  # "SMA_20", "RSI_14", etc.
    symbol: str
    value: float
    timestamp: datetime
    
    # Optional metadata
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 3. Trading Signal Events
```python
@dataclass
class TradingSignal:
    """Trading signal from strategy"""
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    
    # Optional fields
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    strategy_id: str = ""
    regime_context: Optional[str] = None
```

#### 4. Order Events
```python
@dataclass
class OrderEvent:
    """Order to be executed"""
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    order_type: Literal["MARKET", "LIMIT", "STOP"]
    
    # Risk management
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size_pct: float = 0.02
```

#### 5. Fill Events
```python
@dataclass
class FillEvent:
    """Executed trade result"""
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    commission: float
    timestamp: datetime
```

#### 6. Portfolio Events
```python
@dataclass
class PortfolioUpdateEvent:
    """Portfolio state change"""
    cash: float
    total_value: float
    positions: Dict[str, int]
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
```

## 🔄 Event Bus Architecture

### Isolated Event Buses

Each container has its own isolated event bus:

```python
class ContainerEventBus:
    """Isolated event bus for a single container"""
    
    def __init__(self):
        self.subscribers = {}  # event_type -> [handlers]
        self.event_queue = Queue()
        self.correlation_tracker = {}
        
    def subscribe(self, event_type: type, handler: callable):
        """Subscribe to specific event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        
    def emit(self, event: Any):
        """Emit event to all subscribers"""
        event_type = type(event)
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    self.handle_error(event, handler, e)
```

### Cross-Container Event Routing

When containers need to communicate, events go through the **Event Router**:

```python
class EventRouter:
    """Routes events between containers"""
    
    def __init__(self):
        self.routes = {}  # source_container -> [(target_container, event_filter)]
        self.containers = {}  # container_name -> container_instance
        
    def add_route(self, source: str, target: str, event_filter: callable = None):
        """Add routing rule between containers"""
        if source not in self.routes:
            self.routes[source] = []
        self.routes[source].append((target, event_filter))
        
    def route_event(self, source: str, event: Any):
        """Route event from source to target containers"""
        if source in self.routes:
            for target, event_filter in self.routes[source]:
                if not event_filter or event_filter(event):
                    # Serialize for isolation
                    serialized_event = self.serialize_event(event)
                    target_container = self.containers[target]
                    target_container.receive_external_event(serialized_event)
```

## 🎼 Event Orchestration Patterns

### 1. Pipeline Pattern
Events flow sequentially through components:

```yaml
# Configuration for pipeline flow
event_flow:
  type: "pipeline"
  containers: ["data", "indicators", "strategy", "risk", "execution"]
  
# Results in this flow:
# data → indicators → strategy → risk → execution
```

### 2. Broadcast Pattern
One source sends to multiple targets:

```yaml
event_flow:
  type: "broadcast"
  source: "data"
  targets: ["strategy_a", "strategy_b", "strategy_c"]
  
# Results in:
# data → strategy_a
# data → strategy_b  
# data → strategy_c
```

### 3. Hierarchical Pattern
Parent-child relationships with context propagation:

```yaml
event_flow:
  type: "hierarchical"
  parent: "classifier"
  children: ["risk_conservative", "risk_aggressive"]
  
# Results in:
# classifier → risk_conservative (when regime = conservative)
# classifier → risk_aggressive (when regime = aggressive)
```

### 4. Selective Pattern
Content-based routing:

```yaml
event_flow:
  type: "selective"
  source: "signal_generator"
  routes:
    - condition: "signal.strength > 0.8"
      target: "high_conviction_execution"
    - condition: "signal.strength > 0.5"
      target: "medium_conviction_execution"
    - condition: "signal.strength > 0.2"
      target: "low_conviction_execution"
```

## 📊 Event Correlation and Tracing

### Correlation IDs

Events carry correlation IDs to trace related events:

```python
# Original signal
signal = TradingSignal(
    symbol="AAPL",
    action="BUY",
    correlation_id="trade_001"
)

# Related order
order = OrderEvent(
    symbol="AAPL",
    side="BUY",
    correlation_id="trade_001",  # Same correlation ID
    causation_id=signal.event_id  # Links to causing event
)

# Related fill
fill = FillEvent(
    symbol="AAPL",
    side="BUY", 
    correlation_id="trade_001",  # Same correlation ID
    causation_id=order.event_id  # Links to causing event
)
```

### Event Tracing

Track complete event flows:

```python
class EventTracer:
    """Traces event flows for debugging and analysis"""
    
    def __init__(self):
        self.event_history = {}  # correlation_id -> [events]
        
    def record_event(self, event):
        """Record event in trace history"""
        correlation_id = getattr(event, 'correlation_id', None)
        if correlation_id:
            if correlation_id not in self.event_history:
                self.event_history[correlation_id] = []
            self.event_history[correlation_id].append(event)
            
    def get_trace(self, correlation_id: str) -> List[Any]:
        """Get complete event trace"""
        return self.event_history.get(correlation_id, [])
        
    def analyze_latency(self, correlation_id: str) -> Dict[str, float]:
        """Analyze timing between events"""
        events = self.get_trace(correlation_id)
        latencies = {}
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            stage = f"{type(current).__name__} → {type(next_event).__name__}"
            latency = (next_event.timestamp - current.timestamp).total_seconds()
            latencies[stage] = latency
            
        return latencies
```

## ⚡ Performance Optimizations

### Event Pooling

Reuse event objects to reduce garbage collection:

```python
class EventPool:
    """Pool of reusable event objects"""
    
    def __init__(self):
        self.pools = {}  # event_type -> [available_instances]
        
    def get_event(self, event_type: type) -> Any:
        """Get event from pool or create new"""
        if event_type not in self.pools:
            self.pools[event_type] = []
            
        if self.pools[event_type]:
            return self.pools[event_type].pop()
        else:
            return event_type()
            
    def return_event(self, event: Any):
        """Return event to pool for reuse"""
        event_type = type(event)
        # Reset event state
        event.reset()
        self.pools[event_type].append(event)
```

### Batch Processing

Process events in batches for efficiency:

```python
class BatchEventProcessor:
    """Process events in batches for efficiency"""
    
    def __init__(self, batch_size=100, timeout_ms=10):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.pending_events = []
        
    def add_event(self, event):
        """Add event to current batch"""
        self.pending_events.append(event)
        
        if len(self.pending_events) >= self.batch_size:
            self.process_batch()
            
    def process_batch(self):
        """Process all pending events as batch"""
        if self.pending_events:
            batch = self.pending_events[:]
            self.pending_events.clear()
            
            # Process entire batch at once
            self.handle_event_batch(batch)
```

### Async Event Processing

Use async processing for high-throughput scenarios:

```python
import asyncio

class AsyncEventBus:
    """Async event bus for high-performance scenarios"""
    
    def __init__(self):
        self.subscribers = {}
        self.event_queue = asyncio.Queue()
        
    async def emit_async(self, event):
        """Emit event asynchronously"""
        await self.event_queue.put(event)
        
    async def process_events(self):
        """Process events asynchronously"""
        while True:
            event = await self.event_queue.get()
            await self.handle_event_async(event)
            
    async def handle_event_async(self, event):
        """Handle single event asynchronously"""
        event_type = type(event)
        if event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event_type]:
                task = asyncio.create_task(handler(event))
                tasks.append(task)
            await asyncio.gather(*tasks)
```

## 🔧 Configuration-Driven Event Flow

### YAML Event Configuration

Define event flows entirely through configuration:

```yaml
# Complete event flow configuration
event_system:
  # Event bus settings
  event_buses:
    default:
      queue_size: 10000
      batch_size: 100
      timeout_ms: 10
      
  # Event routing
  routing:
    - source: "data_container"
      targets: ["indicator_container"]
      events: ["BarEvent"]
      
    - source: "indicator_container"
      targets: ["strategy_container"]
      events: ["IndicatorEvent"]
      filters:
        - "event.indicator_name in ['SMA_20', 'RSI_14']"
        
    - source: "strategy_container"
      targets: ["risk_container"]
      events: ["TradingSignal"]
      filters:
        - "event.strength > 0.5"
        
  # Event transformation
  transformations:
    - from_event: "TradingSignal"
      to_event: "OrderEvent"
      transformer: "signal_to_order_transformer"
      
  # Monitoring
  monitoring:
    trace_events: true
    correlation_tracking: true
    performance_metrics: true
    dead_letter_queue: true
```

## 🎯 Benefits of Event-Driven Design

### 1. **Loose Coupling**
Components don't know about each other, only events:
- Easy to add/remove components
- Easy to modify behavior
- Easy to test in isolation
- Easy to scale independently

### 2. **Natural Parallelism**
Events enable concurrent processing:
- Multiple strategies can process same data simultaneously
- Risk management can run in parallel with signal generation
- Portfolio updates don't block order generation

### 3. **Production Consistency**
Same event flow works everywhere:
- Backtesting uses same logic as live trading
- No subtle differences between environments
- Reliable behavior across all scenarios

### 4. **Debugging and Monitoring**
Event flows are observable:
- Complete audit trail of all decisions
- Performance analysis at event level
- Easy to replay and debug issues
- Clear separation of concerns

### 5. **Testability**
Easy to test with mock events:
```python
def test_strategy_signal_generation():
    # Create mock events
    bar = BarEvent(symbol="AAPL", close=150.0)
    
    # Inject into strategy
    strategy = MomentumStrategy()
    signal = strategy.on_bar(bar)
    
    # Verify result
    assert signal.action == "BUY"
    assert signal.strength > 0.5
```

## 🛠️ Best Practices

### Event Design
1. **Immutable Events**: Events should not be modified after creation
2. **Rich Context**: Include all necessary information in events
3. **Consistent Schema**: Use consistent field names and types
4. **Metadata**: Include correlation IDs and timestamps

### Error Handling
1. **Graceful Degradation**: Handle missing or invalid events gracefully
2. **Dead Letter Queues**: Route failed events for later analysis
3. **Circuit Breakers**: Stop processing when error rates are high
4. **Retry Logic**: Implement appropriate retry strategies

### Performance
1. **Event Pooling**: Reuse event objects when possible
2. **Batch Processing**: Process events in batches for efficiency
3. **Async When Appropriate**: Use async for high-throughput scenarios
4. **Monitor Latency**: Track event processing times

## 🤔 Common Questions

**Q: Are events slower than direct method calls?**
A: Individual events have minimal overhead (<1ms). The benefits of loose coupling and parallelism far outweigh the small latency cost.

**Q: How do I debug event flows?**
A: Use correlation IDs and event tracing. ADMF-PC provides built-in tools to visualize event flows and analyze performance.

**Q: Can events be lost?**
A: No! ADMF-PC uses reliable event delivery with acknowledgments and dead letter queues for failed events.

**Q: How do I handle event ordering?**
A: Events within a container are processed in order. Cross-container events use timestamps for ordering when needed.

## 🎯 Key Takeaways

1. **Events > Method Calls**: Events provide loose coupling and flexibility
2. **Isolation + Communication**: Containers are isolated but communicate through events
3. **Configuration-Driven**: Event flows defined in YAML, not code
4. **Observable**: Complete audit trail and debugging capabilities
5. **Production-Ready**: Same logic works for backtesting and live trading

Event-driven design is what enables ADMF-PC's zero-code approach to scale while maintaining professional-grade reliability and performance.

---

Next: [Protocol + Composition](protocol-composition.md) - How infinite composability works →