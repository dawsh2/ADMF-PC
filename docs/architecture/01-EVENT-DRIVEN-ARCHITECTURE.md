# Event-Driven Architecture

## Overview

The ADMF-PC system is built on a robust event-driven architecture that enables loose coupling, scalability, and testability. This document details the core concepts, patterns, and implementation guidelines.

## Core Concepts

### 1. Events as First-Class Citizens

Events are the primary mechanism for communication between components:
- **Immutable**: Events are never modified after creation
- **Self-contained**: Events carry all necessary data
- **Timestamped**: Every event has a precise timestamp
- **Typed**: Strong typing ensures compile-time safety

### 2. Event Bus

The event bus is the central nervous system:
```python
class EventBus:
    """Central event routing mechanism"""
    
    def publish(self, event_type: str, event: Event) -> None:
        """Publish event to all subscribers"""
        
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to specific event type"""
```

### 3. Event Isolation

Each container has its own isolated event bus:
- **No cross-container contamination**
- **Hierarchical event propagation**
- **Explicit parent-child relationships**
- **Testable in isolation**

## Event Flow Patterns

### 1. Data Flow Pattern (Complete Cycle)
```
DataSource → [BAR] → Indicator → [INDICATOR_VALUE] → Strategy
                                                     ↓
                                                 [SIGNAL]
                                                     ↓
                                                RiskManager
                                                     ↓
                                                  [ORDER]
                                                     ↓
                                              ExecutionEngine
                                                     ↓
                                                  [FILL]
                                                     ↓
                                                RiskManager
                                                     ↓
                                            [PORTFOLIO_UPDATE]
                                                     ↓
                                             PortfolioState
```

**Critical**: The FILL event completes the cycle by:
- Updating portfolio positions
- Adjusting available capital
- Modifying risk limits
- Enabling accurate position sizing for future signals

### 2. Hierarchical Pattern
```
ParentContainer
    │
    ├── ChildContainer1 [isolated bus]
    │       │
    │       └── [CHILD_EVENT] → Parent subscribes
    │
    └── ChildContainer2 [isolated bus]
            │
            └── [CHILD_EVENT] → Parent subscribes
```

### 3. Aggregation Pattern
```
Strategy1 → [SIGNAL] ┓
                     ├→ SignalAggregator → [CONSENSUS_SIGNAL]
Strategy2 → [SIGNAL] ┛
```

## Event Types

### Market Data Events
```python
@dataclass
class BarEvent(Event):
    """Market data bar event"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
```

### Trading Events
```python
@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    symbol: str
    direction: Direction
    strength: float
    strategy_id: str
    timestamp: datetime

@dataclass
class OrderEvent(Event):
    """Order event from risk manager"""
    order_id: str
    symbol: str
    direction: Direction
    quantity: int
    order_type: OrderType
    limit_price: Optional[float]
    stop_price: Optional[float]
    timestamp: datetime

@dataclass
class FillEvent(Event):
    """Execution fill event - completes the cycle"""
    order_id: str
    symbol: str
    direction: Direction
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    execution_id: str

@dataclass
class PortfolioUpdateEvent(Event):
    """Portfolio state update after fill"""
    symbol: str
    position_quantity: int
    average_price: float
    realized_pnl: float
    unrealized_pnl: float
    total_value: float
    timestamp: datetime
```

### System Events
```python
@dataclass
class RegimeChangeEvent(Event):
    """Market regime change event"""
    old_regime: MarketRegime
    new_regime: MarketRegime
    confidence: float
    classifier_id: str
```

## Event Flow Lifecycle

### Complete Trading Cycle

1. **Market Data Arrival**
   ```python
   # DataSource emits BAR event
   bar_event = BarEvent(symbol="AAPL", close=150.0, ...)
   event_bus.publish("BAR", bar_event)
   ```

2. **Signal Generation**
   ```python
   # Strategy processes bar and emits signal
   signal_event = SignalEvent(symbol="AAPL", direction=Direction.BUY, ...)
   event_bus.publish("SIGNAL", signal_event)
   ```

3. **Risk Management**
   ```python
   # RiskManager validates and sizes position
   order_event = OrderEvent(symbol="AAPL", quantity=100, ...)
   event_bus.publish("ORDER", order_event)
   ```

4. **Execution**
   ```python
   # ExecutionEngine processes order and returns fill
   fill_event = FillEvent(symbol="AAPL", quantity=100, price=150.05, ...)
   event_bus.publish("FILL", fill_event)
   ```

5. **Portfolio Update (Cycle Completion)**
   ```python
   # RiskManager updates portfolio state
   portfolio_update = PortfolioUpdateEvent(
       symbol="AAPL",
       position_quantity=100,
       average_price=150.05,
       total_value=99850.00  # After commission
   )
   event_bus.publish("PORTFOLIO_UPDATE", portfolio_update)
   ```

### Critical Feedback Loop

The FILL → PORTFOLIO_UPDATE loop is essential because:
- **Position Tracking**: Accurate current positions for risk calculations
- **Capital Management**: Updated available capital for position sizing
- **Risk Limits**: Dynamic adjustment based on P&L
- **Performance Metrics**: Real-time strategy performance tracking

## Implementation Guidelines

### 1. Event Creation

Always use factory methods:
```python
def create_signal_event(symbol: str, direction: Direction, 
                       strength: float, strategy_id: str) -> SignalEvent:
    """Factory method ensures consistent event creation"""
    return SignalEvent(
        event_id=generate_event_id(),
        timestamp=datetime.now(),
        symbol=symbol,
        direction=direction,
        strength=strength,
        strategy_id=strategy_id
    )
```

### 2. Event Handling

Use type-safe handlers:
```python
class StrategyEventHandler:
    def on_bar(self, event: BarEvent) -> None:
        """Type-safe bar event handler"""
        # Process bar data
        
    def on_regime_change(self, event: RegimeChangeEvent) -> None:
        """Type-safe regime change handler"""
        # Adjust strategy for new regime
```

### 3. Event Bus Isolation

Always create isolated buses:
```python
class Container:
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.isolation_manager = get_isolation_manager()
        self.event_bus = self.isolation_manager.create_isolated_bus(
            container_id
        )
```

## Testing Event-Driven Systems

### 1. Event Capture
```python
class EventCapture:
    """Capture events for testing"""
    def __init__(self):
        self.captured_events = []
    
    def capture(self, event: Event) -> None:
        self.captured_events.append(event)
```

### 2. Event Injection
```python
def test_strategy_response():
    strategy = TestStrategy()
    test_bar = create_test_bar_event()
    
    # Inject event
    strategy.on_bar(test_bar)
    
    # Verify response
    assert strategy.last_signal is not None
```

### 3. Event Flow Testing
```python
def test_complete_event_cycle():
    """Test the complete event cycle including feedback"""
    # Setup test pipeline
    pipeline = create_test_pipeline()
    
    # Capture events at each stage
    captures = {
        'signal': EventCapture(),
        'order': EventCapture(),
        'fill': EventCapture(),
        'portfolio_update': EventCapture()
    }
    
    # Wire captures
    pipeline.event_bus.subscribe('SIGNAL', captures['signal'].capture)
    pipeline.event_bus.subscribe('ORDER', captures['order'].capture)
    pipeline.event_bus.subscribe('FILL', captures['fill'].capture)
    pipeline.event_bus.subscribe('PORTFOLIO_UPDATE', captures['portfolio_update'].capture)
    
    # Inject test data
    pipeline.inject_bar(test_bar)
    
    # Verify complete cycle
    assert len(captures['signal'].captured_events) > 0
    assert len(captures['order'].captured_events) > 0
    assert len(captures['fill'].captured_events) > 0
    assert len(captures['portfolio_update'].captured_events) > 0
    
    # Verify portfolio state updated
    assert pipeline.portfolio_state.positions['AAPL'].quantity > 0
```

## Performance Considerations

### 1. Event Pooling
```python
class EventPool:
    """Reuse event objects to reduce GC pressure"""
    def __init__(self, event_class: Type[Event], size: int = 1000):
        self.pool = [event_class() for _ in range(size)]
        self.available = deque(self.pool)
```

### 2. Batch Processing
```python
class BatchEventProcessor:
    """Process events in batches for efficiency"""
    def process_batch(self, events: List[Event]) -> None:
        # Process multiple events together
        pass
```

### 3. Async Event Handling
```python
async def async_event_handler(event: Event) -> None:
    """Handle events asynchronously for better throughput"""
    await process_event_async(event)
```

## Best Practices

### 1. Event Naming
- Use descriptive, action-based names
- Include context in event type: `STRATEGY_SIGNAL`, not just `SIGNAL`
- Be consistent with tense: past for completed actions
- Distinguish between forward flow (SIGNAL, ORDER) and feedback (FILL, UPDATE)

### 2. Event Data
- Include all necessary data in the event
- Avoid references to mutable objects
- Keep events small and focused

### 3. Error Handling
```python
def safe_event_handler(event: Event) -> None:
    try:
        process_event(event)
    except Exception as e:
        logger.error(f"Event processing failed: {e}")
        # Emit error event
        error_event = create_error_event(event, e)
        event_bus.publish("ERROR", error_event)
```

### 4. Event Ordering
- Events are processed in order within a container
- No ordering guarantees across containers
- Use sequence numbers if ordering critical

## Common Patterns

### 1. Request-Response Pattern
```python
# Request
request_event = DataRequestEvent(
    request_id=generate_id(),
    symbol="AAPL",
    start_date=start,
    end_date=end
)

# Response  
response_event = DataResponseEvent(
    request_id=request_event.request_id,
    data=market_data
)
```

### 2. Event Sourcing Pattern
```python
class EventStore:
    """Store all events for replay"""
    def store(self, event: Event) -> None:
        self.events.append(event)
    
    def replay(self, from_timestamp: datetime) -> List[Event]:
        return [e for e in self.events if e.timestamp >= from_timestamp]
```

### 3. Event Aggregation Pattern
```python
class EventAggregator:
    """Aggregate multiple events into summary"""
    def aggregate(self, events: List[Event], window: timedelta) -> AggregatedEvent:
        # Combine events within time window
        pass
```

## Implementation Example: Complete Cycle

```python
class TradingSystem:
    """Example showing complete event cycle"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.portfolio_state = PortfolioState()
        
        # Wire up the complete cycle
        self.event_bus.subscribe("BAR", self.strategy.on_bar)
        self.event_bus.subscribe("SIGNAL", self.risk_manager.on_signal)
        self.event_bus.subscribe("ORDER", self.execution_engine.on_order)
        self.event_bus.subscribe("FILL", self.risk_manager.on_fill)  # Critical feedback
        self.event_bus.subscribe("PORTFOLIO_UPDATE", self.portfolio_state.update)
    
    def process_bar(self, bar: Bar):
        """Process new market data through complete cycle"""
        # 1. Emit bar event
        self.event_bus.publish("BAR", BarEvent.from_bar(bar))
        
        # 2-5. Events flow automatically through subscriptions
        # The cycle completes when portfolio state is updated
```

## Debugging Event Systems

### 1. Event Logging
```python
class EventLogger:
    def log_event(self, event_type: str, event: Event) -> None:
        logger.info(
            f"EVENT | {event_type} | {event.timestamp} | "
            f"{event.__class__.__name__} | {event}"
        )
```

### 2. Event Visualization
- Use tools to visualize event flow
- Track event latency and throughput
- Monitor event bus queue depths

### 3. Event Replay
- Replay events to reproduce issues
- Use deterministic timestamps for testing
- Capture event sequences for debugging

## Summary

The event-driven architecture provides:
- **Loose coupling** between components
- **Complete feedback loops** for accurate state management
- **Testability** through event injection
- **Scalability** through async processing
- **Flexibility** through dynamic subscriptions
- **Debuggability** through event logging
- **State consistency** through proper event cycles

By following these patterns and ensuring complete event cycles (including the critical FILL → PORTFOLIO_UPDATE feedback), the system maintains accurate state while enabling sophisticated interactions between components.