# Race Conditions and Pending Orders in ADMF-PC

This document explains how race conditions could occur in the ADMF-PC trading system and how we prevent them using the container architecture and pending order tracking.

## Understanding the Problem

### What Are Race Conditions?

Race conditions occur when multiple concurrent operations access shared state in an unpredictable order, leading to incorrect results. In trading systems, this can result in:
- Over-ordering (creating multiple orders when only one was intended)
- Position inconsistencies (actual position differs from tracked position)
- Risk limit violations (orders approved based on stale state)

### Where Race Conditions Could Occur in ADMF-PC

Let's trace through the event flow to identify potential race points:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Event Flow Timeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  T0: BAR arrives for AAPL                                      │
│   ↓                                                            │
│  T1: Features calculated                                       │
│   ↓                                                            │
│  T2: Multiple strategies receive features simultaneously       │
│      │                                                         │
│      ├─ Strategy_A: Momentum signals BUY AAPL                │
│      └─ Strategy_B: Mean Reversion signals SELL AAPL         │
│                                                                │
│  T3: Both signals arrive at Portfolio Container               │
│      ⚠️ RACE CONDITION WINDOW OPENS HERE                      │
│      │                                                         │
│      ├─ Signal A: Reads position = 100                       │
│      ├─ Signal B: Reads position = 100 (same!)              │
│      ├─ Signal A: Creates ORDER_REQUEST (BUY 50)            │
│      └─ Signal B: Creates ORDER_REQUEST (SELL 150)          │
│                                                                │
│  T4: Both orders sent to Risk Service                         │
│      │                                                         │
│      ├─ Risk sees portfolio state with 100 shares           │
│      └─ Approves both orders (individually valid)           │
│                                                                │
│  T5: Execution processes both orders                          │
│      │                                                         │
│      ├─ FILL: Bought 50 (position now 150)                  │
│      └─ FILL: Sold 150 (position now -0???)                 │
│                                                                │
│  Result: Unintended short position!                           │
└─────────────────────────────────────────────────────────────────┘
```

## The Core Problem: Concurrent Signal Processing

The fundamental issue is that multiple strategies can generate signals for the same symbol simultaneously:

```python
# What happens without protection:

# Time T3.0 - Signal A arrives
def handle_signal_a(signal):
    position = self.positions['AAPL']  # Returns 100
    # ... calculate order size ...
    order = create_order('BUY', 50)
    
# Time T3.1 - Signal B arrives (microseconds later)
def handle_signal_b(signal):
    position = self.positions['AAPL']  # STILL returns 100!
    # ... calculate order size ...
    order = create_order('SELL', 150)
    
# Both orders created based on same position snapshot
```

## How ADMF-PC Prevents Race Conditions

### 1. Container Isolation with Sequential Processing

Each Portfolio Container has its own EventBus that processes events sequentially:

```
┌─────────────────────────────────────────────────────────────────┐
│                  Portfolio Container                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   EventBus Queue (FIFO)                                        │
│   ┌─────────────────────────────────────────┐                  │
│   │ 1. Signal A (BUY AAPL)                 │                  │
│   │ 2. Signal B (SELL AAPL)                │                  │
│   │ 3. FILL from previous order            │                  │
│   │ 4. Signal C (BUY MSFT)                 │                  │
│   └─────────────────────────────────────────┘                  │
│                     ↓                                           │
│   Sequential Processing:                                        │
│   - Process Signal A completely                                 │
│   - THEN process Signal B                                      │
│   - THEN process FILL                                          │
│   - THEN process Signal C                                      │
│                                                                 │
│   No concurrent access to portfolio state!                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Pending Order Tracking (MUST BE IMPLEMENTED)

Even with sequential processing, we need to track pending orders:

```python
@dataclass
class PortfolioState:
    """Portfolio state with pending order tracking."""
    positions: Dict[str, float] = field(default_factory=dict)
    cash: float = 100000.0
    pending_orders: Dict[str, Order] = field(default_factory=dict)  # CRITICAL!
    
    def handle_signal(self, signal_event: Event) -> Optional[Event]:
        """Process signal with pending order check."""
        symbol = signal_event.payload['symbol']
        
        # CHECK 1: Do we have pending orders for this symbol?
        pending_for_symbol = [
            order for order in self.pending_orders.values()
            if order.symbol == symbol
        ]
        
        if pending_for_symbol:
            logger.info(f"Skipping signal for {symbol} - {len(pending_for_symbol)} pending orders")
            return None
            
        # CHECK 2: Calculate order considering current position
        current_position = self.positions.get(symbol, 0)
        order_quantity = self._calculate_order_quantity(signal_event, current_position)
        
        if order_quantity == 0:
            return None
            
        # CREATE ORDER AND TRACK IT
        order_id = f"ORD_{uuid.uuid4().hex[:8]}"
        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=order_quantity,
            signal_id=signal_event.metadata.get('event_id'),
            created_at=datetime.now()
        )
        
        # Add to pending BEFORE publishing
        self.pending_orders[order_id] = order
        
        # Create order request event
        return Event(
            event_type=EventType.ORDER_REQUEST,
            payload={
                'order_id': order_id,
                'symbol': symbol,
                'quantity': order_quantity,
                'portfolio_snapshot': {
                    'positions': self.positions.copy(),
                    'cash': self.cash,
                    'pending_orders': list(self.pending_orders.keys())
                }
            },
            correlation_id=signal_event.correlation_id,
            causation_id=signal_event.metadata.get('event_id')
        )
    
    def handle_fill(self, fill_event: Event) -> None:
        """Update positions and clear pending order."""
        order_id = fill_event.payload['order_id']
        symbol = fill_event.payload['symbol']
        quantity = fill_event.payload['quantity']
        price = fill_event.payload['price']
        
        # Update position
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        self.cash -= quantity * price
        
        # CRITICAL: Remove from pending orders
        self.pending_orders.pop(order_id, None)
        
    def handle_order_rejected(self, rejection_event: Event) -> None:
        """Clear pending order on rejection."""
        order_id = rejection_event.payload['order_id']
        
        # CRITICAL: Remove from pending orders
        self.pending_orders.pop(order_id, None)
```

## Complete Race Condition Prevention Flow

Here's how the system works with proper pending order tracking:

```
┌─────────────────────────────────────────────────────────────────┐
│              Protected Event Flow Timeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  T0: Signal A arrives at Portfolio EventBus                    │
│      → Queued for processing                                   │
│                                                                 │
│  T1: Signal B arrives at Portfolio EventBus                    │
│      → Queued behind Signal A                                  │
│                                                                 │
│  T2: EventBus processes Signal A                               │
│      │                                                         │
│      ├─ Check pending_orders['AAPL'] → None                  │
│      ├─ Read position: 100                                    │
│      ├─ Create ORDER_REQUEST (BUY 50)                        │
│      ├─ Add to pending_orders['order_123']                   │
│      └─ Publish ORDER_REQUEST                                 │
│                                                                │
│  T3: EventBus processes Signal B                               │
│      │                                                         │
│      ├─ Check pending_orders['AAPL'] → ['order_123'] ✓       │
│      └─ SKIP - Pending order exists!                         │
│                                                                │
│  T4: FILL arrives for order_123                               │
│      │                                                         │
│      ├─ Update position: 100 → 150                           │
│      └─ Remove from pending_orders                           │
│                                                                │
│  T5: Next signal for AAPL can now be processed               │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Risk Service Validation

The stateless Risk Service also plays a role in preventing issues:

```python
class RiskService:
    """Stateless risk validation service."""
    
    def validate_order(self, order_request_event: Event) -> Event:
        """Validate order with portfolio snapshot."""
        snapshot = order_request_event.payload['portfolio_snapshot']
        
        # Calculate total exposure INCLUDING pending orders
        symbol = order_request_event.payload['symbol']
        current_position = snapshot['positions'].get(symbol, 0)
        
        # Get all pending orders from snapshot
        pending_order_ids = snapshot['pending_orders']
        # In practice, would need order details - simplified here
        pending_exposure = self._calculate_pending_exposure(pending_order_ids)
        
        new_order_quantity = order_request_event.payload['quantity']
        total_exposure = current_position + pending_exposure + new_order_quantity
        
        # Validate against limits
        if abs(total_exposure) > self.position_limits.get(symbol, 1000):
            return Event(
                event_type=EventType.ORDER_REJECTED,
                payload={
                    'order_id': order_request_event.payload['order_id'],
                    'reason': 'Would exceed position limits',
                    'current_position': current_position,
                    'pending_exposure': pending_exposure,
                    'total_would_be': total_exposure
                }
            )
        
        # Validate against cash
        required_cash = abs(new_order_quantity) * self._estimate_price(symbol)
        available_cash = snapshot['cash'] - self._calculate_pending_cash_usage(pending_order_ids)
        
        if required_cash > available_cash:
            return Event(
                event_type=EventType.ORDER_REJECTED,
                payload={
                    'order_id': order_request_event.payload['order_id'],
                    'reason': 'Insufficient cash'
                }
            )
        
        # Order is valid
        return Event(
            event_type=EventType.ORDER,
            payload=order_request_event.payload
        )
```

## Terminal Event Tracking at Root Level

The root container tracks completion using terminal events:

```python
class TerminalEventTracker:
    """Component that tracks event flow completion."""
    
    def initialize(self, container: Container) -> None:
        self.container = container
        self.pending_orders: Dict[str, str] = {}  # order_id -> correlation_id
        
        # Only need to track 3 event types!
        container.event_bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)
        container.event_bus.subscribe(EventType.FILL, self._on_terminal)
        container.event_bus.subscribe(EventType.ORDER_REJECTED, self._on_terminal)
        
    def _on_order_request(self, event: Event):
        """New order to track."""
        order_id = event.payload.get('order_id')
        correlation_id = event.correlation_id or event.payload.get('bar_sequence')
        
        if order_id and correlation_id:
            self.pending_orders[order_id] = correlation_id
            
    def _on_terminal(self, event: Event):
        """Order reached terminal state."""
        order_id = event.payload.get('order_id')
        
        if order_id in self.pending_orders:
            correlation_id = self.pending_orders.pop(order_id)
            
            # Check if this correlation (e.g., bar) is complete
            remaining = [cid for cid in self.pending_orders.values() 
                        if cid == correlation_id]
            
            if not remaining:
                # All orders for this correlation are complete
                self.container.publish_event(
                    Event(
                        event_type='CORRELATION_COMPLETE',
                        payload={'correlation_id': correlation_id}
                    )
                )
```

## Integration with Event Refactor

This approach is fully compatible with the refactored event system:

1. **EventBus remains pure pub/sub**: No race condition logic in the bus itself
2. **Tracing via observers**: Race condition monitoring can be added as an observer
3. **Composition over inheritance**: TerminalEventTracker is a composable component
4. **Protocol-based**: All components follow defined protocols

Example integration:

```python
# In container factory
container = Container(config)

# Add portfolio state
portfolio_state = PortfolioState(initial_capital=config['initial_capital'])
container.add_component('portfolio_state', portfolio_state)

# Add terminal event tracking at root level
if container.role == ContainerRole.ROOT:
    tracker = TerminalEventTracker()
    container.add_component('terminal_tracker', tracker)

# Enable event tracing if configured
if config.get('enable_event_tracing'):
    container.event_bus.enable_tracing({
        'correlation_id': f'{container.container_id}_trace',
        'events_to_trace': ['ORDER_REQUEST', 'FILL', 'ORDER_REJECTED'],
        'retention_policy': 'minimal'  # Only keep open orders
    })
```

## Summary

Race conditions are prevented through:

1. **Container Architecture**: Each portfolio has isolated state with sequential event processing
2. **Pending Order Tracking**: Portfolio state tracks orders from creation to terminal events
3. **Terminal Event Monitoring**: Root container ensures all orders complete before proceeding
4. **Stateless Risk Validation**: Risk service validates using complete state snapshots

**Critical Implementation Note**: The `pending_orders` tracking in PortfolioState MUST be implemented. Without it, even sequential processing cannot prevent over-ordering when multiple signals arrive between order creation and fill.

The architecture already provides the foundation (isolated event buses, sequential processing) - we just need to add the pending order tracking logic to complete the protection.