# Position Tracking Refactor: Single Source of Truth

## Problem with Current Architecture

The current system has **duplicate position tracking**:
1. Risk module tracks positions with `Decimal` precision
2. Execution module tracks positions with `float` 
3. Both update independently → synchronization issues

## Refactored Architecture: Single Source of Truth

### Core Principle
**Risk Module's PortfolioState is the ONLY authoritative position tracker**

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Risk Module                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │            PortfolioState (SSOT)                 │   │
│  │  - Positions (Decimal precision)                 │   │
│  │  - Cash tracking                                 │   │
│  │  - P&L calculation                               │   │
│  │  - Risk metrics                                  │   │
│  └─────────────────────────────────────────────────┘   │
│                          ▲                              │
│                          │ Updates                      │
└──────────────────────────┼──────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────┐
│                    Execution Module                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │              BacktestBroker                      │   │
│  │  - Order tracking only                           │   │
│  │  - References PortfolioState for validation     │   │
│  │  - Records fills                                 │   │
│  │  - NO position tracking                          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Order Submission**
   ```python
   # Broker validates against portfolio state
   position = portfolio_state.get_position(symbol)
   if order.side == SELL and quantity > position.quantity:
       reject_order()
   ```

2. **Fill Execution**
   ```python
   # Broker records fill
   broker.execute_fill(fill)
   
   # Risk module updates positions
   portfolio_state.update_position(
       symbol=fill.symbol,
       quantity_delta=fill.quantity,
       price=fill.price,
       timestamp=fill.executed_at
   )
   ```

3. **Position Query**
   ```python
   # Always query from portfolio state
   positions = portfolio_state.get_all_positions()
   # Broker can convert format if needed
   broker_positions = broker.convert_positions(positions)
   ```

### Implementation Changes

#### 1. Refactored BacktestBroker
```python
class BacktestBrokerRefactored:
    def __init__(self, portfolio_state: PortfolioStateProtocol):
        self.portfolio_state = portfolio_state  # Reference, not own
        self.order_tracker = OrderTracker()     # Orders only
        # NO position tracking
```

#### 2. Execution Engine Integration
```python
class ExecutionEngine:
    async def process_order(self, order: Order):
        # Submit to broker
        order_id = await self.broker.submit_order(order)
        
        # Simulate execution
        fill = await self.market_simulator.simulate_fill(order)
        
        # Update portfolio state (single source of truth)
        self.portfolio_state.update_position(
            symbol=fill.symbol,
            quantity_delta=fill.quantity,
            price=fill.price,
            timestamp=fill.executed_at
        )
```

#### 3. Risk-Execution Contract
```python
# Risk module provides interface
class PortfolioStateProtocol:
    def get_position(symbol: str) -> Position
    def update_position(...) -> Position
    def get_cash_balance() -> Decimal
    
# Execution module consumes interface
class BacktestBroker:
    def __init__(self, portfolio_state: PortfolioStateProtocol):
        # Uses portfolio state for all position queries
```

### Benefits

1. **No Synchronization Issues**: Single source of truth
2. **Consistent Precision**: Always use Decimal from Risk module
3. **Simpler Architecture**: Less code to maintain
4. **Clear Ownership**: Risk module owns positions
5. **Better Testing**: Mock single interface

### Migration Path

1. **Phase 1**: Create refactored broker alongside existing
2. **Phase 2**: Update execution engine to use refactored broker
3. **Phase 3**: Update tests
4. **Phase 4**: Remove old broker and Position class from execution

### Example Usage

```python
# Initialize with single portfolio state
portfolio_state = PortfolioState(initial_capital=Decimal("100000"))
broker = BacktestBrokerRefactored(portfolio_state)
execution_engine = ExecutionEngine(broker, portfolio_state)

# Process order
order = Order(symbol="AAPL", side=BUY, quantity=100)
fill = await execution_engine.execute(order)

# Position is updated in one place
position = portfolio_state.get_position("AAPL")
print(f"Position: {position.quantity} @ {position.average_price}")
```

### Key Principles

1. **Risk Module Owns Positions**: All position state lives in Risk module
2. **Execution Module is Stateless**: Only tracks orders and fills
3. **Clear Interfaces**: Use protocols for clean separation
4. **Decimal Everywhere**: No float/Decimal conversion issues
5. **Event-Driven Updates**: Portfolio state emits events on changes