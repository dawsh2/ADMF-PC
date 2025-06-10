# Portfolio Module

## Overview

The Portfolio module is the **central but simple** hub for ADMF-PC. It tracks portfolio state and forwards signals/state to the Risk module for sophisticated analysis, meta-analysis, strategy ensembles, and emergent property discovery. 

Portfolio's job: Track state, forward to Risk for the smart stuff.

## Architecture

### Module Structure

```
src/portfolio/
├── __init__.py          # Module exports  
├── protocols.py         # Simple portfolio protocols and types
├── state.py            # PortfolioState - the core state tracker
└── README.md           # This documentation
```

### Key Components

#### PortfolioState (state.py)
- **THE canonical portfolio state tracker**
- Tracks positions, cash, P&L
- Race condition prevention for pending orders
- Basic risk metrics calculation
- Forwards state to Risk module for analysis
- Event-driven processing (fills, market data)

#### Portfolio Protocols (protocols.py)
- `PortfolioStateProtocol`: Simple state tracking interface
- **Only defines portfolio-specific types**: `Position`, `RiskMetrics`  
- **No duplicate types** - uses canonical Event system
- No complex logic - just data structures

## Key Features

### Position Management
- Automatic position tracking with average price calculation
- Partial position closing with realized P&L calculation
- Position flipping (long to short) support
- Market value and cost basis calculations

### Risk Metrics
- Real-time portfolio value calculation
- Sharpe ratio calculation (when sufficient history available)
- Maximum and current drawdown tracking
- Value at Risk (VaR) estimation
- Leverage and concentration calculations

### Race Condition Prevention
- Pending order tracking to prevent duplicate orders
- Thread-safe position updates
- Atomic operations for portfolio state changes

## Integration with Risk Module

Portfolio is the **central hub** that forwards everything to Risk for the smart analysis:

- **Portfolio Module**: Simple state tracking, forward signals/state to Risk
- **Risk Module**: **ALL the intelligence** - sizing, limits, validation, meta-analysis, strategy ensembles, emergent properties

```python
# Portfolio: Just track state and forward
portfolio_state = PortfolioState(initial_capital=100000)
position = portfolio_state.update_position(symbol, quantity, price, timestamp)

# Risk gets all the signals and state for analysis
signals = strategy.generate_signals(data)
orders = risk_module.process_signals_with_portfolio_oversight(
    signals=signals,
    portfolio_state=portfolio_state,
    market_data=market_data
)

# Risk module does ALL the smart stuff:
# - Position sizing based on portfolio state
# - Risk limit validation  
# - Signal ensemble analysis
# - Meta-analysis of signal vs returns
# - Emergent pattern detection
```

## Usage Examples

### Basic Portfolio Tracking

```python
from portfolio import PortfolioState, Position
from decimal import Decimal
from datetime import datetime

# Initialize portfolio
portfolio = PortfolioState(initial_capital=Decimal("100000"))

# Update position from a trade
position = portfolio.update_position(
    symbol="SPY",
    quantity_delta=Decimal("100"),  # Buy 100 shares
    price=Decimal("400.50"),
    timestamp=datetime.now()
)

# Get portfolio metrics
metrics = portfolio.get_risk_metrics()
print(f"Total Value: ${metrics.total_value}")
print(f"Unrealized P&L: ${metrics.unrealized_pnl}")
```

### Race Condition Prevention

```python
# Before creating an order, check if one is already pending
if portfolio.can_create_order("SPY"):
    # Create and track new order
    order = create_order(symbol="SPY", ...)
    portfolio.add_pending_order(order)
    
    # Submit order to execution
    execution_engine.submit_order(order)
else:
    logger.info("Order already pending for SPY, skipping")

# When order fills or is rejected
portfolio.remove_pending_order(order.order_id)
```

### Performance Analysis

```python
# Get comprehensive performance summary
summary = portfolio.get_performance_summary()
print(f"Total Return: {summary['total_return']}")
print(f"Max Drawdown: {summary['max_drawdown']}")  
print(f"Sharpe Ratio: {summary['sharpe_ratio']}")
```

## Migration Notes

### From Risk Module

The following components were moved from `src/risk/` to `src/portfolio/`:

- `portfolio_state.py` → `state.py`
- `models.py` → `models.py` (portfolio-specific portions)
- Portfolio-related protocols moved to `protocols.py`

### Import Updates

Old imports:
```python
from risk.portfolio_state import PortfolioState
from risk.protocols import PortfolioStateProtocol
```

New imports:
```python
from portfolio import PortfolioState
from portfolio.protocols import PortfolioStateProtocol
```

## Design Principles

### Single Responsibility
- Portfolio module: State tracking and metrics
- Risk module: Validation and meta-analysis

### Protocol + Composition
- Clean separation between interfaces and implementations
- Composition over inheritance for extending functionality
- No "enhanced" or "improved" versions

### Thread Safety
- All portfolio state updates are atomic
- Pending order tracking prevents race conditions
- Immutable return types where possible

## Future Enhancements

### Planned Features
- Multi-currency portfolio support
- Portfolio-level reporting and visualization
- Historical performance attribution
- Advanced risk metrics (Beta, Alpha, Information Ratio)

### Architecture Growth
- Portfolio will naturally expand as needs arise
- Risk module will focus on meta-analysis of portfolio behavior
- Clear separation enables independent development

## Testing

### Unit Tests
- Position tracking accuracy
- P&L calculation correctness
- Race condition prevention
- Metrics calculation validation

### Integration Tests
- Portfolio-Risk module interaction
- Order lifecycle management
- Performance metrics accuracy

### Test Utilities
```python
from portfolio.models import create_test_signal, create_test_order, create_test_fill

# Create test data
signal = create_test_signal(symbol="SPY", strength=0.8)
order = create_test_order(symbol="SPY", quantity=100)
fill = create_test_fill(symbol="SPY", price=400.50)
```

---

This portfolio module follows ADMF-PC's architecture standards and provides the foundation for sophisticated portfolio management while maintaining clean separation from risk validation logic.