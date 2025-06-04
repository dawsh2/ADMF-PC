# Risk Management Module

This module provides THE canonical risk management implementations for ADMF-PC.

## Architecture Reference
- **System Architecture**: docs/SYSTEM_ARCHITECTURE_v5.MD#risk-module  
- **Style Guide**: STYLE.md - Canonical risk implementations
- **Core Patterns**: docs/new/arch-101.md - Protocol + Composition

## Module Overview

The Risk module implements comprehensive risk management using Protocol + Composition patterns with **no inheritance**. All components are designed as composable, protocol-compliant building blocks that can be mixed and matched through configuration.

## Core Principle: Mixed Stateful/Stateless Design

This module contains both **stateful** and **stateless** risk components, each serving different purposes:

### Stateful Components
Components that legitimately need to maintain state across signals/time:
- **Portfolio State Tracking**: Current positions, cash, P&L, value history
- **Violation Tracking**: Daily loss limits, circuit breakers, breach monitoring
- **Rate Limiting**: Order timing, cooldown periods, frequency constraints
- **Adaptive Models**: Learning risk models that evolve with market conditions
- **Correlation Tracking**: Relationship matrices that build over time

### Stateless Components  
Components that perform pure calculations based on inputs:
- **Position Sizing**: Calculate position sizes based on current portfolio state
- **Basic Limits**: Check constraints like max position, max exposure
- **Risk Metrics**: VaR calculations, concentration analysis

## Files

### Core Implementations
- **`portfolio_state.py`** - THE portfolio state tracking implementation
  - `PortfolioState`: Manages positions, cash, P&L, performance metrics
  - Stateful component: Maintains positions, value history, returns history
  - Provides risk metrics: drawdown, Sharpe ratio, VaR, leverage

- **`position_sizing.py`** - THE position sizing strategies
  - `FixedPositionSizer`: Fixed position sizes
  - `PercentagePositionSizer`: Portfolio percentage-based sizing
  - `KellyCriterionSizer`: Kelly criterion optimal sizing
  - `VolatilityBasedSizer`: Risk parity approach
  - `ATRBasedSizer`: ATR-based position sizing
  - Stateless components: Calculate based on signal + portfolio state

- **`limits.py`** - THE risk limit implementations
  - `MaxPositionLimit`: Position size constraints
  - `MaxDrawdownLimit`: Drawdown protection
  - `VaRLimit`: Value at Risk constraints
  - `MaxExposureLimit`: Total exposure limits
  - `ConcentrationLimit`: Portfolio concentration limits
  - `LeverageLimit`: Leverage constraints
  - `DailyLossLimit`: Daily loss tracking (stateful)
  - `SymbolRestrictionLimit`: Trading symbol restrictions
  - Mixed stateful/stateless: Basic checks are stateless, violation tracking is stateful

### Protocol Definitions
- **`protocols.py`** - THE risk management protocols
  - `RiskPortfolioProtocol`: Combined risk + portfolio management interface
  - `PositionSizerProtocol`: Position sizing calculation interface
  - `RiskLimitProtocol`: Risk constraint checking interface
  - `PortfolioStateProtocol`: Portfolio state management interface
  - `SignalProcessorProtocol`: Signal-to-order conversion interface

### Data Models
- **`models.py`** - Risk-specific data structures
  - `TradingSignal`: Signal from strategy to risk system
  - `Order`: Risk-adjusted order for execution
  - `Fill`: Execution result for portfolio updates
  - `Position`: Position tracking data structure
  - `RiskConfig`: Risk management configuration

## Usage Examples

### Basic Risk Management
```python
from src.risk import (
    PortfolioState, 
    PercentagePositionSizer, 
    MaxPositionLimit,
    MaxDrawdownLimit
)

# Initialize components
portfolio = PortfolioState(initial_capital=Decimal("100000"))
position_sizer = PercentagePositionSizer(percentage=Decimal("0.02"))
risk_limits = [
    MaxPositionLimit(max_position_percent=Decimal("0.10")),
    MaxDrawdownLimit(max_drawdown=Decimal("0.20"))
]

# Process trading signal
def process_signal(signal, market_data):
    # Calculate position size
    size = position_sizer.calculate_size(signal, portfolio, market_data)
    
    # Create proposed order
    order = Order(
        symbol=signal.symbol,
        side=signal.side,
        quantity=size,
        order_type=OrderType.MARKET
    )
    
    # Check risk limits
    for limit in risk_limits:
        approved, reason = limit.check_limit(order, portfolio, market_data)
        if not approved:
            return None  # Risk rejected
    
    return order
```

### Advanced Risk Configuration
```python
from src.risk import RiskConfig

# Configuration-driven risk management
config = RiskConfig(
    initial_capital=100000.0,
    sizing_method="volatility",
    max_position_size=0.10,
    max_portfolio_risk=0.02,
    max_correlation=0.7,
    max_drawdown=0.15,
    max_leverage=1.0,
    max_concentration=0.20,
    default_stop_loss_pct=0.05
)

# Components configured from settings
portfolio = PortfolioState(
    initial_capital=Decimal(str(config.initial_capital))
)

if config.sizing_method == "volatility":
    sizer = VolatilityBasedSizer(
        target_volatility=Decimal(str(config.max_portfolio_risk))
    )
elif config.sizing_method == "fixed":
    sizer = FixedPositionSizer(
        size=Decimal(str(config.fixed_position_size))
    )
```

### Container Integration
```python
# Risk management in container context
from src.core.containers import Container
from src.risk.protocols import RiskPortfolioProtocol

class RiskContainer(Container):
    """Risk management container with isolated event bus"""
    
    def __init__(self, config: RiskConfig):
        super().__init__()
        
        # Initialize risk components
        self.portfolio = PortfolioState(
            initial_capital=Decimal(str(config.initial_capital))
        )
        
        self.position_sizer = self._create_position_sizer(config)
        self.risk_limits = self._create_risk_limits(config)
        
        # Register event handlers
        self.event_bus.subscribe("SIGNAL", self._handle_signal)
        self.event_bus.subscribe("FILL", self._handle_fill)
    
    def _handle_signal(self, signal_event):
        """Process trading signal through risk management"""
        signal = signal_event.data
        market_data = signal_event.metadata.get("market_data", {})
        
        # Risk processing
        order = self.process_signal(signal, market_data)
        
        if order:
            # Emit order event
            self.event_bus.emit("ORDER", order)
        else:
            # Log risk rejection
            self.logger.info(f"Signal rejected by risk management: {signal.symbol}")
```

## Protocol Compliance

All risk components implement standard protocols:

```python
# Position sizers implement PositionSizerProtocol
def calculate_size(
    self, 
    signal: Signal, 
    portfolio_state: PortfolioStateProtocol,
    market_data: Dict[str, Any]
) -> Decimal:
    """Calculate position size based on signal and current state"""

# Risk limits implement RiskLimitProtocol  
def check_limit(
    self,
    order: Order,
    portfolio_state: PortfolioStateProtocol,
    market_data: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """Check if order violates risk constraints"""

# Portfolio state implements PortfolioStateProtocol
def get_risk_metrics(self) -> RiskMetrics:
    """Get current portfolio risk metrics"""
```

## Configuration Patterns

### YAML Configuration
```yaml
risk_management:
  portfolio:
    initial_capital: 100000
    base_currency: "USD"
  
  position_sizing:
    method: "percentage"
    percentage: 0.02
    max_position: 0.10
  
  limits:
    - type: "max_position"
      max_position_percent: 0.10
    - type: "max_drawdown" 
      max_drawdown: 0.20
    - type: "daily_loss"
      max_daily_loss: 0.05
  
  features:
    - portfolio_tracking
    - violation_monitoring
    - adaptive_sizing
```

### Factory Pattern
```python
def create_risk_manager(config: RiskConfig) -> RiskPortfolioProtocol:
    """Factory function for risk management components"""
    
    # Create portfolio state
    portfolio = PortfolioState(
        initial_capital=Decimal(str(config.initial_capital))
    )
    
    # Create position sizer based on config
    if config.sizing_method == "fixed":
        sizer = FixedPositionSizer(Decimal(str(config.fixed_position_size)))
    elif config.sizing_method == "percentage":
        sizer = PercentagePositionSizer(Decimal(str(config.max_position_size)))
    elif config.sizing_method == "volatility":
        sizer = VolatilityBasedSizer(Decimal(str(config.max_portfolio_risk)))
    
    # Create risk limits
    limits = []
    if config.max_position_size > 0:
        limits.append(MaxPositionLimit(
            max_position_percent=Decimal(str(config.max_position_size))
        ))
    if config.max_drawdown > 0:
        limits.append(MaxDrawdownLimit(
            max_drawdown=Decimal(str(config.max_drawdown))
        ))
    
    return RiskManager(portfolio, sizer, limits)
```

## Testing Strategy

### Unit Testing
```python
def test_percentage_position_sizer():
    """Test position sizing calculation"""
    sizer = PercentagePositionSizer(percentage=Decimal("0.02"))
    
    # Create test signal and portfolio state
    signal = create_test_signal(symbol="SPY", strength=0.8)
    portfolio = PortfolioState(initial_capital=Decimal("100000"))
    market_data = {"prices": {"SPY": 400}}
    
    # Test calculation
    size = sizer.calculate_size(signal, portfolio, market_data)
    expected_size = Decimal("40")  # (100000 * 0.02 * 0.8) / 400
    
    assert size == expected_size

def test_risk_limit_violation():
    """Test risk limit enforcement"""
    limit = MaxPositionLimit(max_position_percent=Decimal("0.10"))
    
    # Create test order that exceeds limit
    order = create_test_order(symbol="SPY", quantity=Decimal("500"))
    portfolio = PortfolioState(initial_capital=Decimal("100000"))
    market_data = {"prices": {"SPY": 400}}
    
    # Test limit check
    passes, reason = limit.check_limit(order, portfolio, market_data)
    
    assert not passes
    assert "exceeds limit" in reason
```

### Integration Testing
```python
def test_full_risk_pipeline():
    """Test complete signal-to-order pipeline"""
    # Setup risk management components
    portfolio = PortfolioState(initial_capital=Decimal("100000"))
    sizer = PercentagePositionSizer(percentage=Decimal("0.02"))
    limits = [MaxPositionLimit(max_position_percent=Decimal("0.10"))]
    
    # Process signal through complete pipeline
    signal = create_test_signal(symbol="SPY", strength=0.8)
    market_data = {"prices": {"SPY": 400}}
    
    # Calculate size
    size = sizer.calculate_size(signal, portfolio, market_data)
    
    # Create order
    order = Order(
        order_id="test_001",
        symbol=signal.symbol,
        side=signal.side,
        quantity=size,
        order_type=OrderType.MARKET
    )
    
    # Check limits
    for limit in limits:
        passes, reason = limit.check_limit(order, portfolio, market_data)
        assert passes, f"Risk limit failed: {reason}"
    
    # Simulate fill and portfolio update
    fill = Fill(
        fill_id="fill_001",
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=Decimal("400"),
        timestamp=datetime.now()
    )
    
    # Update portfolio
    initial_cash = portfolio.get_cash_balance()
    portfolio.update_position(
        symbol=fill.symbol,
        quantity_delta=fill.quantity if fill.side == OrderSide.BUY else -fill.quantity,
        price=fill.price,
        timestamp=fill.timestamp
    )
    
    # Verify portfolio state
    assert portfolio.get_cash_balance() < initial_cash
    position = portfolio.get_position("SPY")
    assert position is not None
    assert position.quantity == fill.quantity
```

## Performance Considerations

- **Memory Usage**: Portfolio state maintains limited history (252 days by default)
- **Calculation Efficiency**: Position sizers are lightweight calculation functions
- **State Management**: Portfolio state updates are atomic and consistent
- **Risk Limit Performance**: Limits check current state without expensive computations

## No "Enhanced" Versions

Do not create `enhanced_risk_manager.py`, `improved_portfolio_state.py`, etc. Use composition and configuration to add capabilities to the canonical implementations in this module.

## Integration Points

- **Data Module**: Receives market data for risk calculations
- **Strategy Module**: Processes trading signals from strategies  
- **Execution Module**: Sends approved orders to execution engines
- **Core Events**: Integrates with event bus for container communication

---

## Addendum: Stateless Risk Architecture Proposal

### Alternative Architecture: Stateless Risk + Separate Portfolio Module

During the architectural review, we explored an alternative approach where **Risk becomes purely stateless** and **Portfolio state is separated into its own module**. This section documents that proposal for future consideration.

#### Proposed Module Separation

**Option: Split into separate modules**
```
src/portfolio/          # Stateful portfolio tracking
├── __init__.py
├── protocols.py        # PortfolioStateProtocol, Position protocols  
├── state.py           # PortfolioState implementation
├── models.py          # Position, Fill data models
└── README.md

src/risk/              # Stateless risk calculations
├── __init__.py  
├── protocols.py       # RiskLimitProtocol, PositionSizerProtocol
├── position_sizing.py # Stateless sizing strategies
├── limits.py          # Stateless risk limits
└── README.md
```

#### Stateless Risk Data Flow

In this alternative architecture, the data flow becomes:
```
BAR → SIGNAL → Execution.executeOrder(Risk.check(Portfolio.getState()))
```

Where:
- **Risk components** are pure functions that receive portfolio state as input
- **Portfolio module** owns all state management (positions, cash, P&L)  
- **Execution module** orchestrates between stateless Risk and stateful Portfolio

#### Benefits of Stateless Approach

1. **Pure Risk Functions**: All risk calculations become testable pure functions
2. **Clear Separation**: Portfolio = state management, Risk = constraint enforcement
3. **Parallel Execution**: Multiple risk calculations can run simultaneously
4. **Easier Testing**: No mocking required for risk function tests
5. **Reusable Calculations**: Same risk functions work with any portfolio state

#### Implementation Example

```python
# Stateless risk functions
class PositionSizer:
    def calculate_size(self, signal: Signal, portfolio_state: PortfolioSnapshot) -> Decimal:
        """Pure function - no state modification"""
        
class RiskLimits:
    def check_limits(self, order: Order, portfolio_state: PortfolioSnapshot) -> bool:
        """Pure function - no state modification"""

# Execution orchestrates everything
class ExecutionEngine:
    def process_signal(self, signal: Signal):
        # Get immutable state snapshot
        state = self.portfolio.get_current_state()
        
        # Pure risk calculations
        size = self.risk.calculate_size(signal, state)
        approved = self.risk.check_limits(order, state)
        
        # Execute and update state
        if approved:
            order = self.create_order(signal, size)
            fill = self.broker.execute(order)
            self.portfolio.update_state(fill)  # Only here is state modified
```

#### When to Consider Stateless Risk

The stateless approach makes sense when:
- Risk components primarily do **calculations** rather than **tracking**
- You want **maximum testability** and **parallel execution**
- Portfolio state can be **cleanly separated** from risk logic
- You're building a **research-focused** system with lots of parameter variations

#### Why We Chose Stateful Risk (For Now)

We decided to keep the current **stateful risk architecture** because:

1. **Real-world risk needs state**: Daily loss limits, rate limiting, violation tracking, adaptive models
2. **Simpler integration**: Components can maintain their own necessary state
3. **Clearer responsibility**: Risk components own their risk-specific state  
4. **Proven pattern**: Current architecture works well for production trading systems
5. **Future flexibility**: Can always refactor to stateless if research needs change

#### Migration Path

If future requirements favor the stateless approach:

1. **Phase 1**: Extract portfolio state into separate module
2. **Phase 2**: Convert basic risk limits to stateless functions
3. **Phase 3**: Keep stateful components (rate limiting, adaptive models) as special cases
4. **Phase 4**: Execution module becomes the orchestrator between Portfolio and Risk

This addendum preserves the stateless risk proposal for future architectural decisions while documenting why we chose the current stateful approach.