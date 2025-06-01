# Step 2: Add Risk Container

**Status**: Foundation Step
**Complexity**: Low-Medium
**Prerequisites**: [Step 1: Core Pipeline Test](step-01-core-pipeline.md) completed
**Architecture Ref**: [CONTAINER-HIERARCHY.md](../../architecture/02-CONTAINER-HIERARCHY.md)

## 🎯 Objective

Add a proper Risk Container that:
- Encapsulates risk management logic
- Maintains isolated state
- Transforms signals into risk-adjusted orders
- Tracks portfolio exposure
- Enforces position limits

## 📋 Required Reading

Before starting:
1. [BACKTEST_README.md](../../BACKTEST_README.md#risk-container)
2. [Container Hierarchy](../../architecture/02-CONTAINER-HIERARCHY.md)
3. [Risk Module Documentation](../../risk/README.md)

## 🏗️ Implementation Tasks

### 1. Create Risk Container

```python
# src/risk/risk_container.py
class RiskContainer:
    """
    Encapsulates all risk management components.
    Maintains portfolio state and enforces risk limits.
    """
    
    def __init__(self, container_id: str, config: RiskConfig):
        self.container_id = container_id
        self.config = config
        
        # Create isolated event bus
        self.isolation_manager = get_isolation_manager()
        self.event_bus = self.isolation_manager.create_isolated_bus(
            f"{container_id}_risk"
        )
        
        # Initialize components
        self.portfolio_state = PortfolioState(container_id)
        self.position_sizer = PositionSizer(config.sizing_method)
        self.risk_limits = RiskLimits(config.limits)
        self.order_manager = OrderManager(container_id)
        
        # Setup logging
        self.logger = ComponentLogger("RiskContainer", container_id)
        
        # Wire internal events
        self._setup_internal_events()
    
    def on_signal(self, signal: TradingSignal) -> None:
        """Process trading signal through risk pipeline"""
        self.logger.log_event_flow(
            "SIGNAL_RECEIVED", "strategy", "risk", 
            f"Signal: {signal.direction} {signal.symbol}"
        )
        
        # Check risk limits
        if not self.risk_limits.can_trade(self.portfolio_state, signal):
            self.logger.warning(f"Signal rejected by risk limits: {signal}")
            return
        
        # Calculate position size
        size = self.position_sizer.calculate_size(
            signal, self.portfolio_state
        )
        
        if size > 0:
            # Create order
            order = self.order_manager.create_order(
                signal, size, self.portfolio_state.current_prices
            )
            
            # Update portfolio state optimistically
            self.portfolio_state.add_pending_order(order)
            
            # Emit order event
            self.event_bus.publish("ORDER", order)
```

### 2. Portfolio State Management

```python
# src/risk/portfolio_state.py
class PortfolioState:
    """Tracks current portfolio positions and P&L"""
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.cash = 100000.0  # Starting capital
        self.total_value = self.cash
        self.logger = ComponentLogger("PortfolioState", container_id)
    
    def update_position(self, fill: Fill) -> None:
        """Update position based on fill"""
        symbol = fill.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        position = self.positions[symbol]
        position.update(fill)
        
        # Update cash
        self.cash -= fill.quantity * fill.price * fill.direction.value
        
        # Log state change
        self.logger.info(
            f"Position updated: {symbol} qty={position.quantity} "
            f"avg_price={position.avg_price:.2f}"
        )
    
    def calculate_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            pos.quantity * current_prices.get(symbol, pos.avg_price)
            for symbol, pos in self.positions.items()
        )
        self.total_value = self.cash + position_value
        return self.total_value
```

### 3. Risk Limits Implementation

```python
# src/risk/risk_limits.py
class RiskLimits:
    """Enforces various risk constraints"""
    
    def __init__(self, config: RiskLimitConfig):
        self.max_position_size = config.max_position_size
        self.max_portfolio_risk = config.max_portfolio_risk
        self.max_correlation = config.max_correlation
        self.max_drawdown = config.max_drawdown
        
    def can_trade(self, portfolio: PortfolioState, signal: TradingSignal) -> bool:
        """Check if trade is allowed under current risk limits"""
        
        # Check position limit
        if not self._check_position_limit(portfolio, signal):
            return False
        
        # Check portfolio risk
        if not self._check_portfolio_risk(portfolio, signal):
            return False
        
        # Check drawdown limit
        if not self._check_drawdown_limit(portfolio):
            return False
        
        return True
    
    def _check_position_limit(self, portfolio: PortfolioState, 
                            signal: TradingSignal) -> bool:
        """Ensure position size is within limits"""
        current_position = portfolio.positions.get(signal.symbol)
        if current_position:
            position_value = abs(current_position.quantity * 
                               portfolio.current_prices[signal.symbol])
            position_pct = position_value / portfolio.total_value
            return position_pct < self.max_position_size
        return True
```

### 4. Container Integration

```python
# src/containers/risk_container_factory.py
def create_risk_container(container_id: str, config: Dict) -> RiskContainer:
    """Factory function to create configured risk container"""
    
    risk_config = RiskConfig(
        sizing_method=config.get('sizing_method', 'fixed'),
        limits=RiskLimitConfig(
            max_position_size=config.get('max_position_size', 0.1),
            max_portfolio_risk=config.get('max_portfolio_risk', 0.02),
            max_correlation=config.get('max_correlation', 0.7),
            max_drawdown=config.get('max_drawdown', 0.2)
        )
    )
    
    container = RiskContainer(container_id, risk_config)
    
    # Register with lifecycle manager
    lifecycle_manager = get_lifecycle_manager()
    lifecycle_manager.register_container(container_id, container)
    
    return container
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step2_risk_components.py`:

```python
class TestPortfolioState:
    """Test portfolio state management"""
    
    def test_position_update(self):
        """Test position tracking"""
        portfolio = PortfolioState("test")
        
        # Add buy fill
        fill = Fill(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            direction=Direction.BUY
        )
        portfolio.update_position(fill)
        
        # Verify position
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100
        assert portfolio.cash == 85000.0  # 100000 - (100 * 150)

class TestRiskLimits:
    """Test risk limit enforcement"""
    
    def test_position_size_limit(self):
        """Test max position size enforcement"""
        config = RiskLimitConfig(max_position_size=0.1)
        limits = RiskLimits(config)
        
        # Create portfolio with large position
        portfolio = create_test_portfolio_with_position(
            symbol="AAPL", 
            value=15000,  # 15% of 100k portfolio
            total_value=100000
        )
        
        # New signal should be rejected
        signal = TradingSignal("AAPL", Direction.BUY)
        assert not limits.can_trade(portfolio, signal)
```

### Integration Tests

Create `tests/integration/test_step2_risk_container.py`:

```python
def test_risk_container_signal_processing():
    """Test complete signal processing through risk container"""
    # Setup
    container_id = "test_risk_integration"
    config = {
        'max_position_size': 0.1,
        'sizing_method': 'fixed'
    }
    
    risk_container = create_risk_container(container_id, config)
    
    # Capture emitted orders
    orders = []
    risk_container.event_bus.subscribe(
        "ORDER", lambda order: orders.append(order)
    )
    
    # Send test signal
    signal = TradingSignal(
        symbol="AAPL",
        direction=Direction.BUY,
        strength=0.8
    )
    risk_container.on_signal(signal)
    
    # Verify order was created
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].quantity > 0

def test_risk_container_isolation():
    """Test container event isolation"""
    # Create two risk containers
    container1 = create_risk_container("risk1", {})
    container2 = create_risk_container("risk2", {})
    
    # Subscribe to container1 events
    container1_orders = []
    container1.event_bus.subscribe(
        "ORDER", lambda o: container1_orders.append(o)
    )
    
    # Send signal to container2
    signal = TradingSignal("AAPL", Direction.BUY)
    container2.on_signal(signal)
    
    # Verify no leakage
    assert len(container1_orders) == 0
```

### System Tests

Create `tests/system/test_step2_risk_integration.py`:

```python
def test_strategy_to_risk_to_execution():
    """Test full pipeline with risk container"""
    # Setup complete system
    system = create_test_system_with_risk()
    
    # Run with synthetic data
    data = SyntheticDataGenerator.create_volatile_market()
    results = system.run_backtest(data)
    
    # Verify risk limits were applied
    assert results['max_position_size'] <= 0.1
    assert results['max_drawdown'] <= 0.2
    assert all(trade['risk_adjusted'] for trade in results['trades'])
```

## ✅ Validation Checklist

### Container Validation
- [ ] Risk container properly encapsulated
- [ ] Event bus isolation verified
- [ ] Lifecycle management working
- [ ] State properly maintained

### Risk Management Validation
- [ ] Position sizing calculates correctly
- [ ] Risk limits enforced properly
- [ ] Portfolio state tracks accurately
- [ ] Orders include risk adjustments

### Testing Validation
- [ ] Unit tests: Portfolio state management
- [ ] Unit tests: Risk limit calculations
- [ ] Integration tests: Signal to order flow
- [ ] System tests: End-to-end with risk
- [ ] Test coverage > 90%

### Performance Validation
- [ ] Risk calculations < 1ms per signal
- [ ] Memory usage stable
- [ ] No calculation bottlenecks

## 🐛 Common Issues

1. **Portfolio State Synchronization**
   - Ensure fills update state atomically
   - Handle partial fills correctly
   - Reconcile pending vs executed orders

2. **Risk Limit Edge Cases**
   - Test with zero positions
   - Test at exactly the limit
   - Test with multiple correlated positions

3. **Event Flow Issues**
   - Ensure order events reach execution
   - Verify fill events update portfolio
   - Check for event loops

## 📊 Memory & Performance

### Memory Monitoring
```python
@memory_profile
def process_signals(risk_container, signals):
    """Monitor memory usage during signal processing"""
    for signal in signals:
        risk_container.on_signal(signal)
```

### Performance Benchmarks
- Process 1000 signals/second
- Risk calculations < 1ms per signal  
- State updates < 0.1ms per fill
- Total memory < 50MB for 10k positions

## 🎯 Success Criteria

Step 2 is complete when:
1. ✅ Risk container encapsulates all risk logic
2. ✅ Portfolio state accurately tracked
3. ✅ Risk limits properly enforced
4. ✅ All test tiers pass
5. ✅ Performance requirements met

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 2.5: Walk-Forward Foundation](step-02.5-walk-forward.md)

## 📚 Additional Resources

- [Risk Module Deep Dive](../../risk/README.md)
- [Container Patterns](../../core/containers/patterns.md)
- [Event Isolation Guide](../validation-framework/event-bus-isolation.md)