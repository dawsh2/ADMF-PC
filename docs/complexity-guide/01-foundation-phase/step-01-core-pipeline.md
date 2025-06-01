# Step 1: Core Pipeline Test

**Status**: Foundation Step
**Complexity**: Low
**Prerequisites**: [Pre-Implementation Setup](../00-pre-implementation/README.md) completed
**Architecture Ref**: [EVENT-DRIVEN-ARCHITECTURE.md](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)

## 🎯 Objective

Build and validate the fundamental event-driven pipeline:
- Data source emits market data events
- Indicator consumes data and calculates values
- Strategy generates signals based on indicators
- Risk manager transforms signals into orders
- Execution engine processes orders
- **Execution engine returns fills to complete the cycle**
- **Risk manager updates portfolio state based on fills**

## 📋 Required Reading

Before starting:
1. [BACKTEST_README.md](../../BACKTEST_README.md#event-flow)
2. [Event Bus Isolation](../validation-framework/event-bus-isolation.md)
3. [Three-Tier Testing Strategy](../testing-framework/three-tier-strategy.md)

## 🏗️ Implementation Tasks

### 1. Create Core Components

#### 1.1 Simple Moving Average Indicator
```python
# src/strategy/indicators.py
class SimpleMovingAverage:
    """Basic SMA indicator for testing"""
    
    def __init__(self, period: int, container_id: str):
        self.period = period
        self.container_id = container_id
        self.values = deque(maxlen=period)
        self.logger = ComponentLogger("SMA", container_id)
        
    def on_bar(self, bar: Bar) -> None:
        """Process new market data"""
        self.values.append(bar.close)
        if len(self.values) == self.period:
            self.current_value = sum(self.values) / self.period
            self.logger.log_event_flow(
                "SMA_CALCULATED", "indicator", "strategy", 
                f"SMA={self.current_value:.2f}"
            )
```

#### 1.2 Basic Trend Strategy
```python
# src/strategy/strategies/trend_following.py
class SimpleTrendStrategy:
    """SMA crossover strategy for testing"""
    
    def __init__(self, fast_period: int, slow_period: int, container_id: str):
        self.fast_sma = SimpleMovingAverage(fast_period, container_id)
        self.slow_sma = SimpleMovingAverage(slow_period, container_id)
        self.position = 0
        self.event_bus = None  # Injected by container
        
    def on_bar(self, bar: Bar) -> None:
        """Process market data and generate signals"""
        self.fast_sma.on_bar(bar)
        self.slow_sma.on_bar(bar)
        
        if self._should_generate_signal():
            signal = self._create_signal(bar)
            self.event_bus.publish("SIGNAL", signal)
```

### 2. Event Flow Setup

```python
# src/core/events/event_flow.py
def setup_core_pipeline(container_id: str) -> Dict[str, Any]:
    """Wire up the core event pipeline with complete cycle"""
    
    # Create isolated event bus
    isolation_manager = get_isolation_manager()
    event_bus = isolation_manager.create_isolated_bus(container_id)
    
    # Create components
    data_source = DataSource(container_id)
    indicator = SimpleMovingAverage(20, container_id)
    strategy = SimpleTrendStrategy(10, 20, container_id)
    risk_manager = RiskManager(container_id)
    execution = ExecutionEngine(container_id)
    portfolio_state = PortfolioState(container_id)
    
    # Wire up event flow (including feedback loop)
    event_bus.subscribe("BAR", indicator.on_bar)
    event_bus.subscribe("BAR", strategy.on_bar)
    event_bus.subscribe("SIGNAL", risk_manager.on_signal)
    event_bus.subscribe("ORDER", execution.on_order)
    event_bus.subscribe("FILL", risk_manager.on_fill)  # Critical feedback
    event_bus.subscribe("PORTFOLIO_UPDATE", portfolio_state.update)
    
    # Inject event bus into components
    strategy.event_bus = event_bus
    risk_manager.event_bus = event_bus
    execution.event_bus = event_bus  # Execution needs to emit fills
    
    return {
        'event_bus': event_bus,
        'components': {
            'data_source': data_source,
            'indicator': indicator,
            'strategy': strategy,
            'risk_manager': risk_manager,
            'execution': execution,
            'portfolio_state': portfolio_state
        }
    }
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step1_components.py`:

```python
class TestSimpleMovingAverage:
    """Unit tests for SMA indicator"""
    
    def test_sma_calculation(self):
        """Test SMA calculates correct values"""
        sma = SimpleMovingAverage(period=3, container_id="test")
        
        # Add values: [10, 20, 30]
        bars = [
            Bar(close=10), Bar(close=20), Bar(close=30)
        ]
        
        for bar in bars:
            sma.on_bar(bar)
        
        # Expected: (10 + 20 + 30) / 3 = 20
        assert sma.current_value == 20.0

class TestSimpleTrendStrategy:
    """Unit tests for trend strategy"""
    
    def test_signal_generation(self):
        """Test strategy generates signals correctly"""
        strategy = SimpleTrendStrategy(2, 4, "test")
        strategy.event_bus = Mock()
        
        # Create upward trend crossing
        bars = create_trend_crossing_data()
        
        for bar in bars:
            strategy.on_bar(bar)
        
        # Verify signal was generated
        strategy.event_bus.publish.assert_called_with("SIGNAL", ANY)
```

### Integration Tests

Create `tests/integration/test_step1_event_flow.py`:

```python
def test_signal_to_order_flow():
    """Test complete event flow from data to execution"""
    # Setup
    container_id = "test_integration"
    pipeline = setup_core_pipeline(container_id)
    
    # Inject test data
    test_bars = SyntheticDataGenerator.create_sma_crossover_scenario()
    
    # Run pipeline
    for bar in test_bars:
        pipeline['components']['data_source'].emit_bar(bar)
    
    # Verify execution received orders
    execution = pipeline['components']['execution']
    assert len(execution.processed_orders) == 2  # Buy and sell
    
    # Verify event isolation
    isolation_manager = get_isolation_manager()
    assert isolation_manager.check_isolation(container_id)
```

### System Tests

Create `tests/system/test_step1_full_backtest.py`:

```python
def test_basic_backtest_with_known_results():
    """Test full backtest produces expected results"""
    # Setup
    config = {
        'strategy': 'simple_trend',
        'fast_period': 10,
        'slow_period': 20,
        'risk_limit': 0.02
    }
    
    # Use deterministic data
    data = SyntheticDataGenerator.create_simple_trend()
    
    # Pre-computed expected results
    expected = {
        'total_trades': 4,
        'winning_trades': 3,
        'losing_trades': 1,
        'final_value': 11234.56,
        'max_drawdown': 0.05
    }
    
    # Run backtest
    results = run_simple_backtest(config, data)
    
    # Verify exact match
    assert results['total_trades'] == expected['total_trades']
    assert abs(results['final_value'] - expected['final_value']) < 0.01
```

## ✅ Validation Checklist

Before proceeding to Step 2:

### Component Validation
- [ ] SMA indicator calculates correctly
- [ ] Strategy generates signals at right times
- [ ] Risk manager applies position limits
- [ ] Execution engine processes orders

### Event Flow Validation
- [ ] Events flow in correct order
- [ ] Fill events properly emitted by execution
- [ ] Portfolio state updated from fills
- [ ] Complete cycle validated (BAR → SIGNAL → ORDER → FILL → UPDATE)
- [ ] No event leakage between containers
- [ ] All events logged properly
- [ ] Event bus isolation verified

### Testing Validation
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All system tests pass
- [ ] Code coverage > 90%
- [ ] Performance < 100ms per backtest day

### Memory Validation
- [ ] No memory leaks detected
- [ ] Memory usage < 100MB for test data
- [ ] Proper cleanup on completion

## 📊 Performance Requirements

- Process 1 year of daily data in < 1 second
- Process 1 year of minute data in < 10 seconds
- Memory usage scales linearly with data size
- CPU usage remains constant regardless of data size

## 🐛 Common Issues

1. **Event Bus Not Injected**
   - Ensure strategy and risk manager have event_bus set
   - Check container wiring in setup

2. **SMA Not Ready**
   - SMA needs `period` bars before calculating
   - Handle warm-up period correctly

3. **Event Isolation Failures**
   - Each test must use unique container_id
   - Always cleanup in teardown

## 📝 Documentation Requirements

### File Headers
Every file must have:
```python
"""
File: src/strategy/indicators.py
Status: ACTIVE
Architecture Ref: BACKTEST_README.md#indicators
Step: 1 - Core Pipeline Test
Dependencies: core.events, data.models
"""
```

### Logging Standards
- Log every event publication
- Log every signal generation
- Log every order creation
- Include container_id in all logs

## 🎯 Success Criteria

Step 1 is complete when:
1. ✅ Basic pipeline processes data end-to-end
2. ✅ All three test tiers pass
3. ✅ Event isolation validated
4. ✅ Performance requirements met
5. ✅ Documentation complete

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 2: Add Risk Container](step-02-risk-container.md)

## 📚 Additional Resources

- [Event Bus Documentation](../../core/events/README.md)
- [Testing Best Practices](../testing-framework/best-practices.md)
- [Performance Optimization Guide](../optimization/performance-guide.md)