# Three-Tier Testing Strategy

## Overview

The three-tier testing strategy ensures comprehensive validation at every level of the system, from individual components to full end-to-end workflows. This approach creates a safety net that catches issues early and ensures system reliability.

## The Three Tiers

### Tier 1: Unit Tests - Component-Level Correctness

**Purpose**: Validate individual components work correctly in isolation

**Characteristics**:
- Test single classes/functions
- Mock all dependencies
- Execute in milliseconds
- Focus on business logic
- No external resources (files, network, database)

**Example**:
```python
def test_sma_indicator_calculation():
    """Test SMA calculation with known values"""
    sma = SimpleMovingAverage(period=3)
    
    # Add values: [10, 20, 30]
    sma.add_value(10)
    sma.add_value(20)
    sma.add_value(30)
    
    # Expected: (10 + 20 + 30) / 3 = 20
    assert sma.value == 20.0
    
    # Add another: [20, 30, 40]
    sma.add_value(40)
    
    # Expected: (20 + 30 + 40) / 3 = 30
    assert sma.value == 30.0
```

### Tier 2: Integration Tests - Container Interaction Validation

**Purpose**: Validate components work together correctly

**Characteristics**:
- Test interaction between components
- Use real implementations (not mocks)
- Test event flows
- Validate container isolation
- Use synthetic data

**Example**:
```python
def test_signal_to_order_flow():
    """Test signal flows correctly from strategy to risk manager"""
    # Setup containers
    container = TestContainer("integration_test")
    strategy = TrendStrategy()
    risk_manager = RiskManager()
    
    # Wire up event flow
    container.event_bus.subscribe("SIGNAL", risk_manager.on_signal)
    
    # Inject test data
    test_bar = Bar(symbol="SPY", close=100.0)
    strategy.on_bar(test_bar)
    
    # Verify signal was processed
    assert risk_manager.last_signal is not None
    assert risk_manager.last_order.symbol == "SPY"
```

### Tier 3: System Tests - End-to-End Workflow Validation

**Purpose**: Validate complete workflows produce expected results

**Characteristics**:
- Test full system behavior
- Use deterministic inputs
- Validate exact outputs
- Test optimization reproducibility
- Ensure production readiness

**Example**:
```python
def test_complete_backtest_workflow():
    """Test full backtest produces expected results"""
    # Setup
    config = load_config("test_configs/simple_backtest.yaml")
    synthetic_data = SyntheticDataGenerator.create_known_pattern()
    
    # Pre-computed expected results
    expected = {
        'total_trades': 10,
        'final_value': 11234.56,
        'sharpe_ratio': 1.45,
        'max_drawdown': 0.05
    }
    
    # Run full backtest
    results = run_backtest(config, synthetic_data)
    
    # Validate exact match
    assert results['total_trades'] == expected['total_trades']
    assert abs(results['final_value'] - expected['final_value']) < 0.01
    assert abs(results['sharpe_ratio'] - expected['sharpe_ratio']) < 0.001
```

## Test Development Process

### 1. Specification First

Before writing any code, specify what should be tested:

```python
# test_specifications.py
"""
Step 1 Testing Specifications

Unit Tests:
- SMA indicator calculates correct values
- Strategy generates signals at right times
- Risk manager applies position limits

Integration Tests:
- Events flow from data → indicator → strategy → risk
- Container isolation is maintained
- Multiple strategies coordinate properly

System Tests:
- Basic backtest completes successfully
- Results match hand-calculated expectations
- Performance meets requirements
"""
```

### 2. Parallel Development

Tests and implementation are developed together:

```bash
# Terminal 1: Write tests
vim tests/unit/test_sma_indicator.py

# Terminal 2: Write implementation  
vim src/indicators/sma.py

# Terminal 3: Run tests continuously
pytest-watch tests/unit/test_sma_indicator.py
```

### 3. Test-Driven Development Cycle

```python
# 1. Write failing test
def test_sma_handles_empty_data():
    sma = SimpleMovingAverage(period=5)
    assert sma.value is None  # No data yet

# 2. Implement minimal code to pass
class SimpleMovingAverage:
    def __init__(self, period):
        self.period = period
        self._value = None
    
    @property
    def value(self):
        return self._value

# 3. Refactor and enhance
# 4. Repeat
```

## Testing Patterns

### Unit Test Pattern
```python
class TestComponent:
    """Unit tests for Component"""
    
    def setup_method(self):
        """Setup before each test"""
        self.component = Component()
        self.mock_dependency = Mock()
    
    def test_normal_operation(self):
        """Test normal behavior"""
        result = self.component.process(valid_input)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test boundary conditions"""
        result = self.component.process(edge_input)
        assert result == edge_output
    
    def test_error_handling(self):
        """Test error conditions"""
        with pytest.raises(ValueError):
            self.component.process(invalid_input)
```

### Integration Test Pattern
```python
class TestIntegration:
    """Integration tests for component interaction"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_context = TestContext("integration")
        self.container = self.test_context.create_container()
        
    def test_event_flow(self):
        """Test events flow correctly"""
        # Setup components
        source = EventSource()
        handler = EventHandler()
        
        # Wire up
        self.container.event_bus.subscribe("TEST", handler.handle)
        
        # Trigger event
        source.emit_event("TEST", {"data": "test"})
        
        # Verify handling
        assert handler.received_count == 1
        assert handler.last_event.data == "test"
    
    def teardown_method(self):
        """Cleanup after test"""
        self.test_context.cleanup()
```

### System Test Pattern
```python
class TestSystem:
    """System tests for complete workflows"""
    
    @pytest.fixture
    def synthetic_scenario(self):
        """Provide deterministic test scenario"""
        return {
            'data': SyntheticData.trending_market(),
            'config': Config.simple_strategy(),
            'expected': PreComputed.trending_results()
        }
    
    def test_end_to_end_workflow(self, synthetic_scenario):
        """Test complete workflow with known results"""
        # Run system
        results = run_complete_system(
            synthetic_scenario['config'],
            synthetic_scenario['data']
        )
        
        # Validate against expected
        assert_results_match(
            results, 
            synthetic_scenario['expected'],
            tolerance=1e-10
        )
```

## Coverage Requirements

### Unit Test Coverage
- Minimum 90% line coverage
- 100% coverage for critical paths
- All error conditions tested
- All edge cases covered

### Integration Test Coverage
- All event flows tested
- All component interactions verified
- Container isolation validated
- Error propagation tested

### System Test Coverage
- All user workflows tested
- All optimization paths validated
- Performance requirements verified
- Production scenarios simulated

## Performance Requirements

### Test Execution Times
- Unit tests: < 100ms each
- Integration tests: < 1s each
- System tests: < 10s each
- Full test suite: < 5 minutes

### Memory Requirements
- Unit tests: < 50MB per test
- Integration tests: < 200MB per test
- System tests: < 1GB per test
- No memory leaks allowed

## Continuous Integration

### Test Automation
```yaml
# .github/workflows/tests.yml
name: Three-Tier Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: make test-unit
      - name: Check coverage
        run: make coverage-check

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run integration tests
        run: make test-integration

  system-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run system tests
        run: make test-system
```

### Test Reports
- Coverage reports generated for each tier
- Performance metrics tracked over time
- Failed test trends analyzed
- Flaky tests identified and fixed

## Benefits

1. **Early Bug Detection**: Unit tests catch logic errors immediately
2. **Integration Confidence**: Know components work together
3. **System Reliability**: End-to-end validation ensures correctness
4. **Refactoring Safety**: Tests ensure changes don't break functionality
5. **Documentation**: Tests serve as usage examples
6. **Performance Tracking**: Benchmarks prevent degradation

## Summary

The three-tier testing strategy provides comprehensive validation:
- **Unit tests** ensure components are correct
- **Integration tests** ensure components work together
- **System tests** ensure the system produces correct results

Together, they create a safety net that enables confident development and refactoring while maintaining system reliability.