# Testing Standards

Comprehensive testing standards for ADMF-PC using the three-tier testing strategy and protocol-based testing patterns.

## Three-Tier Testing Strategy

### Overview

Every component must have tests at three levels:

1. **Unit Tests**: Component logic in isolation
2. **Integration Tests**: Component interactions
3. **System Tests**: End-to-end workflows

### Testing Pyramid

```
         /\
        /  \  System Tests (10%)
       /    \ - Full workflows
      /      \ - Real data
     /--------\
    /          \ Integration Tests (30%)
   /            \ - Component interactions
  /              \ - Event flow validation
 /                \ - Container isolation
/------------------\
                    Unit Tests (60%)
                    - Component logic
                    - Protocol compliance
                    - Edge cases
```

## Unit Testing

### Unit Test Structure

```python
"""
Test file: test_{component_name}.py
Location: tests/unit/{module}/test_{component_name}.py
"""

import pytest
from unittest.mock import Mock, MagicMock

class TestComponent:
    """Unit tests for Component"""
    
    @pytest.fixture
    def component(self):
        """Create component with mocked dependencies"""
        mock_event_bus = Mock()
        config = {"param1": "value1"}
        return Component(config, mock_event_bus)
    
    def test_initialization(self, component):
        """Test component initializes correctly"""
        assert component.state == "created"
        assert component.config["param1"] == "value1"
    
    def test_protocol_compliance(self, component):
        """Test component implements required protocol"""
        assert hasattr(component, 'generate_signal')
        assert callable(component.generate_signal)
    
    def test_signal_generation(self, component):
        """Test signal generation logic"""
        test_data = create_test_data()
        signal = component.generate_signal(test_data)
        
        assert signal['action'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal['strength'] <= 1
```

### Protocol-Based Testing

```python
def test_any_signal_generator(generator):
    """Test anything that generates signals (duck typing)"""
    
    test_data = pd.DataFrame({
        'close': [100, 101, 102, 103],
        'volume': [1000, 1100, 900, 1200]
    })
    
    # Duck typing - work with any implementation
    if hasattr(generator, 'generate_signal'):
        signal = generator.generate_signal(test_data)
    elif callable(generator):
        signal = generator(test_data)
    else:
        pytest.skip("Not a signal generator")
    
    # Validate signal format
    assert isinstance(signal, dict)
    assert 'action' in signal
    assert 'strength' in signal

# Test multiple implementations with same test
@pytest.mark.parametrize("generator", [
    MomentumStrategy(),
    mean_reversion_strategy,  # Function
    lambda d: {'action': 'BUY', 'strength': 0.5},  # Lambda
    MLModel().predict,  # Method
])
def test_signal_generators(generator):
    test_any_signal_generator(generator)
```

### Edge Case Testing

```python
class TestEdgeCases:
    """Test component behavior with edge cases"""
    
    def test_empty_data(self, component):
        """Test with empty data"""
        empty_data = pd.DataFrame()
        signal = component.generate_signal(empty_data)
        assert signal['action'] == 'HOLD'
    
    def test_missing_values(self, component):
        """Test with missing data"""
        data_with_nans = pd.DataFrame({
            'close': [100, np.nan, 102, 103],
            'volume': [1000, 1100, np.nan, 1200]
        })
        signal = component.generate_signal(data_with_nans)
        assert signal is not None  # Should handle gracefully
    
    def test_extreme_values(self, component):
        """Test with extreme market conditions"""
        extreme_data = pd.DataFrame({
            'close': [100, 10000, 1, 100],  # 100x jump and crash
            'volume': [1000, 0, 1e9, 1000]
        })
        signal = component.generate_signal(extreme_data)
        assert signal['strength'] <= 1.0  # Should cap strength
```

## Integration Testing

### Integration Test Structure

```python
"""
Test file: test_{module}_integration.py
Location: tests/integration/test_{module}_integration.py
"""

class TestStrategyRiskIntegration:
    """Test strategy and risk module integration"""
    
    @pytest.fixture
    def test_container(self):
        """Create isolated test container"""
        container = TestContainer("integration_test")
        container.event_bus = EventBus()
        return container
    
    def test_signal_to_order_flow(self, test_container):
        """Test complete signal to order flow"""
        # Create components
        strategy = MomentumStrategy(config, test_container.event_bus)
        risk_manager = RiskManager(config, test_container.event_bus)
        
        # Track events
        events_captured = []
        test_container.event_bus.subscribe(
            "ORDER", 
            lambda e: events_captured.append(e)
        )
        
        # Generate signal
        strategy.process_bar(test_bar_data)
        
        # Verify event flow
        assert len(events_captured) == 1
        assert events_captured[0].type == "ORDER"
        assert events_captured[0].data['size'] > 0
```

### Container Isolation Testing

```python
class TestContainerIsolation:
    """Test container isolation properties"""
    
    def test_parallel_containers_isolated(self):
        """Test parallel containers don't interfere"""
        
        # Create multiple containers
        containers = [
            BacktestContainer(f"test_{i}")
            for i in range(5)
        ]
        
        # Run in parallel
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(run_backtest, container, test_data)
                for container in containers
            ]
            results = [f.result() for f in futures]
        
        # Verify isolation
        for i, container in enumerate(containers):
            # Each container should have unique results
            assert results[i]['container_id'] == f"test_{i}"
            # No shared state
            assert container.event_bus != containers[(i+1)%5].event_bus
```

### Event Flow Testing

```python
class TestEventFlow:
    """Test event flow through system"""
    
    def test_complete_event_chain(self):
        """Test full event chain from data to fill"""
        
        # Set up event tracking
        event_chain = []
        
        def track_event(event):
            event_chain.append({
                'type': event.type,
                'source': event.source,
                'timestamp': event.timestamp
            })
        
        # Subscribe to all events
        for event_type in ['BAR_DATA', 'INDICATOR', 'SIGNAL', 'ORDER', 'FILL']:
            event_bus.subscribe(event_type, track_event)
        
        # Run test
        run_single_bar_test(test_bar)
        
        # Verify event sequence
        event_types = [e['type'] for e in event_chain]
        assert event_types == ['BAR_DATA', 'INDICATOR', 'SIGNAL', 'ORDER', 'FILL']
        
        # Verify timing
        for i in range(len(event_chain)-1):
            assert event_chain[i]['timestamp'] <= event_chain[i+1]['timestamp']
```

## System Testing

### System Test Structure

```python
"""
Test file: test_{workflow}_system.py
Location: tests/system/test_{workflow}_system.py
"""

class TestCompleteBacktestSystem:
    """Test complete backtest workflow"""
    
    def test_end_to_end_backtest(self):
        """Test full backtest with real data"""
        
        # Load real configuration
        config = load_config("config/test_backtest.yaml")
        
        # Run complete backtest
        coordinator = YAMLCoordinator()
        results = coordinator.run(config)
        
        # Verify results
        assert results['status'] == 'completed'
        assert results['total_trades'] > 0
        assert results['sharpe_ratio'] > 0
        assert 'equity_curve' in results
        
        # Verify reproducibility
        results2 = coordinator.run(config)
        assert results['final_value'] == results2['final_value']
```

### Multi-Phase Workflow Testing

```python
class TestOptimizationWorkflow:
    """Test multi-phase optimization workflow"""
    
    def test_complete_optimization(self):
        """Test full optimization workflow"""
        
        config = load_config("config/test_optimization.yaml")
        
        # Phase 1: Parameter discovery
        phase1_results = run_phase(config, "parameter_discovery")
        assert len(phase1_results['parameter_sets']) > 0
        
        # Phase 2: Signal capture
        signal_log = run_phase(config, "signal_capture")
        assert os.path.exists(signal_log['path'])
        
        # Phase 3: Ensemble optimization
        ensemble_results = run_phase(config, "ensemble_optimization")
        assert ensemble_results['best_weights'] is not None
        
        # Phase 4: Validation
        validation_results = run_phase(config, "validation")
        assert validation_results['sharpe_ratio'] > 0
```

## Test Data Management

### Test Data Fixtures

```python
@pytest.fixture(scope="session")
def sample_market_data():
    """Provide consistent test data"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

@pytest.fixture
def test_signal():
    """Standard test signal"""
    return {
        'action': 'BUY',
        'symbol': 'TEST',
        'strength': 0.75,
        'timestamp': datetime.now(),
        'metadata': {'strategy': 'test'}
    }
```

### Synthetic Data Generation

```python
class SyntheticDataGenerator:
    """Generate synthetic data for testing"""
    
    @staticmethod
    def trending_market(periods=1000, trend=0.0001):
        """Generate trending market data"""
        prices = [100]
        for _ in range(periods-1):
            change = np.random.normal(trend, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        })
    
    @staticmethod
    def volatile_market(periods=1000, volatility=0.05):
        """Generate volatile market data"""
        return pd.DataFrame({
            'close': 100 + np.random.normal(0, volatility, periods).cumsum(),
            'volume': np.random.randint(1000, 10000, periods)
        })
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks"""
    
    def test_signal_generation_speed(self, benchmark):
        """Benchmark signal generation"""
        strategy = MomentumStrategy(config)
        data = create_large_dataset(10000)
        
        result = benchmark(strategy.generate_signal, data)
        
        # Assert performance requirements
        assert benchmark.stats['mean'] < 0.001  # < 1ms average
    
    def test_backtest_throughput(self, benchmark):
        """Benchmark complete backtest"""
        
        def run_backtest():
            container = BacktestContainer("perf_test")
            return container.run(test_data_1year)
        
        result = benchmark(run_backtest)
        
        # Should process 1 year in < 5 seconds
        assert benchmark.stats['mean'] < 5.0
```

### Memory Testing

```python
@pytest.mark.memory
class TestMemoryUsage:
    """Test memory usage patterns"""
    
    def test_container_memory_isolation(self):
        """Test containers don't leak memory"""
        
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many containers
        for i in range(100):
            container = BacktestContainer(f"mem_test_{i}")
            container.run(small_test_data)
            container.cleanup()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not leak more than 50MB for 100 containers
        assert memory_increase < 50
```

## Mock Objects

### Standard Mocks

```python
class MockEventBus:
    """Mock event bus for testing"""
    
    def __init__(self):
        self.published_events = []
        self.subscribers = defaultdict(list)
    
    def publish(self, event: Event):
        self.published_events.append(event)
        for handler in self.subscribers[event.type]:
            handler(event)
    
    def subscribe(self, event_type: str, handler: Callable):
        self.subscribers[event_type].append(handler)
    
    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.published_events if e.type == event_type]
        return self.published_events

class MockDataProvider:
    """Mock data provider for testing"""
    
    def __init__(self, data: pd.DataFrame = None):
        self.data = data or create_test_data()
    
    def get_data(self, symbol: str, start: Any, end: Any):
        return self.data
```

## Test Coverage Requirements

### Coverage Standards

- **Overall Coverage**: Minimum 80%
- **Critical Components**: Minimum 90%
  - Risk management
  - Order execution
  - Portfolio tracking
- **New Code**: Minimum 85%

### Coverage Configuration

```ini
# .coveragerc
[run]
source = src/
omit = 
    */tests/*
    */test_*.py
    */__pycache__/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

### Running Coverage

```bash
# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Run specific module coverage
pytest tests/unit/strategy --cov=src/strategy --cov-report=term-missing

# Generate coverage badge
coverage-badge -o coverage.svg
```

## Continuous Integration

### CI Test Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        # Unit tests
        pytest tests/unit -v
        
        # Integration tests
        pytest tests/integration -v
        
        # System tests (only on main)
        if [ "${{ github.ref }}" == "refs/heads/main" ]; then
          pytest tests/system -v
        fi
    
    - name: Coverage
      run: |
        pytest --cov=src --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Test Organization

### Directory Structure

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests
│   ├── core/
│   │   ├── test_containers.py
│   │   ├── test_events.py
│   │   └── test_components.py
│   ├── strategy/
│   │   ├── test_momentum.py
│   │   └── test_mean_reversion.py
│   └── risk/
│       └── test_position_sizing.py
├── integration/         # Integration tests
│   ├── test_strategy_risk.py
│   ├── test_data_flow.py
│   └── test_event_chain.py
└── system/             # System tests
    ├── test_backtest.py
    ├── test_optimization.py
    └── test_live_trading.py
```

## Testing Best Practices

### 1. Test Independence

```python
# Good: Each test is independent
def test_signal_generation():
    strategy = create_strategy()  # Fresh instance
    result = strategy.generate_signal(test_data)
    assert result['action'] == 'BUY'

# Bad: Tests depend on shared state
class TestStrategy:
    strategy = create_strategy()  # Shared!
    
    def test_one(self):
        self.strategy.process(data1)  # Modifies state
    
    def test_two(self):
        self.strategy.process(data2)  # Depends on test_one
```

### 2. Clear Test Names

```python
# Good: Descriptive test names
def test_momentum_strategy_generates_buy_signal_on_uptrend():
    pass

def test_risk_manager_rejects_signal_exceeding_position_limit():
    pass

# Bad: Unclear names
def test_1():
    pass

def test_strategy():
    pass
```

### 3. Arrange-Act-Assert

```python
def test_position_sizing():
    # Arrange
    risk_manager = RiskManager({"max_position_pct": 2.0})
    signal = {"strength": 0.8, "action": "BUY"}
    portfolio_value = 100000
    
    # Act
    position_size = risk_manager.calculate_position_size(
        signal, 
        portfolio_value
    )
    
    # Assert
    assert position_size <= 2000  # Max 2% of portfolio
    assert position_size > 0      # Some position taken
```

### 4. Use Fixtures Effectively

```python
@pytest.fixture
def market_data():
    """Reusable market data"""
    return create_test_market_data()

@pytest.fixture
def configured_strategy(market_data):
    """Strategy with dependencies"""
    return MomentumStrategy({
        "fast_period": 10,
        "slow_period": 30
    })

def test_strategy_with_fixtures(configured_strategy, market_data):
    """Use fixtures for clean tests"""
    signal = configured_strategy.generate_signal(market_data)
    assert signal is not None
```

## Summary

Effective testing in ADMF-PC:

1. **Three-tier approach**: Unit, Integration, System
2. **Protocol-based**: Test behaviors, not implementations
3. **Container isolation**: Test in isolated environments
4. **Duck typing**: Test any compatible component
5. **Comprehensive coverage**: Minimum 80% overall
6. **Performance testing**: Ensure speed requirements
7. **Clear organization**: Logical test structure

Follow these standards to ensure reliable, maintainable, and comprehensive tests.