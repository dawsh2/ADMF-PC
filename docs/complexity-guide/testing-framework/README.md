# Testing Framework

This directory contains the three-tier testing approach that must be developed in parallel with every implementation step.

## 📋 Contents

1. **[three-tier-strategy.md](three-tier-strategy.md)** - Overview of the testing approach
2. **[unit-test-templates.md](unit-test-templates.md)** - Component-level testing patterns
3. **[integration-test-patterns.md](integration-test-patterns.md)** - Container interaction testing
4. **[system-test-scenarios.md](system-test-scenarios.md)** - End-to-end workflow validation
5. **[test-automation-scripts.md](test-automation-scripts.md)** - CI/CD integration and automation

## 🎯 Three-Tier Testing Strategy

### 1. Unit Tests: Component-Level Correctness
- Test individual functions/classes in isolation
- Use mocks for dependencies
- Fast execution (<1s per test)
- Focus on business logic correctness

### 2. Integration Tests: Container Interaction Validation
- Test event flow between components
- Validate container isolation
- Use real containers with synthetic data
- Verify protocol implementations

### 3. System Tests: End-to-End Workflow Validation
- Test complete workflows with deterministic results
- Validate optimization reproducibility
- Use full system with controlled inputs
- Ensure results match expectations exactly

## 🚀 Test-First Development

**CRITICAL**: Tests must be written BEFORE or WITH implementation, never after.

### Development Process
1. Write test specifications first
2. Implement tests in parallel with code
3. All three tiers must pass before proceeding
4. Use synthetic data framework for determinism

### Example Workflow
```bash
# 1. Write unit test specs
vim tests/unit/test_step1_components.py

# 2. Write integration test specs
vim tests/integration/test_step1_event_flow.py

# 3. Write system test specs
vim tests/system/test_step1_full_workflow.py

# 4. Implement feature with tests
# Tests and implementation developed together

# 5. Run all tests
make test-step1

# 6. Only proceed when all pass
```

## ✅ Testing Requirements

Every step must have:
- [ ] Unit tests with >90% coverage
- [ ] Integration tests for all event flows
- [ ] System tests with synthetic data validation
- [ ] Performance benchmarks met
- [ ] Memory usage within limits
- [ ] Container isolation verified

## 📁 Test File Structure

```
tests/
├── fixtures/                    # Shared test data and utilities
│   ├── synthetic_data.py       # Synthetic data generators
│   ├── expected_results.py     # Pre-computed results
│   └── test_helpers.py         # Common test utilities
│
├── unit/                       # Component-level tests
│   ├── core/                   # Core module tests
│   ├── data/                   # Data module tests
│   ├── strategy/               # Strategy module tests
│   └── risk/                   # Risk module tests
│
├── integration/                # Container interaction tests
│   ├── test_event_flows.py     # Event routing tests
│   ├── test_container_isolation.py  # Isolation tests
│   └── test_signal_to_order.py     # Signal flow tests
│
└── system/                     # End-to-end tests
    ├── test_basic_backtest.py  # Simple workflow tests
    ├── test_optimization.py    # Optimization workflow tests
    └── test_production_ready.py # Production simulation tests
```

## 🔧 Test Utilities

### TestContext Manager
```python
class TestContext:
    """Manages test environment setup and teardown"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.containers = []
        self.temp_files = []
    
    def __enter__(self):
        self.setup_isolation()
        self.setup_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_containers()
        self.cleanup_files()
        self.validate_no_leaks()
```

### Synthetic Data Fixtures
```python
@pytest.fixture
def simple_market_data():
    """Provides deterministic market data"""
    return SyntheticDataGenerator.create_simple_trend()

@pytest.fixture
def expected_sma_signals():
    """Pre-computed SMA crossover signals"""
    return {
        'signals': [
            {'timestamp': '2024-01-05', 'action': 'BUY'},
            {'timestamp': '2024-01-15', 'action': 'SELL'}
        ],
        'trades': 2,
        'final_value': 10250.00
    }
```

## 🏃 Running Tests

### Run All Tests for a Step
```bash
# Run all three tiers for step 1
make test-step1

# Run specific tier
make test-unit-step1
make test-integration-step1
make test-system-step1
```

### Continuous Testing
```bash
# Watch mode - re-run on file changes
make test-watch-step1

# Coverage report
make test-coverage-step1

# Performance benchmarks
make test-performance-step1
```

## 📊 Success Metrics

Tests are considered passing when:
- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ All system tests pass
- ✅ Code coverage > 90%
- ✅ Performance benchmarks met
- ✅ No memory leaks detected
- ✅ Container isolation verified

## Next Steps

1. Review the three-tier strategy
2. Use templates for consistent tests
3. Integrate with CI/CD pipeline
4. Monitor test metrics
5. Continuously improve coverage