# Strategy Testing Process

## Overview

This document formalizes the testing process for all trading strategies in ADMF-PC, ensuring comprehensive test coverage whenever new strategies are added.

## Testing Requirements

### 1. Every Strategy MUST Have Tests

For each strategy function decorated with `@strategy`, there must be corresponding tests that verify:

- **Basic functionality**: Strategy returns valid Signal objects
- **Edge cases**: Handles empty data, insufficient data, NaN values
- **Market conditions**: Performs correctly in trending, ranging, volatile markets
- **Parameter validation**: Handles invalid or extreme parameter values
- **Performance characteristics**: Executes within reasonable time limits

### 2. Test File Structure

```
tests/
└── unit/
    └── strategy/
        └── indicators/
            ├── __init__.py
            ├── conftest.py              # Shared fixtures
            ├── test_crossovers.py       # Tests for crossovers.py
            ├── test_momentum.py         # Tests for momentum.py
            ├── test_oscillators.py      # Tests for oscillators.py
            ├── test_structure.py        # Tests for structure.py
            ├── test_trend.py            # Tests for trend.py
            ├── test_volatility.py       # Tests for volatility.py
            └── test_volume.py           # Tests for volume.py
```

### 3. Test Template

Each test file should follow this template:

```python
"""
Unit tests for [module] indicator strategies.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategy.strategies.indicators.[module] import *
from src.strategy.types import Signal

class Test[Module]Strategies(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with sample market data."""
        # Create various market conditions
        
    def test_[strategy_name](self):
        """Test [strategy_name] strategy."""
        # Test basic functionality
        # Test edge cases
        # Test specific market conditions
        
    def _test_strategy_basic(self, strategy_func, **kwargs):
        """Reusable basic test template."""
        # Verify Signal output
        # Test with empty/insufficient data
        # Validate signal properties
```

## Automated Testing Process

### 1. Generate Tests for New Strategies

When adding a new strategy:

```bash
# 1. Add your strategy to the appropriate module
vim src/strategy/strategies/indicators/your_module.py

# 2. Generate test template
python generate_indicator_tests.py

# 3. Customize the generated tests for your specific strategy
vim tests/unit/strategy/indicators/test_your_module.py

# 4. Run tests to verify
python run_indicator_tests.py
```

### 2. Pre-commit Hook

Install the pre-commit hook to ensure tests exist:

```bash
# Copy the hook to your git hooks directory
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The hook will:
- Check for new/modified strategies
- Verify corresponding tests exist
- Block commits if tests are missing

### 3. CI/CD Integration

All strategy tests run automatically on:
- Every pull request
- Every commit to main branch
- Nightly comprehensive test runs

## Test Coverage Requirements

### Minimum Coverage Criteria

Each strategy must have tests that achieve:
- **Line coverage**: ≥ 90%
- **Branch coverage**: ≥ 85%
- **Edge case coverage**: All identified edge cases tested

### Coverage Report

Generate coverage reports:

```bash
# Run tests with coverage
pytest tests/unit/strategy/indicators/ --cov=src/strategy/strategies/indicators --cov-report=html

# View report
open htmlcov/index.html
```

## Adding a New Strategy: Step-by-Step

### 1. Create the Strategy

```python
# src/strategy/strategies/indicators/momentum.py

@strategy(
    name="new_momentum_indicator",
    description="My new momentum strategy",
    parameters=[...]
)
def new_momentum_indicator(data: pd.DataFrame, period: int = 20) -> Signal:
    """Calculate momentum signal."""
    # Implementation
    return Signal(direction=direction, magnitude=magnitude)
```

### 2. Generate Test Template

```bash
python generate_indicator_tests.py
# This updates tests/unit/strategy/indicators/test_momentum.py
```

### 3. Customize Tests

```python
# tests/unit/strategy/indicators/test_momentum.py

def test_new_momentum_indicator(self):
    """Test new momentum indicator strategy."""
    # Test normal conditions
    signal = new_momentum_indicator(self.data, period=20)
    self.assertIsInstance(signal, Signal)
    
    # Test specific conditions
    trending_data = self.create_trending_data()
    signal = new_momentum_indicator(trending_data, period=20)
    self.assertEqual(signal.direction, 1)  # Expect bullish
    
    # Test edge cases
    self._test_strategy_edge_cases(new_momentum_indicator, period=20)
```

### 4. Run Tests

```bash
# Run specific test
python -m pytest tests/unit/strategy/indicators/test_momentum.py::test_new_momentum_indicator -v

# Run all indicator tests
python run_indicator_tests.py
```

### 5. Verify Coverage

```bash
# Check coverage for your module
pytest tests/unit/strategy/indicators/test_momentum.py --cov=src/strategy/strategies/indicators/momentum
```

## Test Data Fixtures

### Standard Fixtures (in conftest.py)

- `sample_ohlcv_data`: 100 days of random OHLCV data
- `trending_data`: Steady uptrend data
- `ranging_data`: Sideways market data
- `volatile_data`: High volatility data
- `low_volume_data`: Thin market conditions

### Creating Custom Fixtures

```python
@pytest.fixture
def flash_crash_data():
    """Create data with sudden price drop."""
    data = sample_ohlcv_data()
    # Drop price by 10% at day 50
    data.loc[50:, 'close'] *= 0.9
    return data
```

## Performance Testing

### Strategy Execution Time

Each strategy should execute within:
- **Simple strategies**: < 10ms for 1000 bars
- **Complex strategies**: < 50ms for 1000 bars
- **ML-based strategies**: < 200ms for 1000 bars

### Performance Test Template

```python
def test_strategy_performance(self):
    """Test strategy execution time."""
    import time
    
    # Create large dataset
    large_data = self.create_large_dataset(10000)  # 10k bars
    
    start = time.time()
    signal = strategy_function(large_data)
    elapsed = time.time() - start
    
    self.assertLess(elapsed, 0.1)  # Should complete in 100ms
```

## Continuous Improvement

### Monthly Review Process

1. **Coverage Analysis**: Review test coverage reports
2. **Failed Test Analysis**: Investigate recurring failures
3. **Performance Regression**: Check for slowdowns
4. **New Edge Cases**: Add tests for production issues

### Test Maintenance

- Update tests when strategy behavior changes
- Remove tests for deprecated strategies
- Refactor common test patterns into utilities
- Keep test data realistic and current

## Tools and Scripts

### Available Scripts

1. **generate_indicator_tests.py**: Creates test templates
2. **run_indicator_tests.py**: Runs all indicator tests
3. **check_test_coverage.py**: Verifies coverage requirements
4. **validate_strategy_tests.py**: Pre-commit validation

### Running Tests

```bash
# Run all strategy tests
python -m pytest tests/unit/strategy/ -v

# Run specific module tests
python -m pytest tests/unit/strategy/indicators/test_momentum.py -v

# Run with coverage
python -m pytest tests/unit/strategy/ --cov=src/strategy --cov-report=term-missing

# Run in parallel
python -m pytest tests/unit/strategy/ -n auto
```

## Best Practices

### DO

- ✅ Write tests BEFORE merging new strategies
- ✅ Test edge cases and error conditions
- ✅ Use meaningful test names that describe what's being tested
- ✅ Keep tests fast and independent
- ✅ Use fixtures for common test data
- ✅ Test with realistic market data patterns

### DON'T

- ❌ Skip tests because "it's a simple strategy"
- ❌ Use production data in tests
- ❌ Write tests that depend on external services
- ❌ Ignore flaky tests
- ❌ Test implementation details instead of behavior

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **Fixture Not Found**: Check conftest.py is in the right location
3. **Slow Tests**: Profile and optimize or mark as slow
4. **Flaky Tests**: Fix random seeds, avoid time-dependent logic

### Getting Help

- Check existing test examples in `tests/unit/strategy/`
- Review pytest documentation
- Ask in #testing channel
- Consult CONTRIBUTING.md for standards

## Enforcement

### Automated Checks

1. **Pre-commit hooks**: Verify tests exist for new strategies
2. **CI pipeline**: Run all tests on every PR
3. **Coverage gates**: Block PRs below coverage threshold
4. **Performance benchmarks**: Alert on regression

### Manual Review

Code reviewers must verify:
- Tests exist for all new strategies
- Tests cover edge cases
- Tests follow naming conventions
- Tests are maintainable

---

**Remember**: Well-tested strategies lead to confident trading. Every strategy deserves comprehensive tests!