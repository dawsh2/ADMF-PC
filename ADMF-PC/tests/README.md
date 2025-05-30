# ADMF-PC Test Suite

This directory contains the comprehensive test suite for the ADMF-PC (Adaptive Dynamic Multi-Factor Protocol - Polymorphic Composition) trading system.

## Test Structure

```
tests/
├── test_config.py                  # Test configuration and utilities
├── test_data.py                    # Data integration tests (requires pytest)
├── run_tests.py                    # Test runner script
├── validate_imports.py             # Import validation utility
│
├── test_risk/                      # Risk module tests
│   ├── test_signal_advanced.py     # Advanced signal processing tests
│   └── test_signal_flow.py         # Signal flow management tests
│
├── test_execution/                 # Execution module tests
│   ├── test_market_simulation.py   # Market simulation tests
│   ├── test_order_flow.py          # Order management tests
│   └── test_unified_backtest.py    # Unified backtest engine tests
│
├── test_integration/               # Integration tests
│   ├── test_full_backtest_flow.py  # Complete backtest workflow tests
│   ├── test_risk_execution_integration.py  # Risk-Execution integration
│   └── test_signal_to_fill_flow.py # Signal to fill flow tests
│
└── test_strategies/                # Strategy tests
    └── test_example_strategies.py  # Example strategy implementations
```

## Test Coverage

### Risk Module Tests
- **Signal Processing**: Router, validator, cache, prioritizer
- **Signal Flow**: Flow manager, multi-symbol routing
- **Portfolio Management**: Position tracking, risk limits
- **Advanced Features**: Signal aggregation, risk adjustments

### Execution Module Tests
- **Order Management**: Creation, validation, lifecycle
- **Market Simulation**: Slippage models, commission models
- **Backtest Engine**: Unified architecture, event flow
- **Broker Integration**: Fill simulation, position updates

### Integration Tests
- **Signal to Fill Flow**: Complete workflow from strategy signal to executed fill
- **Risk-Execution Integration**: Risk limits, position sizing, order approval
- **Full Backtest Flow**: End-to-end backtesting with multiple strategies
- **Multi-Strategy Flow**: Signal aggregation and weighted processing

### Strategy Tests
- **Momentum Strategy**: Signal generation based on price momentum
- **Mean Reversion Strategy**: Signal generation based on statistical properties
- **Strategy Integration**: Multiple strategies working together

## Running Tests

### Using unittest (no external dependencies)
```bash
# Run all tests
python3 tests/run_tests.py

# Run specific test category
python3 tests/run_tests.py --category risk
python3 tests/run_tests.py --category execution
python3 tests/run_tests.py --category integration

# Run specific test file
python3 -m unittest tests.test_risk.test_signal_advanced
```

### Validate imports
```bash
python3 tests/validate_imports.py
```

## Test Design Principles

1. **PC Architecture Compliance**: All tests follow Protocol + Composition pattern
2. **No Inheritance**: Tests use composition and protocols only
3. **Isolated Components**: Each test validates a specific component in isolation
4. **Integration Coverage**: Separate tests validate component interactions
5. **Realistic Scenarios**: Tests use realistic market data and trading scenarios

## Key Test Scenarios

### Risk Management
- Signal validation and deduplication
- Risk limit enforcement (position limits, exposure limits)
- Position sizing strategies
- Portfolio state consistency

### Order Execution
- Order lifecycle management
- Fill simulation with slippage and commission
- Partial fill handling
- Order status tracking

### Signal Flow
- Multi-strategy signal collection
- Signal aggregation and prioritization
- Routing to appropriate processors
- Event-driven communication

### Backtesting
- Historical data processing
- Strategy performance calculation
- Risk metrics computation
- Transaction cost modeling

## Test Utilities

### MockMarketData
Provides realistic market data for testing:
- Price generation with volatility
- Volume simulation
- Spread calculation

### TestConfig
Common test configuration:
- Default test parameters
- Mock object factories
- Assertion helpers

### TestSignals
Pre-defined test signals for common scenarios:
- Entry/exit signals
- Risk management signals
- Multi-strategy signals

## Dependencies

### Core Dependencies (No external packages)
- unittest (Python standard library)
- datetime, decimal, typing, etc.

### Optional Dependencies
- pytest: Enhanced test runner (test_data.py)
- numpy: Numerical computations (some integration tests)

## Adding New Tests

1. Follow the existing structure and naming conventions
2. Use composition, not inheritance
3. Mock external dependencies appropriately
4. Include both positive and negative test cases
5. Document test purpose and expected behavior
6. Ensure tests are independent and can run in any order

## Test Maintenance

- Run `validate_imports.py` after adding new tests
- Update this README when adding new test categories
- Keep tests focused and fast-running
- Remove or update obsolete tests promptly