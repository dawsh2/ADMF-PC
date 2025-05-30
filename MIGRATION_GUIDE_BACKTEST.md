# Backtest Module Migration Guide

## Overview

The backtest functionality has been moved into the execution module as `UnifiedBacktestEngine`. This eliminates duplicate code and ensures consistent execution paths between backtest and live trading.

## Key Changes

### 1. Module Location
- **Old**: `src.backtest.BacktestEngine`
- **New**: `src.execution.UnifiedBacktestEngine`

### 2. Position Tracking
- **Old**: BacktestEngine maintained its own position tracking
- **New**: Delegates to Risk module's RiskPortfolioContainer

### 3. Order Execution
- **Old**: Direct execution logic in BacktestEngine
- **New**: Uses ExecutionEngine for all order processing

### 4. Event System
- **Old**: Mix of direct calls and events
- **New**: Fully event-driven through ExecutionEngine

## Migration Examples

### Old Code
```python
from src.backtest import BacktestEngine, BacktestConfig
from src.data import DataLoader
from src.strategy import MyStrategy

# Configure backtest
config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=100000,
    symbols=["AAPL", "GOOGL"],
    commission=0.001,
    slippage=0.0005
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run(MyStrategy(), DataLoader())
```

### New Code
```python
from src.execution import UnifiedBacktestEngine, BacktestConfig
from src.data import DataLoader
from src.strategy import MyStrategy
from decimal import Decimal

# Configure backtest (now uses Decimal for precision)
config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=Decimal("100000"),
    symbols=["AAPL", "GOOGL"],
    commission=Decimal("0.001"),
    slippage=Decimal("0.0005")
)

# Run backtest
engine = UnifiedBacktestEngine(config)
results = engine.run(MyStrategy(), DataLoader())
```

## Architecture Benefits

### 1. Single Source of Truth
- Portfolio state managed by Risk module
- No duplicate position tracking
- Consistent P&L calculations

### 2. Realistic Execution
- Same execution path as live trading
- Proper order lifecycle management
- Realistic market simulation

### 3. Better Integration
- Works seamlessly with Risk module limits
- Supports complex position sizing
- Integrates with event system

## Detailed Component Flow

```
UnifiedBacktestEngine
    │
    ├─> Generates signals via Strategy
    │
    ├─> Processes signals through RiskPortfolioContainer
    │   └─> Position sizing
    │   └─> Risk limit checks
    │   └─> Generates orders
    │
    ├─> Submits orders to ExecutionEngine
    │   └─> Order validation
    │   └─> Market simulation
    │   └─> Generates fills
    │
    └─> Updates portfolio via BacktestBrokerRefactored
        └─> Delegates to Risk module's PortfolioState
```

## Feature Comparison

| Feature | Old BacktestEngine | UnifiedBacktestEngine |
|---------|-------------------|----------------------|
| Position Tracking | Internal | Risk Module |
| Order Processing | Direct | ExecutionEngine |
| Event System | Partial | Full |
| Market Simulation | Internal | MarketSimulator |
| Risk Integration | Limited | Full |
| Decimal Precision | No | Yes |
| Thread Safety | No | Yes |

## API Compatibility

The new engine maintains API compatibility for basic usage:
- `run()` method signature unchanged
- `BacktestConfig` enhanced but backward compatible
- `BacktestResults` structure preserved

## Advanced Features

### 1. Container Integration
```python
from src.core.containers import UniversalScopedContainer

container = UniversalScopedContainer("backtest_container")
engine = UnifiedBacktestEngine(config, container=container)

# Now supports event monitoring
container.event_bus.subscribe(EventType.INFO, progress_handler)
```

### 2. Custom Risk Limits
```python
engine = UnifiedBacktestEngine(config)

# Add custom risk limits
engine.risk_portfolio.add_risk_limit(MaxDrawdownLimit(Decimal("0.20")))
engine.risk_portfolio.add_risk_limit(MaxPositionLimit(Decimal("10000")))
```

### 3. Multiple Position Sizers
```python
# Configure different sizers for different conditions
engine.risk_portfolio.set_position_sizer(
    VolatilityBasedSizer(risk_per_trade=Decimal("0.02"))
)
```

## Performance Improvements

1. **Memory Efficiency**: Single portfolio state reduces memory usage
2. **Speed**: Event batching improves performance
3. **Accuracy**: Decimal arithmetic eliminates floating-point errors

## Deprecation Timeline

1. **Phase 1** (Current): Both engines available
2. **Phase 2** (Next Release): Old engine marked deprecated
3. **Phase 3** (Future): Old engine removed

## Support

For migration assistance:
- Review test examples in `tests/execution/test_backtest_engine.py`
- Check updated documentation
- File issues for migration problems