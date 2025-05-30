# Risk Management Module

The Risk module provides a unified Risk & Portfolio management system that manages the critical path from signal generation to order creation.

## Architecture Overview

The module implements a layered signal processing pipeline:

```
Strategies → Signal Flow → Risk & Portfolio → Orders → Execution
```

### Core Components

#### 1. **RiskPortfolioContainer**
The main container that manages:
- Multiple strategy components
- Signal to order conversion (with veto capability)
- Portfolio state tracking
- Position sizing
- Risk limit enforcement

#### 2. **Signal Processing Pipeline**

The signal processing is separated into multiple specialized components:

##### Basic Processing
- **SignalProcessor**: Core signal to order conversion
- **SignalAggregator**: Combines signals from multiple strategies

##### Advanced Processing
- **SignalValidator**: Validates signal integrity and feasibility
- **SignalCache**: Deduplicates signals to prevent double processing
- **SignalPrioritizer**: Orders signals by priority (exits before entries)
- **SignalRouter**: Routes signals to specialized processors
- **RiskAdjustedSignalProcessor**: Enhanced processor with risk adjustments

##### Flow Management
- **SignalFlowManager**: Orchestrates the complete signal flow
- **MultiSymbolSignalFlow**: Manages flows for multiple symbols/classifiers

#### 3. **Portfolio State Management**
- **PortfolioState**: Tracks positions, P&L, and risk metrics
- Thread-safe operations based on execution context
- Real-time position and cash tracking

#### 4. **Position Sizing Strategies**
- **FixedPositionSizer**: Fixed quantity per trade
- **PercentagePositionSizer**: Percentage of portfolio value
- **VolatilityBasedSizer**: Size based on volatility (ATR)
- **KellyCriterionSizer**: Optimal sizing based on win rate

#### 5. **Risk Limits**
- **MaxPositionLimit**: Maximum position size
- **MaxDrawdownLimit**: Stop trading at drawdown threshold
- **ConcentrationLimit**: Limit exposure to single positions
- **LeverageLimit**: Control overall leverage
- **VaRLimit**: Value at Risk limits

## Signal Flow Architecture

### Standard Flow
```python
# 1. Strategies generate signals
signal = Signal(
    signal_id="sig_001",
    strategy_id="momentum",
    symbol="AAPL",
    signal_type=SignalType.ENTRY,
    side=OrderSide.BUY,
    strength=Decimal("0.85"),
    timestamp=datetime.now(),
    metadata={...}
)

# 2. Signals collected by flow manager
await flow_manager.collect_signal(signal)

# 3. Periodic processing converts signals to orders
orders = await flow_manager.process_signals(
    portfolio_state=portfolio_state,
    position_sizer=position_sizer,
    risk_limits=risk_limits,
    market_data=market_data
)

# 4. Orders sent to execution
for order in orders:
    await execution_engine.submit_order(order)
```

### Multi-Symbol Flow
```python
# Create flow managers per classifier
flow_manager = multi_flow.create_flow_manager(
    classifier_id="tech_stocks",
    config={
        "enable_aggregation": True,
        "aggregation_method": "weighted_average"
    }
)

# Map symbols to classifiers
multi_flow.map_symbol_to_classifier("AAPL", "tech_stocks")
multi_flow.map_symbol_to_classifier("GOOGL", "tech_stocks")

# Route signals automatically
await multi_flow.route_signal(signal)

# Process all classifiers
orders_by_classifier = await multi_flow.process_all_signals(
    portfolio_states=portfolio_states,
    position_sizers=position_sizers,
    risk_limits=risk_limits,
    market_data=market_data
)
```

## Signal Aggregation Methods

The system supports multiple aggregation methods when combining signals from multiple strategies:

1. **Weighted Average**: Combines signals based on strength and optional weights
2. **Majority Vote**: Takes the direction with most votes
3. **Unanimous**: Requires all strategies to agree
4. **First**: Takes the first signal (no aggregation)

## Signal Prioritization

Signals are prioritized to ensure proper execution order:

1. **CRITICAL**: Risk exits, stop losses
2. **HIGH**: Strong signals, regular exits
3. **NORMAL**: Regular entry signals
4. **LOW**: Weak signals, rebalances

## Risk Checks Pipeline

Each signal goes through multiple risk checks:

1. **Validation**: Signal integrity and timestamp checks
2. **Deduplication**: Prevent processing duplicate signals
3. **Position Logic**: Ensure signal makes sense given current positions
4. **Position Sizing**: Calculate appropriate size
5. **Risk Limits**: Check all configured risk limits
6. **Order Creation**: Generate order with all metadata

## Thread Safety

The module is designed for concurrent operation:
- Thread-safe based on ExecutionContext
- Lock-free design where possible
- Async-first architecture
- Batch processing for efficiency

## Example Usage

### Basic Risk & Portfolio Setup
```python
from src.risk import (
    RiskPortfolioContainer,
    PercentagePositionSizer,
    MaxDrawdownLimit,
    SignalFlowManager
)

# Create Risk & Portfolio container
risk_portfolio = RiskPortfolioContainer(
    name="Conservative",
    initial_capital=Decimal("100000")
)

# Configure position sizing
risk_portfolio.set_position_sizer(
    PercentagePositionSizer(percentage=Decimal("0.02"))  # 2% per position
)

# Add risk limits
risk_portfolio.add_risk_limit(
    MaxDrawdownLimit(
        max_drawdown_pct=Decimal("10"),  # 10% max drawdown
        reduce_at_pct=Decimal("8")        # Start reducing at 8%
    )
)

# Create signal flow manager
flow_manager = SignalFlowManager(
    enable_aggregation=True,
    aggregation_method="weighted_average"
)

# Register strategies
flow_manager.register_strategy("momentum", weight=Decimal("0.6"))
flow_manager.register_strategy("mean_reversion", weight=Decimal("0.4"))
```

### Processing Signals
```python
# Collect signals from strategies
await flow_manager.collect_signal(momentum_signal)
await flow_manager.collect_signal(mean_reversion_signal)

# Process into orders
orders = await flow_manager.process_signals(
    portfolio_state=risk_portfolio.get_portfolio_state(),
    position_sizer=risk_portfolio._position_sizer,
    risk_limits=risk_portfolio._risk_limits,
    market_data=current_market_data
)

# Update portfolio with fills
fills = await execution_engine.execute_orders(orders)
risk_portfolio.update_fills(fills)
```

## Best Practices

1. **Signal Generation**
   - Include all necessary metadata in signals
   - Use appropriate signal types (ENTRY, EXIT, RISK_EXIT)
   - Normalize signal strength to [-1, 1]

2. **Risk Configuration**
   - Start with conservative limits
   - Use multiple complementary risk limits
   - Monitor risk metrics regularly

3. **Performance**
   - Enable signal caching to prevent duplicates
   - Use batch processing for efficiency
   - Configure appropriate aggregation methods

4. **Monitoring**
   - Track signal approval rates
   - Monitor risk limit violations
   - Review position sizing effectiveness