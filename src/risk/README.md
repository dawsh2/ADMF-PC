# Risk Module Documentation

## Overview

The Risk module manages risk controls and limits for trading strategies. It works in conjunction with separate Portfolio containers to enforce risk constraints while allowing flexible portfolio allocation strategies.

## Architecture

### Risk Container with Separate Portfolio Containers

```
Classifier Container
└── Risk Container
    ├── Risk Limit Enforcement  
    ├── Signal → Order Conversion
    └── Portfolio Container Pool
        ├── Portfolio Container A (Conservative allocation)
        ├── Portfolio Container B (Aggressive allocation)
        └── Portfolio Container N (Custom allocation)
```

**Key Principle**: Risk and Portfolio are separate concerns - Risk enforces limits and constraints, while Portfolio manages allocation strategies and position tracking. This separation enables testing different portfolio approaches under the same risk framework.

**Note**: While logically separate containers, the current implementation files are co-located under `src/risk/` for convenience. This can be refactored to separate modules later without affecting the container architecture.

## Core Components

### 1. RiskContainer (`risk_container.py`)

The main risk management component that:
- Manages multiple portfolio containers as children
- Enforces risk limits across all portfolios
- Converts signals to orders (with veto capability)
- Monitors aggregate exposure and limits
- Coordinates with separate portfolio containers

```python
# Example usage
risk_container = RiskContainer(
    container_id="main_risk",
    parent_container=classifier_container
)

# Add portfolio containers with different strategies
conservative_portfolio = PortfolioContainer(
    container_id="conservative_portfolio",
    initial_capital=Decimal("50000"),
    allocation_strategy="conservative"
)
aggressive_portfolio = PortfolioContainer(
    container_id="aggressive_portfolio", 
    initial_capital=Decimal("50000"),
    allocation_strategy="aggressive"
)

risk_container.add_portfolio(conservative_portfolio)
risk_container.add_portfolio(aggressive_portfolio)

# Add risk limits that apply to all portfolios
risk_container.add_risk_limit(MaxExposureLimit(max_exposure_pct=Decimal("20")))
risk_container.add_risk_limit(MaxDrawdownLimit(max_drawdown_pct=Decimal("10")))
```

### 2. Position Sizing Strategies (`position_sizing.py`)

Determine how much to trade based on portfolio state and risk parameters:

- **FixedPositionSizer**: Fixed number of shares
- **PercentagePositionSizer**: Percentage of portfolio value
- **VolatilityBasedSizer**: Risk parity approach using volatility
- **KellyCriterionSizer**: Optimal sizing based on edge and probability
- **ATRBasedSizer**: Position size based on Average True Range

### 3. Risk Limits (`risk_limits.py`)

Enforce portfolio-wide constraints:

- **MaxPositionLimit**: Maximum shares per position
- **MaxExposureLimit**: Maximum percentage exposure
- **MaxDrawdownLimit**: Stop trading or reduce size at drawdown threshold
- **VaRLimit**: Value at Risk constraints
- **ConcentrationLimit**: Prevent over-concentration in single positions
- **LeverageLimit**: Control overall leverage
- **DailyLossLimit**: Stop trading after daily loss threshold
- **SymbolRestrictionLimit**: Whitelist/blacklist symbols

### 4. Portfolio State (`portfolio_state.py`)

Tracks:
- Current positions and P&L
- Cash balance
- Performance history
- Risk metrics (exposure, drawdown, etc.)

### 5. Signal Processing (`signal_processing.py`)

Pipeline for converting signals to orders:
- Signal validation
- Position sizing
- Risk limit checking
- Order generation

## Event Flow

```
1. Strategy generates SIGNAL event
   {
     "signal_id": "sig_001",
     "strategy_id": "momentum_strategy",
     "symbol": "AAPL",
     "side": "BUY",
     "strength": 0.85
   }

2. Risk & Portfolio processes signal
   - Checks current portfolio state
   - Applies position sizing rules
   - Verifies risk limits
   - May veto signal if limits exceeded

3. If approved, generates ORDER event
   {
     "order_id": "ord_001",
     "signal_id": "sig_001",
     "symbol": "AAPL",
     "side": "BUY",
     "quantity": 100,
     "order_type": "MARKET"
   }

4. Receives FILL event from execution
   {
     "order_id": "ord_001",
     "price": 150.25,
     "quantity": 100,
     "commission": 1.0
   }

5. Updates portfolio state
   - Records position
   - Updates cash
   - Recalculates metrics
```

## Configuration

### Using RiskPortfolioCapability

Apply the capability to any container to transform it into a Risk & Portfolio container:

```python
from src.risk import RiskPortfolioCapability

# Apply to container
factory.apply_capability(
    container,
    RiskPortfolioCapability(),
    {
        'initial_capital': 100000,
        'position_sizers': [
            {
                'name': 'default',
                'type': 'percentage',
                'percentage': 2.0  # 2% per position
            },
            {
                'name': 'momentum',
                'type': 'volatility',
                'risk_per_trade': 1.0  # 1% risk per trade
            }
        ],
        'risk_limits': [
            {
                'type': 'position',
                'max_position': 5000
            },
            {
                'type': 'exposure', 
                'max_exposure_pct': 20  # 20% max total exposure
            },
            {
                'type': 'drawdown',
                'max_drawdown_pct': 10,  # 10% max drawdown
                'reduce_at_pct': 8       # Start reducing at 8%
            },
            {
                'type': 'concentration',
                'max_position_pct': 5,   # Max 5% in any position
                'max_sector_pct': 20     # Max 20% in any sector
            }
        ]
    }
)
```

### YAML Configuration Example

```yaml
risk_portfolio:
  initial_capital: 100000
  
  position_sizers:
    - name: default
      type: percentage
      percentage: 2.0
      
    - name: momentum
      type: volatility
      risk_per_trade: 1.0
      lookback_period: 20
      
    - name: mean_reversion
      type: fixed
      size: 100
  
  risk_limits:
    - type: position
      max_position: 5000
      
    - type: exposure
      max_exposure_pct: 20
      
    - type: drawdown
      max_drawdown_pct: 10
      reduce_at_pct: 8
      
    - type: daily_loss
      max_daily_loss: 2000
      max_daily_loss_pct: 2
      
    - type: symbol_restriction
      allowed_symbols: ["AAPL", "GOOGL", "MSFT", "SPY"]
      blocked_symbols: ["MEME", "PENNY"]
```

## Multi-Strategy Management

The Risk & Portfolio container manages multiple strategies, each generating signals independently:

```python
# Strategies generate signals
momentum_signal = {
    'strategy_id': 'momentum',
    'symbol': 'AAPL',
    'side': 'BUY',
    'strength': 0.9
}

mean_reversion_signal = {
    'strategy_id': 'mean_reversion',
    'symbol': 'SPY',
    'side': 'SELL',
    'strength': 0.7
}

# Risk & Portfolio processes both
# - Uses strategy-specific position sizers
# - Enforces portfolio-wide risk limits
# - Ensures total exposure stays within bounds
```

## Thread Safety

The module automatically handles thread safety based on ExecutionContext:

- **Backtest Mode**: Single-threaded, no locks needed
- **Live Trading**: Multi-threaded, automatic locking
- **Optimization**: Process-level parallelism

## Key Benefits

1. **Global Portfolio View**: Risk decisions based on entire portfolio, not individual positions
2. **Strategy Independence**: Strategies focus only on signal generation
3. **Flexible Risk Controls**: Mix and match position sizing and risk limits
4. **Clean Separation**: Clear boundary between alpha generation and risk management
5. **Reusability**: Same risk framework works for backtest and live trading

## Common Patterns

### Conservative Risk Profile
```python
{
    'position_sizers': [
        {'type': 'percentage', 'percentage': 1.0}  # 1% per position
    ],
    'risk_limits': [
        {'type': 'exposure', 'max_exposure_pct': 10},  # 10% max exposure
        {'type': 'drawdown', 'max_drawdown_pct': 5},   # 5% max drawdown
        {'type': 'concentration', 'max_position_pct': 2}  # 2% max per position
    ]
}
```

### Aggressive Risk Profile
```python
{
    'position_sizers': [
        {'type': 'kelly', 'kelly_fraction': 0.25}  # Kelly criterion
    ],
    'risk_limits': [
        {'type': 'exposure', 'max_exposure_pct': 50},  # 50% max exposure
        {'type': 'drawdown', 'max_drawdown_pct': 20},  # 20% max drawdown
        {'type': 'leverage', 'max_leverage': 2.0}      # 2x leverage allowed
    ]
}
```

### Sector-Neutral Profile
```python
{
    'position_sizers': [
        {'type': 'volatility', 'risk_per_trade': 0.5}  # Risk parity
    ],
    'risk_limits': [
        {'type': 'concentration', 'max_sector_pct': 15},  # Max 15% per sector
        {'type': 'exposure', 'max_exposure_pct': 30},     # 30% gross exposure
        {'type': 'symbol_restriction', 'allowed_symbols': sector_neutral_universe}
    ]
}
```

## Testing

Run the test suite:
```bash
pytest src/risk/test_risk_portfolio.py -v
```

Key test areas:
- Signal to order conversion
- Risk limit enforcement
- Portfolio state tracking
- Multi-strategy coordination
- Edge cases (insufficient capital, position limits, etc.)

## Future Enhancements

1. **Dynamic Risk Adjustment**: Adjust limits based on market conditions
2. **Correlation-Based Sizing**: Consider position correlations
3. **Options Support**: Handle multi-leg option strategies
4. **Real-time Risk Dashboard**: WebSocket-based risk monitoring
5. **Machine Learning Integration**: ML-based position sizing

## Summary

The Risk & Portfolio module is the heart of the trading system, sitting between signal generation and execution. By unifying risk management and portfolio tracking into a single component, it ensures that every trading decision considers the full portfolio context, enabling sophisticated multi-strategy trading while maintaining strict risk controls.