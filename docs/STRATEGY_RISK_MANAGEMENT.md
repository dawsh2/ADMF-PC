# Strategy-Specific Risk Management

This document describes the enhanced risk management system that supports per-strategy risk constraints, exit criteria, and performance-based adjustments.

## Overview

The strategy-aware risk module extends the existing stateless risk architecture to support:

- **Strategy-specific position sizing** based on strategy type and performance
- **Dynamic exit criteria** including MAE/MFE thresholds and time-based exits
- **Performance-driven adjustments** that adapt risk based on recent strategy performance
- **Correlation-aware position sizing** to prevent over-concentration
- **Data-mining resistant design** with configurable exit criteria validation

## Key Components

### 1. Strategy Risk Profiles (`StrategyRiskProfile`)

Each strategy can have its own risk profile containing:

```python
@dataclass
class StrategyRiskProfile:
    strategy_id: str
    strategy_type: StrategyType  # momentum, mean_reversion, breakout, etc.
    
    position_sizing: PositionSizingRules
    exit_rules: ExitRules
    performance_tracking: PerformanceTracking
    correlation_matrix: Dict[str, float]
```

### 2. Position Sizing Rules

Configure how much capital to allocate to each strategy:

```python
position_sizing = PositionSizingRules(
    base_position_percent=0.02,        # 2% of portfolio per trade
    max_position_percent=0.10,         # Never exceed 10%
    strategy_type_multiplier=1.2,      # Adjust by strategy type
    use_signal_strength=True,          # Scale by signal confidence
    performance_adjustment_factor=0.5   # How much to adjust based on recent performance
)
```

### 3. Exit Criteria

Define when to exit positions:

```python
exit_rules = ExitRules(
    max_holding_bars=20,               # Exit after 20 bars max
    max_adverse_excursion_pct=0.05,    # Stop loss at -5%
    min_favorable_excursion_pct=0.08,  # Profit target at +8%
    profit_take_at_mfe_pct=0.06,       # Take partial profits at +6%
    min_exit_signal_strength=0.5       # Require strong exit signals
)
```

### 4. Performance Tracking

Monitor and adapt to strategy performance:

```python
performance_tracking = PerformanceTracking(
    short_term_window=10,              # Recent performance (10 trades)
    medium_term_window=50,             # Medium term (50 trades)
    track_win_rate=True,               # Monitor success rate
    track_avg_return=True,             # Monitor profitability
    track_return_volatility=True,      # Monitor consistency
    performance_review_frequency=20    # Adjust every 20 trades
)
```

## Usage Examples

### Basic Setup

```python
from src.risk.strategy_risk_manager import StrategyRiskManager
from src.risk.strategy_risk_config import StrategyType

# Initialize the risk manager
risk_manager = StrategyRiskManager()

# Create a risk profile for a momentum strategy
momentum_profile = risk_manager.config_manager.create_profile(
    strategy_id='momentum_scalp_v1',
    strategy_type=StrategyType.MOMENTUM,
    template='aggressive_momentum',  # Use a predefined template
    base_position_percent=0.03,      # Override: use 3% position size
)

# Validate a trading signal
signal = {
    'strategy_id': 'momentum_scalp_v1',
    'symbol': 'AAPL',
    'direction': 'long',
    'strength': 0.8,
    'price': 150.0
}

portfolio_state = {
    'total_value': 100000,
    'positions': {},
    'cash': 100000
}

market_data = {
    'close': 150.0,
    'timestamp': datetime.now()
}

# Validate the signal
validation_result = risk_manager.validate_signal(
    signal, portfolio_state, market_data
)

if validation_result['approved']:
    # Calculate position size
    position_size = risk_manager.calculate_position_size(
        signal, portfolio_state, market_data
    )
    print(f"Position size: {position_size} shares")
else:
    print(f"Signal rejected: {validation_result['reason']}")
```

### Creating Profiles from Backtest Results

```python
# Analyze backtest results and create optimal risk profile
backtest_results = {
    'trades': [
        {'mae_pct': -0.03, 'mfe_pct': 0.05, 'duration_bars': 15, 'return_pct': 0.02},
        {'mae_pct': -0.02, 'mfe_pct': 0.08, 'duration_bars': 22, 'return_pct': 0.06},
        # ... more trades
    ],
    'metrics': {
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.08
    }
}

# Create optimized profile
optimized_profile = risk_manager.create_risk_profile_from_backtest(
    strategy_id='mean_reversion_optimized',
    strategy_type='mean_reversion',
    backtest_results=backtest_results,
    risk_tolerance='moderate'  # or 'conservative' / 'aggressive'
)
```

### Real-time Exit Criteria Checking

```python
# Check if current position should be exited
position = {
    'symbol': 'AAPL',
    'quantity': 100,
    'entry_price': 148.0,
    'entry_time': datetime.now() - timedelta(minutes=30),
    'strategy_id': 'momentum_scalp_v1',
    'bars_held': 5
}

current_signal = {
    'strategy_id': 'momentum_scalp_v1',
    'symbol': 'AAPL',
    'exit_signal': True,
    'exit_strength': 0.7
}

market_data = {
    'close': 151.5,  # Currently profitable
    'timestamp': datetime.now()
}

# Check exit criteria
exit_recommendation = risk_manager.check_exit_criteria(
    position, current_signal, portfolio_state, market_data
)

if exit_recommendation['should_exit']:
    print(f"Exit recommended: {exit_recommendation['reason']}")
    print(f"Exit type: {exit_recommendation['exit_type']}")
    print(f"Urgency: {exit_recommendation['urgency']}")
```

### Performance-Based Position Adjustment

```python
# Simulate updating strategy performance after a trade
trade_result = {
    'strategy_id': 'momentum_scalp_v1',
    'return_pct': 0.035,  # 3.5% return
    'pnl': 525.0,         # $525 profit
    'duration_bars': 12,  # Held for 12 bars
    'mae_pct': -0.015,    # Max adverse: -1.5%
    'mfe_pct': 0.042,     # Max favorable: +4.2%
    'exit_type': 'profit_taking',
    'exit_time': datetime.now()
}

# Update performance tracking
risk_manager.update_strategy_performance(
    'momentum_scalp_v1', trade_result
)

# Next position will be sized considering this performance
next_signal = {
    'strategy_id': 'momentum_scalp_v1',
    'symbol': 'MSFT',
    'direction': 'long',
    'strength': 0.9
}

# Position size will be adjusted based on recent performance
adjusted_size = risk_manager.calculate_position_size(
    next_signal, portfolio_state, market_data,
    use_performance_adjustment=True
)
```

## Configuration Templates

### Aggressive Momentum Strategy

```python
aggressive_momentum = {
    'base_position_percent': 0.04,     # 4% positions
    'max_position_percent': 0.15,      # Up to 15% max
    'strategy_type_multiplier': 1.5,   # 50% larger positions
    'exit_rules': {
        'max_holding_bars': 30,        # Hold up to 30 bars
        'max_adverse_excursion_pct': 0.08,  # -8% stop loss
        'min_favorable_excursion_pct': 0.15, # +15% profit target
        'profit_take_at_mfe_pct': 0.12      # Take profits at +12%
    }
}
```

### Conservative Mean Reversion

```python
conservative_mean_reversion = {
    'base_position_percent': 0.015,    # 1.5% positions
    'max_position_percent': 0.08,      # Max 8%
    'strategy_type_multiplier': 0.8,   # 20% smaller positions
    'exit_rules': {
        'max_holding_bars': 8,         # Quick exits
        'max_adverse_excursion_pct': 0.025,  # -2.5% stop
        'min_favorable_excursion_pct': 0.03,  # +3% target
        'profit_take_at_mfe_pct': 0.025      # Take profits at +2.5%
    }
}
```

### Scalping Breakout

```python
scalping_breakout = {
    'base_position_percent': 0.01,     # 1% positions
    'max_position_percent': 0.05,      # Max 5%
    'strategy_type_multiplier': 2.0,   # 2x larger positions for strong signals
    'use_signal_strength': True,       # Scale heavily by signal strength
    'exit_rules': {
        'max_holding_bars': 5,         # Very quick exits
        'max_adverse_excursion_pct': 0.015,  # -1.5% stop
        'min_favorable_excursion_pct': 0.02,  # +2% target
        'min_exit_signal_strength': 0.8      # Strong exit signals required
    }
}
```

## Integration with Existing Architecture

The strategy-aware risk system integrates seamlessly with the existing stateless architecture:

1. **Stateless Validators**: All risk functions remain pure with no internal state
2. **Portfolio State Integration**: Risk parameters are injected into existing portfolio state
3. **Signal Enhancement**: Signals are enhanced with strategy context for validation
4. **Performance Tracking**: Separate tracking maintains strategy performance history

## Benefits for Data Mining Prevention

1. **Configurable Exit Criteria**: Prevents overfitting by using configurable rather than optimized exits
2. **Performance-Based Adaptation**: Reduces position size for strategies showing poor recent performance
3. **Correlation Awareness**: Prevents over-allocation to similar strategies
4. **Time-Based Exits**: Forces position turnover to prevent indefinite holding
5. **MAE/MFE Validation**: Uses realistic loss/profit expectations based on historical data

## Configuration File Format

Risk profiles can be saved/loaded as JSON:

```json
{
  "profiles": {
    "momentum_v1": {
      "strategy_id": "momentum_v1",
      "strategy_type": "momentum",
      "position_sizing": {
        "base_position_percent": 0.03,
        "max_position_percent": 0.12,
        "use_signal_strength": true
      },
      "exit_rules": {
        "max_holding_bars": 25,
        "max_adverse_excursion_pct": 0.06,
        "min_favorable_excursion_pct": 0.10
      }
    }
  }
}
```

This approach provides a robust, configurable risk management system that can adapt to different strategy characteristics while maintaining the benefits of the stateless architecture.