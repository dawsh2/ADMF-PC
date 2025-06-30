"""Optimization modules for stops, targets, and parameters."""

from .stop_loss import (
    optimize_stop_loss,
    backtest_with_stops,
    analyze_stop_effectiveness
)

from .targets import (
    optimize_profit_target,
    optimize_stop_target_grid,
    calculate_risk_reward_ratio
)

from .regime_specific import (
    optimize_stops_by_regime,
    get_regime_optimal_params,
    apply_regime_stops
)

__all__ = [
    # Stop loss
    'optimize_stop_loss',
    'backtest_with_stops',
    'analyze_stop_effectiveness',
    
    # Targets
    'optimize_profit_target',
    'optimize_stop_target_grid',
    'calculate_risk_reward_ratio',
    
    # Regime specific
    'optimize_stops_by_regime',
    'get_regime_optimal_params',
    'apply_regime_stops'
]