"""
Strategy-aware risk validators for per-strategy risk management.

These validators implement pure functions that consider strategy-specific
risk parameters and constraints. Each strategy can have different:
- Position sizing rules
- Exit criteria (max bars, MAE/MFE thresholds)
- Risk limits based on strategy performance
- Dynamic adjustment based on historical performance
"""

from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime, timedelta
import logging

from ..core.components.discovery import risk_validator

logger = logging.getLogger(__name__)


@risk_validator(
    name='strategy_aware_position_sizer',
    validation_types=['position_sizing', 'strategy_specific']
)
def calculate_strategy_position_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> float:
    """
    Calculate position size considering strategy-specific parameters.
    
    Args:
        signal: Trading signal with strategy_id and performance context
        portfolio_state: Current portfolio state
        risk_params: Risk parameters including strategy-specific configs
        market_data: Current market prices
        
    Returns:
        Position size adjusted for strategy characteristics
    """
    # Get strategy-specific risk parameters
    strategy_id = signal.get('strategy_id', 'default')
    strategy_params = risk_params.get('strategy_configs', {}).get(strategy_id, {})
    
    # Base position sizing parameters
    total_value = Decimal(str(portfolio_state.get('total_value', 0)))
    price = Decimal(str(market_data.get('close', market_data.get('price', 0))))
    
    if total_value <= 0 or price <= 0:
        return 0
    
    # Strategy-specific base position size
    base_position_pct = Decimal(str(strategy_params.get('base_position_percent', 
                                                        risk_params.get('base_position_percent', 0.02))))
    
    # Adjust by strategy type
    strategy_type = signal.get('strategy_type', 'unknown')
    position_multipliers = {
        'momentum': strategy_params.get('momentum_multiplier', 1.0),
        'mean_reversion': strategy_params.get('mean_reversion_multiplier', 0.8),
        'breakout': strategy_params.get('breakout_multiplier', 1.2),
        'ma_crossover': strategy_params.get('crossover_multiplier', 0.9),
    }
    
    strategy_multiplier = Decimal(str(position_multipliers.get(strategy_type, 1.0)))
    
    # Performance-based adjustment
    strategy_performance = portfolio_state.get('strategy_performance', {}).get(strategy_id, {})
    if strategy_performance:
        win_rate = strategy_performance.get('win_rate', 0.5)
        avg_return = strategy_performance.get('avg_return', 0.0)
        
        # Increase size for profitable strategies, decrease for unprofitable
        performance_multiplier = Decimal(str(min(2.0, max(0.5, 1.0 + avg_return))))
        
        # Reduce size if win rate is very low
        if win_rate < 0.3:
            performance_multiplier *= Decimal('0.7')
        elif win_rate > 0.7:
            performance_multiplier *= Decimal('1.3')
    else:
        performance_multiplier = Decimal('1.0')
    
    # Calculate final position size
    adjusted_position_pct = base_position_pct * strategy_multiplier * performance_multiplier
    position_value = total_value * adjusted_position_pct
    
    # Apply signal strength if available
    if 'strength' in signal and strategy_params.get('use_signal_strength', True):
        strength = Decimal(str(signal['strength']))
        position_value *= strength
    
    # Convert to shares
    shares = int(position_value / price)
    
    return max(0, shares)


@risk_validator(
    name='strategy_exit_validator',
    validation_types=['exit_criteria', 'strategy_specific']
)
def validate_exit_criteria(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate if exit criteria are met for strategy positions.
    
    Checks strategy-specific exit rules:
    - Maximum holding period (bars)
    - MAE (Maximum Adverse Excursion) limits
    - MFE (Maximum Favorable Excursion) targets
    - Profit taking thresholds
    """
    strategy_id = signal.get('strategy_id', 'default')
    symbol = signal.get('symbol', 'UNKNOWN')
    
    # Get current position
    positions = portfolio_state.get('positions', {})
    position = positions.get(symbol, {})
    
    if not position or position.get('quantity', 0) == 0:
        return {
            'should_exit': False,
            'reason': 'No position to exit',
            'exit_type': None
        }
    
    # Get strategy-specific exit parameters
    strategy_params = risk_params.get('strategy_configs', {}).get(strategy_id, {})
    exit_rules = strategy_params.get('exit_rules', {})
    
    current_price = Decimal(str(market_data.get('close', 0)))
    entry_price = Decimal(str(position.get('entry_price', 0)))
    entry_time = position.get('entry_time')
    current_time = market_data.get('timestamp', datetime.now())
    
    if entry_price <= 0:
        return {
            'should_exit': False,
            'reason': 'Invalid entry price',
            'exit_type': None
        }
    
    # Calculate current P&L
    quantity = position.get('quantity', 0)
    if quantity > 0:  # Long position
        unrealized_pnl = float((current_price - entry_price) / entry_price)
    else:  # Short position
        unrealized_pnl = float((entry_price - current_price) / entry_price)
    
    # Check maximum holding period
    max_bars = exit_rules.get('max_holding_bars')
    if max_bars and entry_time:
        bars_held = position.get('bars_held', 0)
        if bars_held >= max_bars:
            return {
                'should_exit': True,
                'reason': f'Maximum holding period reached: {bars_held} >= {max_bars} bars',
                'exit_type': 'time_based',
                'urgency': 'high'
            }
    
    # Check MAE (stop loss)
    max_adverse_pct = exit_rules.get('max_adverse_excursion_pct')
    if max_adverse_pct and unrealized_pnl < -max_adverse_pct:
        return {
            'should_exit': True,
            'reason': f'MAE limit hit: {unrealized_pnl:.2%} < -{max_adverse_pct:.2%}',
            'exit_type': 'stop_loss',
            'urgency': 'immediate'
        }
    
    # Check MFE (profit taking)
    min_favorable_pct = exit_rules.get('min_favorable_excursion_pct')
    if min_favorable_pct and unrealized_pnl > min_favorable_pct:
        # Check if we should take profits
        profit_take_pct = exit_rules.get('profit_take_at_mfe_pct', min_favorable_pct * 0.8)
        if unrealized_pnl >= profit_take_pct:
            return {
                'should_exit': True,
                'reason': f'Profit target reached: {unrealized_pnl:.2%} >= {profit_take_pct:.2%}',
                'exit_type': 'profit_taking',
                'urgency': 'normal'
            }
    
    # Check strategy-specific exit signals
    if 'exit_signal' in signal and signal['exit_signal']:
        exit_strength = signal.get('exit_strength', 1.0)
        min_exit_strength = exit_rules.get('min_exit_signal_strength', 0.5)
        
        if exit_strength >= min_exit_strength:
            return {
                'should_exit': True,
                'reason': f'Exit signal triggered with strength {exit_strength:.2f}',
                'exit_type': 'signal_based',
                'urgency': 'normal'
            }
    
    # No exit criteria met
    return {
        'should_exit': False,
        'reason': 'No exit criteria triggered',
        'exit_type': None,
        'metrics': {
            'unrealized_pnl_pct': unrealized_pnl,
            'bars_held': position.get('bars_held', 0),
            'max_bars_allowed': max_bars
        }
    }


@risk_validator(
    name='strategy_performance_adjuster',
    validation_types=['position_sizing', 'performance_based']
)
def adjust_size_by_performance(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> float:
    """
    Adjust position size based on strategy recent performance.
    
    Uses rolling window of recent trades to adjust position sizing:
    - Increase size for strategies with good recent performance
    - Decrease size for strategies with poor recent performance
    - Consider win rate, average return, and volatility
    """
    strategy_id = signal.get('strategy_id', 'default')
    
    # Get recent performance data
    strategy_performance = portfolio_state.get('strategy_performance', {}).get(strategy_id, {})
    recent_trades = strategy_performance.get('recent_trades', [])
    
    # Base position size calculation
    base_size = calculate_strategy_position_size(
        signal, portfolio_state, risk_params, market_data
    )
    
    if len(recent_trades) < 5:  # Not enough data for adjustment
        return base_size
    
    # Calculate recent performance metrics
    recent_returns = [trade.get('return_pct', 0) for trade in recent_trades[-20:]]  # Last 20 trades
    recent_win_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)
    avg_return = sum(recent_returns) / len(recent_returns)
    return_volatility = statistics.stdev(recent_returns) if len(recent_returns) > 1 else 0.1
    
    # Performance-based adjustment
    performance_params = risk_params.get('performance_adjustment', {})
    
    # Win rate adjustment
    target_win_rate = performance_params.get('target_win_rate', 0.5)
    win_rate_factor = min(1.5, max(0.5, recent_win_rate / target_win_rate))
    
    # Return adjustment
    if avg_return > 0:
        return_factor = min(1.5, 1.0 + avg_return * 10)  # Cap at 50% increase
    else:
        return_factor = max(0.3, 1.0 + avg_return * 5)  # Floor at 70% decrease
    
    # Volatility adjustment (reduce size for high volatility strategies)
    max_volatility = performance_params.get('max_volatility', 0.3)
    volatility_factor = max(0.5, 1.0 - (return_volatility / max_volatility) * 0.5)
    
    # Combine adjustments
    total_adjustment = win_rate_factor * return_factor * volatility_factor
    
    # Apply bounds
    min_adjustment = performance_params.get('min_adjustment_factor', 0.2)
    max_adjustment = performance_params.get('max_adjustment_factor', 2.0)
    total_adjustment = max(min_adjustment, min(max_adjustment, total_adjustment))
    
    adjusted_size = int(base_size * total_adjustment)
    
    logger.debug(f"Strategy {strategy_id} performance adjustment: "
                f"win_rate={recent_win_rate:.2f}, avg_return={avg_return:.3f}, "
                f"volatility={return_volatility:.3f}, adjustment={total_adjustment:.2f}")
    
    return max(0, adjusted_size)


@risk_validator(
    name='strategy_correlation_validator',
    validation_types=['position_sizing', 'correlation_based']
)
def validate_strategy_correlation(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate position size considering strategy correlation.
    
    Reduces position size when multiple correlated strategies are active
    to prevent over-concentration in similar market conditions.
    """
    strategy_id = signal.get('strategy_id', 'default')
    strategy_type = signal.get('strategy_type', 'unknown')
    symbol = signal.get('symbol', 'UNKNOWN')
    
    # Get current positions by strategy
    positions = portfolio_state.get('positions', {})
    strategy_positions = portfolio_state.get('strategy_positions', {})
    
    # Define strategy correlation matrix
    correlation_matrix = risk_params.get('strategy_correlations', {
        'momentum': {'momentum': 1.0, 'breakout': 0.7, 'trend_following': 0.8},
        'mean_reversion': {'mean_reversion': 1.0, 'rsi_strategy': 0.6},
        'breakout': {'breakout': 1.0, 'momentum': 0.7, 'volatility_breakout': 0.8},
        'ma_crossover': {'ma_crossover': 1.0, 'trend_following': 0.5}
    })
    
    # Calculate correlation exposure
    current_correlations = correlation_matrix.get(strategy_type, {})
    total_correlated_exposure = 0
    
    for other_strategy, positions_list in strategy_positions.items():
        if other_strategy == strategy_id:
            continue
            
        other_strategy_type = other_strategy.split('_')[0]  # Extract type from ID
        correlation = current_correlations.get(other_strategy_type, 0)
        
        if correlation > 0.3:  # Consider correlations above 30%
            exposure = sum(abs(pos.get('value', 0)) for pos in positions_list)
            total_correlated_exposure += exposure * correlation
    
    # Calculate correlation penalty
    total_portfolio_value = portfolio_state.get('total_value', 1)
    correlation_exposure_pct = total_correlated_exposure / total_portfolio_value
    
    max_correlation_exposure = risk_params.get('max_correlation_exposure', 0.3)
    
    if correlation_exposure_pct > max_correlation_exposure:
        penalty_factor = max_correlation_exposure / correlation_exposure_pct
        return {
            'approved': False,
            'reason': f'High correlation exposure: {correlation_exposure_pct:.1%} > {max_correlation_exposure:.1%}',
            'suggested_reduction': 1.0 - penalty_factor,
            'correlation_metrics': {
                'correlation_exposure_pct': correlation_exposure_pct,
                'max_allowed': max_correlation_exposure,
                'correlated_strategies': list(current_correlations.keys())
            }
        }
    
    return {
        'approved': True,
        'reason': 'Correlation exposure within limits',
        'correlation_metrics': {
            'correlation_exposure_pct': correlation_exposure_pct,
            'max_allowed': max_correlation_exposure
        }
    }


def create_strategy_risk_config(
    strategy_id: str,
    strategy_type: str,
    base_position_pct: float = 0.02,
    exit_rules: Optional[Dict[str, Any]] = None,
    performance_tracking: bool = True
) -> Dict[str, Any]:
    """
    Helper function to create strategy-specific risk configuration.
    
    Args:
        strategy_id: Unique strategy identifier
        strategy_type: Type of strategy (momentum, mean_reversion, etc.)
        base_position_pct: Base position size as percentage of portfolio
        exit_rules: Dictionary of exit criteria
        performance_tracking: Whether to track and adjust based on performance
        
    Returns:
        Strategy risk configuration dictionary
    """
    if exit_rules is None:
        # Default exit rules by strategy type
        exit_rules = {
            'momentum': {
                'max_holding_bars': 20,
                'max_adverse_excursion_pct': 0.05,
                'min_favorable_excursion_pct': 0.08,
                'profit_take_at_mfe_pct': 0.06
            },
            'mean_reversion': {
                'max_holding_bars': 10,
                'max_adverse_excursion_pct': 0.03,
                'min_favorable_excursion_pct': 0.04,
                'profit_take_at_mfe_pct': 0.03
            },
            'breakout': {
                'max_holding_bars': 50,
                'max_adverse_excursion_pct': 0.08,
                'min_favorable_excursion_pct': 0.15,
                'profit_take_at_mfe_pct': 0.12
            }
        }.get(strategy_type, {
            'max_holding_bars': 30,
            'max_adverse_excursion_pct': 0.05,
            'min_favorable_excursion_pct': 0.10,
            'profit_take_at_mfe_pct': 0.08
        })
    
    return {
        'strategy_id': strategy_id,
        'strategy_type': strategy_type,
        'base_position_percent': base_position_pct,
        'exit_rules': exit_rules,
        'performance_tracking': performance_tracking,
        'use_signal_strength': True,
        'correlation_sensitivity': 0.7,
        'performance_lookback_trades': 20
    }


# Example usage and configuration templates
STRATEGY_RISK_TEMPLATES = {
    'aggressive_momentum': {
        'base_position_percent': 0.04,
        'momentum_multiplier': 1.5,
        'exit_rules': {
            'max_holding_bars': 30,
            'max_adverse_excursion_pct': 0.08,
            'min_favorable_excursion_pct': 0.15
        }
    },
    'conservative_mean_reversion': {
        'base_position_percent': 0.015,
        'mean_reversion_multiplier': 0.8,
        'exit_rules': {
            'max_holding_bars': 8,
            'max_adverse_excursion_pct': 0.025,
            'min_favorable_excursion_pct': 0.03
        }
    },
    'scalping_breakout': {
        'base_position_percent': 0.01,
        'breakout_multiplier': 2.0,
        'exit_rules': {
            'max_holding_bars': 5,
            'max_adverse_excursion_pct': 0.015,
            'min_favorable_excursion_pct': 0.02
        }
    }
}