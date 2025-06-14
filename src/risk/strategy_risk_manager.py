"""
Strategy-aware risk manager that integrates with the existing stateless risk system.

This manager coordinates strategy-specific risk validation and position sizing
while maintaining the stateless architecture principles.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from .strategy_risk_config import StrategyRiskConfigManager, StrategyRiskProfile
from .strategy_aware_validators import (
    calculate_strategy_position_size,
    validate_exit_criteria,
    adjust_size_by_performance,
    validate_strategy_correlation
)
from .validators import (
    validate_max_position,
    validate_drawdown,
    calculate_position_size
)

logger = logging.getLogger(__name__)


class StrategyRiskManager:
    """
    Coordinates strategy-specific risk management.
    
    Maintains strategy risk profiles and coordinates validation
    while preserving stateless architecture.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize strategy risk manager.
        
        Args:
            config_file: Optional path to load risk configurations from
        """
        self.config_manager = StrategyRiskConfigManager()
        
        # Load configurations if provided
        if config_file:
            try:
                self.config_manager.load_from_file(config_file)
                logger.info(f"Loaded strategy risk configurations from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        
        # Registry of available validators
        self.validators = {
            'position_size': calculate_strategy_position_size,
            'max_position': validate_max_position,
            'drawdown': validate_drawdown,
            'exit_criteria': validate_exit_criteria,
            'performance_adjustment': adjust_size_by_performance,
            'correlation': validate_strategy_correlation
        }
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.recent_trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def validate_signal(
        self,
        signal: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_data: Dict[str, Any],
        validation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate a trading signal using strategy-specific risk rules.
        
        Args:
            signal: Trading signal with strategy context
            portfolio_state: Current portfolio state
            market_data: Current market data
            validation_types: Optional list of specific validations to run
            
        Returns:
            Validation result with approval status and metrics
        """
        strategy_id = signal.get('strategy_id', 'default')
        
        # Get risk parameters for this strategy
        risk_params = self.config_manager.get_risk_params_for_strategy(strategy_id)
        
        # Add current performance data to portfolio state
        enhanced_portfolio_state = portfolio_state.copy()
        enhanced_portfolio_state['strategy_performance'] = self.strategy_performance
        
        # Default validation types
        if validation_types is None:
            validation_types = ['position_size', 'max_position', 'drawdown', 'correlation']
        
        # Run validations
        results = {}
        overall_approved = True
        combined_metrics = {}
        
        for validation_type in validation_types:
            if validation_type not in self.validators:
                logger.warning(f"Unknown validation type: {validation_type}")
                continue
            
            validator = self.validators[validation_type]
            
            try:
                if validation_type == 'position_size':
                    # Position sizing returns a number
                    size = validator(signal, enhanced_portfolio_state, risk_params, market_data)
                    results[validation_type] = {
                        'approved': size > 0,
                        'position_size': size,
                        'reason': 'Position size calculated' if size > 0 else 'Zero position size calculated'
                    }
                else:
                    # Other validators return validation dictionaries
                    result = validator(signal, enhanced_portfolio_state, risk_params, market_data)
                    results[validation_type] = result
                    
                    if not result.get('approved', True):
                        overall_approved = False
                    
                    # Collect metrics
                    if 'risk_metrics' in result:
                        combined_metrics.update(result['risk_metrics'])
                    if 'correlation_metrics' in result:
                        combined_metrics.update(result['correlation_metrics'])
                        
            except Exception as e:
                logger.error(f"Error in {validation_type} validation: {e}")
                results[validation_type] = {
                    'approved': False,
                    'reason': f'Validation error: {str(e)}'
                }
                overall_approved = False
        
        # Aggregate results
        failed_validations = [vtype for vtype, result in results.items() 
                            if not result.get('approved', True)]
        
        return {
            'approved': overall_approved,
            'strategy_id': strategy_id,
            'failed_validations': failed_validations,
            'validation_details': results,
            'risk_metrics': combined_metrics,
            'reason': 'All validations passed' if overall_approved else f"Failed: {', '.join(failed_validations)}"
        }
    
    def calculate_position_size(
        self,
        signal: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_data: Dict[str, Any],
        use_performance_adjustment: bool = True
    ) -> float:
        """
        Calculate optimal position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            market_data: Current market data
            use_performance_adjustment: Whether to apply performance-based adjustments
            
        Returns:
            Recommended position size
        """
        strategy_id = signal.get('strategy_id', 'default')
        risk_params = self.config_manager.get_risk_params_for_strategy(strategy_id)
        
        # Enhanced portfolio state with performance data
        enhanced_portfolio_state = portfolio_state.copy()
        enhanced_portfolio_state['strategy_performance'] = self.strategy_performance
        
        # Calculate base size
        base_size = calculate_strategy_position_size(
            signal, enhanced_portfolio_state, risk_params, market_data
        )
        
        if not use_performance_adjustment or base_size == 0:
            return base_size
        
        # Apply performance adjustment
        try:
            adjusted_size = adjust_size_by_performance(
                signal, enhanced_portfolio_state, risk_params, market_data
            )
            return adjusted_size
        except Exception as e:
            logger.error(f"Error in performance adjustment: {e}")
            return base_size
    
    def check_exit_criteria(
        self,
        position: Dict[str, Any],
        signal: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if exit criteria are met for a position.
        
        Args:
            position: Current position information
            signal: Current signal (may contain exit signals)
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Exit recommendation with reasoning
        """
        strategy_id = position.get('strategy_id', signal.get('strategy_id', 'default'))
        risk_params = self.config_manager.get_risk_params_for_strategy(strategy_id)
        
        # Add position info to signal for validation
        exit_signal = signal.copy()
        exit_signal.update({
            'strategy_id': strategy_id,
            'symbol': position.get('symbol', signal.get('symbol'))
        })
        
        return validate_exit_criteria(
            exit_signal, portfolio_state, risk_params, market_data
        )
    
    def update_strategy_performance(
        self,
        strategy_id: str,
        trade_result: Dict[str, Any]
    ):
        """
        Update performance tracking for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            trade_result: Trade outcome with P&L, duration, etc.
        """
        # Add to recent trades
        self.recent_trades[strategy_id].append({
            'timestamp': trade_result.get('exit_time', datetime.now()),
            'return_pct': trade_result.get('return_pct', 0.0),
            'pnl': trade_result.get('pnl', 0.0),
            'duration_bars': trade_result.get('duration_bars', 0),
            'mae_pct': trade_result.get('mae_pct', 0.0),
            'mfe_pct': trade_result.get('mfe_pct', 0.0),
            'exit_type': trade_result.get('exit_type', 'unknown')
        })
        
        # Keep only recent trades (configurable window)
        profile = self.config_manager.get_profile(strategy_id)
        max_trades = 200
        if profile:
            max_trades = profile.performance_tracking.long_term_window
        
        if len(self.recent_trades[strategy_id]) > max_trades:
            self.recent_trades[strategy_id] = self.recent_trades[strategy_id][-max_trades:]
        
        # Calculate performance metrics
        self._calculate_performance_metrics(strategy_id)
    
    def _calculate_performance_metrics(self, strategy_id: str):
        """Calculate and cache performance metrics for a strategy."""
        trades = self.recent_trades.get(strategy_id, [])
        
        if len(trades) < 5:  # Not enough data
            return
        
        # Calculate metrics for different windows
        windows = [10, 20, 50]  # Short, medium, long term
        
        for window in windows:
            recent_trades = trades[-window:] if len(trades) >= window else trades
            
            if not recent_trades:
                continue
            
            returns = [t['return_pct'] for t in recent_trades]
            
            metrics = {
                'trade_count': len(recent_trades),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'avg_return': sum(returns) / len(returns),
                'total_return': sum(returns),
                'best_trade': max(returns),
                'worst_trade': min(returns),
                'avg_duration': sum(t['duration_bars'] for t in recent_trades) / len(recent_trades)
            }
            
            # Volatility calculation
            if len(returns) > 1:
                mean_return = metrics['avg_return']
                variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
                metrics['volatility'] = variance ** 0.5
            else:
                metrics['volatility'] = 0.0
            
            # MAE/MFE analysis
            mae_values = [t['mae_pct'] for t in recent_trades if t['mae_pct'] != 0]
            mfe_values = [t['mfe_pct'] for t in recent_trades if t['mfe_pct'] != 0]
            
            if mae_values:
                metrics['avg_mae'] = sum(mae_values) / len(mae_values)
                metrics['max_mae'] = max(mae_values)
            
            if mfe_values:
                metrics['avg_mfe'] = sum(mfe_values) / len(mfe_values)
                metrics['max_mfe'] = max(mfe_values)
            
            # Exit type analysis
            exit_types = defaultdict(int)
            for trade in recent_trades:
                exit_types[trade['exit_type']] += 1
            
            metrics['exit_type_distribution'] = dict(exit_types)
            
            # Store metrics
            window_key = f'window_{window}'
            if strategy_id not in self.strategy_performance:
                self.strategy_performance[strategy_id] = {}
            
            self.strategy_performance[strategy_id][window_key] = metrics
        
        # Store recent trades for validators
        self.strategy_performance[strategy_id]['recent_trades'] = trades[-50:]  # Keep 50 most recent
    
    def get_strategy_summary(self, strategy_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of strategy risk profile and performance."""
        profile = self.config_manager.get_profile(strategy_id)
        performance = self.strategy_performance.get(strategy_id, {})
        
        summary = {
            'strategy_id': strategy_id,
            'profile_exists': profile is not None,
            'performance_data_available': bool(performance),
            'recent_trade_count': len(self.recent_trades.get(strategy_id, []))
        }
        
        if profile:
            summary.update({
                'strategy_type': profile.strategy_type.value,
                'base_position_pct': profile.position_sizing.base_position_percent,
                'max_holding_bars': profile.exit_rules.max_holding_bars,
                'stop_loss_pct': profile.exit_rules.max_adverse_excursion_pct,
                'profit_target_pct': profile.exit_rules.min_favorable_excursion_pct
            })
        
        if performance:
            # Use the most comprehensive window available
            for window in ['window_50', 'window_20', 'window_10']:
                if window in performance:
                    window_perf = performance[window]
                    summary.update({
                        'recent_win_rate': window_perf.get('win_rate', 0),
                        'recent_avg_return': window_perf.get('avg_return', 0),
                        'recent_volatility': window_perf.get('volatility', 0),
                        'avg_trade_duration': window_perf.get('avg_duration', 0),
                        'performance_window': window
                    })
                    break
        
        return summary
    
    def create_risk_profile_from_backtest(
        self,
        strategy_id: str,
        strategy_type: str,
        backtest_results: Dict[str, Any],
        risk_tolerance: str = 'moderate'
    ) -> StrategyRiskProfile:
        """
        Create a risk profile based on backtest results.
        
        Args:
            strategy_id: Strategy identifier
            strategy_type: Type of strategy
            backtest_results: Historical performance data
            risk_tolerance: 'aggressive', 'moderate', or 'conservative'
            
        Returns:
            New StrategyRiskProfile optimized for the strategy's characteristics
        """
        # Analyze backtest results
        trades = backtest_results.get('trades', [])
        metrics = backtest_results.get('metrics', {})
        
        # Calculate optimal exit criteria based on MAE/MFE distribution
        mae_values = [t.get('mae_pct', 0) for t in trades if t.get('mae_pct', 0) < 0]
        mfe_values = [t.get('mfe_pct', 0) for t in trades if t.get('mfe_pct', 0) > 0]
        durations = [t.get('duration_bars', 0) for t in trades]
        
        # Risk tolerance adjustments
        tolerance_multipliers = {
            'conservative': {'position': 0.7, 'stop': 0.8, 'target': 0.8, 'holding': 0.8},
            'moderate': {'position': 1.0, 'stop': 1.0, 'target': 1.0, 'holding': 1.0},
            'aggressive': {'position': 1.3, 'stop': 1.2, 'target': 1.2, 'holding': 1.2}
        }
        
        multipliers = tolerance_multipliers.get(risk_tolerance, tolerance_multipliers['moderate'])
        
        # Calculate optimal parameters
        base_position_pct = min(0.05, max(0.005, 0.02 * multipliers['position']))
        
        if mae_values:
            # Set stop loss at 90th percentile of MAE
            mae_sorted = sorted(mae_values)
            stop_loss_pct = abs(mae_sorted[int(len(mae_sorted) * 0.9)]) * multipliers['stop']
        else:
            stop_loss_pct = 0.05 * multipliers['stop']
        
        if mfe_values:
            # Set profit target at 70th percentile of MFE
            mfe_sorted = sorted(mfe_values, reverse=True)
            profit_target_pct = mfe_sorted[int(len(mfe_sorted) * 0.3)] * multipliers['target']
        else:
            profit_target_pct = 0.08 * multipliers['target']
        
        if durations:
            # Set max holding period at 80th percentile
            duration_sorted = sorted(durations)
            max_holding_bars = int(duration_sorted[int(len(duration_sorted) * 0.8)] * multipliers['holding'])
        else:
            max_holding_bars = 30
        
        # Create the profile
        profile = self.config_manager.create_profile(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            base_position_percent=base_position_pct,
            exit_rules={
                'max_holding_bars': max_holding_bars,
                'max_adverse_excursion_pct': stop_loss_pct,
                'min_favorable_excursion_pct': profit_target_pct,
                'profit_take_at_mfe_pct': profit_target_pct * 0.8
            }
        )
        
        logger.info(f"Created risk profile for {strategy_id}: "
                   f"position={base_position_pct:.3f}, stop={stop_loss_pct:.3f}, "
                   f"target={profit_target_pct:.3f}, max_bars={max_holding_bars}")
        
        return profile


# Example integration with existing risk workflow
def integrate_with_portfolio_manager(
    strategy_risk_manager: StrategyRiskManager,
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[int], Dict[str, Any]]:
    """
    Example integration function showing how to use strategy-specific risk
    management within the existing portfolio workflow.
    
    Returns:
        (approved, position_size, risk_info)
    """
    # Validate the signal
    validation_result = strategy_risk_manager.validate_signal(
        signal, portfolio_state, market_data
    )
    
    if not validation_result['approved']:
        return False, None, validation_result
    
    # Calculate position size
    position_size = strategy_risk_manager.calculate_position_size(
        signal, portfolio_state, market_data
    )
    
    if position_size <= 0:
        return False, None, {
            'approved': False,
            'reason': 'Zero position size calculated',
            'validation_result': validation_result
        }
    
    return True, position_size, {
        'approved': True,
        'validation_result': validation_result,
        'position_size': position_size,
        'strategy_id': signal.get('strategy_id', 'default')
    }