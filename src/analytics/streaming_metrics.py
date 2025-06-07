"""
Streaming metrics calculation for memory-efficient performance tracking.

This component calculates metrics without storing full history, using
Welford's algorithm for numerically stable running statistics.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import math
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamingMetrics:
    """
    Calculate performance metrics without storing full history.
    
    Uses Welford's algorithm for numerically stable calculation of
    mean and variance without storing all data points.
    
    Attributes:
        total_return: Cumulative return
        n_trades: Number of trades executed
        winning_trades: Number of profitable trades
        max_drawdown: Maximum peak-to-trough decline
        current_drawdown: Current drawdown from peak
        peak_value: Historical peak portfolio value
        n_returns: Number of return observations
        mean_return: Running mean of returns
        m2: Sum of squares of differences (for variance)
        sum_returns: Sum of all returns (for total return)
    """
    
    # Performance tracking
    total_return: float = 0.0
    n_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Drawdown tracking
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_value: float = 0.0
    
    # Return statistics (Welford's algorithm)
    n_returns: int = 0
    mean_return: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from mean
    sum_returns: float = 0.0
    
    # Additional metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    # Time tracking
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    # Configuration
    annualization_factor: float = 252.0  # Default for daily returns
    min_periods: int = 20  # Minimum periods for meaningful statistics
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.start_time is None:
            self.start_time = datetime.now()
        self.last_update = self.start_time
    
    def update(self, returns: float, portfolio_value: float, trade_pnl: Optional[float] = None) -> None:
        """
        Update metrics with new return data.
        
        Args:
            returns: Period return (as decimal, e.g., 0.01 for 1%)
            portfolio_value: Current portfolio value
            trade_pnl: P&L from trades in this period (optional)
        """
        self.last_update = datetime.now()
        
        # Update return statistics using Welford's algorithm
        self.n_returns += 1
        self.sum_returns += returns
        
        # Welford's algorithm for running mean and variance
        delta = returns - self.mean_return
        self.mean_return += delta / self.n_returns
        delta2 = returns - self.mean_return
        self.m2 += delta * delta2
        
        # Update total return
        # Using log returns for better numerical stability
        # total_return = exp(sum(log(1 + r))) - 1
        self.total_return = (1 + self.total_return) * (1 + returns) - 1
        
        # Update drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Update trade statistics if trade occurred
        if trade_pnl is not None and trade_pnl != 0:
            self.n_trades += 1
            self.total_pnl += trade_pnl
            
            if trade_pnl > 0:
                self.winning_trades += 1
                self.gross_profit += trade_pnl
            else:
                self.losing_trades += 1
                self.gross_loss += abs(trade_pnl)
    
    def update_from_trade(self, trade_result: Dict[str, Any]) -> None:
        """
        Update metrics from a trade result.
        
        Args:
            trade_result: Dict containing trade information
                - pnl: Trade P&L
                - return: Trade return
                - portfolio_value: Portfolio value after trade
        """
        pnl = trade_result.get('pnl', 0.0)
        returns = trade_result.get('return', 0.0)
        portfolio_value = trade_result.get('portfolio_value', self.peak_value)
        
        self.update(returns, portfolio_value, pnl)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metric values.
        
        Returns:
            Dictionary of calculated metrics
        """
        # Calculate variance and standard deviation
        variance = self.m2 / (self.n_returns - 1) if self.n_returns > 1 else 0.0
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        
        # Calculate Sharpe ratio (annualized)
        if std_dev > 0 and self.n_returns >= self.min_periods:
            sharpe_ratio = (self.mean_return / std_dev) * math.sqrt(self.annualization_factor)
        else:
            sharpe_ratio = 0.0
        
        # Calculate win rate
        win_rate = self.winning_trades / self.n_trades if self.n_trades > 0 else 0.0
        
        # Calculate profit factor
        profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf')
        if profit_factor == float('inf') and self.gross_profit == 0:
            profit_factor = 0.0
        
        # Calculate average trade metrics
        avg_win = self.gross_profit / self.winning_trades if self.winning_trades > 0 else 0.0
        avg_loss = self.gross_loss / self.losing_trades if self.losing_trades > 0 else 0.0
        avg_trade = self.total_pnl / self.n_trades if self.n_trades > 0 else 0.0
        
        # Calculate expectancy (edge per trade)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if self.n_trades > 0 else 0.0
        
        # Time-based metrics
        days_active = (self.last_update - self.start_time).days if self.start_time else 0
        
        return {
            # Core performance metrics
            'sharpe_ratio': sharpe_ratio,
            'total_return': self.total_return,
            'annualized_return': self._annualize_return(self.total_return, days_active),
            'volatility': std_dev * math.sqrt(self.annualization_factor),
            
            # Drawdown metrics
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            
            # Trade statistics
            'total_trades': self.n_trades,
            'win_rate': win_rate,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            
            # P&L metrics
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'average_trade': avg_trade,
            
            # Statistical metrics
            'mean_return': self.mean_return,
            'std_dev': std_dev,
            'variance': variance,
            'n_periods': self.n_returns,
            
            # Sortino ratio (downside deviation)
            'sortino_ratio': self._calculate_sortino_ratio(),
            
            # Calmar ratio (return / max drawdown)
            'calmar_ratio': self.total_return / self.max_drawdown if self.max_drawdown > 0 else 0.0,
        }
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get key summary metrics only.
        
        Returns:
            Subset of most important metrics
        """
        full_metrics = self.get_metrics()
        
        return {
            'sharpe_ratio': full_metrics['sharpe_ratio'],
            'total_return': full_metrics['total_return'],
            'max_drawdown': full_metrics['max_drawdown'],
            'win_rate': full_metrics['win_rate'],
            'profit_factor': full_metrics['profit_factor'],
            'total_trades': full_metrics['total_trades'],
        }
    
    def _annualize_return(self, total_return: float, days: int) -> float:
        """
        Annualize a return based on number of days.
        
        Args:
            total_return: Total return as decimal
            days: Number of days
            
        Returns:
            Annualized return
        """
        if days <= 0:
            return 0.0
        
        years = days / 365.25
        if years <= 0:
            return 0.0
        
        # Compound annual growth rate
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_sortino_ratio(self) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Note: This is a simplified version. Full implementation would
        track downside returns separately.
        """
        # For now, return 0 as we'd need to track negative returns separately
        # This could be enhanced by maintaining separate downside statistics
        return 0.0
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.__init__(
            annualization_factor=self.annualization_factor,
            min_periods=self.min_periods
        )
    
    def merge(self, other: 'StreamingMetrics') -> 'StreamingMetrics':
        """
        Merge metrics from another instance.
        
        Useful for combining metrics from multiple portfolios.
        
        Args:
            other: Another StreamingMetrics instance
            
        Returns:
            New StreamingMetrics with combined data
        """
        # This is a simplified merge - full implementation would
        # properly combine Welford statistics
        merged = StreamingMetrics(
            annualization_factor=self.annualization_factor,
            min_periods=self.min_periods
        )
        
        # Simple averaging for now
        merged.total_return = (self.total_return + other.total_return) / 2
        merged.n_trades = self.n_trades + other.n_trades
        merged.winning_trades = self.winning_trades + other.winning_trades
        merged.losing_trades = self.losing_trades + other.losing_trades
        merged.max_drawdown = max(self.max_drawdown, other.max_drawdown)
        merged.total_pnl = self.total_pnl + other.total_pnl
        merged.gross_profit = self.gross_profit + other.gross_profit
        merged.gross_loss = self.gross_loss + other.gross_loss
        
        return merged


class PortfolioMetricsTracker:
    """
    Helper class to track portfolio metrics using StreamingMetrics.
    
    Integrates with portfolio state updates to automatically calculate metrics.
    Conditionally tracks equity curve based on results configuration.
    """
    
    def __init__(self, initial_capital: float = 100000.0, results_config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize portfolio metrics tracker.
        
        Args:
            initial_capital: Starting portfolio value
            results_config: Results collection configuration
            **kwargs: Additional arguments for StreamingMetrics
        """
        self.initial_capital = initial_capital
        self.previous_value = initial_capital
        self.metrics = StreamingMetrics(**kwargs)
        
        # Results configuration
        self.results_config = results_config or {}
        collection_config = self.results_config.get('collection', {})
        
        # Equity curve tracking configuration
        self.store_equity_curve = collection_config.get('store_equity_curve', False)
        self.snapshot_interval = collection_config.get('snapshot_interval', 100)
        self.max_equity_points = collection_config.get('max_equity_points', 10000)
        
        # Equity curve storage (only if enabled)
        self.equity_curve: List[Dict[str, Any]] = [] if self.store_equity_curve else None
        self.update_counter = 0
        self.last_snapshot_counter = 0
        
        # Trade tracking configuration
        self.store_trades = collection_config.get('store_trades', True)
        self.trade_summary_only = collection_config.get('trade_summary_only', False)
        self.trades: List[Dict[str, Any]] = [] if self.store_trades else None
        
        logger.info(f"Initialized PortfolioMetricsTracker with capital: {initial_capital}, "
                   f"equity_curve: {self.store_equity_curve}, trades: {self.store_trades}")
    
    def update_portfolio_value(self, current_value: float, trade_pnl: Optional[float] = None, 
                              timestamp: Optional[datetime] = None) -> None:
        """
        Update metrics based on new portfolio value.
        
        Args:
            current_value: Current portfolio value
            trade_pnl: P&L from trades in this period
            timestamp: Current timestamp (for equity curve)
        """
        # Calculate period return
        returns = (current_value - self.previous_value) / self.previous_value if self.previous_value > 0 else 0.0
        
        # Update streaming metrics (always)
        self.metrics.update(returns, current_value, trade_pnl)
        
        # Conditionally update equity curve
        if self.store_equity_curve:
            self.update_counter += 1
            
            # Check if we should take a snapshot
            if self.update_counter >= self.last_snapshot_counter + self.snapshot_interval:
                equity_point = {
                    'timestamp': (timestamp or datetime.now()).isoformat(),
                    'value': current_value,
                    'returns': returns,
                    'drawdown': self.metrics.current_drawdown,
                    'total_return': self.metrics.total_return
                }
                self.equity_curve.append(equity_point)
                self.last_snapshot_counter = self.update_counter
                
                # Prevent memory overflow
                if len(self.equity_curve) > self.max_equity_points:
                    self._downsample_equity_curve()
        
        # Update previous value
        self.previous_value = current_value
    
    def _downsample_equity_curve(self) -> None:
        """Downsample equity curve to prevent memory overflow."""
        if len(self.equity_curve) <= 2:
            return
            
        # Keep first and last, sample the middle
        first = self.equity_curve[0]
        last = self.equity_curve[-1]
        middle = self.equity_curve[1:-1:2]  # Every other point
        self.equity_curve = [first] + middle + [last]
        
        # Double the snapshot interval
        self.snapshot_interval *= 2
        logger.debug(f"Downsampled equity curve to {len(self.equity_curve)} points, "
                    f"new snapshot interval: {self.snapshot_interval}")
    
    def on_trade_complete(self, trade_result: Dict[str, Any]) -> None:
        """
        Update metrics when a trade completes.
        
        Args:
            trade_result: Trade execution result
        """
        self.metrics.update_from_trade(trade_result)
        
        # Conditionally store trade
        if self.store_trades:
            if self.trade_summary_only:
                # Store summary only
                trade_summary = {
                    'id': trade_result.get('id'),
                    'symbol': trade_result.get('symbol'),
                    'entry_time': trade_result.get('entry_time'),
                    'exit_time': trade_result.get('exit_time'),
                    'pnl': trade_result.get('pnl', 0.0),
                    'return': trade_result.get('return', 0.0),
                    'direction': trade_result.get('direction')
                }
                self.trades.append(trade_summary)
            else:
                # Store full trade details
                self.trades.append(trade_result)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current portfolio metrics."""
        metrics = self.metrics.get_metrics()
        
        # Add portfolio-specific metrics
        metrics['initial_capital'] = self.initial_capital
        metrics['current_value'] = self.previous_value
        metrics['net_pnl'] = self.previous_value - self.initial_capital
        metrics['net_return'] = (self.previous_value - self.initial_capital) / self.initial_capital
        
        # Add collection status
        metrics['equity_curve_enabled'] = self.store_equity_curve
        metrics['equity_curve_points'] = len(self.equity_curve) if self.equity_curve else 0
        metrics['trades_stored'] = len(self.trades) if self.trades else 0
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics."""
        summary = self.metrics.get_summary_metrics()
        
        # Add collection summary
        summary['collection_status'] = {
            'equity_curve': {
                'enabled': self.store_equity_curve,
                'points': len(self.equity_curve) if self.equity_curve else 0,
                'snapshot_interval': self.snapshot_interval
            },
            'trades': {
                'enabled': self.store_trades,
                'count': len(self.trades) if self.trades else 0,
                'summary_only': self.trade_summary_only
            }
        }
        
        return summary
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get complete results including metrics, equity curve, and trades.
        
        This is what gets saved between phases or at completion.
        """
        results = {
            'metrics': self.get_metrics(),
            'summary': self.get_summary()
        }
        
        # Add equity curve if enabled
        if self.store_equity_curve and self.equity_curve:
            results['equity_curve'] = self.equity_curve
        
        # Add trades if enabled
        if self.store_trades and self.trades:
            results['trades'] = self.trades
        
        return results