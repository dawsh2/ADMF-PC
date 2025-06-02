"""
File: src/strategy/performance_tracking.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#performance-tracking
Step: 4 - Multiple Strategies
Dependencies: dataclasses, datetime, statistics

Performance tracking for individual strategies and dynamic weight adjustment.
Tracks Sharpe ratios, win rates, and contribution metrics for each strategy.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import statistics
import math

from ..core.logging.structured import ComponentLogger


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy"""
    strategy_id: str
    trade_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    total_return: float = 0.0
    signal_count: int = 0
    consensus_contributions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Rolling windows for metrics
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_signals: deque = field(default_factory=lambda: deque(maxlen=50))
    daily_returns: deque = field(default_factory=lambda: deque(maxlen=252))  # 1 year
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate"""
        return self.win_count / max(1, self.trade_count)
    
    @property
    def avg_return_per_trade(self) -> float:
        """Calculate average return per trade"""
        return self.total_return / max(1, self.trade_count)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from recent returns"""
        if len(self.recent_returns) < 10:
            return 0.0
        
        returns = list(self.recent_returns)
        mean_return = statistics.mean(returns)
        
        if len(returns) < 2:
            return 0.0
        
        try:
            std_return = statistics.stdev(returns)
            if std_return == 0:
                return 0.0
            
            # Annualize assuming daily returns
            annual_return = mean_return * 252
            annual_std = std_return * math.sqrt(252)
            
            return (annual_return - risk_free_rate) / annual_std
        except statistics.StatisticsError:
            return 0.0
    
    def calculate_information_ratio(self, benchmark_returns: List[float]) -> float:
        """Calculate information ratio vs benchmark"""
        if len(self.recent_returns) < 10 or len(benchmark_returns) < 10:
            return 0.0
        
        strategy_returns = list(self.recent_returns)
        min_length = min(len(strategy_returns), len(benchmark_returns))
        
        if min_length < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = [
            strategy_returns[i] - benchmark_returns[i] 
            for i in range(min_length)
        ]
        
        try:
            mean_excess = statistics.mean(excess_returns)
            std_excess = statistics.stdev(excess_returns)
            
            if std_excess == 0:
                return 0.0
            
            return mean_excess / std_excess
        except statistics.StatisticsError:
            return 0.0
    
    def add_return(self, return_value: float, timestamp: datetime = None) -> None:
        """Add a return observation"""
        self.recent_returns.append(return_value)
        self.total_return += return_value
        self.last_updated = timestamp or datetime.now()
    
    def add_trade(self, pnl: float, is_winner: bool, timestamp: datetime = None) -> None:
        """Add a completed trade"""
        self.trade_count += 1
        self.total_pnl += pnl
        
        if is_winner:
            self.win_count += 1
        
        # Calculate return as percentage
        # Simplified - in practice would need position size
        return_pct = pnl / 10000.0  # Assume $10k base
        self.add_return(return_pct, timestamp)
    
    def add_signal(self, signal_strength: float, contributed_to_consensus: bool = False) -> None:
        """Add a signal observation"""
        self.signal_count += 1
        self.recent_signals.append(signal_strength)
        
        if contributed_to_consensus:
            self.consensus_contributions += 1
    
    @property
    def consensus_contribution_rate(self) -> float:
        """Rate of signals that contributed to consensus"""
        return self.consensus_contributions / max(1, self.signal_count)


class StrategyPerformanceTracker:
    """Tracks individual strategy performance for weight adjustment"""
    
    def __init__(self, container_id: str = "global"):
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.rolling_window = 100  # trades
        self.min_trades_for_weight_adjustment = 20
        self.weight_decay_factor = 0.95  # For exponential decay of poor performers
        
        # Benchmark tracking
        self.benchmark_returns: deque = deque(maxlen=252)
        self.portfolio_returns: deque = deque(maxlen=252)
        
        self.logger = ComponentLogger("StrategyPerformanceTracker", container_id)
    
    def track_signal(self, strategy_id: str, signal_strength: float, 
                    contributed_to_consensus: bool = False) -> None:
        """Track a signal from a strategy"""
        if strategy_id not in self.strategy_metrics:
            self.strategy_metrics[strategy_id] = StrategyMetrics(strategy_id)
        
        metrics = self.strategy_metrics[strategy_id]
        metrics.add_signal(signal_strength, contributed_to_consensus)
    
    def track_consensus_signal(self, consensus, contributing_signals) -> None:
        """Track which strategies contributed to consensus"""
        contributing_strategy_ids = {s.strategy_id for s in contributing_signals}
        
        for strategy_id in self.strategy_metrics:
            contributed = strategy_id in contributing_strategy_ids
            # Find the signal strength if contributed
            signal_strength = 0.0
            if contributed:
                for sig in contributing_signals:
                    if sig.strategy_id == strategy_id:
                        signal_strength = sig.signal.strength
                        break
            
            self.track_signal(strategy_id, signal_strength, contributed)
    
    def update_on_fill(self, strategy_id: str, fill_data: Dict[str, Any]) -> None:
        """Update strategy metrics based on fill"""
        if strategy_id not in self.strategy_metrics:
            self.strategy_metrics[strategy_id] = StrategyMetrics(strategy_id)
        
        metrics = self.strategy_metrics[strategy_id]
        
        # Extract relevant data from fill
        pnl = fill_data.get('pnl', 0.0)
        is_winner = pnl > 0
        
        metrics.add_trade(pnl, is_winner)
        
        self.logger.debug(
            f"Updated {strategy_id} metrics: "
            f"trades={metrics.trade_count}, win_rate={metrics.win_rate:.2f}"
        )
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on performance"""
        if not self.strategy_metrics:
            return {}
        
        # Filter strategies with sufficient data
        eligible_strategies = {
            sid: metrics for sid, metrics in self.strategy_metrics.items()
            if metrics.trade_count >= self.min_trades_for_weight_adjustment
        }
        
        if not eligible_strategies:
            # Equal weights if insufficient data
            return {sid: 1.0 / len(self.strategy_metrics) 
                   for sid in self.strategy_metrics}
        
        # Calculate composite scores
        strategy_scores = {}
        
        for sid, metrics in eligible_strategies.items():
            # Components of the score
            sharpe_ratio = max(0, metrics.calculate_sharpe_ratio())
            win_rate = metrics.win_rate
            contribution_rate = metrics.consensus_contribution_rate
            avg_return = metrics.avg_return_per_trade
            
            # Composite score with weights
            score = (
                sharpe_ratio * 0.4 +
                win_rate * 0.3 +
                contribution_rate * 0.2 +
                max(0, avg_return) * 0.1
            )
            
            strategy_scores[sid] = max(0.01, score)  # Minimum weight
        
        # Normalize to weights
        total_score = sum(strategy_scores.values())
        
        if total_score == 0:
            # Equal weights if all scores are zero
            return {sid: 1.0 / len(eligible_strategies) 
                   for sid in eligible_strategies}
        
        weights = {
            sid: score / total_score 
            for sid, score in strategy_scores.items()
        }
        
        # Include non-eligible strategies with minimum weight
        for sid in self.strategy_metrics:
            if sid not in weights:
                weights[sid] = 0.05  # Minimal weight for new strategies
        
        # Renormalize
        total_weight = sum(weights.values())
        weights = {sid: w / total_weight for sid, w in weights.items()}
        
        self.logger.info(f"Updated strategy weights: {weights}")
        return weights
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'timestamp': datetime.now(),
            'total_strategies': len(self.strategy_metrics),
            'strategies': {}
        }
        
        for sid, metrics in self.strategy_metrics.items():
            report['strategies'][sid] = {
                'trade_count': metrics.trade_count,
                'win_rate': metrics.win_rate,
                'total_pnl': metrics.total_pnl,
                'avg_return': metrics.avg_return_per_trade,
                'sharpe_ratio': metrics.calculate_sharpe_ratio(),
                'signal_count': metrics.signal_count,
                'consensus_contributions': metrics.consensus_contributions,
                'contribution_rate': metrics.consensus_contribution_rate,
                'last_updated': metrics.last_updated
            }
        
        # Add portfolio-level metrics
        if self.strategy_metrics:
            total_trades = sum(m.trade_count for m in self.strategy_metrics.values())
            total_pnl = sum(m.total_pnl for m in self.strategy_metrics.values())
            total_signals = sum(m.signal_count for m in self.strategy_metrics.values())
            
            report['portfolio'] = {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'total_signals': total_signals,
                'avg_pnl_per_trade': total_pnl / max(1, total_trades)
            }
        
        return report
    
    def reset_strategy_metrics(self, strategy_id: str) -> None:
        """Reset metrics for a strategy"""
        if strategy_id in self.strategy_metrics:
            del self.strategy_metrics[strategy_id]
            self.logger.info(f"Reset metrics for strategy: {strategy_id}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.strategy_metrics:
            return {'strategies': 0}
        
        total_trades = sum(m.trade_count for m in self.strategy_metrics.values())
        total_signals = sum(m.signal_count for m in self.strategy_metrics.values())
        
        return {
            'strategies': len(self.strategy_metrics),
            'total_trades': total_trades,
            'total_signals': total_signals,
            'avg_trades_per_strategy': total_trades / len(self.strategy_metrics),
            'avg_signals_per_strategy': total_signals / len(self.strategy_metrics)
        }


def create_performance_tracker(container_id: str = "global") -> StrategyPerformanceTracker:
    """Factory function to create performance tracker"""
    return StrategyPerformanceTracker(container_id)