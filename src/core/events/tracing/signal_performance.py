"""
Signal performance tracking for risk-aware decision making.

This module extends signal storage to track performance metrics that can be
used by risk functions to make more sophisticated decisions.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalPerformance:
    """Tracks performance metrics for a specific signal/strategy."""
    
    strategy_id: str
    strategy_name: str
    parameters: Dict[str, Any]
    
    # Performance tracking
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    
    # Detailed metrics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Per-regime performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Recent performance window
    recent_window: int = 20
    recent_signals: List[Dict[str, Any]] = field(default_factory=list)
    recent_win_rate: float = 0.0
    
    # Confidence metrics
    confidence_score: float = 1.0  # 0-1 score for risk sizing
    
    def update_with_result(self, signal: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Update performance metrics with a signal result.
        
        Args:
            signal: Original signal dict with entry info
            result: Trade result with exit info and P&L
        """
        self.total_signals += 1
        
        # Extract P&L
        pnl = result.get('pnl', 0.0)
        pnl_pct = result.get('pnl_pct', 0.0)
        
        # Update win/loss counts
        if pnl > 0:
            self.winning_signals += 1
            self.avg_win = ((self.avg_win * (self.winning_signals - 1)) + pnl_pct) / self.winning_signals
        else:
            self.losing_signals += 1
            self.avg_loss = ((self.avg_loss * (self.losing_signals - 1)) + abs(pnl_pct)) / self.losing_signals
        
        # Update win rate
        self.win_rate = self.winning_signals / self.total_signals if self.total_signals > 0 else 0
        
        # Update profit factor
        total_wins = self.winning_signals * self.avg_win
        total_losses = self.losing_signals * self.avg_loss
        self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Track regime-specific performance
        regime = signal.get('classifier_states', {}).get('trend', 'unknown')
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                'signals': 0,
                'wins': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0
            }
        
        regime_stats = self.regime_performance[regime]
        regime_stats['signals'] += 1
        if pnl > 0:
            regime_stats['wins'] += 1
        regime_stats['win_rate'] = regime_stats['wins'] / regime_stats['signals']
        regime_stats['avg_pnl'] = ((regime_stats['avg_pnl'] * (regime_stats['signals'] - 1)) + pnl_pct) / regime_stats['signals']
        
        # Update recent performance window
        self.recent_signals.append({
            'timestamp': signal.get('timestamp'),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'regime': regime
        })
        
        # Keep only recent window
        if len(self.recent_signals) > self.recent_window:
            self.recent_signals.pop(0)
        
        # Calculate recent win rate
        recent_wins = sum(1 for s in self.recent_signals if s['pnl'] > 0)
        self.recent_win_rate = recent_wins / len(self.recent_signals) if self.recent_signals else 0
        
        # Update confidence score
        self._update_confidence()
    
    def _update_confidence(self) -> None:
        """
        Update confidence score based on various factors.
        
        Confidence is used by risk functions to adjust position sizing.
        """
        if self.total_signals < 10:
            # Not enough data
            self.confidence_score = 0.5
            return
        
        # Base confidence on win rate
        base_confidence = self.win_rate
        
        # Adjust for recent performance
        if self.recent_win_rate > self.win_rate:
            # Recent performance better than average
            confidence_boost = min(0.2, (self.recent_win_rate - self.win_rate))
            base_confidence += confidence_boost
        else:
            # Recent performance worse
            confidence_penalty = min(0.3, (self.win_rate - self.recent_win_rate))
            base_confidence -= confidence_penalty
        
        # Adjust for profit factor
        if self.profit_factor > 2.0:
            base_confidence += 0.1
        elif self.profit_factor < 1.0:
            base_confidence -= 0.2
        
        # Clamp between 0 and 1
        self.confidence_score = max(0.1, min(1.0, base_confidence))
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get metrics relevant for risk management decisions.
        
        Returns:
            Dict with risk-relevant metrics
        """
        return {
            'confidence_score': self.confidence_score,
            'win_rate': self.win_rate,
            'recent_win_rate': self.recent_win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_signals': self.total_signals,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown
        }
    
    def get_regime_metrics(self, regime: str) -> Optional[Dict[str, float]]:
        """Get performance metrics for a specific regime."""
        return self.regime_performance.get(regime)
    
    def should_take_signal(self, min_confidence: float = 0.3) -> bool:
        """
        Simple decision helper based on confidence.
        
        Risk functions can use this or implement their own logic.
        """
        return self.confidence_score >= min_confidence
    
    def save(self, filepath: Path) -> None:
        """Save performance metrics to JSON."""
        data = {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'parameters': self.parameters,
            'metrics': {
                'total_signals': self.total_signals,
                'win_rate': self.win_rate,
                'recent_win_rate': self.recent_win_rate,
                'profit_factor': self.profit_factor,
                'confidence_score': self.confidence_score,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown
            },
            'regime_performance': self.regime_performance,
            'recent_signals': self.recent_signals[-self.recent_window:],  # Only save recent
            'updated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: Path) -> None:
        """Load performance metrics from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        
        self.strategy_id = data['strategy_id']
        self.strategy_name = data['strategy_name']
        self.parameters = data['parameters']
        
        metrics = data['metrics']
        self.total_signals = metrics['total_signals']
        self.winning_signals = int(metrics['win_rate'] * self.total_signals)
        self.losing_signals = self.total_signals - self.winning_signals
        self.win_rate = metrics['win_rate']
        self.recent_win_rate = metrics['recent_win_rate']
        self.profit_factor = metrics['profit_factor']
        self.confidence_score = metrics['confidence_score']
        self.avg_win = metrics['avg_win']
        self.avg_loss = metrics['avg_loss']
        self.sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        self.max_drawdown = metrics.get('max_drawdown', 0.0)
        
        self.regime_performance = data.get('regime_performance', {})
        self.recent_signals = data.get('recent_signals', [])


@dataclass
class PerformanceAwareSignalIndex:
    """
    Enhanced signal index that tracks performance for risk decisions.
    
    Extends SignalIndex with performance tracking capabilities.
    """
    
    strategy_name: str
    strategy_id: str
    parameters: Dict[str, Any]
    
    # Signal storage
    signals: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    performance: SignalPerformance = field(init=False)
    
    # Trade results mapping
    signal_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # bar_idx -> result
    
    def __post_init__(self):
        """Initialize performance tracker."""
        self.performance = SignalPerformance(
            strategy_id=self.strategy_id,
            strategy_name=self.strategy_name,
            parameters=self.parameters
        )
    
    def append_signal(self, bar_idx: int, signal: Dict[str, Any]) -> None:
        """Append a signal with metadata."""
        enhanced_signal = signal.copy()
        enhanced_signal['bar_idx'] = bar_idx
        enhanced_signal['timestamp'] = datetime.now().isoformat()
        enhanced_signal['confidence'] = self.performance.confidence_score
        
        self.signals.append(enhanced_signal)
    
    def record_result(self, bar_idx: int, result: Dict[str, Any]) -> None:
        """
        Record the result of a signal.
        
        Args:
            bar_idx: Bar index of the original signal
            result: Trade result with exit info and P&L
        """
        # Find the original signal
        signal = None
        for s in self.signals:
            if s['bar_idx'] == bar_idx:
                signal = s
                break
        
        if signal:
            # Update performance metrics
            self.performance.update_with_result(signal, result)
            
            # Store result
            self.signal_results[bar_idx] = result
            
            logger.info(f"Recorded result for {self.strategy_id} at bar {bar_idx}: "
                       f"P&L={result.get('pnl_pct', 0):.2%}, "
                       f"New confidence={self.performance.confidence_score:.2f}")
    
    def get_signal_with_performance(self, bar_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get signal with current performance metrics.
        
        This is what risk functions would use to make decisions.
        """
        for signal in self.signals:
            if signal['bar_idx'] == bar_idx:
                return {
                    **signal,
                    'performance': self.performance.get_risk_metrics(),
                    'regime_performance': self.performance.get_regime_metrics(
                        signal.get('classifier_states', {}).get('trend', 'unknown')
                    )
                }
        return None
    
    def save(self, base_path: Path) -> None:
        """Save signals and performance data."""
        # Save signals
        if self.signals:
            signals_df = pd.DataFrame(self.signals)
            signals_path = base_path / f"{self.strategy_id}_signals.parquet"
            signals_df.to_parquet(signals_path, compression='snappy')
        
        # Save performance
        perf_path = base_path / f"{self.strategy_id}_performance.json"
        self.performance.save(perf_path)
        
        # Save results
        if self.signal_results:
            results_df = pd.DataFrame([
                {'bar_idx': k, **v} for k, v in self.signal_results.items()
            ])
            results_path = base_path / f"{self.strategy_id}_results.parquet"
            results_df.to_parquet(results_path, compression='snappy')
    
    def load(self, base_path: Path) -> None:
        """Load signals and performance data."""
        # Load signals
        signals_path = base_path / f"{self.strategy_id}_signals.parquet"
        if signals_path.exists():
            signals_df = pd.read_parquet(signals_path)
            self.signals = signals_df.to_dict('records')
        
        # Load performance
        perf_path = base_path / f"{self.strategy_id}_performance.json"
        if perf_path.exists():
            self.performance.load(perf_path)
        
        # Load results
        results_path = base_path / f"{self.strategy_id}_results.parquet"
        if results_path.exists():
            results_df = pd.read_parquet(results_path)
            self.signal_results = {
                row['bar_idx']: {k: v for k, v in row.items() if k != 'bar_idx'}
                for _, row in results_df.iterrows()
            }


def create_risk_aware_signal(
    base_signal: Dict[str, Any],
    performance: SignalPerformance,
    current_regime: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an enhanced signal with performance context for risk functions.
    
    Args:
        base_signal: Original signal from strategy
        performance: Performance tracker for the strategy
        current_regime: Current market regime
        
    Returns:
        Enhanced signal with risk metrics
    """
    risk_metrics = performance.get_risk_metrics()
    
    # Get regime-specific metrics if available
    regime_metrics = None
    if current_regime:
        regime_metrics = performance.get_regime_metrics(current_regime)
    
    return {
        **base_signal,
        'risk_context': {
            'confidence': performance.confidence_score,
            'win_rate': performance.win_rate,
            'recent_win_rate': performance.recent_win_rate,
            'profit_factor': performance.profit_factor,
            'total_signals': performance.total_signals,
            'regime_performance': regime_metrics,
            'should_take': performance.should_take_signal(),
            'suggested_size_multiplier': min(2.0, max(0.5, performance.confidence_score * 1.5))
        }
    }