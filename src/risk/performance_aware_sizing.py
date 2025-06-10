"""
Performance-aware position sizing for risk management.

This module provides position sizing that adapts based on signal performance history,
allowing more sophisticated risk management decisions.
"""

from typing import Dict, Any, Optional, Protocol
from dataclasses import dataclass
import logging
from decimal import Decimal

from ..core.events.tracing.signal_performance import SignalPerformance

logger = logging.getLogger(__name__)


class PerformanceProvider(Protocol):
    """Protocol for accessing signal performance data."""
    
    def get_performance(self, strategy_id: str) -> Optional[SignalPerformance]:
        """Get performance metrics for a strategy."""
        ...
    
    def get_recent_performance(self, strategy_id: str, window: int = 20) -> Optional[Dict[str, float]]:
        """Get recent performance metrics."""
        ...


@dataclass
class PerformanceAwarePositionSizer:
    """
    Position sizer that adjusts size based on signal performance.
    
    This sizer considers:
    - Historical win rate and profit factor
    - Recent performance trends
    - Regime-specific performance
    - Confidence scores from signal performance
    """
    
    # Base sizing parameters
    base_size_pct: float = 0.02  # 2% base position size
    min_size_pct: float = 0.005  # 0.5% minimum
    max_size_pct: float = 0.05   # 5% maximum
    
    # Performance adjustments
    enable_performance_sizing: bool = True
    min_confidence_for_trade: float = 0.3
    min_signals_for_full_size: int = 30
    
    # Scaling factors
    confidence_weight: float = 0.4  # How much confidence affects size
    win_rate_weight: float = 0.3   # How much win rate affects size
    recent_perf_weight: float = 0.3 # How much recent performance affects size
    
    # Kelly criterion parameters (optional)
    use_kelly_criterion: bool = False
    kelly_fraction: float = 0.25  # Fraction of Kelly to use (conservative)
    
    def calculate_position_size(
        self,
        signal: Dict[str, Any],
        portfolio_value: float,
        performance: Optional[SignalPerformance] = None,
        existing_exposure: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate position size with performance awareness.
        
        Args:
            signal: Trading signal with risk_context if available
            portfolio_value: Total portfolio value
            performance: Optional performance data (can be in signal)
            existing_exposure: Current exposures by symbol
            
        Returns:
            Dict with:
                - size_pct: Position size as percentage of portfolio
                - size_value: Position size in currency
                - confidence: Confidence in this sizing
                - adjustments: Dict of adjustment factors applied
        """
        # Start with base size
        base_size = self.base_size_pct
        adjustments = {}
        
        # Extract performance from signal if not provided
        if not performance and 'risk_context' in signal:
            # Signal has embedded performance data
            risk_context = signal['risk_context']
            confidence = risk_context.get('confidence', 0.5)
            win_rate = risk_context.get('win_rate', 0.5)
            recent_win_rate = risk_context.get('recent_win_rate', win_rate)
            profit_factor = risk_context.get('profit_factor', 1.0)
            total_signals = risk_context.get('total_signals', 0)
            
            # Check if we should skip based on confidence
            if confidence < self.min_confidence_for_trade:
                return {
                    'size_pct': 0.0,
                    'size_value': 0.0,
                    'confidence': confidence,
                    'skip_reason': f'Confidence too low: {confidence:.2f}',
                    'adjustments': {}
                }
        elif performance:
            # Use provided performance object
            metrics = performance.get_risk_metrics()
            confidence = metrics['confidence_score']
            win_rate = metrics['win_rate']
            recent_win_rate = metrics['recent_win_rate']
            profit_factor = metrics['profit_factor']
            total_signals = metrics['total_signals']
        else:
            # No performance data - use base sizing
            return {
                'size_pct': base_size,
                'size_value': portfolio_value * base_size,
                'confidence': 0.5,
                'adjustments': {'no_data': True}
            }
        
        if not self.enable_performance_sizing:
            # Performance sizing disabled
            return {
                'size_pct': base_size,
                'size_value': portfolio_value * base_size,
                'confidence': confidence,
                'adjustments': {'performance_sizing': False}
            }
        
        # Apply adjustments
        size_multiplier = 1.0
        
        # 1. Confidence adjustment
        confidence_factor = confidence  # Linear scaling
        confidence_adjustment = self.confidence_weight * (confidence_factor - 0.5) * 2
        size_multiplier *= (1 + confidence_adjustment)
        adjustments['confidence'] = confidence_adjustment
        
        # 2. Win rate adjustment
        win_rate_factor = (win_rate - 0.5) * 2  # Convert to -1 to 1 scale
        win_rate_adjustment = self.win_rate_weight * win_rate_factor
        size_multiplier *= (1 + win_rate_adjustment)
        adjustments['win_rate'] = win_rate_adjustment
        
        # 3. Recent performance adjustment
        recent_factor = (recent_win_rate - win_rate) / max(win_rate, 0.1)  # Relative change
        recent_adjustment = self.recent_perf_weight * recent_factor
        size_multiplier *= (1 + recent_adjustment)
        adjustments['recent_performance'] = recent_adjustment
        
        # 4. Experience adjustment (reduce size for new strategies)
        if total_signals < self.min_signals_for_full_size:
            experience_factor = total_signals / self.min_signals_for_full_size
            size_multiplier *= experience_factor
            adjustments['experience'] = experience_factor
        
        # 5. Profit factor adjustment
        if profit_factor < 1.0:
            # Losing strategy - reduce size
            pf_adjustment = max(0.5, profit_factor)
            size_multiplier *= pf_adjustment
            adjustments['profit_factor'] = pf_adjustment
        elif profit_factor > 2.0:
            # Very profitable - can increase slightly
            pf_adjustment = min(1.2, 1 + (profit_factor - 2) * 0.1)
            size_multiplier *= pf_adjustment
            adjustments['profit_factor'] = pf_adjustment
        
        # Apply Kelly criterion if enabled
        if self.use_kelly_criterion and win_rate > 0 and performance:
            kelly_size = self._calculate_kelly_size(
                win_rate=win_rate,
                avg_win=performance.avg_win,
                avg_loss=performance.avg_loss
            )
            
            if kelly_size > 0:
                # Use fraction of Kelly
                kelly_adjusted = kelly_size * self.kelly_fraction
                # Blend with performance-based size
                final_size = (size_multiplier * base_size + kelly_adjusted) / 2
                adjustments['kelly'] = kelly_adjusted / base_size
            else:
                final_size = size_multiplier * base_size
        else:
            final_size = size_multiplier * base_size
        
        # Apply limits
        final_size = max(self.min_size_pct, min(self.max_size_pct, final_size))
        
        # Check existing exposure
        if existing_exposure:
            symbol = signal.get('symbol')
            current_exposure = existing_exposure.get(symbol, 0.0)
            
            # Reduce size if already have exposure
            if current_exposure > 0:
                exposure_factor = max(0.5, 1 - current_exposure / 0.10)  # Reduce up to 50%
                final_size *= exposure_factor
                adjustments['existing_exposure'] = exposure_factor
        
        return {
            'size_pct': final_size,
            'size_value': portfolio_value * final_size,
            'confidence': confidence,
            'adjustments': adjustments,
            'base_size': base_size,
            'multiplier': final_size / base_size
        }
    
    def _calculate_kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate position size using Kelly criterion.
        
        Kelly formula: f = (p * b - q) / b
        where:
            f = fraction of capital to wager
            p = probability of winning
            b = odds (avg_win / avg_loss)
            q = probability of losing (1 - p)
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss
        
        kelly = (p * b - q) / b
        
        # Kelly can suggest negative sizes (don't trade) or very large sizes
        # Limit to reasonable range
        return max(0.0, min(0.25, kelly))  # Max 25% even with full Kelly
    
    def adjust_for_regime(
        self,
        base_size: float,
        regime: str,
        regime_performance: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Adjust size based on regime-specific performance.
        
        Args:
            base_size: Base position size
            regime: Current market regime
            regime_performance: Performance in this regime
            
        Returns:
            Adjusted size
        """
        if not regime_performance:
            return base_size
        
        regime_win_rate = regime_performance.get('win_rate', 0.5)
        regime_signals = regime_performance.get('signals', 0)
        
        # Need enough data in this regime
        if regime_signals < 10:
            return base_size * 0.8  # Reduce size in untested regime
        
        # Adjust based on regime performance
        if regime_win_rate < 0.3:
            # Poor performance in this regime
            return base_size * 0.5
        elif regime_win_rate > 0.7:
            # Excellent performance in this regime
            return base_size * 1.2
        else:
            return base_size
    
    def get_size_explanation(self, sizing_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation of sizing decision."""
        size_pct = sizing_result['size_pct']
        confidence = sizing_result['confidence']
        adjustments = sizing_result['adjustments']
        
        explanation = f"Position size: {size_pct:.1%} of portfolio (confidence: {confidence:.2f})\n"
        
        if size_pct == 0:
            explanation += f"Reason: {sizing_result.get('skip_reason', 'Unknown')}\n"
        else:
            explanation += "Adjustments applied:\n"
            for adj_name, adj_value in adjustments.items():
                if isinstance(adj_value, bool):
                    explanation += f"  - {adj_name}: {'Yes' if adj_value else 'No'}\n"
                else:
                    explanation += f"  - {adj_name}: {adj_value:+.1%}\n"
            
            multiplier = sizing_result.get('multiplier', 1.0)
            explanation += f"Total multiplier: {multiplier:.2f}x base size\n"
        
        return explanation


# Example integration with existing position sizing
def create_performance_aware_sizer(
    base_config: Dict[str, Any],
    enable_kelly: bool = False
) -> PerformanceAwarePositionSizer:
    """
    Factory function to create performance-aware sizer with config.
    
    Args:
        base_config: Base configuration dict
        enable_kelly: Whether to use Kelly criterion
        
    Returns:
        Configured PerformanceAwarePositionSizer
    """
    return PerformanceAwarePositionSizer(
        base_size_pct=base_config.get('base_size_pct', 0.02),
        min_size_pct=base_config.get('min_size_pct', 0.005),
        max_size_pct=base_config.get('max_size_pct', 0.05),
        enable_performance_sizing=base_config.get('enable_performance_sizing', True),
        min_confidence_for_trade=base_config.get('min_confidence', 0.3),
        use_kelly_criterion=enable_kelly,
        kelly_fraction=base_config.get('kelly_fraction', 0.25)
    )