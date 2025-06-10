"""
Example of using signal performance tracking for sophisticated risk decisions.

This shows how risk functions can make decisions based on:
- Historical win rate
- Recent performance trends
- Regime-specific performance
- Confidence scores
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.core.events.tracing.signal_performance import (
    SignalPerformance,
    PerformanceAwareSignalIndex,
    create_risk_aware_signal
)
from src.risk.validators import RiskValidator


@dataclass
class PerformanceAwareRiskValidator(RiskValidator):
    """
    Risk validator that uses signal performance history to make decisions.
    
    This validator adjusts position sizing and can reject signals based on:
    - Strategy confidence scores
    - Recent performance
    - Regime-specific performance
    - Risk limits that adapt to performance
    """
    
    # Base risk parameters
    base_position_size: float = 0.02  # 2% of portfolio
    max_position_size: float = 0.05   # 5% max
    min_confidence: float = 0.3       # Minimum confidence to take signal
    
    # Performance adjustments
    use_performance_sizing: bool = True
    use_regime_filtering: bool = True
    min_signals_for_confidence: int = 20
    
    def validate_order(
        self, 
        order: Dict[str, Any], 
        portfolio_state: Dict[str, Any],
        signal_with_performance: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate order with performance-aware logic.
        
        Args:
            order: Order to validate
            portfolio_state: Current portfolio state
            signal_with_performance: Enhanced signal with performance metrics
            
        Returns:
            Validation result with adjusted parameters
        """
        # Start with base validation
        result = {
            'is_valid': True,
            'adjusted_size': None,
            'rejection_reason': None,
            'confidence': 1.0
        }
        
        # If no performance data, use standard validation
        if not signal_with_performance or 'risk_context' not in signal_with_performance:
            return self._standard_validation(order, portfolio_state)
        
        risk_context = signal_with_performance['risk_context']
        
        # 1. Check confidence threshold
        confidence = risk_context['confidence']
        if confidence < self.min_confidence:
            result['is_valid'] = False
            result['rejection_reason'] = (
                f"Signal confidence too low: {confidence:.2f} < {self.min_confidence}"
            )
            return result
        
        # 2. Check if we have enough history
        total_signals = risk_context['total_signals']
        if total_signals < self.min_signals_for_confidence:
            # Reduce size for new strategies
            result['confidence'] = 0.5
            result['adjusted_size'] = self.base_position_size * 0.5
        
        # 3. Performance-based sizing
        if self.use_performance_sizing:
            size_multiplier = risk_context.get('suggested_size_multiplier', 1.0)
            
            # Further adjust based on recent vs overall performance
            win_rate = risk_context['win_rate']
            recent_win_rate = risk_context['recent_win_rate']
            
            if recent_win_rate < win_rate * 0.7:  # Recent performance much worse
                size_multiplier *= 0.7
            elif recent_win_rate > win_rate * 1.3:  # Recent performance much better
                size_multiplier *= 1.2
            
            # Apply profit factor adjustment
            profit_factor = risk_context.get('profit_factor', 1.0)
            if profit_factor < 1.0:
                size_multiplier *= 0.8
            elif profit_factor > 2.0:
                size_multiplier *= 1.1
            
            # Calculate final size
            adjusted_size = self.base_position_size * size_multiplier
            adjusted_size = max(self.base_position_size * 0.5,  # Min 50% of base
                               min(self.max_position_size, adjusted_size))  # Max limit
            
            result['adjusted_size'] = adjusted_size
            result['confidence'] = confidence
        
        # 4. Regime-specific filtering
        if self.use_regime_filtering and risk_context.get('regime_performance'):
            regime_perf = risk_context['regime_performance']
            regime_win_rate = regime_perf.get('win_rate', 0)
            
            # Skip signal in regimes where strategy performs poorly
            if regime_perf['signals'] > 10 and regime_win_rate < 0.3:
                result['is_valid'] = False
                result['rejection_reason'] = (
                    f"Poor performance in current regime: {regime_win_rate:.1%} win rate"
                )
                return result
        
        # 5. Risk capacity check with performance adjustment
        current_risk = portfolio_state.get('total_risk', 0)
        max_risk = portfolio_state.get('max_risk', 0.20)  # 20% max portfolio risk
        
        # Tighten risk limits if recent performance is poor
        if recent_win_rate < 0.4:
            max_risk *= 0.8
        
        position_risk = result.get('adjusted_size', self.base_position_size)
        if current_risk + position_risk > max_risk:
            result['is_valid'] = False
            result['rejection_reason'] = f"Would exceed risk limit: {current_risk + position_risk:.1%} > {max_risk:.1%}"
        
        return result
    
    def _standard_validation(self, order: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Standard validation without performance data."""
        return {
            'is_valid': True,
            'adjusted_size': self.base_position_size,
            'rejection_reason': None,
            'confidence': 0.5  # Default confidence without history
        }


def demonstrate_performance_aware_risk():
    """Show how performance tracking enhances risk decisions."""
    
    # Create performance tracker for a strategy
    performance = SignalPerformance(
        strategy_id="momentum_20_50",
        strategy_name="momentum_crossover",
        parameters={'fast_period': 20, 'slow_period': 50}
    )
    
    # Simulate some trading history
    print("=== Simulating Trading History ===")
    
    # Good initial performance
    for i in range(15):
        signal = {
            'bar_idx': i,
            'symbol': 'SPY',
            'classifier_states': {'trend': 'bull'}
        }
        
        # 70% win rate initially
        if i % 10 < 7:
            result = {'pnl': 100, 'pnl_pct': 0.02}  # 2% win
        else:
            result = {'pnl': -50, 'pnl_pct': -0.01}  # 1% loss
        
        performance.update_with_result(signal, result)
    
    print(f"After 15 trades: Win rate={performance.win_rate:.1%}, "
          f"Confidence={performance.confidence_score:.2f}")
    
    # Recent poor performance
    for i in range(15, 25):
        signal = {
            'bar_idx': i,
            'symbol': 'SPY',
            'classifier_states': {'trend': 'bear'}
        }
        
        # Only 20% win rate recently
        if i % 10 < 2:
            result = {'pnl': 100, 'pnl_pct': 0.02}
        else:
            result = {'pnl': -50, 'pnl_pct': -0.01}
        
        performance.update_with_result(signal, result)
    
    print(f"After 25 trades: Win rate={performance.win_rate:.1%}, "
          f"Recent win rate={performance.recent_win_rate:.1%}, "
          f"Confidence={performance.confidence_score:.2f}")
    
    # Create risk validator
    risk_validator = PerformanceAwareRiskValidator(
        base_position_size=0.02,
        min_confidence=0.3
    )
    
    # Test different scenarios
    print("\n=== Risk Validation Scenarios ===")
    
    # Scenario 1: Current signal with poor recent performance
    current_signal = {
        'symbol': 'SPY',
        'direction': 'long',
        'strength': 0.8
    }
    
    enhanced_signal = create_risk_aware_signal(
        current_signal,
        performance,
        current_regime='bear'
    )
    
    portfolio_state = {
        'total_risk': 0.10,  # 10% current risk
        'max_risk': 0.20     # 20% max risk
    }
    
    order = {
        'symbol': 'SPY',
        'quantity': 100,
        'side': 'buy'
    }
    
    result = risk_validator.validate_order(order, portfolio_state, enhanced_signal)
    print(f"\nScenario 1 - Poor recent performance:")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Adjusted size: {result.get('adjusted_size', 0):.1%}")
    print(f"  Reason: {result.get('rejection_reason', 'N/A')}")
    
    # Scenario 2: Different regime with good historical performance
    # Update regime performance
    performance.regime_performance['sideways'] = {
        'signals': 20,
        'wins': 16,
        'win_rate': 0.80,
        'avg_pnl': 0.015
    }
    
    enhanced_signal = create_risk_aware_signal(
        current_signal,
        performance,
        current_regime='sideways'
    )
    
    result = risk_validator.validate_order(order, portfolio_state, enhanced_signal)
    print(f"\nScenario 2 - Good regime performance:")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Adjusted size: {result.get('adjusted_size', 0):.1%}")
    print(f"  Confidence: {result.get('confidence', 0):.2f}")
    
    # Scenario 3: High confidence scenario
    # Simulate recovery
    performance.confidence_score = 0.85
    performance.recent_win_rate = 0.75
    
    enhanced_signal = create_risk_aware_signal(
        current_signal,
        performance,
        current_regime='bull'
    )
    
    result = risk_validator.validate_order(order, portfolio_state, enhanced_signal)
    print(f"\nScenario 3 - High confidence:")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Adjusted size: {result.get('adjusted_size', 0):.1%}")
    print(f"  Size multiplier: {result.get('adjusted_size', 0) / 0.02:.1f}x")
    
    # Show risk metrics
    print("\n=== Risk Metrics Summary ===")
    risk_metrics = performance.get_risk_metrics()
    for key, value in risk_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Show regime breakdown
    print("\n=== Regime Performance ===")
    for regime, stats in performance.regime_performance.items():
        print(f"{regime}: {stats}")


def demonstrate_signal_storage_integration():
    """Show how signal storage integrates with performance tracking."""
    
    print("\n=== Signal Storage with Performance ===")
    
    # Create performance-aware signal index
    signal_index = PerformanceAwareSignalIndex(
        strategy_name="mean_reversion",
        strategy_id="mean_rev_bollinger_2std",
        parameters={'period': 20, 'std_dev': 2.0}
    )
    
    # Generate and store signals
    for i in range(50):
        signal = {
            'symbol': 'QQQ',
            'direction': 'long' if i % 3 == 0 else 'short',
            'strength': 0.5 + (i % 5) * 0.1,
            'classifier_states': {'volatility': 'high' if i % 4 == 0 else 'normal'}
        }
        
        signal_index.append_signal(i, signal)
        
        # Simulate some results for closed trades
        if i > 10 and i % 5 == 0:
            # Record result for signal from 10 bars ago
            old_idx = i - 10
            pnl = 50 if i % 3 == 0 else -30
            result = {
                'entry_idx': old_idx,
                'exit_idx': i,
                'pnl': pnl,
                'pnl_pct': pnl / 1000.0
            }
            signal_index.record_result(old_idx, result)
    
    # Retrieve signal with performance context
    enhanced = signal_index.get_signal_with_performance(45)
    if enhanced:
        print(f"\nSignal at bar 45:")
        print(f"  Direction: {enhanced['direction']}")
        print(f"  Confidence: {enhanced['performance']['confidence_score']:.2f}")
        print(f"  Strategy win rate: {enhanced['performance']['win_rate']:.1%}")
        print(f"  Recent win rate: {enhanced['performance']['recent_win_rate']:.1%}")
    
    # Save and demonstrate persistence
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)
        signal_index.save(save_path)
        print(f"\nSaved signal index with {len(signal_index.signals)} signals")
        print(f"Performance confidence: {signal_index.performance.confidence_score:.2f}")


if __name__ == '__main__':
    demonstrate_performance_aware_risk()
    demonstrate_signal_storage_integration()