#!/usr/bin/env python3
"""
Signal Observer Workflow Example

Demonstrates how containers can use SignalObserver to enable performance-aware
risk decisions. Shows the full flow from signal generation through risk validation
with performance context.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our components
from src.core.events import Event, EventType, EventBus
from src.core.events.tracing.observers import SignalObserver
from src.core.events.tracing.signal_performance import SignalPerformance
from src.core.containers.components.signal_generator import SignalGeneratorComponent
from src.risk.performance_aware_sizing import PerformanceAwarePositionSizer


def create_sample_features(timestamp: datetime, symbol: str = "SPY") -> Dict[str, Any]:
    """Create sample feature data for strategies."""
    return {
        symbol: {
            'sma_20': 420.0 + random.uniform(-5, 5),
            'sma_50': 415.0 + random.uniform(-5, 5),
            'rsi': random.uniform(30, 70),
            'volume_ratio': random.uniform(0.8, 1.2),
            'atr': random.uniform(2, 4),
            'bb_width': random.uniform(0.01, 0.03),
            'momentum': random.uniform(-0.02, 0.02)
        }
    }


def create_sample_bars(timestamp: datetime, symbol: str = "SPY") -> Dict[str, Any]:
    """Create sample bar data."""
    base_price = 420.0
    return {
        symbol: {
            'timestamp': timestamp,
            'open': base_price + random.uniform(-2, 2),
            'high': base_price + random.uniform(0, 3),
            'low': base_price + random.uniform(-3, 0),
            'close': base_price + random.uniform(-2, 2),
            'volume': random.randint(50000000, 100000000),
            'symbol': symbol,
            'timeframe': '5m',
            'index': 1000
        }
    }


# Sample strategy functions
def momentum_strategy(features: Dict[str, Any], bar: Dict[str, Any], 
                     params: Dict[str, Any]) -> Dict[str, Any]:
    """Simple momentum strategy."""
    # Extract parameters
    fast_period = params.get('fast_period', 20)
    slow_period = params.get('slow_period', 50)
    
    # Get moving averages
    sma_fast = features.get(f'sma_{fast_period}', 0)
    sma_slow = features.get(f'sma_{slow_period}', 0)
    
    # Generate signal based on crossover
    if sma_fast > sma_slow * 1.01:  # Fast above slow by 1%
        return {
            'value': 1.0,  # Buy signal
            'symbol': bar['symbol'],
            'confidence': min(abs(sma_fast - sma_slow) / sma_slow, 1.0)
        }
    elif sma_fast < sma_slow * 0.99:  # Fast below slow by 1%
        return {
            'value': -1.0,  # Sell signal
            'symbol': bar['symbol'],
            'confidence': min(abs(sma_slow - sma_fast) / sma_slow, 1.0)
        }
    
    return {'value': 0}  # No signal


def mean_reversion_strategy(features: Dict[str, Any], bar: Dict[str, Any],
                           params: Dict[str, Any]) -> Dict[str, Any]:
    """Simple mean reversion strategy."""
    rsi = features.get('rsi', 50)
    bb_width = features.get('bb_width', 0.02)
    
    # Generate signal based on RSI extremes
    if rsi < params.get('oversold', 30):
        return {
            'value': 1.0,  # Buy when oversold
            'symbol': bar['symbol'],
            'confidence': (30 - rsi) / 30  # More oversold = higher confidence
        }
    elif rsi > params.get('overbought', 70):
        return {
            'value': -1.0,  # Sell when overbought
            'symbol': bar['symbol'],
            'confidence': (rsi - 70) / 30  # More overbought = higher confidence
        }
    
    return {'value': 0}


class PerformanceAwareRiskFunction:
    """
    Risk function that uses signal performance data to make sophisticated decisions.
    
    This demonstrates the "risk as meta-strategy" concept where risk management
    becomes a complex decision-making layer based on historical performance.
    """
    
    def __init__(self, signal_observer: SignalObserver, 
                 position_sizer: PerformanceAwarePositionSizer,
                 max_positions: int = 5):
        self.signal_observer = signal_observer
        self.position_sizer = position_sizer
        self.max_positions = max_positions
        self.open_positions = {}
        
    def validate_signal(self, signal: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance signal with performance-aware risk decisions.
        
        Returns enhanced signal with position sizing and risk metadata.
        """
        # Get enhanced signal with performance data
        enhanced_signal = self.signal_observer.enhance_signal_with_performance(signal)
        
        # Extract performance metrics
        strategy_id = enhanced_signal.get('strategy_id')
        performance = enhanced_signal.get('performance', {})
        
        # Make risk decisions based on performance
        logger.info(f"\n{'='*60}")
        logger.info(f"Risk Analysis for {strategy_id}")
        logger.info(f"{'='*60}")
        logger.info(f"Win Rate: {performance.get('win_rate', 0):.1%}")
        logger.info(f"Confidence Score: {performance.get('confidence_score', 1.0):.2f}")
        logger.info(f"Recent Performance: {performance.get('recent_performance', 0):.1%}")
        logger.info(f"Regime Performance: {performance.get('regime_performance', {})}")
        
        # 1. Check if strategy is performing well enough
        min_win_rate = 0.45  # Minimum 45% win rate
        if performance.get('win_rate', 0) < min_win_rate and performance.get('total_signals', 0) > 10:
            logger.warning(f"Rejecting signal - win rate {performance['win_rate']:.1%} below threshold")
            return None
            
        # 2. Check recent performance (drawdown protection)
        if performance.get('recent_performance', 0) < -0.10:  # -10% recent performance
            logger.warning(f"Rejecting signal - recent performance {performance['recent_performance']:.1%} too negative")
            return None
            
        # 3. Calculate position size based on performance
        portfolio_value = portfolio_state.get('portfolio_value', 100000)
        existing_exposure = sum(pos.get('value', 0) for pos in self.open_positions.values())
        
        size_pct, size_value, confidence, adjustments = self.position_sizer.calculate_position_size(
            signal=enhanced_signal,
            portfolio_value=portfolio_value,
            performance=self.signal_observer.get_performance(strategy_id),
            existing_exposure=existing_exposure
        )
        
        # 4. Apply portfolio-level constraints
        if len(self.open_positions) >= self.max_positions:
            logger.warning(f"Rejecting signal - max positions ({self.max_positions}) reached")
            return None
            
        # 5. Check correlation with existing positions (simplified)
        # In real system, would check actual correlations
        same_strategy_positions = sum(1 for pos in self.open_positions.values() 
                                    if pos.get('strategy_id') == strategy_id)
        if same_strategy_positions >= 2:
            logger.warning(f"Rejecting signal - already have {same_strategy_positions} positions from {strategy_id}")
            return None
            
        # 6. Create validated signal with risk metadata
        validated_signal = enhanced_signal.copy()
        validated_signal.update({
            'position_size_pct': size_pct,
            'position_size_value': size_value,
            'risk_confidence': confidence,
            'risk_adjustments': adjustments,
            'risk_metadata': {
                'win_rate_check': performance.get('win_rate', 0) >= min_win_rate,
                'recent_performance_check': performance.get('recent_performance', 0) >= -0.10,
                'position_count': len(self.open_positions),
                'strategy_exposure': same_strategy_positions,
                'performance_metrics': performance
            }
        })
        
        logger.info(f"\nRisk Decision: APPROVED")
        logger.info(f"Position Size: {size_pct:.1%} (${size_value:,.2f})")
        logger.info(f"Confidence: {confidence:.2f}")
        logger.info(f"Adjustments: {adjustments}")
        logger.info(f"{'='*60}\n")
        
        return validated_signal


def simulate_trading_session():
    """Simulate a trading session with performance-aware risk management."""
    
    # 1. Create event bus and attach signal observer
    event_bus = EventBus("main_bus")
    signal_observer = SignalObserver(
        retention_policy="trade_complete",
        recent_window_size=20,
        min_trades_for_confidence=5
    )
    event_bus.attach_observer(signal_observer)
    
    # 2. Create signal generator component
    signal_gen = SignalGeneratorComponent(
        storage_enabled=False  # Not storing for this example
    )
    
    # Mock container reference
    class MockContainer:
        container_id = "feature_container_1"
    
    signal_gen.initialize(MockContainer())
    
    # 3. Register strategies
    signal_gen.register_strategy(
        name="Momentum Crossover",
        strategy_id="momentum_1",
        func=momentum_strategy,
        parameters={'fast_period': 20, 'slow_period': 50}
    )
    
    signal_gen.register_strategy(
        name="RSI Mean Reversion", 
        strategy_id="mean_rev_1",
        func=mean_reversion_strategy,
        parameters={'oversold': 30, 'overbought': 70}
    )
    
    # 4. Create performance-aware risk components
    position_sizer = PerformanceAwarePositionSizer(
        base_size_pct=0.02,  # 2% base position size
        max_size_pct=0.05,   # 5% max position size
        use_kelly=True,
        confidence_scaling=True
    )
    
    risk_function = PerformanceAwareRiskFunction(
        signal_observer=signal_observer,
        position_sizer=position_sizer,
        max_positions=5
    )
    
    # 5. Simulate trading over time
    logger.info("\n" + "="*80)
    logger.info("STARTING PERFORMANCE-AWARE TRADING SIMULATION")
    logger.info("="*80 + "\n")
    
    portfolio_value = 100000
    current_time = datetime.now()
    
    # First, generate some historical trades to build performance data
    logger.info("Phase 1: Building Performance History")
    logger.info("-" * 40)
    
    for i in range(20):  # Generate 20 historical trades
        timestamp = current_time - timedelta(hours=100-i*5)
        
        # Generate signals
        bars = create_sample_bars(timestamp)
        features = create_sample_features(timestamp)
        events = signal_gen.process_synchronized_bars(bars, features)
        
        # Publish signal events
        for event in events:
            event_bus.publish(event)
            
            # Simulate position lifecycle
            correlation_id = f"trade_{i}"
            event.correlation_id = correlation_id
            
            # Position open
            open_event = Event(
                event_type=EventType.POSITION_OPEN,
                payload={
                    'strategy_id': event.payload['strategy_id'],
                    'symbol': event.payload['symbol'],
                    'price': bars['SPY']['close'],
                    'quantity': 100,
                    'direction': 'BUY' if event.payload['direction'] == 'BUY' else 'SELL'
                },
                correlation_id=correlation_id
            )
            event_bus.publish(open_event)
            
            # Position close with random P&L
            pnl = random.uniform(-200, 300)  # Random P&L between -$200 and $300
            close_price = bars['SPY']['close'] * (1 + pnl/10000)  # Approximate
            
            close_event = Event(
                event_type=EventType.POSITION_CLOSE,
                payload={
                    'strategy_id': event.payload['strategy_id'],
                    'symbol': event.payload['symbol'],
                    'price': close_price,
                    'quantity': 100,
                    'pnl': pnl,
                    'pnl_pct': pnl / 10000  # Approximate percentage
                },
                correlation_id=correlation_id
            )
            event_bus.publish(close_event)
    
    # Show performance summary after history
    logger.info("\nPerformance Summary After History:")
    summary = signal_observer.get_summary()
    for strategy_id, perf in summary['performance_summary'].items():
        logger.info(f"{strategy_id}: Win Rate={perf['win_rate']:.1%}, "
                   f"Confidence={perf['confidence']:.2f}, PF={perf['profit_factor']:.2f}")
    
    # 6. Now simulate real-time trading with performance-aware decisions
    logger.info("\n\nPhase 2: Real-Time Trading with Performance-Aware Risk")
    logger.info("-" * 40)
    
    for i in range(10):  # Generate 10 new signals
        timestamp = current_time + timedelta(minutes=i*5)
        
        # Generate market data
        bars = create_sample_bars(timestamp)
        features = create_sample_features(timestamp)
        
        # Generate signals
        events = signal_gen.process_synchronized_bars(bars, features)
        
        # Process each signal through risk management
        for event in events:
            if event.event_type == EventType.SIGNAL:
                signal = event.payload
                
                # Apply performance-aware risk validation
                portfolio_state = {
                    'portfolio_value': portfolio_value,
                    'cash': portfolio_value * 0.5,  # 50% cash
                    'positions': risk_function.open_positions
                }
                
                validated_signal = risk_function.validate_signal(signal, portfolio_state)
                
                if validated_signal:
                    # Signal approved - would create order here
                    logger.info(f"✓ Signal approved: {validated_signal['strategy_id']} "
                               f"{validated_signal['direction']} "
                               f"Size: {validated_signal['position_size_pct']:.1%}")
                    
                    # Track as open position
                    position_id = f"pos_{timestamp.timestamp()}"
                    risk_function.open_positions[position_id] = {
                        'strategy_id': validated_signal['strategy_id'],
                        'symbol': validated_signal['symbol'],
                        'value': validated_signal['position_size_value'],
                        'timestamp': timestamp
                    }
                else:
                    logger.info(f"✗ Signal rejected by risk management")
    
    # 7. Final summary
    logger.info("\n\n" + "="*80)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*80)
    
    final_summary = signal_observer.get_summary()
    logger.info(f"\nSignals Observed: {final_summary['signals_observed']}")
    logger.info(f"Trades Completed: {final_summary['trades_completed']}")
    logger.info(f"Strategies Tracked: {final_summary['strategies_tracked']}")
    
    logger.info("\nFinal Performance by Strategy:")
    for strategy_id, perf in final_summary['performance_summary'].items():
        logger.info(f"\n{strategy_id}:")
        logger.info(f"  Total Signals: {perf['total_signals']}")
        logger.info(f"  Win Rate: {perf['win_rate']:.1%}")
        logger.info(f"  Confidence: {perf['confidence']:.2f}")
        logger.info(f"  Profit Factor: {perf['profit_factor']:.2f}")


def demonstrate_edge_cases():
    """Demonstrate how the system handles edge cases."""
    logger.info("\n\n" + "="*80)
    logger.info("EDGE CASE DEMONSTRATIONS")
    logger.info("="*80 + "\n")
    
    # Create components
    event_bus = EventBus("edge_case_bus")
    signal_observer = SignalObserver()
    event_bus.attach_observer(signal_observer)
    
    position_sizer = PerformanceAwarePositionSizer()
    
    # Edge Case 1: New strategy with no history
    logger.info("Edge Case 1: New Strategy (No History)")
    logger.info("-" * 40)
    
    new_signal = {
        'strategy_id': 'new_strategy_1',
        'strategy_name': 'Brand New Strategy',
        'symbol': 'AAPL',
        'direction': 'BUY',
        'value': 1.0,
        'parameters': {'threshold': 0.02}
    }
    
    enhanced = signal_observer.enhance_signal_with_performance(new_signal)
    logger.info(f"Original signal: {new_signal}")
    logger.info(f"Enhanced signal has performance data: {'performance' in enhanced}")
    logger.info("Result: New strategies get default confidence score\n")
    
    # Edge Case 2: Strategy with poor recent performance
    logger.info("Edge Case 2: Poor Recent Performance")
    logger.info("-" * 40)
    
    # Simulate a strategy with recent losses
    strategy_id = 'losing_streak_1'
    
    # Create performance history
    perf = SignalPerformance(
        strategy_id=strategy_id,
        strategy_name='Losing Streak Strategy',
        parameters={}
    )
    
    # Add losing trades
    for i in range(5):
        signal = {'value': 1.0, 'symbol': 'SPY'}
        result = {'pnl': -100, 'pnl_pct': -0.01}
        perf.update_with_result(signal, result)
    
    signal_observer.strategy_performance[strategy_id] = perf
    
    poor_signal = {
        'strategy_id': strategy_id,
        'symbol': 'SPY',
        'direction': 'BUY',
        'value': 1.0
    }
    
    enhanced = signal_observer.enhance_signal_with_performance(poor_signal)
    logger.info(f"Performance after 5 losses:")
    logger.info(f"  Win Rate: {enhanced['performance']['win_rate']:.1%}")
    logger.info(f"  Confidence: {enhanced['performance']['confidence_score']:.2f}")
    logger.info(f"  Recent Performance: {enhanced['performance']['recent_performance']:.1%}")
    logger.info("Result: Low confidence due to losing streak\n")
    
    # Edge Case 3: Strategy with excellent regime-specific performance
    logger.info("Edge Case 3: Regime-Specific Performance")
    logger.info("-" * 40)
    
    regime_strategy_id = 'regime_aware_1'
    regime_perf = SignalPerformance(
        strategy_id=regime_strategy_id,
        strategy_name='Regime Aware Strategy',
        parameters={}
    )
    
    # Add trades in different regimes
    # Bullish regime - excellent performance
    for i in range(10):
        signal = {'value': 1.0, 'symbol': 'SPY', 'classifier_states': {'trend': 'bullish'}}
        result = {'pnl': 200, 'pnl_pct': 0.02}
        regime_perf.update_with_result(signal, result)
    
    # Bearish regime - poor performance
    for i in range(10):
        signal = {'value': 1.0, 'symbol': 'SPY', 'classifier_states': {'trend': 'bearish'}}
        result = {'pnl': -150, 'pnl_pct': -0.015}
        regime_perf.update_with_result(signal, result)
    
    signal_observer.strategy_performance[regime_strategy_id] = regime_perf
    
    # Test in bullish regime
    bullish_signal = {
        'strategy_id': regime_strategy_id,
        'symbol': 'SPY',
        'direction': 'BUY',
        'value': 1.0,
        'classifier_states': {'trend': 'bullish'}
    }
    
    enhanced = signal_observer.enhance_signal_with_performance(bullish_signal)
    logger.info(f"Performance in BULLISH regime:")
    logger.info(f"  Overall Win Rate: {enhanced['performance']['win_rate']:.1%}")
    logger.info(f"  Regime Win Rate: {enhanced['performance']['regime_performance'].get('win_rate', 0):.1%}")
    logger.info(f"  Confidence: {enhanced['performance']['confidence_score']:.2f}")
    
    # Test in bearish regime
    bearish_signal = bullish_signal.copy()
    bearish_signal['classifier_states'] = {'trend': 'bearish'}
    
    enhanced = signal_observer.enhance_signal_with_performance(bearish_signal)
    logger.info(f"\nPerformance in BEARISH regime:")
    logger.info(f"  Overall Win Rate: {enhanced['performance']['win_rate']:.1%}")
    logger.info(f"  Regime Win Rate: {enhanced['performance']['regime_performance'].get('win_rate', 0):.1%}")
    logger.info(f"  Confidence: {enhanced['performance']['confidence_score']:.2f}")
    logger.info("Result: Same strategy gets different confidence in different regimes\n")


if __name__ == "__main__":
    # Run the main simulation
    simulate_trading_session()
    
    # Demonstrate edge cases
    demonstrate_edge_cases()
    
    logger.info("\n" + "="*80)
    logger.info("Example complete! This demonstrates how SignalObserver enables")
    logger.info("sophisticated risk decisions based on real-time performance tracking.")
    logger.info("="*80)