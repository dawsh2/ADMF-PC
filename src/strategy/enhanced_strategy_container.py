"""
File: src/strategy/enhanced_strategy_container.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#enhanced-strategy-container
Step: 4 - Multiple Strategies
Dependencies: events, logging, protocols, datetime

Enhanced strategy container with lifecycle management and multi-strategy support.
Extends the base StrategyContainer with performance tracking and adaptive behavior.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass
import uuid
import asyncio

from ..core.events.isolation import get_enhanced_isolation_manager
from ..core.logging.structured import ComponentLogger
from ..data.models import Bar
from .protocols import Strategy, TradingSignal
from .performance_tracking import StrategyMetrics, create_performance_tracker


@dataclass
class StrategyContainerConfig:
    """Configuration for enhanced strategy container"""
    strategy_type: str
    strategy_params: Dict[str, Any]
    initial_weight: float = 1.0
    enable_performance_tracking: bool = True
    min_confidence_threshold: float = 0.5
    max_signal_frequency: Optional[int] = None  # Max signals per minute
    lifecycle_management: bool = True


class EnhancedStrategyContainer:
    """
    Enhanced container for individual strategy with lifecycle management.
    
    Features:
    - Performance tracking and metrics
    - Adaptive signal filtering
    - Lifecycle management (active/inactive/paused)
    - Resource monitoring
    - Error isolation and recovery
    """
    
    def __init__(self, container_id: str, config: StrategyContainerConfig):
        self.container_id = container_id
        self.config = config
        
        # Create isolated components
        self.isolation_manager = get_enhanced_isolation_manager()
        self.event_bus = self.isolation_manager.create_container_bus(
            f"{container_id}_strategy"
        )
        
        # Strategy instance
        self.strategy: Optional[Strategy] = None
        self.strategy_state = "initializing"
        
        # Lifecycle management
        self.is_active = True
        self.is_paused = False
        self.last_error: Optional[Exception] = None
        self.error_count = 0
        self.max_errors = 5
        
        # Performance metrics
        self.signals_generated = 0
        self.signals_accepted = 0  # Above confidence threshold
        self.last_signal_time: Optional[datetime] = None
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        
        # Signal rate limiting
        self.signal_history: List[datetime] = []
        self.signal_rate_window = 60  # seconds
        
        # Performance tracking
        if config.enable_performance_tracking:
            self.performance_tracker = create_performance_tracker(container_id)
            self.metrics = StrategyMetrics(container_id)
        else:
            self.performance_tracker = None
            self.metrics = None
        
        # Current market state
        self._current_indicators: Dict[str, Any] = {}
        self._last_bar: Optional[Bar] = None
        
        # Setup logging
        self.logger = ComponentLogger("EnhancedStrategyContainer", container_id)
        
        # Initialize strategy
        self._initialize_strategy()
    
    def _initialize_strategy(self) -> None:
        """Initialize the strategy instance"""
        try:
            self.strategy = self._create_strategy(
                self.config.strategy_type, 
                self.config.strategy_params
            )
            self.strategy_state = "active"
            self.logger.info(f"Strategy {self.config.strategy_type} initialized successfully")
        except Exception as e:
            self.strategy_state = "error"
            self.last_error = e
            self.logger.error(f"Failed to initialize strategy: {e}")
            raise
    
    def _create_strategy(self, strategy_type: str, params: Dict[str, Any]) -> Strategy:
        """Factory method for strategy creation"""
        strategy_map = {
            'momentum': self._create_momentum_strategy,
            'mean_reversion': self._create_mean_reversion_strategy,
            'trend_following': self._create_trend_following_strategy,
            'pairs_trading': self._create_pairs_trading_strategy
        }
        
        if strategy_type not in strategy_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_map[strategy_type](params)
    
    def _create_momentum_strategy(self, params: Dict[str, Any]) -> Strategy:
        """Create momentum strategy"""
        from .strategies.momentum import MomentumStrategy
        return MomentumStrategy(**params)
    
    def _create_mean_reversion_strategy(self, params: Dict[str, Any]) -> Strategy:
        """Create mean reversion strategy"""
        from .strategies.mean_reversion import MeanReversionStrategy
        return MeanReversionStrategy(**params)
    
    def _create_trend_following_strategy(self, params: Dict[str, Any]) -> Strategy:
        """Create trend following strategy"""
        from .strategies.trend_following import TrendFollowingStrategy
        return TrendFollowingStrategy(**params)
    
    def _create_pairs_trading_strategy(self, params: Dict[str, Any]) -> Strategy:
        """Create pairs trading strategy"""
        from .strategies.pairs_trading import PairsTradingStrategy
        return PairsTradingStrategy(**params)
    
    def on_bar(self, bar: Bar) -> None:
        """Process market data"""
        if not self._should_process():
            return
        
        start_time = datetime.now()
        
        try:
            self._last_bar = bar
            
            # Generate signal
            signal = self._generate_signal(bar)
            
            if signal and self._should_emit_signal(signal):
                self._emit_signal(signal)
            
            # Update processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_metrics(processing_time)
            
        except Exception as e:
            self._handle_error(e)
    
    def _generate_signal(self, bar: Bar) -> Optional[TradingSignal]:
        """Generate signal from strategy"""
        if not self.strategy:
            return None
        
        try:
            # Prepare strategy input
            strategy_input = {
                'market_data': {bar.symbol: {
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'price': bar.close  # Convenience field
                }},
                'indicators': self._current_indicators,
                'timestamp': bar.timestamp
            }
            
            # Generate signals
            signals = self.strategy.generate_signals(strategy_input)
            
            if signals and len(signals) > 0:
                # Return first signal (most strategies generate one signal per bar)
                signal = signals[0]
                self.signals_generated += 1
                
                # Track in performance metrics if enabled
                if self.metrics:
                    self.metrics.add_signal(
                        signal.strength, 
                        contributed_to_consensus=False  # Will be updated later
                    )
                
                return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            raise
        
        return None
    
    def _should_process(self) -> bool:
        """Check if container should process data"""
        return (
            self.is_active and 
            not self.is_paused and 
            self.strategy_state == "active" and
            self.error_count < self.max_errors
        )
    
    def _should_emit_signal(self, signal: TradingSignal) -> bool:
        """Check if signal should be emitted"""
        # Check confidence threshold
        if signal.strength < self.config.min_confidence_threshold:
            return False
        
        # Check rate limiting
        if self.config.max_signal_frequency and self._is_rate_limited():
            self.logger.debug("Signal rate limited")
            return False
        
        return True
    
    def _is_rate_limited(self) -> bool:
        """Check if signal rate is limited"""
        if not self.config.max_signal_frequency:
            return False
        
        now = datetime.now()
        
        # Remove old signals from history
        cutoff = now.timestamp() - self.signal_rate_window
        self.signal_history = [
            ts for ts in self.signal_history 
            if ts.timestamp() > cutoff
        ]
        
        # Check if we're at the limit
        signals_per_minute = len(self.signal_history)
        return signals_per_minute >= self.config.max_signal_frequency
    
    def _emit_signal(self, signal: TradingSignal) -> None:
        """Emit signal through event bus"""
        self.signals_accepted += 1
        self.last_signal_time = datetime.now()
        
        # Add to rate limiting history
        self.signal_history.append(self.last_signal_time)
        
        # Emit signal
        self.event_bus.publish("SIGNAL", signal)
        
        self.logger.info(
            f"Signal emitted: {signal.direction} {signal.symbol} "
            f"strength={signal.strength:.2f}"
        )
    
    def _update_processing_metrics(self, processing_time: float) -> None:
        """Update processing time metrics"""
        self.total_processing_time += processing_time
        
        # Calculate rolling average
        if self.signals_generated > 0:
            self.avg_processing_time = (
                self.total_processing_time / self.signals_generated
            )
    
    def _handle_error(self, error: Exception) -> None:
        """Handle processing errors"""
        self.error_count += 1
        self.last_error = error
        
        self.logger.error(f"Strategy error (count: {self.error_count}): {error}")
        
        if self.error_count >= self.max_errors:
            self.is_active = False
            self.strategy_state = "disabled"
            self.logger.error(f"Strategy disabled due to too many errors")
    
    def update_indicators(self, indicators: Dict[str, Any]) -> None:
        """Update current indicators"""
        self._current_indicators.update(indicators)
        self.logger.debug(f"Updated indicators: {indicators}")
    
    def pause(self) -> None:
        """Pause strategy processing"""
        self.is_paused = True
        self.strategy_state = "paused"
        self.logger.info("Strategy paused")
    
    def resume(self) -> None:
        """Resume strategy processing"""
        self.is_paused = False
        if self.is_active and self.error_count < self.max_errors:
            self.strategy_state = "active"
            self.logger.info("Strategy resumed")
    
    def deactivate(self) -> None:
        """Deactivate strategy"""
        self.is_active = False
        self.strategy_state = "inactive"
        
        if self.strategy and hasattr(self.strategy, 'cleanup'):
            self.strategy.cleanup()
        
        self.logger.info("Strategy deactivated")
    
    def reset_errors(self) -> None:
        """Reset error count"""
        self.error_count = 0
        self.last_error = None
        
        if self.is_active and not self.is_paused:
            self.strategy_state = "active"
            
        self.logger.info("Error count reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get comprehensive container state"""
        state = {
            'container_id': self.container_id,
            'strategy_type': self.config.strategy_type,
            'strategy_state': self.strategy_state,
            'is_active': self.is_active,
            'is_paused': self.is_paused,
            'signals_generated': self.signals_generated,
            'signals_accepted': self.signals_accepted,
            'acceptance_rate': (
                self.signals_accepted / max(1, self.signals_generated)
            ),
            'last_signal_time': self.last_signal_time,
            'error_count': self.error_count,
            'last_error': str(self.last_error) if self.last_error else None,
            'avg_processing_time_ms': self.avg_processing_time * 1000,
            'current_indicators_count': len(self._current_indicators)
        }
        
        # Add strategy-specific state if available
        if self.strategy and hasattr(self.strategy, 'get_state'):
            state['strategy_internal_state'] = self.strategy.get_state()
        
        # Add performance metrics if available
        if self.metrics:
            state['performance_metrics'] = {
                'trade_count': self.metrics.trade_count,
                'win_rate': self.metrics.win_rate,
                'sharpe_ratio': self.metrics.calculate_sharpe_ratio(),
                'total_return': self.metrics.total_return,
                'consensus_contribution_rate': self.metrics.consensus_contribution_rate
            }
        
        return state
    
    def get_capabilities(self) -> Set[str]:
        """Get container capabilities"""
        capabilities = {
            f"strategy.{self.config.strategy_type}",
            "signal_generation",
            "performance_tracking"
        }
        
        if self.strategy and hasattr(self.strategy, 'get_capabilities'):
            capabilities.update(self.strategy.get_capabilities())
        
        return capabilities
    
    def shutdown(self) -> None:
        """Shutdown container and cleanup resources"""
        self.deactivate()
        
        # Clear state
        self._current_indicators.clear()
        self.signal_history.clear()
        
        self.logger.info("Enhanced strategy container shutdown complete")


def create_enhanced_strategy_container(
    strategy_type: str,
    strategy_params: Dict[str, Any],
    container_id: str = None,
    initial_weight: float = 1.0,
    enable_performance_tracking: bool = True
) -> EnhancedStrategyContainer:
    """Factory function to create enhanced strategy container"""
    config = StrategyContainerConfig(
        strategy_type=strategy_type,
        strategy_params=strategy_params,
        initial_weight=initial_weight,
        enable_performance_tracking=enable_performance_tracking
    )
    
    return EnhancedStrategyContainer(
        container_id=container_id or f"strategy_{strategy_type}_{uuid.uuid4().hex[:8]}",
        config=config
    )


def create_test_strategy_containers() -> List[EnhancedStrategyContainer]:
    """Create test strategy containers for development"""
    containers = []
    
    # Momentum strategy
    momentum_container = create_enhanced_strategy_container(
        strategy_type="momentum",
        strategy_params={'lookback_period': 20, 'momentum_threshold': 0.001},
        container_id="test_momentum",
        initial_weight=0.4
    )
    containers.append(momentum_container)
    
    # Mean reversion strategy  
    mean_rev_container = create_enhanced_strategy_container(
        strategy_type="mean_reversion",
        strategy_params={'period': 15, 'threshold': 2.0},
        container_id="test_mean_reversion",
        initial_weight=0.3
    )
    containers.append(mean_rev_container)
    
    # Trend following strategy
    trend_container = create_enhanced_strategy_container(
        strategy_type="trend_following", 
        strategy_params={'period': 50, 'min_trend_strength': 0.7},
        container_id="test_trend_following",
        initial_weight=0.3
    )
    containers.append(trend_container)
    
    return containers