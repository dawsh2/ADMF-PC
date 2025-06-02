"""
File: src/strategy/strategy_coordinator.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#multiple-strategies
Step: 4 - Multiple Strategies
Dependencies: events, logging, containers, protocols

Coordinates multiple strategy containers with signal aggregation.
Manages strategy orchestration, performance tracking, and consensus building.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
import uuid

from ..core.events.isolation import get_enhanced_isolation_manager
from ..core.logging.structured import ComponentLogger
from ..data.models import Bar
from .protocols import TradingSignal
from .signal_aggregation import SignalAggregator, WeightedVotingAggregator
from .performance_tracking import StrategyPerformanceTracker


@dataclass
class AggregatedSignal:
    """Signal with strategy metadata for aggregation"""
    strategy_id: str
    signal: TradingSignal
    weight: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusSignal:
    """Aggregated consensus signal from multiple strategies"""
    symbol: str
    direction: str  # "BUY", "SELL", "HOLD"
    strength: float
    confidence: float
    contributing_strategies: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyCoordinatorConfig:
    """Configuration for strategy coordinator"""
    strategies: Dict[str, Dict[str, Any]]
    aggregation_method: str = "weighted_voting"
    min_consensus_confidence: float = 0.6
    performance_tracking_enabled: bool = True
    dynamic_weight_adjustment: bool = False
    weight_update_frequency: int = 100  # trades


class StrategyCoordinator:
    """
    Coordinates multiple strategy containers.
    Manages signal aggregation and resource allocation.
    """
    
    def __init__(self, container_id: str, config: StrategyCoordinatorConfig):
        self.container_id = container_id
        self.config = config
        
        # Create master event bus
        self.isolation_manager = get_enhanced_isolation_manager()
        self.event_bus = self.isolation_manager.create_container_bus(
            f"{container_id}_coordinator"
        )
        
        # Strategy containers - will be created by factory
        self.strategy_containers: Dict[str, Any] = {}  # Use Any for duck typing
        self.strategy_weights: Dict[str, float] = {}
        
        # Signal aggregation
        self.signal_aggregator = self._create_aggregator(config.aggregation_method)
        self.aggregated_signals: List[AggregatedSignal] = []
        self.pending_signals: Dict[str, List[AggregatedSignal]] = {}  # by symbol
        
        # Performance tracking
        self.performance_tracker = StrategyPerformanceTracker()
        self.total_signals_processed = 0
        self.consensus_signals_generated = 0
        
        # Setup logging
        self.logger = ComponentLogger("StrategyCoordinator", container_id)
        
        self.logger.info(f"Initialized StrategyCoordinator with {len(config.strategies)} strategies")
    
    def _create_aggregator(self, method: str) -> SignalAggregator:
        """Create signal aggregator based on method"""
        if method == "weighted_voting":
            return WeightedVotingAggregator(self.config.min_consensus_confidence)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def add_strategy_container(self, strategy_id: str, container: Any, weight: float = 1.0) -> None:
        """Add a strategy container to coordination"""
        self.strategy_containers[strategy_id] = container
        self.strategy_weights[strategy_id] = weight
        
        # Subscribe to strategy signals if container has event bus
        if hasattr(container, 'event_bus'):
            container.event_bus.subscribe(
                "SIGNAL",
                lambda signal: self.on_strategy_signal(strategy_id, signal)
            )
        
        self.logger.info(f"Added strategy container: {strategy_id} (weight: {weight})")
    
    def on_bar(self, bar: Bar) -> None:
        """Distribute market data to all strategies"""
        # Clear previous signals for this timestamp
        self.pending_signals.clear()
        
        # Send to all strategies
        for strategy_id, container in self.strategy_containers.items():
            try:
                if hasattr(container, 'on_bar'):
                    container.on_bar(bar)
                elif hasattr(container, 'process_bar'):
                    container.process_bar(bar)
            except Exception as e:
                self.logger.error(
                    f"Strategy {strategy_id} failed on bar: {e}"
                )
                # Continue with other strategies
        
        # Process any aggregated signals after a short delay to collect all signals
        self._process_pending_signals(bar.timestamp)
    
    def on_strategy_signal(self, strategy_id: str, signal: TradingSignal) -> None:
        """Handle signal from individual strategy"""
        self.total_signals_processed += 1
        
        self.logger.log_event_flow(
            "SIGNAL_RECEIVED", f"strategy_{strategy_id}", "coordinator",
            f"Signal: {signal.direction} {signal.symbol}"
        )
        
        # Create aggregated signal
        agg_signal = AggregatedSignal(
            strategy_id=strategy_id,
            signal=signal,
            weight=self.strategy_weights.get(strategy_id, 1.0),
            timestamp=datetime.now()
        )
        
        # Group by symbol for aggregation
        symbol = signal.symbol
        if symbol not in self.pending_signals:
            self.pending_signals[symbol] = []
        
        self.pending_signals[symbol].append(agg_signal)
    
    def _process_pending_signals(self, timestamp: datetime) -> None:
        """Aggregate and emit consensus signals"""
        for symbol, signals in self.pending_signals.items():
            if not signals:
                continue
            
            # Aggregate signals for this symbol
            consensus = self.signal_aggregator.aggregate(signals)
            
            if consensus:
                self.consensus_signals_generated += 1
                
                # Log aggregation details
                strategy_ids = [s.strategy_id for s in signals]
                self.logger.info(
                    f"Consensus signal for {symbol}: {consensus.direction} "
                    f"(confidence: {consensus.confidence:.2f}, "
                    f"strategies: {strategy_ids})"
                )
                
                # Emit aggregated signal
                self.event_bus.publish("AGGREGATED_SIGNAL", consensus)
                
                # Track performance if enabled
                if self.config.performance_tracking_enabled:
                    self.performance_tracker.track_consensus_signal(consensus, signals)
    
    def _group_signals_by_symbol(self) -> Dict[str, List[AggregatedSignal]]:
        """Group aggregated signals by symbol"""
        groups = {}
        for signal in self.aggregated_signals:
            symbol = signal.signal.symbol
            if symbol not in groups:
                groups[symbol] = []
            groups[symbol].append(signal)
        return groups
    
    def update_strategy_weights(self, new_weights: Dict[str, float]) -> None:
        """Update strategy weights dynamically"""
        for strategy_id, weight in new_weights.items():
            if strategy_id in self.strategy_weights:
                old_weight = self.strategy_weights[strategy_id]
                self.strategy_weights[strategy_id] = weight
                self.logger.info(
                    f"Updated weight for {strategy_id}: {old_weight:.2f} -> {weight:.2f}"
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get coordinator performance summary"""
        consensus_rate = (
            self.consensus_signals_generated / max(1, self.total_signals_processed)
        )
        
        return {
            'coordinator_id': self.container_id,
            'active_strategies': len(self.strategy_containers),
            'strategy_weights': self.strategy_weights.copy(),
            'total_signals_processed': self.total_signals_processed,
            'consensus_signals_generated': self.consensus_signals_generated,
            'consensus_rate': consensus_rate,
            'aggregation_method': self.config.aggregation_method,
            'min_confidence': self.config.min_consensus_confidence,
            'performance_tracking': self.performance_tracker.get_summary()
        }
    
    def shutdown(self) -> None:
        """Shutdown coordinator and cleanup resources"""
        # Shutdown all strategy containers
        for strategy_id, container in self.strategy_containers.items():
            try:
                if hasattr(container, 'shutdown'):
                    container.shutdown()
                elif hasattr(container, 'deactivate'):
                    container.deactivate()
            except Exception as e:
                self.logger.error(f"Error shutting down {strategy_id}: {e}")
        
        # Clear state
        self.strategy_containers.clear()
        self.pending_signals.clear()
        
        self.logger.info("StrategyCoordinator shutdown complete")


def create_strategy_coordinator(
    container_id: str,
    strategies_config: Dict[str, Dict[str, Any]],
    aggregation_method: str = "weighted_voting",
    min_confidence: float = 0.6
) -> StrategyCoordinator:
    """Factory function to create strategy coordinator"""
    config = StrategyCoordinatorConfig(
        strategies=strategies_config,
        aggregation_method=aggregation_method,
        min_consensus_confidence=min_confidence
    )
    
    return StrategyCoordinator(
        container_id=container_id or f"coordinator_{uuid.uuid4().hex[:8]}",
        config=config
    )


def create_test_multi_strategy_coordinator() -> StrategyCoordinator:
    """Create coordinator for testing purposes"""
    test_config = {
        'momentum': {
            'type': 'momentum',
            'params': {'lookback_period': 20},
            'weight': 0.4
        },
        'mean_reversion': {
            'type': 'mean_reversion', 
            'params': {'period': 15},
            'weight': 0.3
        },
        'trend_following': {
            'type': 'trend_following',
            'params': {'period': 50},
            'weight': 0.3
        }
    }
    
    return create_strategy_coordinator(
        "test_multi_strategy",
        test_config,
        aggregation_method="weighted_voting",
        min_confidence=0.6
    )