"""Signal flow management between strategy and execution components.

This module manages the flow of signals through the system:
- Signal collection from multiple strategies
- Signal validation and deduplication
- Signal aggregation and prioritization
- Signal to order conversion
- Order routing to execution

The flow follows the architecture:
Strategies → Risk & Portfolio → Execution
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional, Set, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import logging
from ..core.events.types import EventType, Event
from ..core.events.event_bus import EventBus

from .protocols import (
    Signal,
    Order,
    SignalType,
    OrderSide,
    PortfolioStateProtocol,
    PositionSizerProtocol,
    RiskLimitProtocol,
)
from .signal_processing import SignalProcessor, SignalAggregator
from .signal_advanced import (
    SignalValidator,
    SignalCache,
    SignalPrioritizer,
    SignalRouter,
    RiskAdjustedSignalProcessor,
)


class SignalFlowManager:
    """Manages the flow of signals through the system."""
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        enable_caching: bool = True,
        enable_validation: bool = True,
        enable_aggregation: bool = True,
        aggregation_method: str = "weighted_average"
    ):
        """Initialize signal flow manager.
        
        Args:
            event_bus: Event bus for publishing events
            enable_caching: Enable signal deduplication
            enable_validation: Enable signal validation
            enable_aggregation: Enable signal aggregation
            aggregation_method: Method for aggregating signals
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        
        # Configuration
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation
        self.enable_aggregation = enable_aggregation
        
        # Core components
        self._signal_processor = RiskAdjustedSignalProcessor()
        self._signal_validator = SignalValidator()
        self._signal_cache = SignalCache() if enable_caching else None
        self._signal_aggregator = SignalAggregator(aggregation_method) if enable_aggregation else None
        self._signal_prioritizer = SignalPrioritizer()
        self._signal_router = SignalRouter()
        
        # Set default processor
        self._signal_router.set_default_processor(self._signal_processor)
        
        # Signal collection
        self._signal_buffer: List[Signal] = []
        self._buffer_lock = asyncio.Lock()
        
        # Strategy registration
        self._registered_strategies: Set[str] = set()
        self._strategy_weights: Dict[str, Decimal] = {}
        
        # Callbacks
        self._order_callbacks: List[Callable[[Order], None]] = []
        
        # Statistics
        self._total_signals_received = 0
        self._total_orders_generated = 0
        self._signals_rejected = 0
    
    def register_strategy(
        self,
        strategy_id: str,
        weight: Decimal = Decimal("1.0")
    ) -> None:
        """Register a strategy for signal collection.
        
        Args:
            strategy_id: Strategy identifier
            weight: Weight for signal aggregation
        """
        self._registered_strategies.add(strategy_id)
        self._strategy_weights[strategy_id] = weight
        
        self.logger.info(
            "strategy_registered",
            strategy_id=strategy_id,
            weight=str(weight)
        )
    
    def unregister_strategy(self, strategy_id: str) -> None:
        """Unregister a strategy."""
        self._registered_strategies.discard(strategy_id)
        self._strategy_weights.pop(strategy_id, None)
        
        self.logger.info("strategy_unregistered", strategy_id=strategy_id)
    
    def add_order_callback(self, callback: Callable[[Order], None]) -> None:
        """Add callback for when orders are generated.
        
        Args:
            callback: Function to call with generated orders
        """
        self._order_callbacks.append(callback)
    
    async def collect_signal(self, signal: Signal) -> None:
        """Collect a signal from a strategy.
        
        This is the entry point for signals into the system.
        Signals are buffered for batch processing.
        
        Args:
            signal: Signal to collect
        """
        self._total_signals_received += 1
        
        # Validate strategy is registered
        if signal.strategy_id not in self._registered_strategies:
            self.logger.warning(
                "signal_from_unregistered_strategy",
                strategy_id=signal.strategy_id,
                signal_id=signal.signal_id
            )
            self._signals_rejected += 1
            return
        
        # Early validation if enabled
        if self.enable_validation:
            is_valid, failures = self._signal_validator.validate(signal)
            if not is_valid:
                self._signals_rejected += 1
                self._emit_event("signal_rejected", {
                    "signal": signal,
                    "reason": "validation_failed",
                    "failures": failures
                })
                return
        
        # Check cache if enabled
        if self.enable_caching and self._signal_cache:
            if self._signal_cache.is_duplicate(signal):
                self._signals_rejected += 1
                self._emit_event("signal_rejected", {
                    "signal": signal,
                    "reason": "duplicate"
                })
                return
            self._signal_cache.add_signal(signal)
        
        # Add to buffer
        async with self._buffer_lock:
            self._signal_buffer.append(signal)
        
        self._emit_event("signal_collected", {
            "signal": signal,
            "buffer_size": len(self._signal_buffer)
        })
    
    async def process_signals(
        self,
        portfolio_state: PortfolioStateProtocol,
        position_sizer: PositionSizerProtocol,
        risk_limits: List[RiskLimitProtocol],
        market_data: Dict[str, Any]
    ) -> List[Order]:
        """Process collected signals into orders.
        
        This should be called periodically (e.g., on each bar) to process
        buffered signals.
        
        Args:
            portfolio_state: Current portfolio state
            position_sizer: Position sizing strategy
            risk_limits: Risk limits to enforce
            market_data: Current market data
            
        Returns:
            List of generated orders
        """
        # Get signals from buffer
        async with self._buffer_lock:
            signals = self._signal_buffer.copy()
            self._signal_buffer.clear()
        
        if not signals:
            return []
        
        self.logger.info(
            "processing_signals",
            signal_count=len(signals),
            strategies=list(set(s.strategy_id for s in signals))
        )
        
        # Aggregate signals if enabled
        if self.enable_aggregation and self._signal_aggregator:
            signals = self._signal_aggregator.aggregate_signals(
                signals, self._strategy_weights
            )
            self.logger.debug(
                "signals_aggregated",
                original_count=len(self._signal_buffer),
                aggregated_count=len(signals)
            )
        
        # Prioritize signals
        signals = self._signal_prioritizer.prioritize(signals)
        
        # Process signals into orders
        orders = []
        for signal in signals:
            try:
                # Route to appropriate processor
                order = self._signal_router.route_signal(
                    signal=signal,
                    portfolio_state=portfolio_state,
                    position_sizer=position_sizer,
                    risk_limits=risk_limits,
                    market_data=market_data
                )
                
                if order:
                    orders.append(order)
                    self._total_orders_generated += 1
                    
                    # Notify callbacks
                    for callback in self._order_callbacks:
                        try:
                            callback(order)
                        except Exception as e:
                            self.logger.error(
                                "order_callback_error",
                                error=str(e),
                                order_id=order.order_id
                            )
                    
                    # Emit event
                    self._emit_event("order_generated", {
                        "order": order,
                        "signal": signal
                    })
                else:
                    self._signals_rejected += 1
                    
            except Exception as e:
                self.logger.error(
                    "signal_processing_error",
                    signal=signal,
                    error=str(e)
                )
                self._signals_rejected += 1
        
        self.logger.info(
            "signals_processed",
            signals_processed=len(signals),
            orders_generated=len(orders),
            approval_rate=f"{len(orders) / len(signals):.1%}" if signals else "N/A"
        )
        
        return orders
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get flow statistics."""
        approval_rate = 0
        if self._total_signals_received > 0:
            approval_rate = self._total_orders_generated / self._total_signals_received
        
        stats = {
            "total_signals_received": self._total_signals_received,
            "total_orders_generated": self._total_orders_generated,
            "signals_rejected": self._signals_rejected,
            "approval_rate": f"{approval_rate:.1%}",
            "buffer_size": len(self._signal_buffer),
            "registered_strategies": len(self._registered_strategies),
            "signal_processor": self._signal_processor.get_statistics()
        }
        
        if self._signal_cache:
            stats["cache"] = self._signal_cache.get_statistics()
        
        return stats
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event through event bus."""
        if self.event_bus:
            self.event_bus.publish(
                Event(
                    event_type=EventType.SYSTEM,
                    source_id="SignalFlowManager",
                    payload={
                        "type": event_type,
                        **data
                    }
                )
            )


class MultiSymbolSignalFlow:
    """Manages signal flow for multiple symbols and classifiers."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize multi-symbol signal flow.
        
        Args:
            event_bus: Event bus for publishing events
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        
        # Flow managers per classifier
        self._flow_managers: Dict[str, SignalFlowManager] = {}
        
        # Symbol to classifier mapping
        self._symbol_classifiers: Dict[str, str] = {}
        
        # Execution context awareness
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def create_flow_manager(
        self,
        classifier_id: str,
        config: Dict[str, Any]
    ) -> SignalFlowManager:
        """Create flow manager for a classifier.
        
        Args:
            classifier_id: Classifier identifier
            config: Flow manager configuration
            
        Returns:
            Created flow manager
        """
        flow_manager = SignalFlowManager(
            event_bus=self.event_bus,
            enable_caching=config.get("enable_caching", True),
            enable_validation=config.get("enable_validation", True),
            enable_aggregation=config.get("enable_aggregation", True),
            aggregation_method=config.get("aggregation_method", "weighted_average")
        )
        
        self._flow_managers[classifier_id] = flow_manager
        
        self.logger.info(
            "flow_manager_created",
            classifier_id=classifier_id,
            config=config
        )
        
        return flow_manager
    
    def map_symbol_to_classifier(
        self,
        symbol: str,
        classifier_id: str
    ) -> None:
        """Map a symbol to a classifier.
        
        Args:
            symbol: Trading symbol
            classifier_id: Classifier to handle this symbol
        """
        self._symbol_classifiers[symbol] = classifier_id
        
        self.logger.info(
            "symbol_mapped",
            symbol=symbol,
            classifier_id=classifier_id
        )
    
    async def route_signal(self, signal: Signal) -> None:
        """Route signal to appropriate flow manager.
        
        Args:
            signal: Signal to route
        """
        # Find classifier for symbol
        classifier_id = self._symbol_classifiers.get(signal.symbol)
        if not classifier_id:
            self.logger.warning(
                "no_classifier_for_symbol",
                symbol=signal.symbol,
                signal_id=signal.signal_id
            )
            return
        
        # Get flow manager
        flow_manager = self._flow_managers.get(classifier_id)
        if not flow_manager:
            self.logger.warning(
                "no_flow_manager",
                classifier_id=classifier_id,
                signal_id=signal.signal_id
            )
            return
        
        # Route to flow manager
        await flow_manager.collect_signal(signal)
    
    async def process_all_signals(
        self,
        portfolio_states: Dict[str, PortfolioStateProtocol],
        position_sizers: Dict[str, PositionSizerProtocol],
        risk_limits: Dict[str, List[RiskLimitProtocol]],
        market_data: Dict[str, Any]
    ) -> Dict[str, List[Order]]:
        """Process signals for all classifiers.
        
        Args:
            portfolio_states: Portfolio states by classifier
            position_sizers: Position sizers by classifier
            risk_limits: Risk limits by classifier
            market_data: Market data
            
        Returns:
            Orders by classifier
        """
        orders_by_classifier = {}
        
        # Process each classifier's signals
        tasks = []
        for classifier_id, flow_manager in self._flow_managers.items():
            # Get components for this classifier
            portfolio_state = portfolio_states.get(classifier_id)
            position_sizer = position_sizers.get(classifier_id)
            limits = risk_limits.get(classifier_id, [])
            
            if not portfolio_state or not position_sizer:
                self.logger.warning(
                    "missing_components",
                    classifier_id=classifier_id
                )
                continue
            
            # Create processing task
            task = flow_manager.process_signals(
                portfolio_state=portfolio_state,
                position_sizer=position_sizer,
                risk_limits=limits,
                market_data=market_data
            )
            tasks.append((classifier_id, task))
        
        # Execute tasks concurrently
        if tasks:
            results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for (classifier_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    self.logger.error(
                        "classifier_processing_error",
                        classifier_id=classifier_id,
                        error=str(result)
                    )
                    orders_by_classifier[classifier_id] = []
                else:
                    orders_by_classifier[classifier_id] = result
        
        return orders_by_classifier
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all flow managers."""
        return {
            classifier_id: flow_manager.get_statistics()
            for classifier_id, flow_manager in self._flow_managers.items()
        }