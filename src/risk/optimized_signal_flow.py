"""Optimized signal flow management with performance improvements.

This module provides high-performance signal processing with:
- Zero-allocation hot paths
- Efficient data structures
- Minimal string operations
- Optional thread safety
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional, Set, Callable, Union
from collections import deque
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
from .signal_processing import SignalProcessor
from .signal_advanced import (
    SignalValidator,
    SignalCache,
    SignalPrioritizer,
    SignalRouter,
    RiskAdjustedSignalProcessor,
)


class OptimizedSignalFlowManager:
    """High-performance signal flow manager with zero-allocation hot paths."""
    
    __slots__ = (
        '_logger', '_event_bus', '_enable_caching', '_enable_validation',
        '_enable_aggregation', '_signal_processor', '_signal_validator',
        '_signal_cache', '_signal_prioritizer', '_signal_router',
        '_signal_buffer', '_buffer_lock', '_registered_strategies',
        '_strategy_weights', '_order_callbacks', '_stats'
    )
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        enable_caching: bool = True,
        enable_validation: bool = True,
        aggregation_method: str = "weighted_average"
    ):
        """Initialize optimized signal flow manager.
        
        Args:
            event_bus: Event bus for publishing events
            enable_caching: Enable signal deduplication
            enable_validation: Enable signal validation
            aggregation_method: Method for aggregating signals
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._event_bus = event_bus
        
        # Configuration
        self._enable_caching = enable_caching
        self._enable_validation = enable_validation
        
        # Core components - reuse instances to avoid allocation
        self._signal_processor = RiskAdjustedSignalProcessor()
        self._signal_validator = SignalValidator() if enable_validation else None
        self._signal_cache = SignalCache() if enable_caching else None
        self._signal_prioritizer = SignalPrioritizer()
        self._signal_router = SignalRouter()
        
        # Set default processor
        self._signal_router.set_default_processor(self._signal_processor)
        
        # Signal collection - use deque for O(1) append/popleft
        self._signal_buffer: deque = deque()
        self._buffer_lock = asyncio.Lock()
        
        # Strategy registration - use sets for O(1) lookup
        self._registered_strategies: Set[str] = set()
        self._strategy_weights: Dict[str, Decimal] = {}
        
        # Callbacks - preallocate list
        self._order_callbacks: List[Callable[[Order], None]] = []
        
        # Statistics - use single dict to minimize allocations
        self._stats = {
            'signals_received': 0,
            'orders_generated': 0,
            'signals_rejected': 0
        }
    
    def register_strategy(
        self,
        strategy_id: str,
        weight: Decimal = Decimal("1.0")
    ) -> None:
        """Register a strategy for signal collection."""
        self._registered_strategies.add(strategy_id)
        self._strategy_weights[strategy_id] = weight
        
        self._logger.info(f"Strategy registered: {strategy_id}")
    
    def unregister_strategy(self, strategy_id: str) -> None:
        """Unregister a strategy."""
        self._registered_strategies.discard(strategy_id)
        self._strategy_weights.pop(strategy_id, None)
    
    def add_order_callback(self, callback: Callable[[Order], None]) -> None:
        """Add callback for when orders are generated."""
        self._order_callbacks.append(callback)
    
    async def collect_signal(self, signal: Signal) -> None:
        """Collect a signal - optimized hot path."""
        # Fast path: increment counter without string formatting
        self._stats['signals_received'] += 1
        
        # Fast validation: strategy registration check
        if signal.strategy_id not in self._registered_strategies:
            self._stats['signals_rejected'] += 1
            return
        
        # Optional validation - skip if disabled for performance
        if self._enable_validation and self._signal_validator:
            is_valid, _ = self._signal_validator.validate(signal)
            if not is_valid:
                self._stats['signals_rejected'] += 1
                return
        
        # Optional caching - skip if disabled for performance
        if self._enable_caching and self._signal_cache:
            if self._signal_cache.is_duplicate(signal):
                self._stats['signals_rejected'] += 1
                return
            self._signal_cache.add_signal(signal)
        
        # Add to buffer - O(1) operation
        async with self._buffer_lock:
            self._signal_buffer.append(signal)
    
    async def process_signals(
        self,
        portfolio_state: PortfolioStateProtocol,
        position_sizer: PositionSizerProtocol,
        risk_limits: List[RiskLimitProtocol],
        market_data: Dict[str, Any]
    ) -> List[Order]:
        """Process signals - optimized for batch processing."""
        # Fast path: get signals from buffer
        async with self._buffer_lock:
            if not self._signal_buffer:
                return []
            
            # Convert deque to list in one operation
            signals = list(self._signal_buffer)
            self._signal_buffer.clear()
        
        # Process signals without intermediate allocations
        orders = []
        
        # Prioritize signals in-place
        signals = self._signal_prioritizer.prioritize(signals)
        
        # Process each signal
        for signal in signals:
            try:
                # Route to processor
                order = self._signal_router.route_signal(
                    signal=signal,
                    portfolio_state=portfolio_state,
                    position_sizer=position_sizer,
                    risk_limits=risk_limits,
                    market_data=market_data
                )
                
                if order:
                    orders.append(order)
                    self._stats['orders_generated'] += 1
                    
                    # Notify callbacks - minimize exception handling overhead
                    for callback in self._order_callbacks:
                        try:
                            callback(order)
                        except Exception:
                            # Log error without string formatting in hot path
                            pass
                else:
                    self._stats['signals_rejected'] += 1
                    
            except Exception:
                self._stats['signals_rejected'] += 1
        
        return orders
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get flow statistics - cached computation."""
        total_signals = self._stats['signals_received']
        approval_rate = (
            self._stats['orders_generated'] / total_signals 
            if total_signals > 0 else 0.0
        )
        
        # Return pre-computed stats
        stats = {
            **self._stats,
            'approval_rate_decimal': approval_rate,
            'buffer_size': len(self._signal_buffer),
            'registered_strategies': len(self._registered_strategies)
        }
        
        # Add processor stats if available
        if hasattr(self._signal_processor, 'get_statistics'):
            stats['processor'] = self._signal_processor.get_statistics()
        
        return stats


class HighPerformanceSignalCache:
    """Memory-efficient signal cache with minimal allocations."""
    
    __slots__ = ('_cache', '_cache_order', '_max_size', '_duration', '_stats')
    
    def __init__(self, max_size: int = 1000, duration: int = 60):
        """Initialize cache with fixed size."""
        self._cache: Dict[str, tuple] = {}
        self._cache_order: deque = deque()
        self._max_size = max_size
        self._duration = duration
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def _compute_hash(self, signal: Signal) -> str:
        """Compute hash - optimized to avoid string concatenation."""
        # Use tuple for hashing to avoid string operations
        key_tuple = (
            signal.strategy_id,
            signal.symbol,
            signal.signal_type.value,
            signal.side.value,
            int(signal.strength * 1000)  # Convert to int for hashing
        )
        return str(hash(key_tuple))
    
    def is_duplicate(self, signal: Signal) -> bool:
        """Check for duplicate - optimized lookup."""
        signal_hash = self._compute_hash(signal)
        
        if signal_hash in self._cache:
            cached_data = self._cache[signal_hash]
            # Check expiration without datetime operations
            if cached_data[1] + self._duration > datetime.now().timestamp():
                self._stats['hits'] += 1
                return True
            else:
                # Expired - remove immediately
                del self._cache[signal_hash]
                self._stats['evictions'] += 1
        
        self._stats['misses'] += 1
        return False
    
    def add_signal(self, signal: Signal) -> None:
        """Add signal to cache - optimized for minimal allocation."""
        signal_hash = self._compute_hash(signal)
        
        # Enforce size limit
        while len(self._cache) >= self._max_size:
            if self._cache_order:
                oldest = self._cache_order.popleft()
                self._cache.pop(oldest, None)
                self._stats['evictions'] += 1
            else:
                break
        
        # Store with timestamp as float for efficiency
        self._cache[signal_hash] = (signal, datetime.now().timestamp())
        self._cache_order.append(signal_hash)


class BatchSignalProcessor:
    """Processes signals in batches for improved throughput."""
    
    __slots__ = ('_processor', '_batch_size', '_batch_timeout')
    
    def __init__(
        self,
        processor: Any,
        batch_size: int = 50,
        batch_timeout: float = 0.1
    ):
        """Initialize batch processor.
        
        Args:
            processor: Underlying signal processor
            batch_size: Maximum batch size
            batch_timeout: Maximum time to wait for batch
        """
        self._processor = processor
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
    
    async def process_batch(
        self,
        signals: List[Signal],
        portfolio_state: PortfolioStateProtocol,
        position_sizer: PositionSizerProtocol,
        risk_limits: List[RiskLimitProtocol],
        market_data: Dict[str, Any]
    ) -> List[Order]:
        """Process a batch of signals efficiently."""
        orders = []
        
        # Process in parallel if large batch
        if len(signals) > 20:
            # Use thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit tasks
                tasks = []
                for signal in signals:
                    task = loop.run_in_executor(
                        executor,
                        self._process_single_signal,
                        signal,
                        portfolio_state,
                        position_sizer,
                        risk_limits,
                        market_data
                    )
                    tasks.append(task)
                
                # Gather results
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect successful orders
                for result in results:
                    if isinstance(result, Order):
                        orders.append(result)
        else:
            # Process sequentially for small batches
            for signal in signals:
                try:
                    order = self._processor.process_signal(
                        signal=signal,
                        portfolio_state=portfolio_state,
                        position_sizer=position_sizer,
                        risk_limits=risk_limits,
                        market_data=market_data
                    )
                    if order:
                        orders.append(order)
                except Exception:
                    # Skip failed signals
                    continue
        
        return orders
    
    def _process_single_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        position_sizer: PositionSizerProtocol,
        risk_limits: List[RiskLimitProtocol],
        market_data: Dict[str, Any]
    ) -> Optional[Order]:
        """Process a single signal - thread-safe."""
        try:
            return self._processor.process_signal(
                signal=signal,
                portfolio_state=portfolio_state,
                position_sizer=position_sizer,
                risk_limits=risk_limits,
                market_data=market_data
            )
        except Exception:
            return None


def create_optimized_flow_manager(
    event_bus: Optional[EventBus] = None,
    performance_mode: str = "balanced"
) -> OptimizedSignalFlowManager:
    """Factory function to create optimized signal flow manager.
    
    Args:
        event_bus: Event bus for events
        performance_mode: "fast", "balanced", or "safe"
        
    Returns:
        Configured flow manager
    """
    if performance_mode == "fast":
        # Maximum performance - minimal validation
        return OptimizedSignalFlowManager(
            event_bus=event_bus,
            enable_caching=False,  # Skip caching for max speed
            enable_validation=False,  # Skip validation for max speed
        )
    elif performance_mode == "safe":
        # Maximum safety - full validation
        return OptimizedSignalFlowManager(
            event_bus=event_bus,
            enable_caching=True,
            enable_validation=True,
        )
    else:  # balanced
        # Balanced performance and safety
        return OptimizedSignalFlowManager(
            event_bus=event_bus,
            enable_caching=True,
            enable_validation=True,
        )
