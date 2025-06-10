# Barrier-Level Duplicate Order Prevention - Protocol + Composition

## Overview

This document describes implementing duplicate order prevention at the barrier level in `src/core/containers/sync.py`. The approach modifies the canonical `StrategyOrchestrator` implementation to include duplicate prevention as a core feature, preventing duplicate orders at the symbol/timeframe level during synchronized bar processing. This follows ADMF-PC's Protocol + Composition architecture with zero inheritance.

## Protocol Definitions

### OrderTrackingProtocol

```python
# Add to src/core/containers/protocols.py

from typing import Protocol, runtime_checkable, Dict, Set, Tuple, Optional, Any

@runtime_checkable
class OrderTrackingProtocol(Protocol):
    """Protocol for tracking pending orders by symbol/timeframe."""
    
    def has_pending_orders(self, container_id: str, symbol: str, timeframe: str) -> bool:
        """Check if container has pending orders for symbol/timeframe."""
        ...
    
    def register_order(self, container_id: str, symbol: str, timeframe: str, 
                      order_id: str, metadata: Optional[Dict] = None) -> None:
        """Register new pending order."""
        ...
    
    def clear_order(self, container_id: str, symbol: str, timeframe: str, order_id: str) -> bool:
        """Clear completed order."""
        ...
    
    def get_pending_summary(self, container_id: str) -> Dict[str, int]:
        """Get summary of pending orders by symbol_timeframe."""
        ...


@runtime_checkable
class DuplicatePreventionProtocol(Protocol):
    """Protocol for filtering strategies to prevent duplicate orders."""
    
    def filter_strategies_for_duplicates(self, strategy_specs: List['StrategySpecification'], 
                                       container_id: str) -> List['StrategySpecification']:
        """Filter strategies to prevent duplicate orders."""
        ...
    
    def should_process_strategy(self, strategy_spec: 'StrategySpecification', 
                              container_id: str) -> bool:
        """Check if strategy should be processed (no pending orders)."""
        ...
    
    def get_prevention_metrics(self) -> Dict[str, Any]:
        """Get duplicate prevention metrics."""
        ...


@runtime_checkable
class BarrierSynchronizationProtocol(Protocol):
    """Protocol for enhanced barrier synchronization with duplicate prevention."""
    
    def process_synchronized_bars_with_prevention(self, aligned_data: Dict[str, Event], 
                                                 sync_time: datetime, 
                                                 container_id: str) -> None:
        """Process synchronized bars with duplicate order prevention."""
        ...
    
    def register_order_tracker(self, order_tracker: OrderTrackingProtocol) -> None:
        """Register order tracker for duplicate prevention."""
        ...
    
    def get_sync_metrics(self) -> Dict[str, Any]:
        """Get synchronization metrics."""
        ...
```

## Implementation Components

### OrderTracker - Pure Composition

```python
# Add to src/core/containers/sync.py

from typing import Dict, Set, List, Tuple, Optional, Any
from collections import defaultdict
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OrderTracker:
    """
    Pure composition component for tracking pending orders.
    Implements OrderTrackingProtocol via composition (no inheritance).
    """
    
    def __post_init__(self):
        # Structure: {container_id: {(symbol, timeframe): {order_id1, order_id2, ...}}}
        self._pending_orders: Dict[str, Dict[Tuple[str, str], Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._order_lock = threading.RLock()
        
        # Metrics
        self._metrics = {
            'duplicates_prevented': 0,
            'orders_tracked': 0,
            'orders_cleared': 0
        }
        
        # Order metadata for debugging
        self._order_metadata: Dict[str, Dict[str, Any]] = {}
    
    # Implement OrderTrackingProtocol
    
    def has_pending_orders(self, container_id: str, symbol: str, timeframe: str) -> bool:
        """Check if container has pending orders for specific symbol/timeframe."""
        with self._order_lock:
            return len(self._pending_orders[container_id][(symbol, timeframe)]) > 0
    
    def register_order(self, container_id: str, symbol: str, timeframe: str, 
                      order_id: str, metadata: Optional[Dict] = None) -> None:
        """Register new pending order."""
        with self._order_lock:
            self._pending_orders[container_id][(symbol, timeframe)].add(order_id)
            self._metrics['orders_tracked'] += 1
            
            # Store metadata for debugging
            order_key = f"{container_id}:{order_id}"
            self._order_metadata[order_key] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
            
            logger.debug(
                f"Registered order {order_id} for {container_id}:{symbol}_{timeframe}. "
                f"Pending count: {len(self._pending_orders[container_id][(symbol, timeframe)])}"
            )
    
    def clear_order(self, container_id: str, symbol: str, timeframe: str, order_id: str) -> bool:
        """Clear completed order."""
        with self._order_lock:
            symbol_tf_key = (symbol, timeframe)
            if order_id in self._pending_orders[container_id][symbol_tf_key]:
                self._pending_orders[container_id][symbol_tf_key].remove(order_id)
                self._metrics['orders_cleared'] += 1
                
                # Clean up metadata
                order_key = f"{container_id}:{order_id}"
                self._order_metadata.pop(order_key, None)
                
                logger.debug(
                    f"Cleared order {order_id} for {container_id}:{symbol}_{timeframe}. "
                    f"Remaining: {len(self._pending_orders[container_id][symbol_tf_key])}"
                )
                return True
            
            return False
    
    def get_pending_summary(self, container_id: str) -> Dict[str, int]:
        """Get summary of pending orders by symbol_timeframe."""
        with self._order_lock:
            summary = {}
            for (symbol, timeframe), order_ids in self._pending_orders[container_id].items():
                if order_ids:  # Only non-empty sets
                    key = f"{symbol}_{timeframe}"
                    summary[key] = len(order_ids)
            return summary
    
    # Additional utility methods
    
    def get_pending_count(self, container_id: str, symbol: str, timeframe: str) -> int:
        """Get count of pending orders for container/symbol/timeframe."""
        with self._order_lock:
            return len(self._pending_orders[container_id][(symbol, timeframe)])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get order tracking metrics."""
        with self._order_lock:
            total_pending = sum(
                len(orders) 
                for container_orders in self._pending_orders.values() 
                for orders in container_orders.values()
            )
            
            return {
                **self._metrics,
                'total_pending': total_pending,
                'containers_with_orders': len([
                    cid for cid, container_orders in self._pending_orders.items()
                    if any(orders for orders in container_orders.values())
                ])
            }


@dataclass
class DuplicateOrderPrevention:
    """
    Pure composition component for preventing duplicate orders.
    Implements DuplicatePreventionProtocol via composition (no inheritance).
    """
    
    order_tracker: OrderTrackingProtocol
    
    def __post_init__(self):
        # Prevention metrics
        self._metrics = {
            'strategies_processed': 0,
            'strategies_filtered': 0,
            'bars_processed': 0
        }
        self._peak_pending_orders = 0
        self._prevented_by_symbol: Dict[str, int] = defaultdict(int)
    
    # Implement DuplicatePreventionProtocol
    
    def filter_strategies_for_duplicates(self, strategy_specs: List[StrategySpecification], 
                                       container_id: str) -> List[StrategySpecification]:
        """Filter strategies to prevent duplicate orders at symbol/timeframe level."""
        if not strategy_specs:
            return strategy_specs
        
        filtered_strategies = []
        
        for strategy_spec in strategy_specs:
            self._metrics['strategies_processed'] += 1
            
            if self.should_process_strategy(strategy_spec, container_id):
                filtered_strategies.append(strategy_spec)
            else:
                self._metrics['strategies_filtered'] += 1
                
                # Log which symbol/timeframes were blocked
                blocked_pairs = []
                for symbol in strategy_spec.data_requirement.symbols:
                    for timeframe in strategy_spec.data_requirement.timeframes:
                        if self.order_tracker.has_pending_orders(container_id, symbol, timeframe):
                            blocked_pairs.append(f"{symbol}_{timeframe}")
                            self._prevented_by_symbol[symbol] += 1
                
                logger.info(
                    f"DUPLICATE PREVENTED: Strategy {strategy_spec.strategy_id} blocked "
                    f"for container {container_id}. Pending orders: {blocked_pairs}"
                )
        
        logger.debug(
            f"Strategy filtering for {container_id}: "
            f"{len(strategy_specs)} input → {len(filtered_strategies)} output "
            f"({len(strategy_specs) - len(filtered_strategies)} duplicates filtered)"
        )
        
        return filtered_strategies
    
    def should_process_strategy(self, strategy_spec: StrategySpecification, 
                              container_id: str) -> bool:
        """Check if strategy should be processed (no pending orders for its symbols/timeframes)."""
        # Check if ANY of the strategy's symbol/timeframe combinations have pending orders
        for symbol in strategy_spec.data_requirement.symbols:
            for timeframe in strategy_spec.data_requirement.timeframes:
                if self.order_tracker.has_pending_orders(container_id, symbol, timeframe):
                    return False
        
        return True
    
    def get_prevention_metrics(self) -> Dict[str, Any]:
        """Get duplicate prevention metrics."""
        strategies_processed = self._metrics['strategies_processed']
        
        # Update peak pending if needed
        if hasattr(self.order_tracker, 'get_metrics'):
            current_pending = self.order_tracker.get_metrics().get('total_pending', 0)
            self._peak_pending_orders = max(self._peak_pending_orders, current_pending)
        
        return {
            **self._metrics,
            'prevention_rate': (
                self._metrics['strategies_filtered'] / max(1, strategies_processed)
            ) * 100,
            'prevented_by_symbol': dict(self._prevented_by_symbol),  # Add breakdown
            'peak_pending': self._peak_pending_orders  # Track peak load
        }
    
    def _get_prevention_by_symbol(self) -> Dict[str, int]:
        """Get prevention breakdown by symbol."""
        return dict(self._prevented_by_symbol)


@dataclass
class StrategyOrchestrator(ContainerComponent):
    """
    StrategyOrchestrator with barrier-level duplicate prevention.
    
    This is the canonical implementation that includes duplicate order
    prevention at the synchronization barrier level.
    """
    
    # Core orchestrator components
    strategies: List[StrategySpecification] = field(default_factory=list)
    feature_cache_size: int = 1000
    enable_execution_tracking: bool = True
    
    # Duplicate prevention components
    order_tracker: Optional[OrderTrackingProtocol] = None
    duplicate_prevention: Optional[DuplicatePreventionProtocol] = None
    enable_duplicate_prevention: bool = True  # Allow disabling for testing/debugging
    
    def __post_init__(self):
        # Original orchestrator initialization
        self.strategy_groups: Dict[Tuple, List[StrategySpecification]] = defaultdict(list)
        self._group_strategies()
        
        self.feature_cache: Dict[str, Any] = {}
        self.classifier_cache: Dict[str, Any] = {}
        self.execution_history: Dict[str, List[datetime]] = defaultdict(list)
        self.signal_counts: Dict[str, int] = defaultdict(int)
        
        # Component references
        self.time_buffer: Optional[TimeAlignmentBuffer] = None
        self.feature_calculator: Optional[Any] = None
        self.classifier: Optional[Any] = None
        
        # Initialize duplicate prevention if enabled and not provided
        if self.enable_duplicate_prevention:
            if self.order_tracker is None:
                self.order_tracker = OrderTracker()
            
            if self.duplicate_prevention is None:
                self.duplicate_prevention = DuplicateOrderPrevention(
                    order_tracker=self.order_tracker
                )
    
    def initialize(self, container: 'Container') -> None:
        """Initialize orchestrator with duplicate prevention."""
        self.container = container
        
        # Original initialization
        self.time_buffer = container.get_component('time_buffer')
        self.feature_calculator = container.get_component('feature_calculator')
        self.classifier = container.get_component('classifier')
        
        if not self.time_buffer:
            raise ValueError("StrategyOrchestrator requires TimeAlignmentBuffer component")
        
        # Register data requirements
        requirements = [s.data_requirement for s in self.strategies if s.enabled]
        self.time_buffer.track_symbols_timeframes(requirements)
        
        # Register for synchronized data with duplicate prevention
        self.time_buffer.register_callback(self.on_synchronized_data_with_prevention)
        
        # Subscribe to FILL and CANCEL events for order clearing
        from ..events.filters import container_filter
        
        self.container.event_bus.subscribe(
            EventType.FILL.value,
            self._handle_fill_for_order_clearing,
            event_filter=container_filter(self.container.container_id)
        )
        
        self.container.event_bus.subscribe(
            EventType.CANCEL.value,
            self._handle_cancel_for_order_clearing,
            event_filter=container_filter(self.container.container_id)
        )
        
        logger.info(
            f"StrategyOrchestrator initialized with {len(self.strategies)} strategies "
            f"and duplicate prevention {'enabled' if self.enable_duplicate_prevention else 'disabled'}"
        )
    
    def on_synchronized_data_with_prevention(self, aligned_data: Dict[str, Event], 
                                           sync_time: datetime) -> None:
        """
        Process synchronized data with barrier-level duplicate prevention.
        
        This is the key method that integrates duplicate prevention with
        the existing strategy orchestration flow.
        """
        self.duplicate_prevention._metrics['bars_processed'] += 1
        
        # Process each strategy group with duplicate prevention
        for group_key, strategies in self.strategy_groups.items():
            # Check if this group has required data
            if self._has_required_data(strategies[0].data_requirement, aligned_data):
                
                # BARRIER-LEVEL DUPLICATE PREVENTION
                if self.enable_duplicate_prevention and self.duplicate_prevention:
                    filtered_strategies = self.duplicate_prevention.filter_strategies_for_duplicates(
                        strategies, self.container.container_id
                    )
                else:
                    filtered_strategies = strategies  # Skip filtering if disabled
                
                if filtered_strategies:
                    # Process only the filtered strategies (no duplicates possible)
                    self._process_strategy_group_filtered(
                        filtered_strategies, aligned_data, sync_time
                    )
    
    def _process_strategy_group_filtered(self, 
                                       strategies: List[StrategySpecification],
                                       aligned_data: Dict[str, Event],
                                       sync_time: datetime) -> None:
        """Process filtered strategies (guaranteed no duplicates)."""
        # Calculate features (with caching) - same as original
        features = self._calculate_features(aligned_data, sync_time)
        
        # Get classification if needed - same as original
        classification = None
        if self.classifier and any(s.classifier_id for s in strategies):
            classification = self._get_classification(features, aligned_data, sync_time)
        
        # Process each filtered strategy
        for strategy in strategies:
            if strategy.enabled:
                self._execute_strategy_with_tracking(
                    strategy, features, classification, aligned_data, sync_time
                )
    
    def _execute_strategy_with_tracking(self,
                                      strategy: StrategySpecification,
                                      features: Dict[str, Any],
                                      classification: Optional[Any],
                                      aligned_data: Dict[str, Event],
                                      sync_time: datetime) -> None:
        """Execute strategy and track orders for duplicate prevention."""
        
        # Original execution constraints check
        if not self._check_execution_allowed(strategy, sync_time):
            return
        
        if strategy.classifier_id and classification:
            if not self._check_classifier_gate(strategy.classifier_id, classification):
                return
        
        try:
            # Prepare strategy inputs - same as original
            strategy_features = self._prepare_strategy_features(strategy, features)
            
            # Call strategy function - same as original
            signal = strategy.strategy_function(
                features=strategy_features,
                classification=classification,
                parameters=strategy.parameters
            )
            
            if signal and self._validate_signal(signal):
                # Register order BEFORE publishing signal
                order_id = self._generate_order_id(signal, strategy, sync_time, aligned_data)
                symbol = signal.get('symbol')
                
                # Extract timeframe from aligned_data
                timeframe = self._extract_timeframe_for_symbol(symbol, aligned_data)
                
                # Register pending order
                self.order_tracker.register_order(
                    container_id=self.container.container_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    order_id=order_id,
                    metadata={
                        'strategy_id': strategy.strategy_id,
                        'signal_strength': signal.get('strength', 0),
                        'sync_time': sync_time.isoformat()
                    }
                )
                
                # Process and publish signal with order tracking
                self._process_signal_with_tracking(
                    signal, strategy, features, classification, 
                    aligned_data, sync_time, order_id
                )
                
                # Update execution tracking - same as original
                if self.enable_execution_tracking:
                    self.execution_history[strategy.strategy_id].append(sync_time)
                self.signal_counts[strategy.strategy_id] += 1
        
        except Exception as e:
            logger.error(f"Error executing strategy {strategy.strategy_id}: {e}", exc_info=True)
    
    def _process_signal_with_tracking(self,
                                    signal: Dict[str, Any],
                                    strategy: StrategySpecification,
                                    features: Dict[str, Any],
                                    classification: Optional[Any],
                                    aligned_data: Dict[str, Event],
                                    sync_time: datetime,
                                    order_id: str) -> None:
        """Process and publish enriched signal with order tracking."""
        
        # Enrich signal with order tracking - enhanced from original
        enriched_signal = {
            **signal,
            'strategy_id': strategy.strategy_id,
            'sync_time': sync_time.isoformat(),
            'classification': str(classification) if classification else None,
            'bar_data': self._extract_bar_data(aligned_data),
            'features': features.get(signal.get('symbol'), {}),
            'parameters': strategy.parameters,
            'order_id': order_id,  # NEW: Include order ID for tracking
            'container_id': self.container.container_id  # NEW: Include container ID
        }
        
        # Create signal event - same as original
        signal_event = Event(
            event_type=EventType.SIGNAL,
            payload=enriched_signal,
            source_id=self.container.container_id if self.container else 'strategy_orchestrator'
        )
        
        # Publish to parent - same as original
        self.container.publish_event(signal_event, target_scope="parent")
        
        logger.debug(
            f"Published signal from {strategy.strategy_id} for {signal['symbol']} "
            f"with order_id {order_id} at {sync_time}"
        )
    
    def _handle_fill_for_order_clearing(self, fill_event: Event) -> None:
        """Handle FILL event and clear order from tracking."""
        order_id = fill_event.payload.get('order_id')
        symbol = fill_event.payload.get('symbol')
        
        if order_id and symbol:
            # Extract timeframe from fill event or guess from container config
            timeframe = fill_event.payload.get('timeframe', '1m')  # Default fallback
            
            cleared = self.order_tracker.clear_order(
                container_id=self.container.container_id,
                symbol=symbol,
                timeframe=timeframe,
                order_id=order_id
            )
            
            if cleared:
                logger.debug(f"Cleared filled order {order_id} for {symbol}_{timeframe}")
            else:
                logger.warning(f"Could not clear filled order {order_id} - not found in tracking")
    
    def _handle_cancel_for_order_clearing(self, cancel_event: Event) -> None:
        """Handle CANCEL event and clear order from tracking."""
        order_id = cancel_event.payload.get('order_id')
        symbol = cancel_event.payload.get('symbol')
        
        if order_id and symbol:
            timeframe = cancel_event.payload.get('timeframe', '1m')
            
            cleared = self.order_tracker.clear_order(
                container_id=self.container.container_id,
                symbol=symbol,
                timeframe=timeframe,
                order_id=order_id
            )
            
            if cleared:
                logger.debug(f"Cleared cancelled order {order_id} for {symbol}_{timeframe}")
    
    # Utility methods
    
    def _generate_order_id(self, signal: Dict[str, Any], strategy: StrategySpecification, 
                          sync_time: datetime, aligned_data: Dict[str, Event]) -> str:
        """Generate unique order ID with timeframe info."""
        import uuid
        symbol = signal.get('symbol')
        timeframe = self._extract_timeframe_for_symbol(symbol, aligned_data)
        return f"order_{strategy.strategy_id}_{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}"
    
    def _extract_timeframe_for_symbol(self, symbol: str, aligned_data: Dict[str, Event]) -> str:
        """Extract timeframe for symbol from aligned data."""
        # Look for symbol_timeframe keys in aligned_data
        for key, event in aligned_data.items():
            if key.startswith(f"{symbol}_"):
                return key.split('_', 1)[1]  # Extract timeframe part
        
        # Fallback to checking event payload
        for event in aligned_data.values():
            if event.payload.get('symbol') == symbol:
                return event.payload.get('timeframe', '1m')
        
        return '1m'  # Default fallback
    
    # Copy all original methods unchanged
    def _calculate_features(self, aligned_data: Dict[str, Event], sync_time: datetime) -> Dict[str, Any]:
        """Calculate features with caching - same as original."""
        # ... (copy from original StrategyOrchestrator)
        cache_key = self._create_cache_key(aligned_data, sync_time)
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = {}
        
        if self.feature_calculator:
            for key, event in aligned_data.items():
                symbol = key.split('_')[0]
                if hasattr(self.feature_calculator, 'calculate_features'):
                    symbol_features = self.feature_calculator.calculate_features(event.payload)
                    features[symbol] = symbol_features
            
            if hasattr(self.feature_calculator, 'calculate_cross_features'):
                cross_features = self.feature_calculator.calculate_cross_features(aligned_data)
                features['_cross'] = cross_features
        
        self.feature_cache[cache_key] = features
        
        if len(self.feature_cache) > self.feature_cache_size:
            oldest = min(self.feature_cache.keys())
            del self.feature_cache[oldest]
        
        return features
    
    def _get_classification(self, features: Dict[str, Any], 
                           aligned_data: Dict[str, Event],
                           sync_time: datetime) -> Any:
        """Get market classification with caching - same as original."""
        cache_key = f"clf_{sync_time.isoformat()}"
        
        if cache_key in self.classifier_cache:
            return self.classifier_cache[cache_key]
        
        classification = None
        if hasattr(self.classifier, 'classify'):
            classification = self.classifier.classify(features, aligned_data)
        
        self.classifier_cache[cache_key] = classification
        
        if len(self.classifier_cache) > 100:
            oldest = min(self.classifier_cache.keys())
            del self.classifier_cache[oldest]
        
        return classification
    
    # ... (copy all other original methods unchanged)
    
    def get_state(self) -> Dict[str, Any]:
        """Get orchestrator state including duplicate prevention metrics."""
        original_state = {
            'active_strategies': sum(1 for s in self.strategies if s.enabled),
            'total_signals': sum(self.signal_counts.values()),
            'feature_cache_size': len(self.feature_cache),
            'execution_tracking': self.enable_execution_tracking
        }
        
        # Add duplicate prevention metrics
        if self.duplicate_prevention:
            prevention_metrics = self.duplicate_prevention.get_prevention_metrics()
            original_state['duplicate_prevention'] = prevention_metrics
        
        if self.order_tracker:
            order_metrics = self.order_tracker.get_metrics()
            original_state['order_tracking'] = order_metrics
            original_state['pending_orders'] = self.order_tracker.get_pending_summary(
                self.container.container_id
            )
        
        return original_state
```

## Integration with Container

### Enhanced Container Setup

```python
# In your container setup code

def setup_feature_container(container: Container, config: Dict[str, Any]) -> None:
    """
    Setup feature container with barrier-level duplicate prevention.
    
    This is the canonical setup function that includes duplicate prevention
    as part of the standard container configuration.
    """
    # Create time alignment buffer (same as original)
    time_buffer = TimeAlignmentBuffer(
        buffer_size=config.get('buffer_size', 1000),
        default_alignment=TimeframeAlignment[config.get('alignment', 'NEAREST')],
        emit_partial=config.get('emit_partial', False)
    )
    container.add_component('time_buffer', time_buffer)
    
    # Create strategy specifications (same as original)
    strategy_specs = []
    for strat_config in config.get('strategies', []):
        data_req = DataRequirement(
            symbols=strat_config['symbols'],
            timeframes=strat_config['timeframes'],
            min_history=strat_config.get('min_history', 1),
            alignment_mode=AlignmentMode[strat_config.get('alignment_mode', 'ALL').upper()],
            timeout_ms=strat_config.get('timeout_ms')
        )
        
        spec = StrategySpecification(
            strategy_id=strat_config['id'],
            strategy_function=strat_config['function'],
            data_requirement=data_req,
            classifier_id=strat_config.get('classifier_id'),
            parameters=strat_config.get('parameters', {}),
            min_time_between_signals=strat_config.get('min_time_between_signals'),
            max_signals_per_day=strat_config.get('max_signals_per_day'),
            allowed_hours=strat_config.get('allowed_hours')
        )
        strategy_specs.append(spec)
    
    # NEW: Create order tracker and duplicate prevention components
    order_tracker = OrderTracker()
    duplicate_prevention = DuplicateOrderPrevention(order_tracker=order_tracker)
    
    # Create orchestrator with duplicate prevention
    orchestrator = StrategyOrchestrator(
        strategies=strategy_specs,
        feature_cache_size=config.get('feature_cache_size', 1000),
        enable_execution_tracking=config.get('enable_execution_tracking', True),
        enable_duplicate_prevention=config.get('enable_duplicate_prevention', True),
        order_tracker=order_tracker,
        duplicate_prevention=duplicate_prevention
    )
    container.add_component('strategy_orchestrator', orchestrator)
    
    # Optional: Add order tracker as separate component for external access
    container.add_component('order_tracker', order_tracker)
    container.add_component('duplicate_prevention', duplicate_prevention)
    
    logger.info(f"Setup feature container with {len(strategy_specs)} strategies "
               f"and barrier-level duplicate prevention")


# Usage in container factory or workflow manager
def create_portfolio_container_with_prevention(config: Dict) -> Container:
    """Create portfolio container with duplicate prevention."""
    
    container_config = ContainerConfig(
        role=ContainerRole.PORTFOLIO,
        name=config['name'],
        container_id=config.get('container_id'),
        config=config
    )
    
    container = Container(container_config)
    
    # Setup with orchestrator
    setup_feature_container(container, config)
    
    return container
```

## Benefits of Protocol + Composition Approach

### 1. **Zero Inheritance**
- All components use composition, not inheritance
- Clear protocol contracts define behavior
- Easy to test and mock individual components

### 2. **Modular Design**
```python
# Each component has single responsibility:
OrderTracker          → Track pending orders by symbol/timeframe
DuplicateOrderPrevention → Filter strategies to prevent duplicates  
StrategyOrchestrator → Coordinate orchestration + prevention
```

### 3. **Flexible Integration**
```python
# Can use components independently:
order_tracker = OrderTracker()  # Just tracking
prevention = DuplicateOrderPrevention(order_tracker)  # Just prevention
orchestrator = StrategyOrchestrator(order_tracker=order_tracker)  # Full integration
```

### 4. **Protocol Verification**
```python
# Runtime verification that components implement protocols
def verify_order_tracking(tracker: Any):
    assert isinstance(tracker, OrderTrackingProtocol)
    # Guaranteed to have required methods

def verify_duplicate_prevention(prevention: Any):
    assert isinstance(prevention, DuplicatePreventionProtocol)
    # Guaranteed to implement filtering logic
```

### 5. **Symbol/Timeframe Awareness**
```python
# Proper filtering by symbol AND timeframe
strategy_spec = StrategySpecification(
    symbols=['AAPL', 'MSFT'],
    timeframes=['1m', '5m']
)

# Will check for pending orders on:
# - AAPL_1m, AAPL_5m, MSFT_1m, MSFT_5m
# Only runs strategy if ALL combinations are clear
```

## Testing Strategy

### Unit Tests for Components

```python
def test_order_tracker_protocol():
    """Test OrderTracker implements protocol correctly."""
    tracker = OrderTracker()
    
    # Verify protocol compliance
    assert isinstance(tracker, OrderTrackingProtocol)
    
    # Test basic functionality
    assert not tracker.has_pending_orders("container_1", "AAPL", "1m")
    
    tracker.register_order("container_1", "AAPL", "1m", "order_123")
    assert tracker.has_pending_orders("container_1", "AAPL", "1m")
    assert tracker.get_pending_count("container_1", "AAPL", "1m") == 1
    
    cleared = tracker.clear_order("container_1", "AAPL", "1m", "order_123")
    assert cleared
    assert not tracker.has_pending_orders("container_1", "AAPL", "1m")


def test_duplicate_prevention_filtering():
    """Test strategy filtering for duplicates."""
    tracker = OrderTracker()
    prevention = DuplicateOrderPrevention(order_tracker=tracker)
    
    # Create strategy that trades AAPL on 1m timeframe
    strategy = StrategySpecification(
        strategy_id="test_strat",
        data_requirement=DataRequirement(symbols=["AAPL"], timeframes=["1m"])
    )
    
    # No pending orders - should allow strategy
    assert prevention.should_process_strategy(strategy, "container_1")
    
    # Add pending order for AAPL_1m
    tracker.register_order("container_1", "AAPL", "1m", "order_123")
    
    # Now should block strategy
    assert not prevention.should_process_strategy(strategy, "container_1")
```

### Integration Tests

```python
def test_orchestrator_duplicate_prevention():
    """Test orchestrator with duplicate prevention."""
    # Setup container with orchestrator
    container = create_test_container()
    
    # Create test strategies
    strategies = [create_test_strategy("AAPL", "1m")]
    orchestrator = StrategyOrchestrator(strategies=strategies)
    
    # Process synchronized bars twice
    aligned_data = create_test_aligned_data("AAPL", "1m")
    
    # First processing should work
    orchestrator.on_synchronized_data_with_prevention(aligned_data, datetime.now())
    assert orchestrator.signal_counts["test_strategy"] == 1
    
    # Second processing should be blocked (duplicate)
    orchestrator.on_synchronized_data_with_prevention(aligned_data, datetime.now())
    assert orchestrator.signal_counts["test_strategy"] == 1  # Still 1, not 2
```

## Migration Path

### Phase 1: Add Components (No Breaking Changes)
1. Add protocol definitions to `protocols.py`
2. Add `OrderTracker` and `DuplicateOrderPrevention` classes to `sync.py`
3. No changes to existing containers

### Phase 2: Update Canonical Orchestrator
1. Modify `StrategyOrchestrator` in `sync.py` to include prevention
2. Update container setup functions to enable prevention
3. Existing containers continue working unchanged

### Phase 3: Full Migration
1. Enable duplicate prevention in all portfolio containers
2. Monitor metrics to verify duplicate prevention is working
3. Remove original orchestrator after verification

This approach provides maximum safety with zero inheritance, full symbol/timeframe awareness, and seamless integration with your existing Protocol + Composition architecture.
