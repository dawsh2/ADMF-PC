"""Event observers - tracer and metrics.
Consolidated from observers/ subdirectory.
"""

from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime
from collections import defaultdict, deque
import json
import logging
import uuid
from dataclasses import dataclass, field

from ..protocols import EventObserverProtocol, EventTracerProtocol, EventStorageProtocol, MetricsCalculatorProtocol
from ..types import Event, EventType
from .storage import create_storage_backend

logger = logging.getLogger(__name__)


class EventTracer(EventObserverProtocol, EventTracerProtocol):
    """
    Enhanced event tracer with sophisticated event enhancement and container isolation.
    
    Features:
    - Automatic correlation ID assignment and sequence numbering
    - Container isolation for parallel execution
    - Causation chain tracking
    - Event enhancement with timing metadata
    - Intelligent batching and storage
    """
    
    def __init__(
        self,
        correlation_id: Optional[str] = None,
        storage: Optional[EventStorageProtocol] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced event tracer.
        
        Args:
            correlation_id: Base correlation ID (auto-generated if None)
            storage: Storage backend for events
            config: Configuration options
        """
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.storage = storage or create_storage_backend('memory', {})
        self.config = config or {}
        
        # Configuration
        self.max_events = self.config.get('max_events', 10000)
        self.events_to_trace = self.config.get('events_to_trace', 'ALL')
        self.retention_policy = self.config.get('retention_policy', 'all')
        self.container_isolation = self.config.get('container_isolation', True)
        
        # Event enhancement state
        self.sequence_counters = defaultdict(int)  # Per-correlation sequence
        self.event_index: Dict[str, Event] = {}  # Fast lookup by event_id
        self.container_events = defaultdict(list)  # Events by container
        self.correlation_chains = defaultdict(list)  # Events by correlation
        
        # In-memory cache for fast access
        self.recent_events = deque(maxlen=self.max_events)
        
        # Statistics
        self.event_counts = defaultdict(int)
        self.container_counts = defaultdict(int)
        self._traced_count = 0
        self._pruned_count = 0
        self._start_time = datetime.now()
        
        logger.info(f"EventTracer created: {self.correlation_id}, "
                   f"retention: {self.retention_policy}, "
                   f"container_isolation: {self.container_isolation}")
    
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for this trace session."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        return f"trace_{timestamp}_{unique_id}"
    
    # EventObserverProtocol implementation
    
    def on_publish(self, event: Event) -> None:
        """Enhance and trace event when published."""
        if self._should_trace(event):
            self._enhance_event(event)
            self._store_enhanced_event(event)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Update event with delivery timing."""
        if self._should_trace(event):
            event.metadata['delivery'] = {
                'delivered_at': datetime.now().isoformat(),
                'handler': getattr(handler, '__name__', str(handler)),
                'delivery_sequence': self._get_next_sequence('delivery')
            }
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Record error in event metadata."""
        if self._should_trace(event):
            event.metadata['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'handler': getattr(handler, '__name__', str(handler)),
                'error_time': datetime.now().isoformat(),
                'error_sequence': self._get_next_sequence('error')
            }
    
    # EventTracerProtocol implementation
    
    def trace_event(self, event: Event) -> None:
        """Enhanced event tracing with full metadata."""
        self._enhance_event(event)
        self._store_enhanced_event(event)
    
    def _enhance_event(self, event: Event) -> None:
        """Enhance event with tracing metadata and sequencing."""
        # Set correlation ID if not present
        if not event.correlation_id:
            event.correlation_id = self.correlation_id
        
        # Set sequence number within correlation
        if event.sequence_number is None:
            event.sequence_number = self._get_next_sequence(event.correlation_id)
        
        # Add comprehensive trace metadata
        event.metadata.update({
            'trace_info': {
                'tracer_id': self.correlation_id,
                'traced_at': datetime.now().isoformat(),
                'trace_sequence': self._traced_count,
                'container_sequence': self._get_next_sequence(f"container_{event.container_id}"),
                'event_enhanced': True
            },
            'timing': {
                'event_created': event.timestamp.isoformat(),
                'trace_enhanced': datetime.now().isoformat()
            }
        })
        
        # Container isolation tracking
        if self.container_isolation and event.container_id:
            event.metadata['isolation'] = {
                'source_container': event.container_id,
                'container_trace_id': f"{self.correlation_id}_{event.container_id}",
                'isolated': True
            }
    
    def _store_enhanced_event(self, event: Event) -> None:
        """Store enhanced event with indexing and caching."""
        # Store in persistent storage
        self.storage.store(event)
        
        # Update in-memory indices
        self.event_index[event.event_id] = event
        self.recent_events.append(event)
        
        # Container isolation
        if event.container_id and self.container_isolation:
            self.container_events[event.container_id].append(event)
            # Enforce per-container limits
            if len(self.container_events[event.container_id]) > self.max_events:
                removed = self.container_events[event.container_id].pop(0)
                self.event_index.pop(removed.event_id, None)
        
        # Correlation tracking
        if event.correlation_id:
            self.correlation_chains[event.correlation_id].append(event)
        
        # Update statistics
        self.event_counts[event.event_type] += 1
        if event.container_id:
            self.container_counts[event.container_id] += 1
        self._traced_count += 1
        
        # Apply retention policy
        self._apply_retention_policy(event)
    
    def _get_next_sequence(self, key: str) -> int:
        """Get next sequence number for given key."""
        self.sequence_counters[key] += 1
        return self.sequence_counters[key]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive trace summary."""
        return {
            'tracer_id': self.correlation_id,
            'events_traced': self._traced_count,
            'events_pruned': self._pruned_count,
            'events_in_storage': self.storage.count() if hasattr(self.storage, 'count') else len(self.recent_events),
            'events_in_memory': len(self.recent_events),
            'retention_policy': self.retention_policy,
            'container_isolation': self.container_isolation,
            'event_counts': dict(self.event_counts),
            'container_counts': dict(self.container_counts),
            'correlations_tracked': len(self.correlation_chains),
            'containers_tracked': len(self.container_events),
            'start_time': self._start_time.isoformat(),
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'config': self.config
        }
    
    def get_events_by_correlation(self, correlation_id: str) -> List[Event]:
        """Get events by correlation ID with fast in-memory lookup."""
        # Try in-memory first
        if correlation_id in self.correlation_chains:
            return self.correlation_chains[correlation_id].copy()
        
        # Fall back to storage query
        return self.storage.query({'correlation_id': correlation_id})
    
    def get_events_by_container(self, container_id: str) -> List[Event]:
        """Get events by container ID (respects isolation)."""
        if self.container_isolation and container_id in self.container_events:
            return self.container_events[container_id].copy()
        
        # Query storage if not in memory
        return self.storage.query({'container_id': container_id})
    
    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Fast event lookup by ID."""
        return self.event_index.get(event_id)
    
    def trace_causation_chain(self, event_id: str) -> List[Event]:
        """Trace complete causation chain for an event."""
        event = self.get_event_by_id(event_id)
        if not event:
            return []
        
        # Find ancestors
        ancestors = []
        current = event
        while current and current.causation_id:
            parent = self.get_event_by_id(current.causation_id)
            if not parent or parent in ancestors:
                break
            ancestors.insert(0, parent)
            current = parent
        
        # Find descendants
        descendants = []
        to_check = [event_id]
        checked = set()
        
        while to_check:
            parent_id = to_check.pop(0)
            if parent_id in checked:
                continue
            checked.add(parent_id)
            
            # Find children
            for evt in self.recent_events:
                if evt.causation_id == parent_id and evt.event_id != parent_id:
                    descendants.append(evt)
                    to_check.append(evt.event_id)
        
        # Sort descendants by sequence
        descendants.sort(key=lambda e: e.sequence_number or 0)
        
        return ancestors + [event] + descendants
    
    def calculate_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate latency statistics by event type."""
        stats = defaultdict(lambda: {'count': 0, 'total': 0.0, 'max': 0.0, 'min': float('inf')})
        
        for event in self.recent_events:
            # Calculate enhancement latency
            timing = event.metadata.get('timing', {})
            if 'event_created' in timing and 'trace_enhanced' in timing:
                try:
                    created = datetime.fromisoformat(timing['event_created'])
                    enhanced = datetime.fromisoformat(timing['trace_enhanced'])
                    latency_ms = (enhanced - created).total_seconds() * 1000
                    
                    event_stats = stats[event.event_type]
                    event_stats['count'] += 1
                    event_stats['total'] += latency_ms
                    event_stats['max'] = max(event_stats['max'], latency_ms)
                    event_stats['min'] = min(event_stats['min'], latency_ms)
                except Exception:
                    pass
        
        # Calculate averages
        result = {}
        for event_type, event_stats in stats.items():
            if event_stats['count'] > 0:
                result[event_type] = {
                    'avg_ms': event_stats['total'] / event_stats['count'],
                    'max_ms': event_stats['max'],
                    'min_ms': event_stats['min'],
                    'count': event_stats['count']
                }
        
        return result
    
    def save_to_file(self, filepath: str) -> None:
        """Save trace to file with metadata."""
        # Use storage export if available
        if hasattr(self.storage, 'export_to_file'):
            self.storage.export_to_file(filepath)
        else:
            # Manual export
            import json
            with open(filepath, 'w') as f:
                # Write summary as first line
                summary = self.get_summary()
                f.write(json.dumps({'type': 'summary', 'data': summary}) + '\n')
                
                # Write events
                for event in self.recent_events:
                    f.write(json.dumps({'type': 'event', 'data': event.to_dict()}) + '\n')
        
        logger.info(f"Saved trace {self.correlation_id} to {filepath}")
    
    def clear(self) -> None:
        """Clear all traced events and reset state."""
        # Clear storage
        if hasattr(self.storage, 'clear'):
            self.storage.clear()
        
        # Clear in-memory state
        self.event_index.clear()
        self.container_events.clear()
        self.correlation_chains.clear()
        self.recent_events.clear()
        self.sequence_counters.clear()
        self.event_counts.clear()
        self.container_counts.clear()
        
        # Reset counters
        self._traced_count = 0
        self._pruned_count = 0
        
        logger.info(f"Cleared trace {self.correlation_id}")
    
    # Private methods
    
    def _should_trace(self, event: Event) -> bool:
        """Check if event should be traced based on config."""
        if self.events_to_trace == 'ALL':
            return True
        
        if isinstance(self.events_to_trace, list):
            # Check if event type is in list
            event_type_str = event.event_type
            if hasattr(event.event_type, 'value'):
                event_type_str = event.event_type.value
            return event_type_str in self.events_to_trace
        
        return False
    
    def _apply_retention_policy(self, event: Event) -> None:
        """Apply retention policy after tracing event."""
        if self.retention_policy == 'trade_complete':
            # Check if this event closes a trade
            if event.event_type == EventType.POSITION_CLOSE.value and event.correlation_id:
                # Prune all events for this trade except the close
                pruned = self.storage.prune({
                    'correlation_id': event.correlation_id,
                    'exclude_event_id': event.metadata.get('event_id')
                })
                self._pruned_count += pruned
                logger.debug(f"Pruned {pruned} events for completed trade {event.correlation_id}")
        
        elif self.retention_policy == 'sliding_window':
            # Keep only last N events
            total_events = self.storage.count()
            if total_events > self.max_events:
                # Prune oldest events
                to_prune = total_events - self.max_events
                if hasattr(self.storage, 'prune_oldest'):
                    pruned = self.storage.prune_oldest(to_prune)
                    self._pruned_count += pruned
        
        elif self.retention_policy == 'minimal':
            # Only keep events for open positions
            if event.event_type == EventType.POSITION_CLOSE.value and event.correlation_id:
                # Remove ALL events for this correlation
                pruned = self.storage.prune({'correlation_id': event.correlation_id})
                self._pruned_count += pruned


@dataclass
class MetricsObserver(EventObserverProtocol):
    """
    Pure observer that delegates to metrics calculator.
    
    This follows Protocol + Composition approach perfectly!
    Observes events and delegates calculation to a separate calculator.
    """
    
    # Composed calculator - not inherited!
    calculator: MetricsCalculatorProtocol
    
    # Configuration
    retention_policy: str = "trade_complete"
    max_events: int = 1000
    
    # Temporary storage for correlation
    active_trades: Dict[str, List[Event]] = field(default_factory=dict)
    _event_count: int = 0
    _pruned_count: int = 0
    
    def on_publish(self, event: Event) -> None:
        """Observe published events."""
        self._event_count += 1
        
        # Store events for active trades based on retention policy
        if self.retention_policy == "trade_complete":
            self._store_trade_event(event)
        elif self.retention_policy == "sliding_window":
            # Different retention strategy
            pass
        
        # Process specific events
        if event.event_type == EventType.FILL.value:
            self._process_fill(event)
        elif event.event_type == EventType.PORTFOLIO_UPDATE.value:
            self._process_portfolio_update(event)
        elif event.event_type == EventType.POSITION_CLOSE.value:
            self._process_position_close(event)
        elif event.event_type == EventType.POSITION_OPEN.value:
            self._process_position_open(event)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Not needed for metrics observation."""
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Log errors but don't affect metrics."""
        logger.warning(f"Error processing {event.event_type}: {error}")
    
    def _store_trade_event(self, event: Event) -> None:
        """Store event if it's part of an active trade."""
        trade_id = event.correlation_id
        
        if not trade_id:
            return
        
        # Events we care about for trade tracking
        relevant_types = [
            EventType.POSITION_OPEN.value,
            EventType.ORDER_REQUEST.value,
            EventType.ORDER.value,
            EventType.FILL.value,
            EventType.POSITION_CLOSE.value
        ]
        
        if event.event_type in relevant_types:
            if trade_id not in self.active_trades:
                self.active_trades[trade_id] = []
            self.active_trades[trade_id].append(event)
    
    def _process_position_open(self, event: Event) -> None:
        """Track new position."""
        # Ensure we're tracking this trade
        trade_id = event.correlation_id
        if trade_id and trade_id not in self.active_trades:
            self.active_trades[trade_id] = [event]
    
    def _process_fill(self, event: Event) -> None:
        """Process fill event."""
        # For now, just ensure it's tracked
        # Actual P&L calculation happens on position close
        pass
    
    def _process_portfolio_update(self, event: Event) -> None:
        """Process portfolio value update."""
        portfolio_value = event.payload.get('portfolio_value')
        timestamp = event.payload.get('timestamp') or event.timestamp
        
        if portfolio_value is not None:
            self.calculator.update_portfolio_value(portfolio_value, timestamp)
    
    def _process_position_close(self, event: Event) -> None:
        """Process position close and update metrics."""
        trade_id = event.correlation_id
        
        if trade_id in self.active_trades:
            # Find the opening event
            events = self.active_trades[trade_id]
            open_event = next(
                (e for e in events if e.event_type == EventType.POSITION_OPEN.value), 
                None
            )
            
            if open_event:
                # Extract data from events
                entry_price = open_event.payload.get('price', 0)
                exit_price = event.payload.get('price', 0)
                quantity = event.payload.get('quantity', 0)
                direction = open_event.payload.get('direction', 'long')
                
                # Delegate calculation to calculator
                self.calculator.update_from_trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    direction=direction
                )
            
            # Apply retention policy
            if self.retention_policy == "trade_complete":
                # Remove all events for this completed trade
                pruned = len(self.active_trades[trade_id])
                del self.active_trades[trade_id]
                self._pruned_count += pruned
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from calculator plus observer stats."""
        return {
            'metrics': self.calculator.get_metrics(),
            'observer_stats': {
                'events_observed': self._event_count,
                'events_pruned': self._pruned_count,
                'active_trades': len(self.active_trades),
                'retention_policy': self.retention_policy
            }
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get complete results (alias for container compatibility)."""
        return self.get_metrics()


# Signal Performance Observer

@dataclass 
class SignalObserver(EventObserverProtocol):
    """
    Observer that tracks signal performance and stores signals for replay.
    
    Primary Purpose:
    - Store ALL signals with full metadata for signal replay
    - Preserve signal context (bar data, features, classifier states)
    - Track performance metrics for risk-aware decisions
    - Enable signal reconstruction without recomputing expensive features
    
    Features:
    - Comprehensive signal storage with full context
    - Performance tracking (win rate, profit factor, confidence)
    - Storage-based retention (never clears by order status)
    - Multi-strategy performance analysis
    """
    
    # Configuration
    retention_policy: str = "trade_complete"
    max_strategies: int = 50
    recent_window_size: int = 20
    min_trades_for_confidence: int = 10
    
    # Storage limits (based on memory/disk, not order lifecycle)
    max_signals_total: int = 100000  # Total signal storage limit
    max_signals_per_strategy: int = 10000  # Per-strategy limit
    
    # Performance tracking state
    strategy_performance: Dict[str, 'SignalPerformance'] = field(default_factory=dict)
    pending_signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # correlation_id -> signal data
    strategy_last_seen: Dict[str, datetime] = field(default_factory=dict)
    
    # Signal storage for replay (full metadata preservation)
    stored_signals: List[Dict[str, Any]] = field(default_factory=list)  # All signals with full context
    signals_by_strategy: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # Per-strategy signals
    
    # Statistics
    _signals_observed: int = 0
    _trades_completed: int = 0
    _strategies_tracked: int = 0
    
    def on_publish(self, event: Event) -> None:
        """Process relevant events for signal storage and performance tracking."""
        
        if event.event_type == EventType.SIGNAL.value:
            self._handle_signal(event)
            
        elif event.event_type == EventType.POSITION_OPEN.value:
            self._handle_position_open(event)
            
        elif event.event_type == EventType.POSITION_CLOSE.value:
            self._handle_position_close(event)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Not needed for signal observation."""
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Log errors related to signal processing."""
        logger.warning(f"Error in signal processing: {error}")
    
    def _handle_signal(self, event: Event) -> None:
        """Store signal with full metadata for replay and track performance."""
        self._signals_observed += 1
        
        strategy_id = event.payload.get('strategy_id')
        if not strategy_id:
            return
            
        # Initialize performance tracking if new strategy
        if strategy_id not in self.strategy_performance:
            self._init_strategy_performance(strategy_id, event.payload)
        
        # Store comprehensive signal data for replay (as per storage doc)
        signal_record = {
            'strategy_id': strategy_id,
            'strategy_name': event.payload.get('strategy_name', strategy_id),
            'correlation_id': event.correlation_id,
            'timestamp': event.timestamp.isoformat() if event.timestamp else datetime.now().isoformat(),
            'signal_value': event.payload.get('signal_value', event.payload.get('value', 0)),
            
            # Full context for replay (as specified in storage doc)
            'bar_data': event.payload.get('bars', event.payload.get('bar_data', {})),
            'features': event.payload.get('features', {}),
            'classifier_states': event.payload.get('classifier_states', {}),
            'strategy_parameters': event.payload.get('parameters', {}),
            
            # Signal metadata
            'confidence': event.payload.get('confidence', 1.0),
            'symbol': event.payload.get('symbol'),
            'timeframe': event.payload.get('timeframe'),
            'direction': event.payload.get('direction'),
            'entry_price': event.payload.get('entry_price'),
            
            # Full event payload for complete reconstruction
            'full_payload': event.payload.copy(),
            
            # Event metadata
            'event_id': event.event_id,
            'source_id': event.source_id,
            'container_id': event.container_id
        }
        
        # Store in global signal list
        self.stored_signals.append(signal_record)
        
        # Store per-strategy for efficient lookup
        if strategy_id not in self.signals_by_strategy:
            self.signals_by_strategy[strategy_id] = []
        self.signals_by_strategy[strategy_id].append(signal_record)
        
        # Apply storage limits (LRU eviction)
        self._apply_storage_limits()
        
        # Store for performance correlation when trade completes
        correlation_id = event.correlation_id
        if correlation_id:
            self.pending_signals[correlation_id] = {
                'strategy_id': strategy_id,
                'strategy_name': event.payload.get('strategy_name', strategy_id),
                'signal_data': event.payload.copy(),
                'timestamp': event.timestamp,
                'classifier_states': event.payload.get('classifier_states', {})
            }
        
        # Update last seen
        self.strategy_last_seen[strategy_id] = datetime.now()
    
    def _handle_position_open(self, event: Event) -> None:
        """Enhance pending signal with position data."""
        correlation_id = event.correlation_id
        
        if correlation_id and correlation_id in self.pending_signals:
            self.pending_signals[correlation_id].update({
                'entry_price': event.payload.get('entry_price', event.payload.get('price')),
                'quantity': event.payload.get('quantity'),
                'direction': event.payload.get('direction'),
                'position_opened_at': event.timestamp
            })
    
    def _handle_position_close(self, event: Event) -> None:
        """Update performance metrics when trade completes."""
        self._trades_completed += 1
        
        correlation_id = event.correlation_id
        strategy_id = event.payload.get('strategy_id')
        
        # Get original signal data
        signal_data = self.pending_signals.get(correlation_id)
        if not signal_data:
            logger.debug(f"No signal data found for correlation_id: {correlation_id}")
            return
            
        # Ensure we have strategy_id
        if not strategy_id:
            strategy_id = signal_data.get('strategy_id')
        
        if not strategy_id:
            return
            
        # Get or create performance tracker
        if strategy_id not in self.strategy_performance:
            self._init_strategy_performance(strategy_id, signal_data.get('signal_data', {}))
        
        perf = self.strategy_performance[strategy_id]
        
        # Create result from signal and close data
        result = {
            'pnl': event.payload.get('pnl', 0),
            'pnl_pct': event.payload.get('pnl_pct', 0),
            'exit_price': event.payload.get('exit_price', event.payload.get('price')),
            'entry_price': signal_data.get('entry_price'),
            'quantity': signal_data.get('quantity'),
            'closed_at': event.timestamp
        }
        
        # Update performance
        perf.update_with_result(signal_data.get('signal_data', {}), result)
        
        # Update last seen
        self.strategy_last_seen[strategy_id] = datetime.now()
        
        # Apply retention policy
        if self.retention_policy == "trade_complete":
            # Remove pending signal data (trade is complete)
            if correlation_id in self.pending_signals:
                del self.pending_signals[correlation_id]
        
        # Check if we need to evict strategies (LRU)
        if len(self.strategy_performance) > self.max_strategies:
            self._evict_lru_strategy()
    
    def _init_strategy_performance(self, strategy_id: str, signal_data: Dict[str, Any]) -> None:
        """Initialize performance tracking for a new strategy."""
        from ..tracing.signal_performance import SignalPerformance
        
        self.strategy_performance[strategy_id] = SignalPerformance(
            strategy_id=strategy_id,
            strategy_name=signal_data.get('strategy_name', strategy_id),
            parameters=signal_data.get('parameters', {})
        )
        self._strategies_tracked += 1
        logger.info(f"Started tracking performance for strategy: {strategy_id}")
    
    def _evict_lru_strategy(self) -> None:
        """Remove least recently used strategy to maintain memory bounds."""
        if not self.strategy_last_seen:
            return
            
        # Find LRU strategy
        lru_strategy = min(self.strategy_last_seen.items(), key=lambda x: x[1])[0]
        
        # Remove it
        if lru_strategy in self.strategy_performance:
            del self.strategy_performance[lru_strategy]
        if lru_strategy in self.strategy_last_seen:
            del self.strategy_last_seen[lru_strategy]
            
        logger.info(f"Evicted LRU strategy: {lru_strategy}")
    
    def _apply_storage_limits(self) -> None:
        """Apply storage limits to prevent unbounded growth."""
        # Global signal limit
        if len(self.stored_signals) > self.max_signals_total:
            excess = len(self.stored_signals) - self.max_signals_total
            # Remove oldest signals (FIFO)
            removed_signals = self.stored_signals[:excess]
            self.stored_signals = self.stored_signals[excess:]
            
            # Also remove from per-strategy storage
            for signal in removed_signals:
                strategy_id = signal['strategy_id']
                if strategy_id in self.signals_by_strategy:
                    try:
                        self.signals_by_strategy[strategy_id].remove(signal)
                    except ValueError:
                        pass  # Signal already removed
            
            logger.info(f"Removed {excess} oldest signals to maintain storage limit")
        
        # Per-strategy limits
        for strategy_id, signals in self.signals_by_strategy.items():
            if len(signals) > self.max_signals_per_strategy:
                excess = len(signals) - self.max_signals_per_strategy
                # Remove oldest signals for this strategy
                removed = signals[:excess]
                self.signals_by_strategy[strategy_id] = signals[excess:]
                
                # Also remove from global storage
                for signal in removed:
                    try:
                        self.stored_signals.remove(signal)
                    except ValueError:
                        pass  # Signal already removed
                
                logger.debug(f"Removed {excess} oldest signals for strategy {strategy_id}")
    
    def get_signals_for_replay(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get stored signals for replay with full metadata.
        
        Args:
            strategy_id: If provided, return only signals for this strategy
            
        Returns:
            List of signal records with full context for replay
        """
        if strategy_id:
            return self.signals_by_strategy.get(strategy_id, []).copy()
        return self.stored_signals.copy()
    
    def get_signal_count(self) -> Dict[str, int]:
        """Get signal counts for monitoring storage usage."""
        return {
            'total_signals': len(self.stored_signals),
            'signals_by_strategy': {
                strategy_id: len(signals) 
                for strategy_id, signals in self.signals_by_strategy.items()
            }
        }
    
    def export_signals_for_replay(self, filepath: str, strategy_id: Optional[str] = None) -> None:
        """
        Export signals to file for replay (following storage doc format).
        
        Args:
            filepath: Path to save signals
            strategy_id: If provided, export only this strategy's signals
        """
        signals = self.get_signals_for_replay(strategy_id)
        
        # Export in format compatible with signal replay
        import json
        with open(filepath, 'w') as f:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'strategy_filter': strategy_id,
                    'signal_count': len(signals),
                    'observer_config': {
                        'max_signals_total': self.max_signals_total,
                        'max_signals_per_strategy': self.max_signals_per_strategy,
                        'retention_policy': self.retention_policy
                    }
                },
                'signals': signals
            }
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(signals)} signals to {filepath}")
    
    def get_performance(self, strategy_id: str) -> Optional['SignalPerformance']:
        """Get performance metrics for a strategy."""
        return self.strategy_performance.get(strategy_id)
    
    def enhance_signal_with_performance(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a signal with performance metrics.
        
        This is called by strategy containers before signal processing.
        """
        strategy_id = signal.get('strategy_id')
        if not strategy_id or strategy_id not in self.strategy_performance:
            # No performance data yet - return signal as-is
            return signal
            
        perf = self.strategy_performance[strategy_id]
        current_regime = signal.get('classifier_states', {}).get('trend')
        
        # Use the existing helper
        from ..tracing.signal_performance import create_risk_aware_signal
        return create_risk_aware_signal(signal, perf, current_regime)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get observer summary statistics."""
        return {
            'signals_observed': self._signals_observed,
            'trades_completed': self._trades_completed,
            'strategies_tracked': self._strategies_tracked,
            'active_strategies': len(self.strategy_performance),
            'pending_signals': len(self.pending_signals),
            
            # Signal storage statistics
            'signal_storage': {
                'total_signals_stored': len(self.stored_signals),
                'storage_utilization': {
                    'total': f"{len(self.stored_signals)}/{self.max_signals_total}",
                    'percentage': f"{len(self.stored_signals)/self.max_signals_total*100:.1f}%"
                },
                'signals_by_strategy': {
                    strategy_id: len(signals) 
                    for strategy_id, signals in self.signals_by_strategy.items()
                }
            },
            
            # Configuration
            'config': {
                'retention_policy': self.retention_policy,
                'max_strategies': self.max_strategies,
                'max_signals_total': self.max_signals_total,
                'max_signals_per_strategy': self.max_signals_per_strategy,
                'storage_based_retention': True  # Emphasize storage-based approach
            },
            
            # Performance tracking
            'performance_summary': {
                strategy_id: {
                    'total_signals': perf.total_signals,
                    'win_rate': perf.win_rate,
                    'confidence': perf.confidence_score,
                    'profit_factor': perf.profit_factor
                }
                for strategy_id, perf in self.strategy_performance.items()
            }
        }
    
    def get_all_performances(self) -> Dict[str, 'SignalPerformance']:
        """Get all strategy performances (for analysis/reporting)."""
        return self.strategy_performance.copy()


# Specialized Observers for Different Metrics

@dataclass
class SharpeRatioObserver(EventObserverProtocol):
    """Observer focused only on Sharpe ratio calculation."""
    
    calculator: 'SharpeRatioCalculator'
    
    def on_publish(self, event: Event) -> None:
        """Only process portfolio updates for returns."""
        if event.event_type == EventType.PORTFOLIO_UPDATE.value:
            value = event.payload.get('portfolio_value')
            if value:
                self.calculator.add_portfolio_value(value)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        pass


@dataclass
class DrawdownObserver(EventObserverProtocol):
    """Observer focused on drawdown tracking."""
    
    calculator: 'DrawdownCalculator'
    
    def on_publish(self, event: Event) -> None:
        """Track portfolio values for drawdown."""
        if event.event_type == EventType.PORTFOLIO_UPDATE.value:
            value = event.payload.get('portfolio_value')
            if value:
                self.calculator.update_value(value)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        pass


def create_tracer_from_config(config: Dict[str, Any]) -> EventTracer:
    """
    Create event tracer from configuration.
    
    Args:
        config: Configuration dict with:
            - correlation_id: Trace identifier
            - max_events: Maximum events to store
            - events_to_trace: List of event types or 'ALL'
            - retention_policy: 'all', 'trade_complete', 'sliding_window', 'minimal'
            - storage_backend: 'memory', 'disk', or 'hierarchical'
            - storage_config: Backend-specific config
            - workflow_id: For hierarchical storage context
            - phase_name: For hierarchical storage context
            - container_id: For hierarchical storage context
    
    Returns:
        Configured EventTracer instance
    """
    # Extract trace ID
    trace_id = config.get('correlation_id', f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create storage backend
    storage_backend = config.get('storage_backend', 'memory')
    storage_config = config.get('storage_config', {})
    storage_config['max_size'] = config.get('max_events', 10000)
    
    # Pass hierarchical storage context if provided
    if storage_backend == 'hierarchical':
        storage_config.update({
            'workflow_id': config.get('workflow_id'),
            'phase_name': config.get('phase_name'),
            'container_id': config.get('container_id')
        })
    
    storage = create_storage_backend(storage_backend, storage_config)
    
    # Create tracer
    tracer_config = {
        'max_events': config.get('max_events', 10000),
        'events_to_trace': config.get('events_to_trace', 'ALL'),
        'retention_policy': config.get('retention_policy', 'all')
    }
    
    return EventTracer(trace_id, storage, tracer_config)