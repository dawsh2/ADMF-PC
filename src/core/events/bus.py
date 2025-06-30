"""
EventBus Enhancement: Required Filtering for SIGNAL Events

Motivation:
-----------
In our event-driven backtest architecture, we have a specific routing challenge:
- Multiple strategies publish SIGNAL events to the root bus
- Multiple portfolio containers subscribe to the root bus
- Each portfolio only manages a subset of strategies

Without filtering, every portfolio receives every signal, leading to:
- Unnecessary processing (portfolios discard irrelevant signals)
- Potential errors (portfolios might process wrong signals)
- Poor performance at scale (N portfolios × M strategies = N×M deliveries)

Solution:
---------
We enhance the EventBus to REQUIRE filters for SIGNAL events, while keeping
other event types unchanged. This enforces correct wiring at subscription time
rather than hoping portfolios filter correctly.
"""

from typing import Dict, List, Set, Optional, Callable, Any, Tuple
from collections import defaultdict
import logging
import uuid
import threading

from .protocols import EventObserverProtocol
from .types import Event, EventType

logger = logging.getLogger(__name__)


class EventBus:
    """
    Pure event bus implementation - no tracing logic.
    
    Tracing and other concerns are added via observers using composition.
    Thread-safe for use within a single container.
    
    ENHANCEMENT: Requires filtering for SIGNAL events to ensure correct routing.
    """
    
    def __init__(self, bus_id: Optional[str] = None):
        """
        Initialize event bus.
        
        Args:
            bus_id: Optional identifier for this bus
        """
        self.bus_id = bus_id or f"bus_{uuid.uuid4().hex[:8]}"
        
        # CHANGE: Store handlers with optional filters as tuples
        # OLD: self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        # NEW: Store (handler, filter_func) tuples
        self._subscribers: Dict[str, List[Tuple[Callable, Optional[Callable]]]] = defaultdict(list)
        
        self._observers: List[EventObserverProtocol] = []
        self._correlation_id: Optional[str] = None
        
        # Basic metrics
        self._event_count = 0
        self._error_count = 0
        # ADDITION: Track filtered events
        self._filtered_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.debug(f"EventBus created: {self.bus_id}")
    
    def attach_observer(self, observer: EventObserverProtocol) -> None:
        """
        Attach an observer for events.
        
        Args:
            observer: Observer implementing EventObserverProtocol
        """
        with self._lock:
            self._observers.append(observer)
            logger.debug(f"Attached observer {type(observer).__name__} to bus {self.bus_id}")
    
    def detach_observer(self, observer: EventObserverProtocol) -> None:
        """
        Detach an observer.
        
        Args:
            observer: Observer to remove
        """
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
                logger.debug(f"Detached observer {type(observer).__name__} from bus {self.bus_id}")
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for events published through this bus."""
        self._correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return self._correlation_id
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        CHANGE: Now checks filters before invoking handlers.
        
        Args:
            event: Event to publish
        """
        with self._lock:
            # Set correlation ID if not set
            if not event.correlation_id and self._correlation_id:
                event.correlation_id = self._correlation_id
            
            # Set container ID if not set
            if not event.container_id and self.bus_id:
                event.container_id = self.bus_id
            
            # Generate event ID if not present
            if 'event_id' not in event.metadata:
                event.metadata['event_id'] = f"{event.event_type}_{uuid.uuid4().hex[:8]}"
            
            # Create snapshots to avoid modification during iteration
            observers_snapshot = list(self._observers)
            
            # Notify observers of publish
            for observer in observers_snapshot:
                try:
                    observer.on_publish(event)
                except Exception as e:
                    logger.error(f"Observer {observer} failed on_publish: {e}")
            
            self._event_count += 1
            
            # Get handlers - already creates new list via concatenation
            handlers = self._subscribers.get(event.event_type, [])
            wildcard_handlers = self._subscribers.get('*', [])
            all_handlers = handlers + wildcard_handlers
            
            # Debug: log FILL event handling
            if event.event_type == "FILL":
                logger.info(f"🔍 Publishing FILL event: {len(handlers)} handlers, {len(wildcard_handlers)} wildcard handlers")
                logger.info(f"   All event types with handlers: {list(self._subscribers.keys())}")
                logger.info(f"   Event type in publish: '{event.event_type}'")
            
            # CHANGE: Deliver to handlers that pass their filters
            for handler, filter_func in all_handlers:  # Now unpacking tuples
                try:
                    # NEW: Check filter before calling handler
                    if filter_func is not None and not filter_func(event):
                        self._filtered_count += 1
                        continue  # Skip this handler
                    
                    # Debug: log handler calls for FILL events
                    if event.event_type == "FILL":
                        logger.debug(f"🔍 Calling handler {handler.__name__} for FILL event")
                    
                    handler(event)
                    
                    # Notify observers of successful delivery
                    for observer in observers_snapshot:
                        try:
                            observer.on_delivered(event, handler)
                        except Exception as e:
                            logger.error(f"Observer {observer} failed on_delivered: {e}")
                            
                except Exception as e:
                    self._error_count += 1
                    
                    # Notify observers of error
                    for observer in observers_snapshot:
                        try:
                            observer.on_error(event, handler, e)
                        except Exception as e2:
                            logger.error(f"Observer {observer} failed on_error: {e2}")
                    
                    logger.error(f"Handler {handler} failed for event {event.event_type}: {e}")
    
    # CHANGE: Updated subscribe method with filter_func parameter
    def subscribe(self, event_type: str, handler: Callable, 
                  filter_func: Optional[Callable[[Event], bool]] = None) -> None:
        """
        Subscribe to events of a specific type.
        
        CHANGE: Now accepts optional filter function.
        SIGNAL events REQUIRE a filter to prevent routing errors.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Function to call when event is published
            filter_func: Optional filter function (REQUIRED for SIGNAL events)
            
        Raises:
            ValueError: If SIGNAL event subscription lacks a filter
            
        Example:
            # SIGNAL events require filter
            bus.subscribe(
                EventType.SIGNAL.value,
                portfolio.receive_event,
                filter_func=lambda e: e.payload.get('strategy_id') in ['strat_1', 'strat_2']
            )
            
            # Other events don't require filter
            bus.subscribe(EventType.FILL.value, portfolio.receive_event)
        """
        # Enforce filtering for SIGNAL events
        if event_type == EventType.SIGNAL.value and filter_func is None:
            raise ValueError(
                "SIGNAL events require a filter function to ensure portfolios "
                "only receive signals from their assigned strategies. "
                "Example: filter_func=lambda e: e.payload.get('strategy_id') in assigned_strategies"
            )
        
        # Enforce filtering for FILL events
        if event_type == EventType.FILL.value and filter_func is None:
            raise ValueError(
                "FILL events require a filter function to prevent portfolio "
                "cross-contamination. Use container_filter(container_id) or "
                "order_filter(order_ids)"
            )
        
        with self._lock:
            # CHANGE: Store handler with its filter
            self._subscribers[event_type].append((handler, filter_func))
            
            # Debug: log FILL subscriptions
            if event_type == "FILL":
                logger.info(f"📝 FILL subscription added: handler={handler.__name__}, filter={filter_func is not None}")
                logger.info(f"   Total FILL handlers now: {len(self._subscribers.get('FILL', []))}")
        
        filter_desc = f" with filter" if filter_func else ""
        logger.debug(f"Subscribed {handler} to {event_type}{filter_desc} on bus {self.bus_id}")
    
    # NEW: Convenience method for signal subscriptions
    def subscribe_to_signals(self, handler: Callable, strategy_ids: List[str]) -> None:
        """
        Convenience method for subscribing to signals from specific strategies.
        
        Args:
            handler: Function to call when matching signal is received
            strategy_ids: List of strategy IDs to receive signals from
            
        Example:
            bus.subscribe_to_signals(
                portfolio.receive_event,
                strategy_ids=['momentum_1', 'pairs_1']
            )
        """
        filter_func = lambda e: e.payload.get('strategy_id') in strategy_ids
        self.subscribe(EventType.SIGNAL.value, handler, filter_func)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """
        Unsubscribe from events.
        
        CHANGE: Updated to work with (handler, filter) tuples.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler to remove
        """
        with self._lock:
            # CHANGE: Filter tuples to remove matching handler
            self._subscribers[event_type] = [
                (h, f) for h, f in self._subscribers[event_type] if h != handler
            ]
        logger.debug(f"Unsubscribed {handler} from {event_type} on bus {self.bus_id}")
    
    def subscribe_all(self, handler: Callable) -> None:
        """
        Subscribe to all events.
        
        NOTE: Wildcard subscriptions don't require filters since they're
        typically used for logging/monitoring, not business logic.
        
        Args:
            handler: Function to call for all events
        """
        self.subscribe('*', handler)
    
    def unsubscribe_all(self, handler: Callable) -> None:
        """
        Unsubscribe handler from all event types.
        
        Args:
            handler: Handler to remove
        """
        with self._lock:
            for event_type in list(self._subscribers.keys()):
                self.unsubscribe(event_type, handler)
    
    def clear(self) -> None:
        """Clear all subscriptions."""
        with self._lock:
            self._subscribers.clear()
        logger.debug(f"Cleared all subscriptions on bus {self.bus_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics.
        
        CHANGE: Now includes filter statistics.
        """
        with self._lock:
            # Count handlers with filters
            total_handlers = sum(len(handlers) for handlers in self._subscribers.values())
            filtered_handlers = sum(
                1 for handlers in self._subscribers.values() 
                for _, filter_func in handlers if filter_func is not None
            )
            
            return {
                'bus_id': self.bus_id,
                'event_count': self._event_count,
                'error_count': self._error_count,
                'filtered_count': self._filtered_count,  # NEW
                'filter_rate': self._filtered_count / max(1, self._event_count),  # NEW
                'subscriber_count': total_handlers,
                'filtered_subscriber_count': filtered_handlers,  # NEW
                'observer_count': len(self._observers),
                'event_types': list(self._subscribers.keys())
            }
    
    # Container integration helpers
    
    def enable_tracing(self, trace_config: Dict[str, Any]) -> None:
        """
        Enable tracing by creating and attaching a tracer.
        
        This is a convenience method for container integration.
        
        Args:
            trace_config: Configuration for tracing
        """
        from .tracing.observers import create_tracer_from_config
        
        tracer = create_tracer_from_config(trace_config)
        self.attach_observer(tracer)
        
        # Store reference for convenience methods
        self._tracer = tracer
        
        # Enable console output if configured
        if trace_config.get('enable_console_output', False):
            from .observers.console import create_console_observer_from_config
            
            console_observer = create_console_observer_from_config(trace_config)
            self.attach_observer(console_observer)
            
            # Store reference for convenience methods
            self._console_observer = console_observer
            logger.info(f"Console output enabled on bus {self.bus_id}")
        
        logger.info(f"Tracing enabled on bus {self.bus_id}")
    
    def disable_tracing(self) -> None:
        """Disable tracing if enabled."""
        if hasattr(self, '_tracer'):
            self.detach_observer(self._tracer)
            delattr(self, '_tracer')
            logger.info(f"Tracing disabled on bus {self.bus_id}")
        
        if hasattr(self, '_console_observer'):
            self.detach_observer(self._console_observer)
            delattr(self, '_console_observer')
            logger.info(f"Console output disabled on bus {self.bus_id}")
    
    def get_tracer_summary(self) -> Optional[Dict[str, Any]]:
        """Get tracer summary if tracing enabled."""
        if hasattr(self, '_tracer'):
            return self._tracer.get_summary()
        return None
    
    def save_trace_to_file(self, filepath: str) -> None:
        """Save trace to file if tracing enabled."""
        if hasattr(self, '_tracer'):
            self._tracer.save_to_file(filepath)
    
    # ==================== Monitoring and Debugging Methods ====================
    
    def get_handler_count(self, event_type: str) -> int:
        """Get number of active handlers for debugging."""
        with self._lock:
            return len(self._subscribers.get(event_type, []))
    
    def list_active_filters(self) -> Dict[str, int]:
        """List active filters for monitoring."""
        filter_counts = {}
        with self._lock:
            for event_type, handlers in self._subscribers.items():
                filtered_count = sum(1 for _, filter_func in handlers if filter_func is not None)
                if filtered_count > 0:
                    filter_counts[event_type] = filtered_count
        return filter_counts
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get detailed subscription statistics for monitoring."""
        with self._lock:
            stats = {
                'total_subscriptions': sum(len(handlers) for handlers in self._subscribers.values()),
                'event_types': len(self._subscribers),
                'filtered_subscriptions': 0,
                'unfiltered_subscriptions': 0,
                'by_event_type': {}
            }
            
            for event_type, handlers in self._subscribers.items():
                filtered = sum(1 for _, filter_func in handlers if filter_func is not None)
                unfiltered = len(handlers) - filtered
                
                stats['filtered_subscriptions'] += filtered
                stats['unfiltered_subscriptions'] += unfiltered
                stats['by_event_type'][event_type] = {
                    'total': len(handlers),
                    'filtered': filtered,
                    'unfiltered': unfiltered
                }
            
            return stats