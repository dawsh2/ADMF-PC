# NOTE: COMPLETE THIS CHECKLIST EVERY TIME!
This guide was created from an agent that did not have full scope of our codebase. When it references some objects or classes, like PortfolioContainer, be mindful that it's just using this code as a vague implementation refernce -- it does not correspond to our actual code. We need to implement this in a manner that is consistent with our _actual_ codebase. Before beginning, please read:
- [ ] src/core/containers/{container.py, factory.py, metrics.py, protocols.py}
- [ ] src/core/routing/{pipe.py, broadcast.py, factory.py, protocols.py}
- [ ] src/core/coordinator/{topology.py, topology_declarative.py} <-- responsible for container construction

We need to ensure that this implementation enables:
- [ ] containers to update metrics from event traces
- [ ] memory-aware event tracing (e.g, limit to only open orders)
- [ ] TopologyBuilder to enable tracing via configuration
- [ ] TopologyBuilder to construct a container with memory and/or disk writing limitations
- [ ] intelligent hierarchical storage of traces (e.g, folder for the root-bus and file per container, so we end up with an organizaed system like (./results/some_naming_convention/{portfolio1, portfolio2, etc}

This refactor is motivated by a) our events module spaghetti'ing and b) realizing we wanted to perform analysis on the full event trace (see: docs/architecture/data-*) while not maintaining two results databases or truth sources. This is our solution, so events are the truth source for performance and we should be able to have granular control over which containers are traced, which 

Related documents in this directory:
-[ ] semantics.md
-[ ] 

# NOTE: NOW UNCHECK THE LIST, AND LET'S GET TO WORK!


# Refactored Events Module Structure
# src/core/events/

# ============================================
# File: src/core/events/__init__.py
# ============================================
"""
Event system for ADMF-PC using Protocol + Composition.

This module provides a clean, composable event infrastructure where:
- EventBus is pure pub/sub with no tracing logic
- Tracing is added via observers using composition
- Everything uses protocols, no inheritance
- Integrates seamlessly with container architecture
"""

from .protocols import (
    EventObserverProtocol,
    EventTracerProtocol,
    EventStorageProtocol,
    EventFilterProtocol
)

from .bus import EventBus

from .types import (
    Event,
    EventType,
    create_market_event,
    create_signal_event,
    create_system_event,
    create_error_event
)

from .observers import (
    EventTracer,
    MetricsObserver,
    create_tracer_from_config
)

from .storage import (
    MemoryEventStorage,
    DiskEventStorage,
    create_storage_backend
)

from .filters import (
    combine_filters,
    any_of_filters,
    strategy_filter,
    symbol_filter,
    classification_filter,
    strength_filter,
    metadata_filter,
    payload_filter,
    custom_filter,
    create_portfolio_filter
)

__all__ = [
    # Protocols
    'EventObserverProtocol',
    'EventTracerProtocol',
    'EventStorageProtocol',
    'EventFilterProtocol',
    
    # Core
    'EventBus',
    'Event',
    'EventType',
    
    # Event creation
    'create_market_event',
    'create_signal_event',
    'create_system_event',
    'create_error_event',
    
    # Observers
    'EventTracer',
    'MetricsObserver',
    'create_tracer_from_config',
    
    # Storage
    'MemoryEventStorage',
    'DiskEventStorage',
    'create_storage_backend',
    
    # Filters
    'combine_filters',
    'any_of_filters',
    'strategy_filter',
    'symbol_filter',
    'classification_filter',
    'strength_filter',
    'metadata_filter',
    'payload_filter',
    'custom_filter',
    'create_portfolio_filter'
]


# ============================================
# File: src/core/events/protocols.py
# ============================================
"""Event system protocols."""

from typing import Protocol, Optional, Dict, Any, List, Callable
from abc import abstractmethod

class EventObserverProtocol(Protocol):
    """Protocol for observing events without coupling to EventBus."""
    
    @abstractmethod
    def on_publish(self, event: 'Event') -> None:
        """Called when event is published."""
        ...
    
    @abstractmethod
    def on_delivered(self, event: 'Event', handler: Callable) -> None:
        """Called when event is delivered to handler."""
        ...
    
    @abstractmethod
    def on_error(self, event: 'Event', handler: Callable, error: Exception) -> None:
        """Called when handler raises error."""
        ...

class EventTracerProtocol(Protocol):
    """Protocol for event tracing functionality."""
    
    @abstractmethod
    def trace_event(self, event: 'Event') -> None:
        """Trace an event."""
        ...
    
    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """Get trace summary."""
        ...
    
    @abstractmethod
    def get_events_by_correlation(self, correlation_id: str) -> List['Event']:
        """Get events by correlation ID."""
        ...
    
    @abstractmethod
    def save_to_file(self, filepath: str) -> None:
        """Save trace to file."""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all traced events."""
        ...

class EventStorageProtocol(Protocol):
    """Protocol for event storage backends."""
    
    @abstractmethod
    def store(self, event: 'Event') -> None:
        """Store an event."""
        ...
    
    @abstractmethod
    def retrieve(self, event_id: str) -> Optional['Event']:
        """Retrieve event by ID."""
        ...
    
    @abstractmethod
    def query(self, criteria: Dict[str, Any]) -> List['Event']:
        """Query events by criteria."""
        ...
    
    @abstractmethod
    def prune(self, criteria: Dict[str, Any]) -> int:
        """Prune events matching criteria, return count pruned."""
        ...
    
    @abstractmethod
    def count(self) -> int:
        """Get total event count."""
        ...
    
    @abstractmethod
    def export_to_file(self, filepath: str) -> None:
        """Export all events to file."""
        ...

class EventFilterProtocol(Protocol):
    """Protocol for event filtering."""
    
    @abstractmethod
    def should_process(self, event: 'Event') -> bool:
        """Check if event should be processed."""
        ...


# ============================================
# File: src/core/events/types.py
# ============================================
"""Event types and creation helpers."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class EventType(Enum):
    """Standard event types in the system."""
    # Market data events
    BAR = "BAR"
    TICK = "TICK"
    
    # Feature events
    FEATURES = "FEATURES"
    
    # Trading events
    SIGNAL = "SIGNAL"
    ORDER_REQUEST = "ORDER_REQUEST"
    ORDER = "ORDER"
    FILL = "FILL"
    
    # Portfolio events
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    POSITION_UPDATE = "POSITION_UPDATE"
    POSITION_OPEN = "POSITION_OPEN"
    POSITION_CLOSE = "POSITION_CLOSE"
    
    # System events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    ERROR = "ERROR"
    
    # Analysis events
    REGIME_CHANGE = "REGIME_CHANGE"
    RISK_BREACH = "RISK_BREACH"


@dataclass
class Event:
    """
    Core event structure.
    
    Attributes:
        event_type: Type of event
        payload: Event data
        source_id: ID of component that created event
        container_id: ID of container that owns the source
        correlation_id: ID to correlate related events
        causation_id: ID of event that caused this event
        timestamp: When event was created
        metadata: Additional event metadata
    """
    event_type: str
    payload: Dict[str, Any]
    source_id: Optional[str] = None
    container_id: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_type': self.event_type,
            'payload': self.payload,
            'source_id': self.source_id,
            'container_id': self.container_id,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        timestamp = data.get('timestamp')
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            event_type=data['event_type'],
            payload=data['payload'],
            source_id=data.get('source_id'),
            container_id=data.get('container_id'),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id'),
            timestamp=timestamp or datetime.now(),
            metadata=data.get('metadata', {})
        )


# Event creation helpers

def create_market_event(
    event_type: EventType,
    symbol: str,
    data: Dict[str, Any],
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create a market data event."""
    return Event(
        event_type=event_type.value,
        payload={
            'symbol': symbol,
            **data
        },
        source_id=source_id,
        container_id=container_id,
        metadata={'category': 'market_data'}
    )

def create_signal_event(
    symbol: str,
    direction: str,
    strength: float,
    strategy_id: str,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None,
    combo_id: Optional[str] = None
) -> Event:
    """Create a trading signal event."""
    return Event(
        event_type=EventType.SIGNAL.value,
        payload={
            'symbol': symbol,
            'direction': direction,
            'strength': strength,
            'strategy_id': strategy_id,
            'combo_id': combo_id
        },
        source_id=source_id,
        container_id=container_id,
        metadata={'category': 'trading'}
    )

def create_system_event(
    event_type: EventType,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create a system event."""
    return Event(
        event_type=event_type.value,
        payload={
            'message': message,
            'details': details or {}
        },
        source_id=source_id,
        container_id=container_id,
        metadata={'category': 'system'}
    )

def create_error_event(
    error_type: str,
    error_message: str,
    original_event: Optional[Event] = None,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create an error event."""
    payload = {
        'error_type': error_type,
        'error_message': error_message
    }
    
    if original_event:
        payload['original_event_type'] = original_event.event_type
        payload['original_event_id'] = original_event.metadata.get('event_id')
    
    return Event(
        event_type=EventType.ERROR.value,
        payload=payload,
        source_id=source_id,
        container_id=container_id,
        causation_id=original_event.metadata.get('event_id') if original_event else None,
        metadata={'category': 'error'}
    )


# ============================================
# File: src/core/events/bus.py
# ============================================
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
        
        logger.debug(f"EventBus created: {self.bus_id}")
    
    def attach_observer(self, observer: EventObserverProtocol) -> None:
        """
        Attach an observer for events.
        
        Args:
            observer: Observer implementing EventObserverProtocol
        """
        self._observers.append(observer)
        logger.debug(f"Attached observer {type(observer).__name__} to bus {self.bus_id}")
    
    def detach_observer(self, observer: EventObserverProtocol) -> None:
        """
        Detach an observer.
        
        Args:
            observer: Observer to remove
        """
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
        # Set correlation ID if not set
        if not event.correlation_id and self._correlation_id:
            event.correlation_id = self._correlation_id
        
        # Set container ID if not set
        if not event.container_id and self.bus_id:
            event.container_id = self.bus_id
        
        # Generate event ID if not present
        if 'event_id' not in event.metadata:
            event.metadata['event_id'] = f"{event.event_type}_{uuid.uuid4().hex[:8]}"
        
        # Notify observers of publish
        for observer in self._observers:
            try:
                observer.on_publish(event)
            except Exception as e:
                logger.error(f"Observer {observer} failed on_publish: {e}")
        
        self._event_count += 1
        
        # Get handlers
        handlers = self._subscribers.get(event.event_type, [])
        wildcard_handlers = self._subscribers.get('*', [])
        all_handlers = handlers + wildcard_handlers
        
        # CHANGE: Deliver to handlers that pass their filters
        for handler, filter_func in all_handlers:  # Now unpacking tuples
            try:
                # NEW: Check filter before calling handler
                if filter_func is not None and not filter_func(event):
                    self._filtered_count += 1
                    continue  # Skip this handler
                
                handler(event)
                
                # Notify observers of successful delivery
                for observer in self._observers:
                    try:
                        observer.on_delivered(event, handler)
                    except Exception as e:
                        logger.error(f"Observer {observer} failed on_delivered: {e}")
                        
            except Exception as e:
                self._error_count += 1
                
                # Notify observers of error
                for observer in self._observers:
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
        # NEW: Enforce filtering for SIGNAL events
        if event_type == EventType.SIGNAL.value and filter_func is None:
            raise ValueError(
                "SIGNAL events require a filter function to ensure portfolios "
                "only receive signals from their assigned strategies. "
                "Example: filter_func=lambda e: e.payload.get('strategy_id') in assigned_strategies"
            )
        
        # CHANGE: Store handler with its filter
        self._subscribers[event_type].append((handler, filter_func))
        
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
        for event_type in list(self._subscribers.keys()):
            self.unsubscribe(event_type, handler)
    
    def clear(self) -> None:
        """Clear all subscriptions."""
        self._subscribers.clear()
        logger.debug(f"Cleared all subscriptions on bus {self.bus_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics.
        
        CHANGE: Now includes filter statistics.
        """
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
        from .observers import create_tracer_from_config
        
        tracer = create_tracer_from_config(trace_config)
        self.attach_observer(tracer)
        
        # Store reference for convenience methods
        self._tracer = tracer
        
        logger.info(f"Tracing enabled on bus {self.bus_id}")
    
    def disable_tracing(self) -> None:
        """Disable tracing if enabled."""
        if hasattr(self, '_tracer'):
            self.detach_observer(self._tracer)
            delattr(self, '_tracer')
            logger.info(f"Tracing disabled on bus {self.bus_id}")
    
    def get_tracer_summary(self) -> Optional[Dict[str, Any]]:
        """Get tracer summary if tracing enabled."""
        if hasattr(self, '_tracer'):
            return self._tracer.get_summary()
        return None
    
    def save_trace_to_file(self, filepath: str) -> None:
        """Save trace to file if tracing enabled."""
        if hasattr(self, '_tracer'):
            self._tracer.save_to_file(filepath)


# ============================================
# File: src/core/events/filters.py
# ============================================
"""
Filter helper functions for event subscriptions.

These helpers make it easy to create and compose filters for
common subscription patterns.
"""

from typing import Callable, List, Optional, Set, Any
from .types import Event


def combine_filters(*filters: Callable[[Event], bool]) -> Callable[[Event], bool]:
    """
    Combine multiple filters with AND logic.
    
    All filters must pass for the event to be delivered.
    
    Args:
        *filters: Variable number of filter functions
        
    Returns:
        Combined filter function
        
    Example:
        # Only receive momentum signals for tech stocks
        filter_func = combine_filters(
            strategy_filter(['momentum_1']),
            symbol_filter(['AAPL', 'MSFT', 'GOOGL'])
        )
    """
    def combined(event: Event) -> bool:
        return all(f(event) for f in filters)
    return combined


def any_of_filters(*filters: Callable[[Event], bool]) -> Callable[[Event], bool]:
    """
    Combine multiple filters with OR logic.
    
    Any filter passing allows the event to be delivered.
    
    Args:
        *filters: Variable number of filter functions
        
    Returns:
        Combined filter function
        
    Example:
        # Receive signals from either momentum OR mean reversion strategies
        filter_func = any_of_filters(
            strategy_filter(['momentum_1']),
            strategy_filter(['mean_reversion_1'])
        )
    """
    def any_of(event: Event) -> bool:
        return any(f(event) for f in filters)
    return any_of


def strategy_filter(strategy_ids: List[str]) -> Callable[[Event], bool]:
    """
    Create a filter for specific strategy IDs.
    
    Args:
        strategy_ids: List of strategy IDs to accept
        
    Returns:
        Filter function
        
    Example:
        filter_func = strategy_filter(['momentum_1', 'pairs_1'])
    """
    strategy_set = set(strategy_ids)
    
    def filter_func(event: Event) -> bool:
        return event.payload.get('strategy_id') in strategy_set
    
    return filter_func


def symbol_filter(symbols: List[str]) -> Callable[[Event], bool]:
    """
    Create a filter for specific symbols.
    
    Args:
        symbols: List of symbols to accept
        
    Returns:
        Filter function
        
    Example:
        filter_func = symbol_filter(['AAPL', 'MSFT', 'GOOGL'])
    """
    symbol_set = set(symbols)
    
    def filter_func(event: Event) -> bool:
        return event.payload.get('symbol') in symbol_set
    
    return filter_func


def classification_filter(classifications: List[str]) -> Callable[[Event], bool]:
    """
    Create a filter for specific market classifications.
    
    Args:
        classifications: List of classifications to accept
        
    Returns:
        Filter function
        
    Example:
        # Only trade in trending markets
        filter_func = classification_filter(['strong_uptrend', 'strong_downtrend'])
    """
    classification_set = set(classifications)
    
    def filter_func(event: Event) -> bool:
        return event.payload.get('classification') in classification_set
    
    return filter_func


def strength_filter(min_strength: float, max_strength: float = 1.0) -> Callable[[Event], bool]:
    """
    Create a filter for signal strength.
    
    Args:
        min_strength: Minimum signal strength
        max_strength: Maximum signal strength
        
    Returns:
        Filter function
        
    Example:
        # Only high conviction signals
        filter_func = strength_filter(0.8)
    """
    def filter_func(event: Event) -> bool:
        strength = event.payload.get('strength', 0.0)
        return min_strength <= strength <= max_strength
    
    return filter_func


def metadata_filter(key: str, value: Any) -> Callable[[Event], bool]:
    """
    Create a filter for specific metadata values.
    
    Args:
        key: Metadata key to check
        value: Required value
        
    Returns:
        Filter function
        
    Example:
        # Only events from backtest containers
        filter_func = metadata_filter('container_type', 'backtest')
    """
    def filter_func(event: Event) -> bool:
        return event.metadata.get(key) == value
    
    return filter_func


def payload_filter(key: str, value: Any) -> Callable[[Event], bool]:
    """
    Create a filter for specific payload values.
    
    Args:
        key: Payload key to check
        value: Required value
        
    Returns:
        Filter function
        
    Example:
        # Only BUY signals
        filter_func = payload_filter('direction', 'BUY')
    """
    def filter_func(event: Event) -> bool:
        return event.payload.get(key) == value
    
    return filter_func


def custom_filter(predicate: Callable[[Event], bool]) -> Callable[[Event], bool]:
    """
    Create a filter with custom logic.
    
    This is just a wrapper for clarity when building filter compositions.
    
    Args:
        predicate: Custom predicate function
        
    Returns:
        Filter function
        
    Example:
        # Complex custom logic
        filter_func = custom_filter(
            lambda e: e.payload.get('volume', 0) > 1000000 and 
                     e.payload.get('price', 0) > 50
        )
    """
    return predicate


# ============================================
# Example Usage Patterns
# ============================================

def create_portfolio_filter(
    strategy_ids: List[str],
    symbols: Optional[List[str]] = None,
    min_strength: float = 0.0,
    classifications: Optional[List[str]] = None
) -> Callable[[Event], bool]:
    """
    Create a comprehensive portfolio filter.
    
    This is an example of how to compose filters for a portfolio container.
    
    Args:
        strategy_ids: Required list of strategy IDs
        symbols: Optional symbol whitelist
        min_strength: Minimum signal strength
        classifications: Optional classification whitelist
        
    Returns:
        Combined filter function
        
    Example:
        filter_func = create_portfolio_filter(
            strategy_ids=['momentum_1', 'pairs_1'],
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            min_strength=0.7,
            classifications=['trending', 'breakout']
        )
    """
    filters = [strategy_filter(strategy_ids)]
    
    if symbols:
        filters.append(symbol_filter(symbols))
    
    if min_strength > 0:
        filters.append(strength_filter(min_strength))
    
    if classifications:
        filters.append(classification_filter(classifications))
    
    return combine_filters(*filters)


# ============================================
# Migration Examples
# ============================================

def migrate_routing_to_filters():
    """
    Example showing how to replace complex routing with filters.
    
    OLD: Using routing infrastructure
    NEW: Using EventBus filters
    """
    # OLD: Complex routing setup
    # route = FilterRoute('signal_route', {...})
    # route.register_requirements('portfolio_1', ['momentum_1'])
    # route.setup(containers)
    # route.start()
    
    # NEW: Simple filter at subscription
    from .bus import EventBus
    
    root_bus = EventBus("root")
    portfolio = ...  # Portfolio container
    
    # Direct subscription with filter
    root_bus.subscribe(
        EventType.SIGNAL.value,
        portfolio.receive_event,
        filter_func=strategy_filter(['momentum_1'])
    )
    
    # Or use convenience method
    root_bus.subscribe_to_signals(
        portfolio.receive_event,
        strategy_ids=['momentum_1']
    )


# ============================================
# File: src/core/events/observers/__init__.py
# ============================================
"""Event observers for composition-based functionality."""

from .tracer import EventTracer, create_tracer_from_config
from .metrics import MetricsObserver

__all__ = [
    'EventTracer',
    'MetricsObserver',
    'create_tracer_from_config'
]


# ============================================
# File: src/core/events/observers/tracer.py
# ============================================
"""Event tracer implementation."""

from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime
import json
import logging

from ..protocols import EventObserverProtocol, EventTracerProtocol, EventStorageProtocol
from ..types import Event, EventType
from ..storage import create_storage_backend

logger = logging.getLogger(__name__)


class EventTracer(EventObserverProtocol, EventTracerProtocol):
    """
    Event tracer that can be attached to any EventBus.
    
    Implements both Observer (for bus integration) and Tracer protocols.
    Uses composition with storage backends for flexibility.
    """
    
    def __init__(
        self,
        trace_id: str,
        storage: EventStorageProtocol,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize event tracer.
        
        Args:
            trace_id: Unique identifier for this trace
            storage: Storage backend for events
            config: Configuration options
        """
        self.trace_id = trace_id
        self.storage = storage
        self.config = config or {}
        
        # Configuration
        self.max_events = self.config.get('max_events', 10000)
        self.events_to_trace = self.config.get('events_to_trace', 'ALL')
        self.retention_policy = self.config.get('retention_policy', 'all')
        
        # Convert string event types to list if needed
        if isinstance(self.events_to_trace, str) and self.events_to_trace != 'ALL':
            self.events_to_trace = [self.events_to_trace]
        
        # Metrics
        self._traced_count = 0
        self._pruned_count = 0
        self._start_time = datetime.now()
        
        logger.info(f"EventTracer created: {trace_id}, "
                   f"retention: {self.retention_policy}, "
                   f"events: {self.events_to_trace}")
    
    # EventObserverProtocol implementation
    
    def on_publish(self, event: Event) -> None:
        """Trace event when published."""
        if self._should_trace(event):
            self.trace_event(event)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Update event with delivery info."""
        # Could update event metadata with delivery timestamp
        if self._should_trace(event):
            event.metadata['delivered_at'] = datetime.now().isoformat()
            event.metadata['delivered_to'] = getattr(handler, '__name__', str(handler))
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Log errors in trace."""
        if self._should_trace(event):
            event.metadata['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'handler': getattr(handler, '__name__', str(handler)),
                'timestamp': datetime.now().isoformat()
            }
    
    # EventTracerProtocol implementation
    
    def trace_event(self, event: Event) -> None:
        """Trace an event."""
        # Add trace metadata
        event.metadata['trace_id'] = self.trace_id
        event.metadata['traced_at'] = datetime.now().isoformat()
        event.metadata['trace_sequence'] = self._traced_count
        
        # Store event
        self.storage.store(event)
        self._traced_count += 1
        
        # Apply retention policy
        self._apply_retention_policy(event)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get trace summary."""
        return {
            'trace_id': self.trace_id,
            'events_traced': self._traced_count,
            'events_pruned': self._pruned_count,
            'events_stored': self.storage.count(),
            'retention_policy': self.retention_policy,
            'start_time': self._start_time.isoformat(),
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'config': self.config
        }
    
    def get_events_by_correlation(self, correlation_id: str) -> List[Event]:
        """Get events by correlation ID."""
        return self.storage.query({'correlation_id': correlation_id})
    
    def save_to_file(self, filepath: str) -> None:
        """Save trace to file."""
        self.storage.export_to_file(filepath)
        logger.info(f"Saved trace {self.trace_id} to {filepath}")
    
    def clear(self) -> None:
        """Clear all traced events."""
        if hasattr(self.storage, 'clear'):
            self.storage.clear()
        self._traced_count = 0
        self._pruned_count = 0
        logger.info(f"Cleared trace {self.trace_id}")
    
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


def create_tracer_from_config(config: Dict[str, Any]) -> EventTracer:
    """
    Create event tracer from configuration.
    
    Args:
        config: Configuration dict with:
            - correlation_id: Trace identifier
            - max_events: Maximum events to store
            - events_to_trace: List of event types or 'ALL'
            - retention_policy: 'all', 'trade_complete', 'sliding_window', 'minimal'
            - storage_backend: 'memory' or 'disk'
            - storage_config: Backend-specific config
    
    Returns:
        Configured EventTracer instance
    """
    # Extract trace ID
    trace_id = config.get('correlation_id', f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create storage backend
    storage_backend = config.get('storage_backend', 'memory')
    storage_config = config.get('storage_config', {})
    storage_config['max_size'] = config.get('max_events', 10000)
    
    storage = create_storage_backend(storage_backend, storage_config)
    
    # Create tracer
    tracer_config = {
        'max_events': config.get('max_events', 10000),
        'events_to_trace': config.get('events_to_trace', 'ALL'),
        'retention_policy': config.get('retention_policy', 'all')
    }
    
    return EventTracer(trace_id, storage, tracer_config)


# ============================================
# File: src/core/events/observers/metrics.py
# ============================================
"""Metrics observer for performance tracking."""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from ..protocols import EventObserverProtocol
from ..types import Event, EventType

logger = logging.getLogger(__name__)


class MetricsObserver(EventObserverProtocol):
    """
    Observer that tracks event metrics.
    
    Separate from tracer - focuses only on counts and performance.
    """
    
    def __init__(self):
        """Initialize metrics observer."""
        self.event_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.handler_times: Dict[str, List[float]] = {}
        self.start_time = datetime.now()
        self.total_events = 0
        self.total_errors = 0
    
    def on_publish(self, event: Event) -> None:
        """Track event publication."""
        event_type = event.event_type
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        self.total_events += 1
        
        # Add publish timestamp
        event.metadata['publish_timestamp'] = datetime.now().timestamp()
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Track successful delivery."""
        # Calculate delivery time if publish timestamp exists
        if 'publish_timestamp' in event.metadata:
            delivery_time = datetime.now().timestamp() - event.metadata['publish_timestamp']
            handler_name = getattr(handler, '__name__', str(handler))
            
            if handler_name not in self.handler_times:
                self.handler_times[handler_name] = []
            self.handler_times[handler_name].append(delivery_time)
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Track errors."""
        event_type = event.event_type
        self.error_counts[event_type] = self.error_counts.get(event_type, 0) + 1
        self.total_errors += 1
        
        logger.error(f"Event {event_type} handler error: {error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'total_events': self.total_events,
            'total_errors': self.total_errors,
            'error_rate': self.total_errors / max(1, self.total_events),
            'events_per_second': self.total_events / max(1, uptime),
            'event_counts': dict(self.event_counts),
            'error_counts': dict(self.error_counts),
            'handler_performance': {
                name: {
                    'count': len(times),
                    'avg_ms': sum(times) / len(times) * 1000 if times else 0,
                    'max_ms': max(times) * 1000 if times else 0
                }
                for name, times in self.handler_times.items()
            },
            'uptime_seconds': uptime
        }


# ============================================
# File: src/core/events/storage/__init__.py
# ============================================
"""Event storage backends."""

from .memory import MemoryEventStorage
from .disk import DiskEventStorage

def create_storage_backend(backend_type: str, config: Dict[str, Any]) -> 'EventStorageProtocol':
    """
    Create storage backend from type and config.
    
    Args:
        backend_type: 'memory' or 'disk'
        config: Backend-specific configuration
    
    Returns:
        Storage backend instance
    """
    if backend_type == 'memory':
        return MemoryEventStorage(
            max_size=config.get('max_size'),
            enable_indices=config.get('enable_indices', True)
        )
    elif backend_type == 'disk':
        return DiskEventStorage(
            directory=config.get('directory', './traces'),
            max_file_size_mb=config.get('max_file_size_mb', 100),
            compression=config.get('compression', True)
        )
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")

__all__ = [
    'MemoryEventStorage',
    'DiskEventStorage',
    'create_storage_backend'
]


# ============================================
# File: src/core/events/storage/memory.py
# ============================================
"""In-memory event storage."""

from typing import Dict, Any, Optional, List, Set
from collections import deque
from datetime import datetime
import json

from ..protocols import EventStorageProtocol
from ..types import Event

class MemoryEventStorage(EventStorageProtocol):
    """
    In-memory event storage with configurable retention.
    
    Used by EventTracer for temporary storage during execution.
    Supports multiple indices for efficient querying.
    """
    
    def __init__(self, max_size: Optional[int] = None, enable_indices: bool = True):
        """
        Initialize memory storage.
        
        Args:
            max_size: Maximum events to store (None for unlimited)
            enable_indices: Whether to maintain indices for fast queries
        """
        self.max_size = max_size
        self.enable_indices = enable_indices
        
        # Primary storage
        self.events: deque = deque(maxlen=max_size) if max_size else deque()
        
        # Indices for fast lookup
        if enable_indices:
            self._event_id_index: Dict[str, Event] = {}
            self._correlation_index: Dict[str, List[Event]] = {}
            self._type_index: Dict[str, List[Event]] = {}
        
        self._total_stored = 0
    
    def store(self, event: Event) -> None:
        """Store an event."""
        # Check if we're at capacity with no maxlen
        if self.max_size is None and len(self.events) >= 1000000:  # 1M safety limit
            # Remove oldest
            oldest = self.events.popleft()
            self._remove_from_indices(oldest)
        
        # Store event
        self.events.append(event)
        self._total_stored += 1
        
        # Update indices
        if self.enable_indices:
            self._add_to_indices(event)
    
    def retrieve(self, event_id: str) -> Optional[Event]:
        """Retrieve event by ID."""
        if self.enable_indices:
            return self._event_id_index.get(event_id)
        
        # Linear search if no indices
        for event in self.events:
            if event.metadata.get('event_id') == event_id:
                return event
        return None
    
    def query(self, criteria: Dict[str, Any]) -> List[Event]:
        """Query events by criteria."""
        # Fast path for indexed queries
        if self.enable_indices:
            if 'correlation_id' in criteria:
                return self._correlation_index.get(criteria['correlation_id'], []).copy()
            
            if 'event_type' in criteria:
                return self._type_index.get(criteria['event_type'], []).copy()
        
        # General query
        results = []
        for event in self.events:
            if self._matches_criteria(event, criteria):
                results.append(event)
        
        return results
    
    def prune(self, criteria: Dict[str, Any]) -> int:
        """Prune events matching criteria."""
        # Special handling for correlation-based pruning
        if 'correlation_id' in criteria and self.enable_indices:
            correlation_id = criteria['correlation_id']
            events_to_prune = self._correlation_index.get(correlation_id, []).copy()
            
            # Check for exclusions
            if 'exclude_event_id' in criteria:
                exclude_id = criteria['exclude_event_id']
                events_to_prune = [e for e in events_to_prune 
                                 if e.metadata.get('event_id') != exclude_id]
            
            # Remove from storage
            pruned = 0
            for event in events_to_prune:
                try:
                    self.events.remove(event)
                    self._remove_from_indices(event)
                    pruned += 1
                except ValueError:
                    pass  # Already removed
            
            return pruned
        
        # General pruning
        to_remove = []
        for event in self.events:
            if self._matches_criteria(event, criteria):
                to_remove.append(event)
        
        for event in to_remove:
            self.events.remove(event)
            self._remove_from_indices(event)
        
        return len(to_remove)
    
    def count(self) -> int:
        """Get total event count."""
        return len(self.events)
    
    def export_to_file(self, filepath: str) -> None:
        """Export all events to file."""
        with open(filepath, 'w') as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict(), default=str) + '\n')
    
    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()
        if self.enable_indices:
            self._event_id_index.clear()
            self._correlation_index.clear()
            self._type_index.clear()
    
    def prune_oldest(self, count: int) -> int:
        """Prune oldest events."""
        pruned = 0
        for _ in range(min(count, len(self.events))):
            event = self.events.popleft()
            self._remove_from_indices(event)
            pruned += 1
        return pruned
    
    # Private methods
    
    def _add_to_indices(self, event: Event) -> None:
        """Add event to indices."""
        # Event ID index
        event_id = event.metadata.get('event_id')
        if event_id:
            self._event_id_index[event_id] = event
        
        # Correlation index
        if event.correlation_id:
            if event.correlation_id not in self._correlation_index:
                self._correlation_index[event.correlation_id] = []
            self._correlation_index[event.correlation_id].append(event)
        
        # Type index
        event_type = event.event_type
        if event_type not in self._type_index:
            self._type_index[event_type] = []
        self._type_index[event_type].append(event)
    
    def _remove_from_indices(self, event: Event) -> None:
        """Remove event from indices."""
        if not self.enable_indices:
            return
        
        # Event ID index
        event_id = event.metadata.get('event_id')
        if event_id and event_id in self._event_id_index:
            del self._event_id_index[event_id]
        
        # Correlation index
        if event.correlation_id and event.correlation_id in self._correlation_index:
            try:
                self._correlation_index[event.correlation_id].remove(event)
                if not self._correlation_index[event.correlation_id]:
                    del self._correlation_index[event.correlation_id]
            except ValueError:
                pass
        
        # Type index
        if event.event_type in self._type_index:
            try:
                self._type_index[event.event_type].remove(event)
                if not self._type_index[event.event_type]:
                    del self._type_index[event.event_type]
            except ValueError:
                pass
    
    def _matches_criteria(self, event: Event, criteria: Dict[str, Any]) -> bool:
        """Check if event matches criteria."""
        for key, value in criteria.items():
            if key.startswith('exclude_'):
                continue
                
            if hasattr(event, key):
                if getattr(event, key) != value:
                    return False
            elif key in event.metadata:
                if event.metadata[key] != value:
                    return False
            elif key in event.payload:
                if event.payload[key] != value:
                    return False
            else:
                return False
        
        return True


# ============================================
# File: src/core/events/storage/disk.py
# ============================================
"""Disk-based event storage."""

import os
import json
import gzip
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..protocols import EventStorageProtocol
from ..types import Event

class DiskEventStorage(EventStorageProtocol):
    """
    Disk-based event storage for persistence.
    
    Stores events in compressed JSON lines format.
    """
    
    def __init__(
        self,
        directory: str = './traces',
        max_file_size_mb: int = 100,
        compression: bool = True
    ):
        """
        Initialize disk storage.
        
        Args:
            directory: Directory to store event files
            max_file_size_mb: Maximum size per file before rotation
            compression: Whether to compress files
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.compression = compression
        
        self.current_file = None
        self.current_file_size = 0
        self.file_count = 0
        self._event_count = 0
        
        self._open_new_file()
    
    def store(self, event: Event) -> None:
        """Store an event to disk."""
        # Convert to JSON
        event_json = json.dumps(event.to_dict(), default=str) + '\n'
        event_bytes = event_json.encode('utf-8')
        
        # Check if we need a new file
        if self.current_file_size + len(event_bytes) > self.max_file_size:
            self._rotate_file()
        
        # Write event
        self.current_file.write(event_bytes)
        self.current_file_size += len(event_bytes)
        self._event_count += 1
        
        # Flush periodically
        if self._event_count % 100 == 0:
            self.current_file.flush()
    
    def retrieve(self, event_id: str) -> Optional[Event]:
        """Retrieve event by ID (requires scanning files)."""
        for file_path in self._get_all_files():
            with self._open_file_for_read(file_path) as f:
                for line in f:
                    event_data = json.loads(line)
                    if event_data.get('metadata', {}).get('event_id') == event_id:
                        return Event.from_dict(event_data)
        return None
    
    def query(self, criteria: Dict[str, Any]) -> List[Event]:
        """Query events by criteria (requires scanning files)."""
        results = []
        
        for file_path in self._get_all_files():
            with self._open_file_for_read(file_path) as f:
                for line in f:
                    event_data = json.loads(line)
                    event = Event.from_dict(event_data)
                    
                    if self._matches_criteria(event, criteria):
                        results.append(event)
        
        return results
    
    def prune(self, criteria: Dict[str, Any]) -> int:
        """Pruning not supported for disk storage."""
        # Disk storage doesn't support pruning individual events
        # Would need to rewrite files
        return 0
    
    def count(self) -> int:
        """Get total event count."""
        return self._event_count
    
    def export_to_file(self, filepath: str) -> None:
        """Export all events to a single file."""
        self.current_file.flush()
        
        with open(filepath, 'wb') as output:
            for file_path in self._get_all_files():
                with open(file_path, 'rb') as input_file:
                    output.write(input_file.read())
    
    # Private methods
    
    def _open_new_file(self) -> None:
        """Open a new file for writing."""
        if self.current_file:
            self.current_file.close()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"events_{timestamp}_{self.file_count:04d}"
        
        if self.compression:
            filename += '.jsonl.gz'
            file_path = self.directory / filename
            self.current_file = gzip.open(file_path, 'wt', encoding='utf-8')
        else:
            filename += '.jsonl'
            file_path = self.directory / filename
            self.current_file = open(file_path, 'w', encoding='utf-8')
        
        self.current_file_size = 0
        self.file_count += 1
    
    def _rotate_file(self) -> None:
        """Rotate to a new file."""
        self._open_new_file()
    
    def _get_all_files(self) -> List[Path]:
        """Get all event files in order."""
        pattern = '*.jsonl.gz' if self.compression else '*.jsonl'
        files = sorted(self.directory.glob(pattern))
        return files
    
    def _open_file_for_read(self, file_path: Path):
        """Open file for reading."""
        if file_path.suffix == '.gz':
            return gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            return open(file_path, 'r', encoding='utf-8')
    
    def _matches_criteria(self, event: Event, criteria: Dict[str, Any]) -> bool:
        """Check if event matches criteria."""
        for key, value in criteria.items():
            if hasattr(event, key):
                if getattr(event, key) != value:
                    return False
            elif key in event.metadata:
                if event.metadata[key] != value:
                    return False
            elif key in event.payload:
                if event.payload[key] != value:
                    return False
            else:
                return False
        return True
    
    def __del__(self):
        """Ensure file is closed on deletion."""
        if hasattr(self, 'current_file') and self.current_file:
            self.current_file.close()


# ============================================
# MIGRATION GUIDE: From Routing to Enhanced EventBus
# ============================================
"""
Migration Guide: Replacing Routing Module with Enhanced EventBus

The routing module is being removed in favor of EventBus with required filtering
for SIGNAL events. This guide shows how to migrate existing code.

Key Changes:
1. No more routing module imports
2. Direct EventBus subscriptions with filters
3. Simpler, cleaner architecture

Migration Steps:

1. REMOVE ROUTING IMPORTS
   OLD:
   ```python
   from src.core.routing import FilterRoute, create_feature_filter
   from src.core.routing.factory import RoutingFactory
   ```
   
   NEW:
   ```python
   from src.core.events import EventBus, strategy_filter
   ```

2. REPLACE FEATURE FILTER ROUTES
   OLD:
   ```python
   # Complex routing setup
   feature_filter = routing_factory.create_route(
       name='feature_filter',
       config={
           'type': 'filter',
           'filter_field': 'payload.features',
           'event_types': [EventType.FEATURES]
       }
   )
   feature_filter.register_requirements(...)
   feature_filter.setup(containers)
   ```
   
   NEW:
   ```python
   # Direct subscription - no routing needed
   # Features are processed in feature containers
   # Strategies subscribe to root bus with filters
   ```

3. REPLACE SIGNAL ROUTING
   OLD:
   ```python
   # Complex signal routing with dynamic strategy creation
   def create_strategy_transform(sid, stype, sconfig):
       def transform(event):
           strategy = create_component(stype, ...)
           return strategy.handle_features(event)
       return transform
   ```
   
   NEW:
   ```python
   # Portfolio subscribes with filter for its strategies
   root_bus.subscribe(
       EventType.SIGNAL.value,
       portfolio.receive_event,
       filter_func=strategy_filter(['momentum_1', 'pairs_1'])
   )
   ```

4. REPLACE RISK SERVICE ROUTE
   OLD:
   ```python
   risk_route = routing_factory.create_route(
       name='risk_service',
       config={
           'type': 'risk_service',
           'risk_validators': risk_validators,
           'root_event_bus': root_event_bus
       }
   )
   ```
   
   NEW:
   ```python
   # Risk validators as simple event handlers
   def risk_handler(event: Event):
       if event.event_type == EventType.ORDER_REQUEST.value:
           # Validate order
           for validator in risk_validators:
               if not validator(event):
                   # Publish rejection
                   return
           # Publish approved ORDER event
   
   root_bus.subscribe(EventType.ORDER_REQUEST.value, risk_handler)
   ```

5. REMOVE BROADCAST ROUTES
   OLD:
   ```python
   fill_broadcast = routing_factory.create_route(
       name='fill_broadcast',
       config={
           'type': 'broadcast',
           'source': 'execution',
           'targets': list(portfolio_containers.keys()),
           'allowed_types': [EventType.FILL]
       }
   )
   ```
   
   NEW:
   ```python
   # Execution publishes to root bus
   # Portfolios subscribe directly
   for portfolio in portfolios:
       root_bus.subscribe(EventType.FILL.value, portfolio.on_fill)
   ```

6. UPDATE TOPOLOGY BUILDERS
   OLD:
   ```python
   # In topology builder
   from src.core.coordinator.topologies.helpers.routing import route_backtest_topology
   routes = route_backtest_topology(containers, config)
   ```
   
   NEW:
   ```python
   # In topology builder
   # Just wire up subscriptions directly
   def setup_subscriptions(containers, root_bus):
       # Portfolios subscribe to signals with filters
       for portfolio_id, portfolio in portfolio_containers.items():
           strategy_ids = portfolio.config.get('strategy_ids', [])
           root_bus.subscribe_to_signals(
               portfolio.receive_event,
               strategy_ids=strategy_ids
           )
       
       # Risk subscribes to ORDER_REQUEST
       if risk_manager:
           root_bus.subscribe(
               EventType.ORDER_REQUEST.value,
               risk_manager.validate_order
           )
       
       # Execution subscribes to ORDER
       root_bus.subscribe(
           EventType.ORDER.value,
           execution.on_order
       )
       
       # Portfolios subscribe to FILL
       for portfolio in portfolio_containers.values():
           root_bus.subscribe(
               EventType.FILL.value,
               portfolio.on_fill
           )
   ```

7. REMOVE ROUTING CLEANUP
   OLD:
   ```python
   # In cleanup
   for route in self.routes:
       route.stop()
   ```
   
   NEW:
   ```python
   # Nothing to clean up - subscriptions cleaned with containers
   ```

Benefits of New Approach:
- Simpler: No routing infrastructure to manage
- Cleaner: Direct subscriptions with filters
- Safer: Required filters prevent routing errors
- Faster: No intermediate routing layers
- Clearer: Event flow is explicit in subscriptions

Common Patterns:

1. Portfolio Signal Filtering:
   ```python
   # Each portfolio only gets its assigned strategies
   bus.subscribe_to_signals(
       portfolio.receive_event,
       strategy_ids=['momentum_1', 'mean_reversion_1']
   )
   ```

2. Multi-Criteria Filtering:
   ```python
   # Complex filtering with composition
   filter_func = combine_filters(
       strategy_filter(['momentum_1']),
       symbol_filter(['AAPL', 'MSFT']),
       strength_filter(0.7, 1.0)
   )
   bus.subscribe(EventType.SIGNAL.value, handler, filter_func)
   ```

3. Conditional Processing:
   ```python
   # Different handlers for different conditions
   bus.subscribe(
       EventType.SIGNAL.value,
       aggressive_handler,
       filter_func=strength_filter(0.9, 1.0)
   )
   
   bus.subscribe(
       EventType.SIGNAL.value,
       conservative_handler,
       filter_func=strength_filter(0.0, 0.5)
   )
   ```

4. Event Transformation:
   ```python
   # Transform events in handlers, not routes
   def order_handler(signal_event):
       # Transform signal to order
       order = create_order_from_signal(signal_event)
       bus.publish(order)
   
   bus.subscribe(EventType.SIGNAL.value, order_handler, 
                filter_func=strategy_filter(['my_strategy']))
   ```

Testing Migration:
1. Remove all routing imports
2. Replace route creation with direct subscriptions
3. Verify SIGNAL subscriptions have filters
4. Test event flow with logging
5. Confirm no events are missed or duplicated
"""