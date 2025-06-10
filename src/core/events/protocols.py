"""Event system protocols."""

from typing import Protocol, Optional, Dict, Any, List, Callable
from abc import abstractmethod

class EventBusProtocol(Protocol):
    """Protocol for event bus implementations."""
    
    @abstractmethod
    def publish(self, event: 'Event') -> None:
        """Publish an event to all subscribers."""
        ...
    
    @abstractmethod
    def subscribe(self, event_type: 'EventType', handler: Callable, filter_func: Optional[Callable] = None) -> None:
        """Subscribe to an event type with optional filter."""
        ...
    
    @abstractmethod
    def unsubscribe(self, event_type: 'EventType', handler: Callable) -> None:
        """Unsubscribe a handler from an event type."""
        ...

class EventHandler(Protocol):
    """Protocol for event handlers."""
    
    @abstractmethod
    def __call__(self, event: 'Event') -> None:
        """Handle an event."""
        ...

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

class MetricsCalculatorProtocol(Protocol):
    """Protocol for metrics calculation."""
    
    @abstractmethod
    def update_from_trade(
        self, 
        entry_price: float, 
        exit_price: float, 
        quantity: float, 
        direction: str
    ) -> None:
        """Update metrics from a completed trade."""
        ...
    
    @abstractmethod
    def update_portfolio_value(
        self, 
        value: float, 
        timestamp: Optional['datetime'] = None
    ) -> None:
        """Update metrics with new portfolio value."""
        ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        ...

class ResultExtractor(Protocol):
    """Protocol for extracting business results from events."""
    
    @abstractmethod
    def extract(self, event: 'Event') -> Optional[Dict[str, Any]]:
        """Extract results from a single event."""
        ...
        
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Get accumulated results."""
        ...