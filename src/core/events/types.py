"""
Event types and protocols for the containerized ADMF-PC system.

This module defines the core event structure and protocols that enable
container-isolated event-driven communication between components.
"""

from __future__ import annotations
from typing import Protocol, Dict, Any, Optional, Union, Callable, runtime_checkable
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field


class EventType(Enum):
    """Standard event types used throughout the system."""
    # Market data events
    BAR = auto()
    TICK = auto()
    QUOTE = auto()
    
    # Trading events
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    CANCEL = auto()
    REJECT = auto()
    
    # Portfolio events
    PORTFOLIO = auto()
    POSITION = auto()
    BALANCE = auto()
    
    # System events
    SYSTEM = auto()
    LIFECYCLE = auto()
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    
    # Analytics events
    CLASSIFICATION = auto()
    REGIME_CHANGE = auto()
    REGIME = auto()  # Alias for REGIME_CHANGE
    INDICATORS = auto()  # For indicator distribution
    INDICATOR = auto()  # Individual indicator update
    META_LABEL = auto()
    METRIC = auto()
    RISK_UPDATE = auto()  # For risk limit changes
    
    # Control events
    START = auto()
    STOP = auto()
    PAUSE = auto()
    RESUME = auto()
    RESET = auto()
    
    # Backtest/Optimization events
    BACKTEST_START = auto()
    BACKTEST_END = auto()
    OPTIMIZATION_START = auto()
    OPTIMIZATION_END = auto()
    TRIAL_START = auto()
    TRIAL_END = auto()


@dataclass
class Event:
    """
    Core event structure for the ADMF-PC system.
    
    Events are the primary communication mechanism between components
    within a container. Each container has its own isolated event space.
    """
    event_type: Union[EventType, str]
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source_id: Optional[str] = None
    container_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate event structure after initialization."""
        if isinstance(self.event_type, str) and not self.event_type:
            raise ValueError("Event type cannot be empty string")
        
        if self.payload is None:
            self.payload = {}
            
        if self.metadata is None:
            self.metadata = {}


@runtime_checkable
class EventHandler(Protocol):
    """Protocol for event handlers."""
    
    def __call__(self, event: Event) -> None:
        """Handle an event."""
        ...


@runtime_checkable
class EventPublisher(Protocol):
    """Protocol for components that can publish events."""
    
    def publish(self, event: Event) -> None:
        """Publish an event to the event bus."""
        ...


@runtime_checkable
class EventSubscriber(Protocol):
    """Protocol for components that can subscribe to events."""
    
    def subscribe(self, event_type: Union[EventType, str], handler: EventHandler) -> None:
        """Subscribe to events of a specific type."""
        ...
    
    def unsubscribe(self, event_type: Union[EventType, str], handler: EventHandler) -> None:
        """Unsubscribe from events of a specific type."""
        ...


@runtime_checkable
class EventBusProtocol(Protocol):
    """Protocol for event bus implementations."""
    
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        ...
    
    def subscribe(self, event_type: Union[EventType, str], handler: EventHandler) -> None:
        """Subscribe a handler to events of a specific type."""
        ...
    
    def unsubscribe(self, event_type: Union[EventType, str], handler: EventHandler) -> None:
        """Unsubscribe a handler from events of a specific type."""
        ...
    
    def unsubscribe_all(self, handler: EventHandler) -> None:
        """Unsubscribe a handler from all event types."""
        ...


@runtime_checkable
class EventCapable(Protocol):
    """Protocol for components that participate in the event system."""
    
    @property
    def event_bus(self) -> EventBusProtocol:
        """Access to the container's event bus."""
        ...
    
    def initialize_events(self) -> None:
        """Initialize event subscriptions."""
        ...
    
    def teardown_events(self) -> None:
        """Clean up event subscriptions."""
        ...


# Common event factory functions

def create_market_event(
    event_type: EventType,
    symbol: str,
    timestamp: datetime,
    data: Dict[str, Any],
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create a market data event."""
    return Event(
        event_type=event_type,
        payload={
            "symbol": symbol,
            "data": data
        },
        timestamp=timestamp,
        source_id=source_id,
        container_id=container_id,
        metadata={"category": "market_data"}
    )


def create_signal_event(
    signal_payload: Dict[str, Any],
    timestamp: datetime,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create a trading signal event."""
    payload = signal_payload
    
    return Event(
        event_type=EventType.SIGNAL,
        payload=payload,
        timestamp=timestamp,
        source_id=source_id,
        container_id=container_id,
        metadata={"category": "trading"}
    )


def create_system_event(
    message: str,
    level: str = "info",
    source_id: Optional[str] = None,
    container_id: Optional[str] = None,
    **kwargs
) -> Event:
    """Create a system event."""
    return Event(
        event_type=EventType.SYSTEM,
        payload={
            "message": message,
            "level": level,
            **kwargs
        },
        timestamp=datetime.now(),
        source_id=source_id,
        container_id=container_id,
        metadata={"category": "system"}
    )


def create_error_event(
    error: Exception,
    error_context: Dict[str, Any],
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create an error event from an exception."""
    return Event(
        event_type=EventType.ERROR,
        payload=error_context,
        timestamp=datetime.now(),
        source_id=source_id,
        container_id=container_id,
        metadata={"category": "error"}
    )