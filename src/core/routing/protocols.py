"""
Protocol definitions for the route-based communication system.

This module defines the protocols (interfaces) that containers and routes
must implement, following ADMF-PC's protocol-based architecture.
"""

from typing import Protocol, runtime_checkable, Dict, Any, Optional
from abc import abstractmethod

from ..types.events import Event, EventType, EventBusProtocol


@runtime_checkable
class Container(Protocol):
    """Minimal container interface for route compatibility.
    
    All containers must implement this protocol to work with routes.
    No inheritance required - any object with these methods/properties
    is a valid container.
    """
    
    @property
    def name(self) -> str:
        """Unique container identifier."""
        ...
    
    @property
    def event_bus(self) -> EventBusProtocol:
        """Container's isolated event bus."""
        ...
    
    def receive_event(self, event: Event) -> None:
        """Receive events from routes.
        
        Args:
            event: Event to process
        """
        ...
    
    def publish_event(self, event: Event) -> None:
        """Publish events to routes.
        
        Args:
            event: Event to publish
        """
        ...
    
    @abstractmethod
    def process(self, event: Event) -> Optional[Event]:
        """Container's business logic.
        
        Args:
            event: Event to process
            
        Returns:
            Optional output event
        """
        ...


@runtime_checkable
class CommunicationRoute(Protocol):
    """Protocol for all communication routes.
    
    Routes implement this protocol to route events between containers.
    No inheritance required - composition over inheritance.
    """
    
    name: str
    config: Dict[str, Any]
    
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure route with container references.
        
        Called once during initialization to establish connections.
        
        Args:
            containers: Map of container names to container instances
        """
        ...
    
    def start(self) -> None:
        """Start route operation.
        
        Called when the system is ready to begin processing events.
        """
        ...
    
    def stop(self) -> None:
        """Stop route operation.
        
        Called during shutdown to clean up resources.
        """
        ...
    
    def handle_event(self, event: Event, source: Container) -> None:
        """Process event with error handling and metrics.
        
        This is the main entry point for events flowing through the route.
        
        Args:
            event: Event to process
            source: Container that published the event
        """
        ...


@runtime_checkable
class RouteMetrics(Protocol):
    """Protocol for route metrics collection."""
    
    def increment_success(self) -> None:
        """Increment successful event counter."""
        ...
        
    def increment_error(self) -> None:
        """Increment error counter."""
        ...
        
    def measure_latency(self):
        """Context manager for measuring latency."""
        ...


@runtime_checkable
class RouteErrorHandler(Protocol):
    """Protocol for route error handling."""
    
    def handle(self, event: Event, error: Exception) -> None:
        """Handle an error that occurred during event processing.
        
        Args:
            event: Event that caused the error
            error: Exception that was raised
        """
        ...


# Additional protocols for route composition

@runtime_checkable
class RouteFactory(Protocol):
    """Protocol for creating routes with standard infrastructure."""
    
    def create_route(self, route_class: type, name: str, config: Dict[str, Any]) -> CommunicationRoute:
        """Create a route with standard infrastructure attached."""
        ...


@runtime_checkable
class EventHandler(Protocol):
    """Protocol for event handling functions."""
    
    def __call__(self, event: Event) -> None:
        """Handle an event."""
        ...