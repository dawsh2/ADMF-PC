"""
Protocol definitions for the adapter-based communication system.

This module defines the protocols (interfaces) that containers and adapters
must implement, following ADMF-PC's protocol-based architecture.
"""

from typing import Protocol, runtime_checkable, Dict, Any, Optional
from abc import abstractmethod

from ..types.events import Event, EventType, EventBusProtocol


@runtime_checkable
class Container(Protocol):
    """Minimal container interface for adapter compatibility.
    
    All containers must implement this protocol to work with adapters.
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
        """Receive events from adapters.
        
        Args:
            event: Event to process
        """
        ...
    
    def publish_event(self, event: Event) -> None:
        """Publish events to adapters.
        
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
class CommunicationAdapter(Protocol):
    """Protocol for all communication adapters.
    
    Adapters implement this protocol to route events between containers.
    No inheritance required - composition over inheritance.
    """
    
    name: str
    config: Dict[str, Any]
    
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure adapter with container references.
        
        Called once during initialization to establish connections.
        
        Args:
            containers: Map of container names to container instances
        """
        ...
    
    def start(self) -> None:
        """Start adapter operation.
        
        Called when the system is ready to begin processing events.
        """
        ...
    
    def stop(self) -> None:
        """Stop adapter operation.
        
        Called during shutdown to clean up resources.
        """
        ...
    
    def handle_event(self, event: Event, source: Container) -> None:
        """Process event with error handling and metrics.
        
        This is the main entry point for events flowing through the adapter.
        
        Args:
            event: Event to process
            source: Container that published the event
        """
        ...


@runtime_checkable
class AdapterMetrics(Protocol):
    """Protocol for adapter metrics collection."""
    
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
class AdapterErrorHandler(Protocol):
    """Protocol for adapter error handling."""
    
    def handle(self, event: Event, error: Exception) -> None:
        """Handle an error that occurred during event processing.
        
        Args:
            event: Event that caused the error
            error: Exception that was raised
        """
        ...