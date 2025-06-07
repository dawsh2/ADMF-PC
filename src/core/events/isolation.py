"""
Event isolation mechanisms for container-based execution.

This module provides the infrastructure to ensure complete event isolation
between containers, preventing any cross-contamination during parallel
execution of backtests or optimization trials.
"""

from __future__ import annotations
from typing import Dict, Optional, Any, Set, Protocol, runtime_checkable
from contextlib import contextmanager
from threading import local
import logging
from weakref import WeakValueDictionary

from ..types.events import Event, EventBusProtocol
from .event_bus import EventBus


logger = logging.getLogger(__name__)


@runtime_checkable
class ContainerProtocol(Protocol):
    """Protocol for containers that host event buses."""
    
    @property
    def container_id(self) -> str:
        """Unique identifier for the container."""
        ...
    
    @property
    def event_bus(self) -> EventBusProtocol:
        """The container's event bus."""
        ...


class EventIsolationManager:
    """
    Manages event isolation between containers.
    
    This class ensures that:
    1. Each container has its own event bus
    2. Events cannot leak between containers
    3. Containers are properly cleaned up
    4. Thread-local storage prevents accidental sharing
    """
    
    def __init__(self):
        """Initialize the isolation manager."""
        # Weak references to allow garbage collection
        self._container_buses: WeakValueDictionary[str, EventBus] = WeakValueDictionary()
        self._active_containers: Set[str] = set()
        
        # Thread-local storage for current container context
        self._local = local()
        
        logger.debug("EventIsolationManager initialized")
    
    def create_container_bus(self, container_id: str) -> EventBus:
        """
        Create an isolated event bus for a container.
        
        Args:
            container_id: Unique identifier for the container
            
        Returns:
            EventBus instance
            
        Raises:
            ValueError: If container_id already exists
        """
        if container_id in self._active_containers:
            raise ValueError(f"Container {container_id} already exists")
        
        event_bus = EventBus(container_id)
        self._container_buses[container_id] = event_bus
        self._active_containers.add(container_id)
        
        logger.info(f"Created isolated event bus for container: {container_id}")
        return event_bus
    
    def get_container_bus(self, container_id: str) -> Optional[EventBus]:
        """
        Get the event bus for a specific container.
        
        Args:
            container_id: The container identifier
            
        Returns:
            The container's event bus or None if not found
        """
        return self._container_buses.get(container_id)
    
    def remove_container_bus(self, container_id: str) -> None:
        """
        Remove a container's event bus.
        
        Args:
            container_id: The container to remove
        """
        if container_id in self._active_containers:
            self._active_containers.remove(container_id)
            
            # Clear the bus if it still exists
            bus = self._container_buses.get(container_id)
            if bus:
                bus.clear()
            
            logger.info(f"Removed event bus for container: {container_id}")
    
    @contextmanager
    def container_context(self, container_id: str):
        """
        Context manager for container-scoped operations.
        
        This ensures that any code executed within the context
        is associated with the specified container.
        
        Args:
            container_id: The container to activate
            
        Yields:
            The container's event bus
        """
        previous_container = getattr(self._local, 'current_container', None)
        self._local.current_container = container_id
        
        try:
            bus = self.get_container_bus(container_id)
            if not bus:
                raise ValueError(f"Container {container_id} not found")
            yield bus
        finally:
            self._local.current_container = previous_container
    
    @property
    def current_container(self) -> Optional[str]:
        """Get the current container ID from thread-local storage."""
        return getattr(self._local, 'current_container', None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about managed containers."""
        stats = {
            "active_containers": len(self._active_containers),
            "container_ids": list(self._active_containers),
            "current_container": self.current_container
        }
        
        # Add per-container stats
        container_stats = {}
        for container_id in self._active_containers:
            bus = self._container_buses.get(container_id)
            if bus:
                container_stats[container_id] = bus.get_stats()
        
        stats["containers"] = container_stats
        return stats


class IsolatedEventPublisher:
    """
    Helper class for publishing events within container isolation.
    
    This ensures events are always published to the correct container's
    event bus and include proper container identification.
    """
    
    def __init__(
        self,
        isolation_manager: EventIsolationManager,
        container_id: str,
        source_id: Optional[str] = None
    ):
        """
        Initialize the publisher.
        
        Args:
            isolation_manager: The isolation manager
            container_id: The container to publish to
            source_id: Optional source identifier
        """
        self.isolation_manager = isolation_manager
        self.container_id = container_id
        self.source_id = source_id
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to the container's bus.
        
        Args:
            event: The event to publish
            
        Raises:
            ValueError: If the container doesn't exist
        """
        # Ensure event has container and source info
        if event.container_id is None:
            event.container_id = self.container_id
        if event.source_id is None and self.source_id:
            event.source_id = self.source_id
        
        # Get the container's bus
        bus = self.isolation_manager.get_container_bus(self.container_id)
        if not bus:
            raise ValueError(f"Container {self.container_id} not found")
        
        # Publish to the isolated bus
        bus.publish(event)


def validate_event_isolation(event: Event, expected_container: str) -> bool:
    """
    Validate that an event belongs to the expected container.
    
    Args:
        event: The event to validate
        expected_container: The expected container ID
        
    Returns:
        True if the event is properly isolated
    """
    if event.container_id != expected_container:
        logger.warning(
            f"Event isolation violation: Event from container "
            f"{event.container_id} received in container {expected_container}"
        )
        return False
    return True


# Global isolation manager instance
_isolation_manager = EventIsolationManager()


def get_isolation_manager() -> EventIsolationManager:
    """Get the global isolation manager instance."""
    return _isolation_manager