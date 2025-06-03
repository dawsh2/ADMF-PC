"""
Integration module to bridge legacy code with protocol-based adapters.

This module provides compatibility layers and factory functions to help
existing code work with the new protocol-based communication system.
"""

from typing import Dict, Any, List, Optional
import logging

from .factory import AdapterFactory, create_adapter_network
from .protocols import Container
from ..events.types import Event

logger = logging.getLogger(__name__)


class EventCommunicationFactory:
    """
    Compatibility layer for legacy code expecting EventCommunicationFactory.
    
    This wraps our new AdapterFactory to provide the expected interface.
    """
    
    def __init__(self, enable_communication: bool = True, **kwargs):
        """Initialize the communication factory.
        
        Args:
            enable_communication: Whether to enable communication
            **kwargs: Additional arguments (coordinator_id, log_manager) for compatibility
        """
        self.enable_communication = enable_communication
        self.adapter_factory = AdapterFactory() if enable_communication else None
        # Store extra args for potential future use
        self.coordinator_id = kwargs.get('coordinator_id')
        self.log_manager = kwargs.get('log_manager')
        
    def create_communication_layer(self, 
                                 config: Dict[str, Any],
                                 containers: Dict[str, Any]) -> 'CommunicationLayer':
        """
        Create a communication layer with adapters.
        
        Args:
            config: Configuration with 'adapters' list
            containers: Map of container name to container instance
            
        Returns:
            CommunicationLayer instance
        """
        if not self.enable_communication:
            return CommunicationLayer(None, {})
            
        # Convert containers to protocol-compliant format if needed
        protocol_containers = self._ensure_protocol_compliance(containers)
        
        # Create adapters using factory
        adapters = self.adapter_factory.create_adapters_from_config(
            config.get('adapters', []),
            protocol_containers
        )
        
        return CommunicationLayer(self.adapter_factory, protocol_containers)
        
    def _ensure_protocol_compliance(self, containers: Dict[str, Any]) -> Dict[str, Container]:
        """
        Ensure containers implement the Container protocol.
        
        This wraps non-compliant containers with a protocol adapter.
        """
        protocol_containers = {}
        
        for name, container in containers.items():
            if hasattr(container, 'name') and hasattr(container, 'event_bus'):
                # Already compliant
                protocol_containers[name] = container
            else:
                # Wrap with adapter
                protocol_containers[name] = ContainerProtocolAdapter(name, container)
                
        return protocol_containers


class CommunicationLayer:
    """
    Compatibility layer representing the communication system.
    
    Provides the interface expected by legacy code.
    """
    
    def __init__(self, adapter_factory: Optional[AdapterFactory], 
                 containers: Dict[str, Container]):
        """Initialize communication layer."""
        self.adapter_factory = adapter_factory
        self.containers = containers
        self.adapters = adapter_factory.active_adapters if adapter_factory else []
        
    async def setup_all_adapters(self) -> None:
        """Setup all adapters (already done by factory)."""
        if self.adapter_factory:
            # Start all adapters
            self.adapter_factory.start_all()
            logger.info(f"Started {len(self.adapters)} communication adapters")
            
    async def cleanup(self) -> None:
        """Clean up all adapters."""
        if self.adapter_factory:
            self.adapter_factory.stop_all()
            logger.info("Stopped all communication adapters")


class ContainerProtocolAdapter:
    """
    Adapter to make legacy containers comply with the Container protocol.
    
    This wraps containers that don't implement our protocol.
    """
    
    def __init__(self, name: str, wrapped_container: Any):
        """Initialize the adapter."""
        self._name = name
        self._wrapped = wrapped_container
        
    @property
    def name(self) -> str:
        """Container name."""
        if hasattr(self._wrapped, 'name'):
            return self._wrapped.name
        elif hasattr(self._wrapped, 'metadata') and hasattr(self._wrapped.metadata, 'name'):
            return self._wrapped.metadata.name
        return self._name
        
    @property
    def event_bus(self):
        """Container's event bus."""
        if hasattr(self._wrapped, 'event_bus'):
            return self._wrapped.event_bus
        elif hasattr(self._wrapped, '_event_bus'):
            return self._wrapped._event_bus
        else:
            # Create a dummy event bus
            return DummyEventBus()
            
    def receive_event(self, event: Event) -> None:
        """Receive events from adapters."""
        if hasattr(self._wrapped, 'receive_event'):
            self._wrapped.receive_event(event)
        elif hasattr(self._wrapped, 'handle_event'):
            self._wrapped.handle_event(event)
        elif hasattr(self._wrapped, 'process_event'):
            self._wrapped.process_event(event)
        else:
            logger.warning(f"Container {self.name} has no event handler method")
            
    def publish_event(self, event: Event) -> None:
        """Publish events to adapters."""
        if hasattr(self._wrapped, 'publish_event'):
            self._wrapped.publish_event(event)
        elif hasattr(self._wrapped, 'emit_event'):
            self._wrapped.emit_event(event)
        elif hasattr(self._wrapped, 'send_event'):
            self._wrapped.send_event(event)
        else:
            # Try through event bus
            if hasattr(self.event_bus, 'publish'):
                self.event_bus.publish(event)
                
    def process(self, event: Event) -> Optional[Event]:
        """Process business logic."""
        if hasattr(self._wrapped, 'process'):
            return self._wrapped.process(event)
        return None


class DummyEventBus:
    """Dummy event bus for containers without one."""
    
    def subscribe(self, event_type, handler):
        """No-op subscribe."""
        pass
        
    def subscribe_all(self, handler):
        """No-op subscribe all."""
        pass
        
    def publish(self, event):
        """No-op publish."""
        pass


# Convenience function for creating pipeline adapter with legacy containers
def create_pipeline_adapter(containers: List[Any], name: str = "main_pipeline") -> Any:
    """
    Create a pipeline adapter from a list of containers.
    
    This is a convenience function for legacy code.
    
    Args:
        containers: List of containers (may not be protocol-compliant)
        name: Pipeline name
        
    Returns:
        Configured pipeline adapter
    """
    from .factory import create_simple_pipeline
    
    # Ensure protocol compliance
    protocol_containers = []
    for i, container in enumerate(containers):
        if hasattr(container, 'name') and hasattr(container, 'event_bus'):
            protocol_containers.append(container)
        else:
            # Determine name
            if hasattr(container, 'name'):
                cont_name = container.name
            elif hasattr(container, 'metadata') and hasattr(container.metadata, 'name'):
                cont_name = container.metadata.name
            else:
                cont_name = f"container_{i}"
                
            protocol_containers.append(ContainerProtocolAdapter(cont_name, container))
    
    return create_simple_pipeline(protocol_containers, name)