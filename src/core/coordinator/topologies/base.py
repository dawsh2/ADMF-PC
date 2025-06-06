"""
Base topology types and utilities
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TopologyDefinition:
    """Standard topology definition structure."""
    containers: Dict[str, Any]
    stateless_components: Dict[str, Any]
    parameter_combinations: List[Dict[str, Any]]
    root_event_bus: Any
    adapters: List[Any] = None
    
    def __post_init__(self):
        if self.adapters is None:
            self.adapters = []
    
    def add_container(self, name: str, container: Any):
        """Add a container to the topology."""
        self.containers[name] = container
        logger.debug(f"Added container '{name}' to topology")
    
    def add_adapter(self, adapter: Any):
        """Add an adapter to the topology."""
        self.adapters.append(adapter)
        logger.debug(f"Added adapter '{adapter.name if hasattr(adapter, 'name') else adapter}' to topology")
    
    def get_containers_by_type(self, container_type: str) -> Dict[str, Any]:
        """Get all containers of a specific type."""
        return {
            name: container 
            for name, container in self.containers.items()
            if container_type in name
        }


@dataclass
class TopologyConfig:
    """Configuration for building a topology."""
    workflow_config: Any
    execution_context: Any
    tracing_enabled: bool = True
    event_tracer: Optional[Any] = None
    
    
def create_traced_event_bus(bus_name: str, event_tracer: Optional[Any] = None) -> Any:
    """Create an event bus with optional tracing."""
    if event_tracer:
        from ...events.tracing import TracedEventBus
        bus = TracedEventBus(bus_name)
        bus.set_tracer(event_tracer)
        logger.info(f"🔍 Created TracedEventBus '{bus_name}' with tracing enabled")
        return bus
    else:
        from ...events import EventBus
        bus = EventBus(bus_name)
        logger.info(f"Created EventBus '{bus_name}' without tracing")
        return bus