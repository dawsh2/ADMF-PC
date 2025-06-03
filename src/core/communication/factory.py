"""
Communication adapter factory using protocol-based design.

This module provides factory functions for creating communication adapters
following ADMF-PC's protocol-based architecture.
"""

from typing import Dict, Any, Type, List, Optional, Callable
import logging

from ..logging.container_logger import ContainerLogger
from ..containers.protocols import Container
from .protocols import CommunicationAdapter
from .helpers import create_adapter_with_logging

# Import protocol-based adapters
from .pipeline_adapter_protocol import (
    PipelineAdapter, 
    create_conditional_pipeline,
    create_parallel_pipeline
)
from .broadcast_adapter import (
    BroadcastAdapter,
    create_filtered_broadcast,
    create_priority_broadcast,
    FanOutAdapter
)
from .hierarchical_adapter import (
    HierarchicalAdapter,
    create_aggregating_hierarchy,
    create_filtered_hierarchy
)
from .selective_adapter import (
    SelectiveAdapter,
    create_capability_based_router,
    create_load_balanced_router,
    create_content_based_router
)


class AdapterFactory:
    """Factory for creating communication adapters.
    
    This factory creates protocol-based adapters without inheritance.
    All adapters follow the CommunicationAdapter protocol but don't
    inherit from any base class.
    """
    
    def __init__(self, logger: Optional[ContainerLogger] = None):
        """Initialize the adapter factory.
        
        Args:
            logger: Logger for factory operations
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Registry of adapter types and their factory functions
        self.adapter_registry: Dict[str, Callable] = {
            # Basic patterns
            'pipeline': lambda n, c: create_adapter_with_logging(PipelineAdapter, n, c),
            'broadcast': lambda n, c: create_adapter_with_logging(BroadcastAdapter, n, c),
            'hierarchical': lambda n, c: create_adapter_with_logging(HierarchicalAdapter, n, c),
            'selective': lambda n, c: create_adapter_with_logging(SelectiveAdapter, n, c),
            
            # Pipeline variants
            'conditional_pipeline': create_conditional_pipeline,
            'parallel_pipeline': create_parallel_pipeline,
            
            # Broadcast variants
            'filtered_broadcast': create_filtered_broadcast,
            'priority_broadcast': create_priority_broadcast,
            'fan_out': lambda n, c: create_adapter_with_logging(FanOutAdapter, n, c),
            
            # Hierarchical variants
            'aggregating_hierarchy': create_aggregating_hierarchy,
            'filtered_hierarchy': create_filtered_hierarchy,
            
            # Selective variants
            'capability_router': create_capability_based_router,
            'load_balanced_router': create_load_balanced_router,
            'content_router': create_content_based_router,
        }
        
        # Active adapter instances
        self.active_adapters: List[Any] = []
        
    def create_adapter(self, name: str, config: Dict[str, Any]) -> Any:
        """Create a communication adapter instance.
        
        Args:
            name: Unique adapter name
            config: Adapter configuration including 'type'
            
        Returns:
            Configured adapter instance
            
        Raises:
            ValueError: If adapter type is unknown
        """
        adapter_type = config.get('type')
        if not adapter_type:
            raise ValueError("Adapter configuration must specify 'type'")
            
        if adapter_type not in self.adapter_registry:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
            
        self.logger.info(f"Creating {adapter_type} adapter: {name}")
        
        # Create adapter using registered factory
        factory_fn = self.adapter_registry[adapter_type]
        adapter = factory_fn(name, config)
        
        # Track active adapter
        self.active_adapters.append(adapter)
        
        return adapter
        
    def register_adapter_type(self, adapter_type: str, factory_fn: Callable):
        """Register a custom adapter type.
        
        Args:
            adapter_type: Type identifier for the adapter
            factory_fn: Factory function that creates adapter instances
        """
        self.adapter_registry[adapter_type] = factory_fn
        self.logger.info(f"Registered adapter type: {adapter_type}")
        
    def create_adapters_from_config(self, 
                                   adapters_config: List[Dict[str, Any]],
                                   containers: Dict[str, Container]) -> List[Any]:
        """Create multiple adapters from configuration.
        
        Args:
            adapters_config: List of adapter configurations
            containers: Available containers
            
        Returns:
            List of configured adapter instances
        """
        adapters = []
        
        for adapter_config in adapters_config:
            name = adapter_config.get('name', f"adapter_{len(adapters)}")
            
            try:
                # Create adapter
                adapter = self.create_adapter(name, adapter_config)
                
                # Setup with containers
                if hasattr(adapter, 'setup'):
                    adapter.setup(containers)
                    
                adapters.append(adapter)
                
            except Exception as e:
                self.logger.error(f"Failed to create adapter {name}: {e}")
                raise
                
        return adapters
        
    def start_all(self) -> None:
        """Start all active adapters."""
        for adapter in self.active_adapters:
            if hasattr(adapter, 'start'):
                # Skip pipeline adapters that have no connections configured
                # They will be started later by the workflow manager after configuration
                if (hasattr(adapter, 'connections') and 
                    hasattr(adapter, 'config') and 
                    adapter.config.get('type') == 'pipeline' and 
                    not adapter.connections):
                    self.logger.info(f"Skipping start of pipeline adapter '{adapter.name}' - will be started after configuration")
                    continue
                    
                adapter.start()
                self.logger.info(f"Started adapter: {adapter.name}")
                
    def stop_all(self) -> None:
        """Stop all active adapters."""
        for adapter in self.active_adapters:
            if hasattr(adapter, 'stop'):
                adapter.stop()
                self.logger.info(f"Stopped adapter: {adapter.name}")
                
        self.active_adapters.clear()


def create_adapter_network(config: Dict[str, Any],
                          containers: Dict[str, Container],
                          logger: Optional[ContainerLogger] = None) -> AdapterFactory:
    """Create a complete adapter network from configuration.
    
    This is a convenience function that creates all adapters and
    wires them up according to the configuration.
    
    Args:
        config: Network configuration with 'adapters' list
        containers: Available containers
        logger: Optional logger
        
    Returns:
        Configured AdapterFactory with all adapters created
    """
    factory = AdapterFactory(logger)
    
    # Create adapters
    adapters_config = config.get('adapters', [])
    adapters = factory.create_adapters_from_config(adapters_config, containers)
    
    # Start all adapters
    factory.start_all()
    
    return factory


# Convenience functions for common patterns

def create_simple_pipeline(containers: List[Container], 
                          name: str = "main_pipeline") -> Any:
    """Create a simple pipeline adapter.
    
    Args:
        containers: List of containers in pipeline order
        name: Pipeline name
        
    Returns:
        Configured pipeline adapter
    """
    config = {
        'type': 'pipeline',
        'containers': [c.name for c in containers]
    }
    
    adapter = create_adapter_with_logging(PipelineAdapter, name, config)
    
    # Setup with container mapping
    container_map = {c.name: c for c in containers}
    adapter.setup(container_map)
    
    return adapter


def create_event_bus(source: Container,
                    targets: List[Container],
                    name: str = "event_bus") -> Any:
    """Create a broadcast adapter acting as an event bus.
    
    Args:
        source: Source container
        targets: Target containers
        name: Bus name
        
    Returns:
        Configured broadcast adapter
    """
    config = {
        'type': 'broadcast',
        'source': source.name,
        'targets': [t.name for t in targets]
    }
    
    adapter = create_adapter_with_logging(BroadcastAdapter, name, config)
    
    # Setup with container mapping
    all_containers = [source] + targets
    container_map = {c.name: c for c in all_containers}
    adapter.setup(container_map)
    
    return adapter


def create_tree_network(root: Container,
                       tree_structure: Dict[str, Any],
                       name: str = "tree_network") -> Any:
    """Create a hierarchical tree network.
    
    Args:
        root: Root container
        tree_structure: Tree configuration
        name: Network name
        
    Returns:
        Configured hierarchical adapter
    """
    config = {
        'type': 'hierarchical',
        'root': root.name,
        'hierarchy': tree_structure
    }
    
    adapter = create_adapter_with_logging(HierarchicalAdapter, name, config)
    
    # Note: setup requires all containers to be provided
    # This is a simplified version - real usage needs full container map
    
    return adapter