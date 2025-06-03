"""Communication adapters for inter-container event routing.

This package provides protocol-based communication adapters that connect
isolated container event buses. Following ADMF-PC's architecture principles:

1. **No Inheritance**: Adapters implement protocols, not inherit from base classes
2. **Composition**: Mix and match adapter behaviors through composition
3. **Type Safety**: Full protocol compliance checking at runtime

Key Components:
- Protocols: Define what adapters and containers must implement
- Adapters: Route events between containers (pipeline, broadcast, etc.)
- Helpers: Utility functions for common adapter operations
- Factory: Create and manage adapter instances

Example:
    ```python
    from src.core.communication import AdapterFactory, create_simple_pipeline
    
    # Create adapters from configuration
    factory = AdapterFactory()
    adapter = factory.create_adapter('main_pipeline', {
        'type': 'pipeline',
        'containers': ['data', 'strategy', 'risk', 'execution']
    })
    
    # Or use convenience functions
    pipeline = create_simple_pipeline([data_container, strategy_container])
    ```
"""

# Protocol definitions
from .protocols import (
    Container,
    CommunicationAdapter,
    AdapterMetrics,
    AdapterErrorHandler,
)

# Helper functions
from .helpers import (
    create_adapter_with_logging,
    handle_event_with_metrics,
    subscribe_to_container_events,
    validate_adapter_config,
    create_forward_handler,
    get_container_connections,
    SimpleAdapterMetrics,
    SimpleAdapterErrorHandler,
)

# Protocol-based adapters
from .pipeline_adapter_protocol import (
    PipelineAdapter,
    create_conditional_pipeline,
    create_parallel_pipeline,
)

from .broadcast_adapter import (
    BroadcastAdapter,
    create_filtered_broadcast,
    create_priority_broadcast,
    FanOutAdapter,
)

from .hierarchical_adapter import (
    HierarchicalAdapter,
    create_aggregating_hierarchy,
    create_filtered_hierarchy,
)

from .selective_adapter import (
    SelectiveAdapter,
    create_capability_based_router,
    create_load_balanced_router,
    create_content_based_router,
)

# Factory and convenience functions
from .factory import (
    AdapterFactory,
    create_adapter_network,
    create_simple_pipeline,
    create_event_bus,
    create_tree_network,
)

# Integration and compatibility
from .integration import (
    EventCommunicationFactory,
    CommunicationLayer,
    ContainerProtocolAdapter,
    create_pipeline_adapter,
)

__all__ = [
    # Protocols
    "Container",
    "CommunicationAdapter",
    "AdapterMetrics", 
    "AdapterErrorHandler",
    
    # Helpers
    "create_adapter_with_logging",
    "handle_event_with_metrics",
    "subscribe_to_container_events",
    "validate_adapter_config",
    "create_forward_handler",
    "get_container_connections",
    "SimpleAdapterMetrics",
    "SimpleAdapterErrorHandler",
    
    # Adapters
    "PipelineAdapter",
    "BroadcastAdapter",
    "HierarchicalAdapter",
    "SelectiveAdapter",
    "FanOutAdapter",
    
    # Adapter variants
    "create_conditional_pipeline",
    "create_parallel_pipeline",
    "create_filtered_broadcast",
    "create_priority_broadcast",
    "create_aggregating_hierarchy",
    "create_filtered_hierarchy",
    "create_capability_based_router",
    "create_load_balanced_router",
    "create_content_based_router",
    
    # Factory
    "AdapterFactory",
    "create_adapter_network",
    "create_simple_pipeline",
    "create_event_bus",
    "create_tree_network",
    
    # Integration/Compatibility
    "EventCommunicationFactory",
    "CommunicationLayer",
    "ContainerProtocolAdapter",
    "create_pipeline_adapter",
]