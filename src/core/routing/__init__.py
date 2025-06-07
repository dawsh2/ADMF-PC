"""Communication routes for inter-container event routing.

This package provides protocol-based communication routes that connect
isolated container event buses. Following ADMF-PC's architecture principles:

1. **No Inheritance**: Routes implement protocols, not inherit from base classes
2. **Composition**: Mix and match route behaviors through composition
3. **Type Safety**: Full protocol compliance checking at runtime

Key Components:
- Protocols: Define what routes and containers must implement
- Routes: Route events between containers (pipeline, broadcast, filter)
- Helpers: Utility functions for common route operations
- Factory: Create and manage route instances

Example:
    ```python
    from src.core.routing import RoutingFactory, create_simple_pipeline
    
    # Create routes from configuration
    factory = RoutingFactory()
    route = factory.create_route('main_pipeline', {
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
    CommunicationRoute,
    RouteMetrics,
    RouteErrorHandler,
)

# Composition utilities
from .composition import (
    compose_route_with_infrastructure,
    wrap_with_metrics,
    create_subscription,
    create_forwarding_handler,
    validate_config,
    extract_connections,
    StandardRouteMetrics,
    StandardRouteErrorHandler,
)

# Core route types
from .pipe import PipelineRoute
from .broadcast import BroadcastRoute
from .filter import FilterRoute, create_feature_filter

# Factory and convenience functions
from .factory import (
    RoutingFactory,
    create_route_network,
    create_simple_pipeline,
    create_event_bus,
)

__all__ = [
    # Protocols
    "Container",
    "CommunicationRoute",
    "RouteMetrics", 
    "RouteErrorHandler",
    
    # Composition utilities
    "compose_route_with_infrastructure",
    "wrap_with_metrics",
    "create_subscription",
    "create_forwarding_handler",
    "validate_config",
    "extract_connections",
    "StandardRouteMetrics",
    "StandardRouteErrorHandler",
    
    # Routes
    "PipelineRoute",
    "BroadcastRoute",
    "FilterRoute",
    
    # Route convenience functions
    "create_feature_filter",
    
    # Factory
    "RoutingFactory",
    "create_route_network",
    "create_simple_pipeline",
    "create_event_bus",
]