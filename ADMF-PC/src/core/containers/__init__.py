"""
Container system for ADMF-PC.

This package provides the containerized execution infrastructure that
ensures complete state isolation between parallel executions while
allowing controlled sharing of read-only services.

Key Components:
- UniversalScopedContainer: Core container with state isolation
- ContainerLifecycleManager: Manages multiple container lifecycles
- ContainerFactory: Creates specialized containers

Example Usage:
    ```python
    # Create a factory
    factory = ContainerFactory()
    
    # Create a backtest container
    container_id = factory.create_backtest_container(
        strategy_spec={
            'class': 'TrendFollowingStrategy',
            'parameters': {'fast_period': 10, 'slow_period': 30}
        },
        shared_services={
            'DataProvider': historical_data,
            'IndicatorHub': shared_indicators
        }
    )
    
    # Get the container
    container = factory.get_container(container_id)
    
    # Start the container
    container.start()
    
    # Run backtest...
    
    # Cleanup
    container.stop()
    factory.dispose_container(container_id)
    ```
"""

from .universal import (
    UniversalScopedContainer,
    ContainerState,
    ContainerType,
    ComponentSpec,
    ContainerMetadata,
    create_backtest_container
)

from .lifecycle import (
    ContainerLifecycleManager,
    LifecycleEvent,
    ContainerInfo,
    get_lifecycle_manager
)

from .factory import (
    ContainerFactory
)


__all__ = [
    # Universal container
    "UniversalScopedContainer",
    "ContainerState",
    "ContainerType",
    "ComponentSpec",
    "ContainerMetadata",
    "create_backtest_container",
    
    # Lifecycle management
    "ContainerLifecycleManager",
    "LifecycleEvent",
    "ContainerInfo",
    "get_lifecycle_manager",
    
    # Factory
    "ContainerFactory"
]