"""
Container system for ADMF-PC.

This package provides the containerized execution infrastructure that
ensures complete state isolation between parallel executions.

Key Components:
- Container: THE canonical container implementation (Protocol + Composition)
- ContainerConfig: Configuration for containers
- ContainerRole: Standard container roles

Example Usage:
    ```python
    from core.containers import Container, ContainerConfig, ContainerRole
    
    # Create a container
    config = ContainerConfig(
        role=ContainerRole.BACKTEST,
        name="my_backtest",
        capabilities={'backtest.execution'}
    )
    container = Container(config)
    
    # Add components
    container.add_component("strategy", MyStrategy())
    container.add_component("data", DataLoader())
    
    # Wire dependencies manually
    container.wire_dependencies("strategy", {"data_source": "data"})
    
    # Use container with adapters for cross-container communication
    await container.initialize()
    await container.start()
    
    # Run backtest...
    
    # Cleanup
    await container.stop()
    ```
"""

from .container import (
    Container,
    ContainerConfig,
    ContainerRole,
    ContainerState,
    # Naming strategy
    ContainerType,
    Phase,
    ClassifierType,
    RiskProfile,
    ContainerNamingStrategy,
    # Convenience functions
    create_backtest_container_id,
    create_optimization_container_id,
    create_signal_analysis_container_id
)

from .protocols import (
    Container as ContainerProtocol,
    ContainerMetadata,
    ContainerLimits
)

__all__ = [
    # Canonical container
    'Container',
    'ContainerConfig', 
    'ContainerRole',
    'ContainerState',
    
    # Naming strategy
    'ContainerType',
    'Phase',
    'ClassifierType',
    'RiskProfile',
    'ContainerNamingStrategy',
    'create_backtest_container_id',
    'create_optimization_container_id',
    'create_signal_analysis_container_id',
    
    # Protocols
    'ContainerProtocol',
    'ContainerMetadata',
    'ContainerLimits'
]