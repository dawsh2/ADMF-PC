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
    
    # Use container with routes for cross-container communication
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
    # Simplified naming strategy
    ContainerType,
    ContainerNamingStrategy,
    # Simplified convenience function
    create_container_id
)

# Import ContainerRole and ContainerState from protocols for backward compatibility
from .protocols import ContainerRole, ContainerState

from .protocols import (
    ContainerProtocol,
    ContainerMetadata,
    ContainerLimits
)

# Import factory for convenience
from .factory import (
    ContainerFactory,
    create_container,
    create_portfolio_container,
    create_strategy_container,
    create_data_container
)

# Exceptions
from .exceptions import (
    ContainerError,
    ComponentAlreadyExistsError,
    ComponentNotFoundError,
    ComponentDependencyError,
    InvalidContainerStateError,
    UnknownContainerRoleError,
    InvalidContainerConfigError,
    CircularContainerDependencyError,
    ParentContainerNotSetError
)

# Types
from .types import (
    ContainerConfigDict,
    ComponentConfigDict,
    ExecutionConfigDict,
    MetricsConfigDict,
    ContainerComponent,
    EventHandler
)

# Synchronization - import from unified barriers module
from ..events.barriers import (
    AlignmentMode,
    TimeframeAlignment,
    DataRequirement,
    BarBuffer,
    BarrierProtocol,
    DataAlignmentBarrier,
    OrderStateBarrier,
    TimingBarrier,
    CompositeBarrier,
    create_standard_barriers,
    setup_barriers_from_config
)

# Container-specific utilities
from .types import StrategySpecification
from .factory import (
    setup_simple_container,
    create_symbol_group_requirement,
    create_multi_timeframe_requirement,
    create_pairs_requirement
)

__all__ = [
    # Canonical container
    'Container',
    'ContainerConfig', 
    'ContainerRole',
    'ContainerState',
    
    # Simplified naming strategy
    'ContainerType',
    'ContainerNamingStrategy',
    'create_container_id',
    
    # Protocols
    'ContainerProtocol',
    'ContainerMetadata',
    'ContainerLimits',
    
    # Factory
    'ContainerFactory',
    'ContainerRegistry',
    'get_global_factory',
    'get_global_registry',
    'compose_pattern',
    
    # Exceptions
    'ContainerError',
    'ComponentAlreadyExistsError',
    'ComponentNotFoundError',
    'ComponentDependencyError',
    'InvalidContainerStateError',
    'UnknownContainerRoleError',
    'InvalidContainerConfigError',
    'CircularContainerDependencyError',
    'ParentContainerNotSetError',
    
    # Types
    'ContainerConfigDict',
    'ComponentConfigDict',
    'ExecutionConfigDict',
    'MetricsConfigDict',
    'ContainerComponent',
    'EventHandler',
    
    # Synchronization - unified barriers system
    'AlignmentMode',
    'TimeframeAlignment',
    'DataRequirement',
    'BarBuffer',
    'BarrierProtocol',
    'DataAlignmentBarrier',
    'OrderStateBarrier', 
    'TimingBarrier',
    'CompositeBarrier',
    'create_standard_barriers',
    'setup_barriers_from_config',
    
    # Container-specific utilities
    'StrategySpecification',
    'setup_simple_container',
    'create_symbol_group_requirement',
    'create_multi_timeframe_requirement',
    'create_pairs_requirement'
]