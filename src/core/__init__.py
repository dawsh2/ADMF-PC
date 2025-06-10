"""
ADMF-PC Core Module

The foundation of the Adaptive Dynamic Multi-Factor Protocol - Polymorphic Composition (ADMF-PC) system.
Provides a sophisticated component-based architecture that enables parallel execution of trading strategies,
backtests, and optimizations with complete state isolation.

Key Features:
- Protocol-based design (no inheritance)
- Container isolation for parallel execution  
- Event-driven communication
- Capability-based component enhancement
- Workflow orchestration
- Type-safe configuration

Architecture Principles:
1. Protocol + Composition over inheritance
2. Container isolation for state separation
3. Event-driven, loosely-coupled communication
4. "Pay for what you use" capability enhancement

Example Usage:
    ```python
    from src.core import Coordinator, WorkflowConfig, WorkflowType
    from src.core.containers import Container, ContainerRole
    from src.core.components import create_component
    
    # Create a container
    container = Container(
        role=ContainerRole.BACKTEST,
        name="my_backtest"
    )
    
    # Add components
    strategy = create_component("MomentumStrategy", capabilities=["events"])
    container.add_component("strategy", strategy)
    
    # Execute workflow
    coordinator = Coordinator()
    config = WorkflowConfig(workflow_type=WorkflowType.BACKTEST)
    result = await coordinator.execute_workflow(config)
    ```
"""

# Core exports - organized by module
from . import components
from . import containers
from . import coordinator
from . import events

# High-level imports for convenience
from .coordinator import Coordinator
from .coordinator.types import WorkflowConfig, WorkflowType, WorkflowPhase

# Container system
from .containers import (
    Container,
    ContainerConfig,
    ContainerRole,
    ContainerState,
    create_container_id
)

# Component system
from .components import (
    Component,
    Lifecycle,
    EventCapable,
    ComponentFactory,
    create_component,
    register_component
)

# Event system
from .events import (
    Event,
    EventType,
    EventBus,
    create_market_event,
    create_signal_event,
    create_system_event
)

# CLI module
from . import cli

__version__ = "1.0.0"

__all__ = [
    # Modules
    "components",
    "containers", 
    "coordinator",
    "events",
    "cli",
    
    # High-level classes
    "Coordinator",
    "WorkflowConfig",
    "WorkflowType", 
    "WorkflowPhase",
    
    # Container system
    "Container",
    "ContainerConfig",
    "ContainerRole",
    "ContainerState",
    "create_container_id",
    
    # Component system
    "Component",
    "Lifecycle",
    "EventCapable",
    "ComponentFactory",
    "create_component",
    "register_component",
    
    # Event system
    "Event",
    "EventType",
    "EventBus",
    "create_market_event",
    "create_signal_event",
    "create_system_event",
    
    # Version
    "__version__",
]