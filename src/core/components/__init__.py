"""
Protocol-based component system for ADMF-PC.

This package provides the infrastructure for discovering, registering,
and creating components based on protocols rather than inheritance.
Components only get the capabilities they need, minimizing overhead.

Key Components:
- Protocols: Define component capabilities
- Registry: Manages component registration and lookup
- Factory: Creates components with automatic capability enhancement
- Discovery: Automatically finds components in the codebase

Example Usage:
    ```python
    # Define a simple component
    class MyStrategy:
        @property
        def component_id(self):
            return "my_strategy"
        
        def generate_signal(self, data):
            return {"action": "buy", "confidence": 0.8}
    
    # Register it
    registry = get_registry()
    registry.register(MyStrategy, tags=["strategy"])
    
    # Create with capabilities
    factory = ComponentFactory()
    strategy = factory.create(
        "MyStrategy",
        context={"event_bus": event_bus},
        capabilities=["events", "lifecycle"]
    )
    
    # Now strategy has event and lifecycle support
    strategy.initialize({})
    strategy.initialize_events()
    ```
"""

from .protocols import (
    # Core protocols
    Component,
    Lifecycle,
    EventCapable,
    Configurable,
    Optimizable,
    Monitorable,
    Stateful,
    
    # Trading protocols
    SignalGenerator,
    DataProvider,
    RiskManager,
    OrderExecutor,
    Portfolio,
    Indicator,
    
    # Capability enumeration
    Capability,
    CAPABILITY_PROTOCOLS,
    
    # Utility functions
    detect_capabilities,
    has_capability,
    require_capability
)

from .factory import (
    ComponentMetadata,
    ComponentRegistry,
    get_registry,
    register_component,
    CapabilityEnhancer,
    LifecycleEnhancer,
    EventEnhancer,
    ComponentFactory,
    create_component,
    create_minimal_component
)


# Add ComponentSpec for backward compatibility
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class ComponentSpec:
    """Specification for creating a component."""
    component_type: str
    component_id: str
    config: Dict[str, Any]
    capabilities: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.metadata is None:
            self.metadata = {}


__all__ = [
    # Protocols
    "Component",
    "Lifecycle",
    "EventCapable",
    "Configurable",
    "Optimizable",
    "Monitorable",
    "Stateful",
    "SignalGenerator",
    "DataProvider",
    "RiskManager",
    "OrderExecutor",
    "Portfolio",
    "Indicator",
    
    # Capabilities
    "Capability",
    "CAPABILITY_PROTOCOLS",
    "detect_capabilities",
    "has_capability",
    "require_capability",
    
    # Registry
    "ComponentMetadata",
    "ComponentRegistry",
    "get_registry",
    "register_component",
    
    # Factory
    "CapabilityEnhancer",
    "LifecycleEnhancer",
    "EventEnhancer",
    "ComponentFactory",
    "create_component",
    "create_minimal_component",
    
    # Specs
    "ComponentSpec"
]