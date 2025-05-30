"""
Universal scoped container implementation for ADMF-PC.

This module provides the core container system that ensures complete
state isolation between parallel executions (backtests, optimization
trials, etc.) while allowing shared read-only services.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Set, Type, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime
import uuid

from ..dependencies import DependencyContainer, ScopedContainer as BaseScopedContainer
from ..components import ComponentFactory, detect_capabilities, Capability
from ..events import EventBus, get_isolation_manager


logger = logging.getLogger(__name__)


class ContainerState(Enum):
    """Container lifecycle states."""
    CREATED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    DISPOSED = auto()
    FAILED = auto()


@dataclass
class ComponentSpec:
    """Specification for creating a component."""
    name: str
    class_name: Union[str, Type[Any]]
    params: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerMetadata:
    """Metadata about a container."""
    container_id: str
    container_type: str
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalScopedContainer:
    """
    Universal container providing complete state isolation.
    
    This container ensures that each execution context (backtest,
    optimization trial, etc.) has its own isolated component instances
    while allowing controlled sharing of read-only services.
    """
    
    def __init__(
        self,
        container_id: Optional[str] = None,
        container_type: str = "generic",
        shared_services: Optional[Dict[str, Any]] = None,
        parent_container: Optional['UniversalScopedContainer'] = None
    ):
        """
        Initialize the universal scoped container.
        
        Args:
            container_id: Unique identifier (auto-generated if None)
            container_type: Type of container (backtest, optimization, etc.)
            shared_services: Read-only services to share
            parent_container: Parent container for hierarchical scoping
        """
        self.container_id = container_id or f"{container_type}_{uuid.uuid4().hex[:8]}"
        self.container_type = container_type
        self.parent = parent_container
        
        # Metadata
        self.metadata = ContainerMetadata(
            container_id=self.container_id,
            container_type=container_type,
            parent_id=parent_container.container_id if parent_container else None
        )
        
        # State management
        self._state = ContainerState.CREATED
        self._state_history: List[tuple[ContainerState, datetime]] = [
            (ContainerState.CREATED, datetime.now())
        ]
        
        # Core infrastructure
        self._dependency_container = BaseScopedContainer(self.container_id)
        self._component_factory = ComponentFactory()
        self._event_isolation = get_isolation_manager()
        
        # Create isolated event bus
        self._event_bus = self._event_isolation.create_container_bus(self.container_id)
        self._dependency_container.register_instance("EventBus", self._event_bus)
        
        # Component tracking
        self._component_specs: Dict[str, ComponentSpec] = {}
        self._component_order: List[str] = []
        self._initialized_components: Set[str] = set()
        
        # Shared services management
        self._shared_services = shared_services or {}
        self._register_shared_services()
        
        logger.info(f"Created UniversalScopedContainer: {self.container_id}")
    
    def create_component(
        self,
        spec: Union[ComponentSpec, Dict[str, Any]],
        initialize: bool = False
    ) -> Any:
        """
        Create a component from specification.
        
        Args:
            spec: Component specification or dict
            initialize: Whether to initialize immediately
            
        Returns:
            The created component instance
        """
        if isinstance(spec, dict):
            spec = ComponentSpec(**spec)
        
        if spec.name in self._component_specs:
            raise ValueError(f"Component '{spec.name}' already exists")
        
        # Store specification
        self._component_specs[spec.name] = spec
        self._component_order.append(spec.name)
        
        # Register in dependency container
        if isinstance(spec.class_name, type):
            # Direct type reference
            metadata: Dict[str, Any] = {
                'singleton': True,
                'params': dict(spec.params)
            }
            self._dependency_container.register_type(
                spec.name,
                spec.class_name,
                dependencies=spec.dependencies,
                metadata=metadata
            )
        else:
            # String reference - will be resolved later
            # For now, just track the spec
            pass
        
        if initialize:
            return self.initialize_component(spec.name)
        
        return None
    
    def initialize_component(self, name: str) -> Any:
        """
        Initialize a specific component.
        
        Args:
            name: Component name to initialize
            
        Returns:
            The initialized component
        """
        if name in self._initialized_components:
            return self._dependency_container.resolve(name)
        
        spec = self._component_specs.get(name)
        if not spec:
            raise ValueError(f"Component '{name}' not found")
        
        # Register if not already registered (for string class names)
        if not self._dependency_container.has(name):
            # Resolve string class name if needed
            if isinstance(spec.class_name, str):
                # Try to get from registry
                from ..components import get_registry
                registry = get_registry()
                component_class = registry.get_class(spec.class_name)
                if not component_class:
                    raise ValueError(f"Component class '{spec.class_name}' not found in registry")
            else:
                component_class = spec.class_name
            
            self._dependency_container.register_type(
                spec.name,
                component_class,
                dependencies=spec.dependencies,
                metadata={'params': spec.params}  # Store params in metadata
            )
        
        # Create context for component
        context: Dict[str, Any] = {
            'component_id': spec.name,
            'config': spec.config,
            'dependencies': spec.dependencies
        }
        # Add shared services to context
        for key, value in self._shared_services.items():
            context[key] = value
        
        # Resolve through dependency container (which will use factory)
        component = self._dependency_container.resolve(name)
        
        # Apply configuration if configurable
        if spec.config and 'configurable' in detect_capabilities(component):
            component.configure(spec.config)
        
        self._initialized_components.add(name)
        logger.debug(f"Initialized component '{name}' in container {self.container_id}")
        
        return component
    
    def initialize_scope(self) -> None:
        """Initialize all components in dependency order."""
        self._transition_state(ContainerState.INITIALIZING)
        
        try:
            # Get initialization order from dependency graph
            order = self._dependency_container.get_dependency_graph().get_initialization_order()
            
            # Initialize components in order
            for name in order:
                if name in self._component_specs and name not in self._initialized_components:
                    self.initialize_component(name)
            
            self._transition_state(ContainerState.INITIALIZED)
            logger.info(f"Container {self.container_id} initialized successfully")
            
        except Exception as e:
            self._transition_state(ContainerState.FAILED)
            logger.error(f"Container {self.container_id} initialization failed: {e}")
            raise
    
    def start(self) -> None:
        """Start all components that have lifecycle capability."""
        if self._state != ContainerState.INITIALIZED:
            raise RuntimeError(f"Cannot start container in state {self._state}")
        
        self._transition_state(ContainerState.STARTING)
        self.metadata.started_at = datetime.now()
        
        try:
            # Start components in initialization order
            order = self._dependency_container.get_dependency_graph().get_initialization_order()
            
            for name in order:
                if name in self._initialized_components:
                    component = self._dependency_container.resolve(name)
                    if hasattr(component, 'start'):
                        component.start()
                        logger.debug(f"Started component '{name}'")
            
            self._transition_state(ContainerState.RUNNING)
            logger.info(f"Container {self.container_id} started successfully")
            
        except Exception as e:
            self._transition_state(ContainerState.FAILED)
            logger.error(f"Container {self.container_id} start failed: {e}")
            raise
    
    def stop(self) -> None:
        """Stop all components in reverse dependency order."""
        if self._state not in (ContainerState.RUNNING, ContainerState.FAILED):
            logger.warning(f"Attempting to stop container in state {self._state}")
            return
        
        self._transition_state(ContainerState.STOPPING)
        self.metadata.stopped_at = datetime.now()
        
        # Get teardown order (reverse of initialization)
        order = self._dependency_container.get_dependency_graph().get_teardown_order()
        
        for name in order:
            if name in self._initialized_components:
                try:
                    component = self._dependency_container.resolve(name)
                    if hasattr(component, 'stop'):
                        component.stop()
                        logger.debug(f"Stopped component '{name}'")
                except Exception as e:
                    logger.error(f"Error stopping component '{name}': {e}")
        
        self._transition_state(ContainerState.STOPPED)
        logger.info(f"Container {self.container_id} stopped")
    
    def reset(self) -> None:
        """Reset all components that support reset."""
        if self._state != ContainerState.STOPPED:
            raise RuntimeError(f"Cannot reset container in state {self._state}")
        
        # Reset in initialization order
        order = self._dependency_container.get_dependency_graph().get_initialization_order()
        
        for name in order:
            if name in self._initialized_components:
                component = self._dependency_container.resolve(name)
                if hasattr(component, 'reset'):
                    component.reset()
                    logger.debug(f"Reset component '{name}'")
        
        # Reset container state
        self._dependency_container.reset()
        self._transition_state(ContainerState.INITIALIZED)
        logger.info(f"Container {self.container_id} reset")
    
    def dispose(self) -> None:
        """Dispose of the container and all its resources."""
        if self._state == ContainerState.DISPOSED:
            return
        
        # Stop if running
        if self._state == ContainerState.RUNNING:
            self.stop()
        
        # Teardown components
        self._dependency_container.teardown()
        
        # Remove event bus
        self._event_isolation.remove_container_bus(self.container_id)
        
        self._transition_state(ContainerState.DISPOSED)
        logger.info(f"Container {self.container_id} disposed")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        if name in self._initialized_components:
            return self._dependency_container.resolve(name)
        return None
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all initialized components."""
        return {
            name: self._dependency_container.resolve(name)
            for name in self._initialized_components
        }
    
    @property
    def state(self) -> ContainerState:
        """Current container state."""
        return self._state
    
    @property
    def event_bus(self) -> EventBus:
        """Container's isolated event bus."""
        return self._event_bus
    
    def get_stats(self) -> Dict[str, Any]:
        """Get container statistics."""
        stats: Dict[str, Any] = {
            'container_id': self.container_id,
            'container_type': self.container_type,
            'state': self._state.name,
            'created_at': self.metadata.created_at.isoformat(),
            'component_count': len(self._component_specs),
            'initialized_count': len(self._initialized_components),
            'event_stats': self._event_bus.get_stats()
        }
        
        if self.metadata.started_at:
            stats['started_at'] = self.metadata.started_at.isoformat()
        if self.metadata.stopped_at:
            stats['stopped_at'] = self.metadata.stopped_at.isoformat()
            
        return stats
    
    # Private methods
    
    def _register_shared_services(self) -> None:
        """Register shared services with the container."""
        for name, service in self._shared_services.items():
            # Register instance first
            self._dependency_container.register_instance(name, service)
            # Then mark as shared (only works with parent container)
            if self._dependency_container.parent:
                self._dependency_container.register_shared(name)
            logger.debug(f"Registered shared service '{name}'")
    
    def _transition_state(self, new_state: ContainerState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._state_history.append((new_state, datetime.now()))
        
        logger.debug(
            f"Container {self.container_id} state transition: "
            f"{old_state.name} -> {new_state.name}"
        )


class ContainerType(Enum):
    """Standard container types."""
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    LIVE_TRADING = "live_trading"
    ANALYSIS = "analysis"
    INDICATOR = "indicator"
    DATA = "data"


def create_backtest_container(
    strategy_spec: Dict[str, Any],
    shared_services: Optional[Dict[str, Any]] = None,
    container_id: Optional[str] = None
) -> UniversalScopedContainer:
    """
    Create a container configured for backtesting.
    
    Args:
        strategy_spec: Strategy specification
        shared_services: Services to share
        container_id: Optional container ID
        
    Returns:
        Configured backtest container
    """
    container = UniversalScopedContainer(
        container_id=container_id,
        container_type=ContainerType.BACKTEST.value,
        shared_services=shared_services
    )
    
    # Standard backtest components
    container.create_component({
        'name': 'Portfolio',
        'class_name': 'Portfolio',
        'params': {'initial_cash': 100000},
        'capabilities': ['lifecycle', 'events', 'reset']
    })
    
    container.create_component({
        'name': 'RiskManager',
        'class_name': 'RiskManager',
        'dependencies': ['Portfolio'],
        'capabilities': ['lifecycle', 'events']
    })
    
    # Create strategy
    container.create_component({
        'name': 'Strategy',
        'class_name': strategy_spec['class'],
        'params': strategy_spec.get('parameters', {}),
        'dependencies': ['Portfolio', 'RiskManager'],
        'capabilities': ['lifecycle', 'events', 'optimization'],
        'config': strategy_spec.get('config', {})
    })
    
    return container