"""
Composable Container Protocol Interface

This module defines the core protocols that enable coordinator-driven 
container composition with full flexibility for arrangement patterns.
"""

from abc import abstractmethod
from typing import Protocol, Dict, List, Any, Optional, Set, Union, TypeVar, Generic
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..events.types import Event


ContainerType = TypeVar('ContainerType', bound='ComposableContainerProtocol')


class ContainerRole(Enum):
    """Standard container roles in the system."""
    BACKTEST = "backtest"  # Root backtest container for peer containers
    DATA = "data"
    INDICATOR = "indicator" 
    CLASSIFIER = "classifier"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    SIGNAL_LOG = "signal_log"
    ENSEMBLE = "ensemble"


class ContainerState(Enum):
    """Container lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ContainerMetadata:
    """Metadata for container identification and management."""
    container_id: str
    role: ContainerRole
    name: str
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class ContainerLimits:
    """Resource limits for container isolation."""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_execution_time_minutes: Optional[int] = None
    max_child_containers: Optional[int] = None
    max_events_per_second: Optional[int] = None


class ComposableContainerProtocol(Protocol):
    """
    Core protocol for composable containers.
    
    All containers must implement this interface to participate 
    in coordinator-driven composition.
    """
    
    @property
    @abstractmethod
    def metadata(self) -> ContainerMetadata:
        """Container identification and metadata."""
        ...
    
    @property
    @abstractmethod
    def state(self) -> ContainerState:
        """Current container state."""
        ...
    
    @property
    @abstractmethod
    def event_bus(self):
        """Container-scoped event bus."""
        ...
    
    @property
    @abstractmethod
    def parent_container(self) -> Optional['ComposableContainerProtocol']:
        """Parent container if nested."""
        ...
    
    @property 
    @abstractmethod
    def child_containers(self) -> List['ComposableContainerProtocol']:
        """Child containers nested within this container."""
        ...
    
    # Lifecycle Management
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize container and all child containers."""
        ...
    
    @abstractmethod
    async def start(self) -> None:
        """Start container processing."""
        ...
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop container processing gracefully."""
        ...
    
    @abstractmethod
    async def dispose(self) -> None:
        """Clean up container resources."""
        ...
    
    # Composition Management
    @abstractmethod
    def add_child_container(self, child: 'ComposableContainerProtocol') -> None:
        """Add a child container."""
        ...
    
    @abstractmethod
    def remove_child_container(self, container_id: str) -> bool:
        """Remove a child container by ID."""
        ...
    
    @abstractmethod
    def get_child_container(self, container_id: str) -> Optional['ComposableContainerProtocol']:
        """Get child container by ID."""
        ...
    
    @abstractmethod
    def find_containers_by_role(self, role: ContainerRole) -> List['ComposableContainerProtocol']:
        """Find all nested containers with specific role."""
        ...
    
    # Event Processing  
    @abstractmethod
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process incoming event and optionally return response."""
        ...
    
    @abstractmethod
    def publish_event(self, event: Event, target_scope: str = "local") -> None:
        """Publish event to specified scope (local, parent, children, broadcast)."""
        ...
    
    # Configuration and Status
    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update container configuration."""
        ...
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current container status and metrics."""
        ...
    
    @abstractmethod
    def get_capabilities(self) -> Set[str]:
        """Get container capabilities/features."""
        ...


class ContainerCompositionProtocol(Protocol):
    """Protocol for composing containers according to patterns."""
    
    @abstractmethod
    def create_container(
        self,
        role: ContainerRole,
        config: Dict[str, Any],
        container_id: Optional[str] = None
    ) -> ComposableContainerProtocol:
        """Create a container of specified role."""
        ...
    
    @abstractmethod
    def compose_pattern(
        self,
        pattern: Dict[str, Any],
        base_config: Dict[str, Any] = None
    ) -> ComposableContainerProtocol:
        """Compose containers according to pattern specification."""
        ...
    
    @abstractmethod
    def validate_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Validate that pattern is valid and composable."""
        ...


@dataclass
class ContainerPattern:
    """Definition of a container arrangement pattern."""
    name: str
    description: str
    structure: Dict[str, Any]
    required_capabilities: Set[str] = field(default_factory=set)
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any]) -> 'ContainerPattern':
        """Create pattern from YAML configuration."""
        return cls(
            name=yaml_data['name'],
            description=yaml_data.get('description', ''),
            structure=yaml_data['structure'],
            required_capabilities=set(yaml_data.get('required_capabilities', [])),
            default_config=yaml_data.get('default_config', {})
        )


class ContainerRegistryProtocol(Protocol):
    """Protocol for container type registry."""
    
    @abstractmethod
    def register_container_type(
        self,
        role: ContainerRole,
        factory_func: callable,
        capabilities: Set[str] = None
    ) -> None:
        """Register a container factory for a role."""
        ...
    
    @abstractmethod
    def get_container_factory(self, role: ContainerRole) -> Optional[callable]:
        """Get factory function for container role."""
        ...
    
    @abstractmethod
    def register_pattern(self, pattern: ContainerPattern) -> None:
        """Register a composition pattern."""
        ...
    
    @abstractmethod
    def get_pattern(self, pattern_name: str) -> Optional[ContainerPattern]:
        """Get pattern by name."""
        ...
    
    @abstractmethod
    def list_available_patterns(self) -> List[str]:
        """List all available pattern names."""
        ...


class DataFlowProtocol(Protocol):
    """Protocol for managing data flow between containers."""
    
    @abstractmethod
    def subscribe_to_data(
        self,
        container_id: str,
        data_types: Set[str],
        callback: callable
    ) -> None:
        """Subscribe container to specific data types."""
        ...
    
    @abstractmethod
    def publish_data(
        self,
        container_id: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> None:
        """Publish data from container."""
        ...
    
    @abstractmethod
    def get_data_dependencies(self, container_id: str) -> Set[str]:
        """Get data types this container depends on."""
        ...


# Base Implementation Helper  
class BaseComposableContainer:
    """
    Base implementation providing common container functionality.
    
    Containers can inherit from this or implement the protocol directly.
    """
    
    def __init__(
        self,
        role: ContainerRole,
        name: str,
        config: Dict[str, Any] = None,
        container_id: str = None,
        limits: ContainerLimits = None
    ):
        # Generate container ID
        container_id_final = container_id or str(uuid.uuid4())
        
        self._metadata = ContainerMetadata(
            container_id=container_id_final,
            role=role,
            name=name,
            config=config or {}
        )
        self._state = ContainerState.UNINITIALIZED
        # Create event bus
        from ..events.event_bus import EventBus
        self._event_bus = EventBus()
        self._parent_container: Optional[ComposableContainerProtocol] = None
        self._child_containers: List[ComposableContainerProtocol] = []
        self._limits = limits or ContainerLimits()
        
        # Resource tracking
        self._metrics = {
            'events_processed': 0,
            'events_published': 0,
            'start_time': None,
            'last_activity': None
        }
        
        # Configure external communication if specified in config
        if config and ('events' in config or 'external_events' in config):
            self.configure_external_communication(config)
    
    @property
    def metadata(self) -> ContainerMetadata:
        return self._metadata
    
    @property
    def state(self) -> ContainerState:
        return self._state
    
    @property
    def event_bus(self):
        """Return the container's event bus."""
        return self._event_bus
    
    @property
    def name(self) -> str:
        """Container name - required by Container protocol."""
        return self._metadata.name
        
    @property
    def parent_container(self) -> Optional[ComposableContainerProtocol]:
        return self._parent_container
    
    @property
    def child_containers(self) -> List[ComposableContainerProtocol]:
        return self._child_containers.copy()
    
    def add_child_container(self, child: ComposableContainerProtocol) -> None:
        """Add child container and set parent relationship."""
        if len(self._child_containers) >= (self._limits.max_child_containers or float('inf')):
            raise ValueError(f"Maximum child containers ({self._limits.max_child_containers}) exceeded")
        
        self._child_containers.append(child)
        if hasattr(child, '_parent_container'):
            child._parent_container = self
        
        # Update parent ID in metadata
        if hasattr(child, '_metadata'):
            child._metadata.parent_id = self._metadata.container_id
        
        # Communication setup will be handled by adapters
    
    def remove_child_container(self, container_id: str) -> bool:
        """Remove child container by ID."""
        for i, child in enumerate(self._child_containers):
            if child.metadata.container_id == container_id:
                # Clean up parent relationship
                if hasattr(child, '_parent_container'):
                    child._parent_container = None
                if hasattr(child, '_metadata'):
                    child._metadata.parent_id = None
                
                self._child_containers.pop(i)
                return True
        return False
    
    def get_child_container(self, container_id: str) -> Optional[ComposableContainerProtocol]:
        """Get child container by ID."""
        for child in self._child_containers:
            if child.metadata.container_id == container_id:
                return child
        return None
    
    def find_containers_by_role(self, role: ContainerRole) -> List[ComposableContainerProtocol]:
        """Find all nested containers with specific role."""
        results = []
        
        # Check direct children
        for child in self._child_containers:
            if child.metadata.role == role:
                results.append(child)
            # Recursively search children
            results.extend(child.find_containers_by_role(role))
        
        return results
    
    def receive_event(self, event: Event) -> None:
        """Receive event from adapters - required by Container protocol."""
        # Publish to local event bus for processing
        self._event_bus.publish(event)
        self._metrics['events_received'] = self._metrics.get('events_received', 0) + 1
        
    def process(self, event: Event) -> Optional[Event]:
        """Process business logic - required by Container protocol."""
        # Default implementation - containers can override
        return None
        
    def publish_event(self, event: Event, target_scope: str = "local") -> None:
        """Publish event to specified scope."""
        self._metrics['events_published'] += 1
        self._metrics['last_activity'] = datetime.now()
        
        if target_scope == "local":
            self._event_bus.publish(event)
        elif target_scope == "parent" and self._parent_container:
            self._parent_container.event_bus.publish(event)
        elif target_scope == "children":
            for child in self._child_containers:
                child.event_bus.publish(event)
        elif target_scope == "broadcast":
            # Publish to all levels
            self._event_bus.publish(event)
            if self._parent_container:
                self._parent_container.event_bus.publish(event)
            for child in self._child_containers:
                child.event_bus.publish(event)
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update container configuration."""
        self._metadata.config.update(config)
    
    def get_status(self) -> Dict[str, Any]:
        """Get container status and metrics."""
        return {
            'metadata': {
                'container_id': self._metadata.container_id,
                'role': self._metadata.role.value,
                'name': self._metadata.name,
                'parent_id': self._metadata.parent_id,
                'created_at': self._metadata.created_at.isoformat(),
                'tags': list(self._metadata.tags)
            },
            'state': self._state.value,
            'metrics': self._metrics.copy(),
            'child_containers': len(self._child_containers),
            'limits': {
                'max_memory_mb': self._limits.max_memory_mb,
                'max_cpu_percent': self._limits.max_cpu_percent,
                'max_execution_time_minutes': self._limits.max_execution_time_minutes,
                'max_child_containers': self._limits.max_child_containers,
                'max_events_per_second': self._limits.max_events_per_second
            }
        }
    
    async def initialize(self) -> None:
        """Initialize container and all children."""
        if self._state != ContainerState.UNINITIALIZED:
            return
        
        self._state = ContainerState.INITIALIZING
        
        try:
            # Initialize self
            await self._initialize_self()
            
            # Subscribe to events on own event bus
            self._setup_event_subscriptions()
            
            # Initialize all children
            for child in self._child_containers:
                await child.initialize()
            
            self._state = ContainerState.INITIALIZED
        except Exception as e:
            self._state = ContainerState.ERROR
            raise e
    
    async def start(self) -> None:
        """Start container processing."""
        if self._state != ContainerState.INITIALIZED:
            await self.initialize()
        
        self._state = ContainerState.RUNNING
        self._metrics['start_time'] = datetime.now()
        
        # Start all children
        for child in self._child_containers:
            await child.start()
    
    async def stop(self) -> None:
        """Stop container processing."""
        if self._state not in [ContainerState.RUNNING, ContainerState.PAUSED]:
            return
        
        self._state = ContainerState.STOPPING
        
        # Stop all children first
        for child in self._child_containers:
            await child.stop()
        
        # Stop self
        await self._stop_self()
        
        self._state = ContainerState.STOPPED
    
    async def dispose(self) -> None:
        """Dispose container and all children."""
        # Stop if running
        if self._state == ContainerState.RUNNING:
            await self.stop()
        
        # Dispose all children
        for child in self._child_containers:
            await child.dispose()
        
        # Clear children
        self._child_containers.clear()
        
        # Dispose self
        await self._dispose_self()
    
    def _setup_event_subscriptions(self) -> None:
        """Subscribe container's process_event method to its event bus."""
        # Subscribe to all event types by default - containers can override this
        # to be more selective about which events they process
        from ..events.types import EventType
        
        # Create async wrapper for process_event
        async def event_handler(event: Event) -> None:
            try:
                await self.process_event(event)
            except Exception as e:
                # Log error but don't break event processing
                import logging
                logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
                logger.error(f"Error processing event {event.event_type}: {e}", exc_info=True)
        
        # Subscribe to all event types
        for event_type in EventType:
            self._event_bus.subscribe(event_type, event_handler)
    
    # Abstract methods to be implemented by specific containers
    async def _initialize_self(self) -> None:
        """Initialize this specific container."""
        pass
    
    async def _stop_self(self) -> None:
        """Stop this specific container."""
        pass
    
    async def _dispose_self(self) -> None:
        """Dispose this specific container."""
        pass
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Default event processing - override in specific containers."""
        self._metrics['events_processed'] += 1
        self._metrics['last_activity'] = datetime.now()
        return None
    
    def get_capabilities(self) -> Set[str]:
        """Default capabilities - override in specific containers."""
        return {f"container.{self._metadata.role.value}"}