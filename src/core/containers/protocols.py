"""
Container protocols for ADMF-PC architecture.

Defines the unified interfaces that all containers must implement.
This is the single source of truth for container protocols.
"""

from typing import Protocol, Dict, Any, Optional, List, Callable, Set
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..types.events import EventBusProtocol


class ContainerRole(Enum):
    """Standard container roles in the system."""
    BACKTEST = "backtest"
    DATA = "data"
    FEATURE = "feature"
    CLASSIFIER = "classifier"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    SIGNAL_LOG = "signal_log"
    ENSEMBLE = "ensemble"
    OPTIMIZATION = "optimization"
    SIGNAL_CAPTURE = "signal_capture"
    SIGNAL_REPLAY = "signal_replay"


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


class Container(Protocol):
    """
    Base container protocol with composition capabilities.
    
    All containers must implement this interface to work with
    the coordinator and container lifecycle manager.
    
    This protocol now includes hierarchical composition, event scoping,
    and advanced lifecycle management for the arch-101.md architecture.
    """
    
    @property
    @abstractmethod
    def container_id(self) -> str:
        """Unique identifier for this container."""
        ...
        
    @property
    @abstractmethod
    def container_type(self) -> str:
        """Type of container (backtest, optimization, etc)."""
        ...
        
    @property
    @abstractmethod
    def event_bus(self) -> EventBusProtocol:
        """Container's scoped event bus."""
        ...
    
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
    def parent_container(self) -> Optional['Container']:
        """Parent container if nested."""
        ...
    
    @property 
    @abstractmethod
    def child_containers(self) -> List['Container']:
        """Child containers nested within this container."""
        ...
        
    @abstractmethod
    def register_singleton(
        self,
        name: str,
        factory: Callable[[], Any]
    ) -> None:
        """
        Register a singleton component.
        
        Args:
            name: Component name
            factory: Factory function to create component
        """
        ...
        
    @abstractmethod
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a registered component.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None
        """
        ...
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the container and all components."""
        ...
        
    @abstractmethod
    async def start(self) -> None:
        """Start the container and begin processing."""
        ...
        
    @abstractmethod
    async def stop(self) -> None:
        """Stop the container gracefully."""
        ...
        
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources and dispose of components."""
        ...
    
    @abstractmethod
    async def dispose(self) -> None:
        """Clean up container resources."""
        ...
    
    # Composition Management
    @abstractmethod
    def add_child_container(self, child: 'Container') -> None:
        """Add a child container."""
        ...
    
    @abstractmethod
    def remove_child_container(self, container_id: str) -> bool:
        """Remove a child container by ID."""
        ...
    
    @abstractmethod
    def get_child_container(self, container_id: str) -> Optional['Container']:
        """Get child container by ID."""
        ...
    
    @abstractmethod
    def find_containers_by_role(self, role: ContainerRole) -> List['Container']:
        """Find all nested containers with specific role."""
        ...
    
    # Event Processing with Scoping
    @abstractmethod
    async def process_event(self, event: Any) -> Optional[Any]:
        """Process incoming event and optionally return response."""
        ...
    
    @abstractmethod
    def publish_event(self, event: Any, target_scope: str = "local") -> None:
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


class BacktestContainer(Container, Protocol):
    """
    Protocol for backtest containers.
    
    Extends base container with backtest-specific methods.
    """
    
    @abstractmethod
    async def prepare_data(self) -> Dict[str, Any]:
        """Prepare data for backtesting."""
        ...
        
    @abstractmethod
    async def execute_backtest(self) -> Dict[str, Any]:
        """Execute the backtest."""
        ...
        
    @abstractmethod
    async def get_results(self) -> Dict[str, Any]:
        """Get backtest results."""
        ...


class SignalGenerationContainer(Container, Protocol):
    """
    Protocol for signal generation containers.
    
    Used for analysis without execution.
    """
    
    @abstractmethod
    async def generate_signals(self) -> Dict[str, Any]:
        """Generate signals without execution."""
        ...
        
    @abstractmethod
    async def analyze_signals(self) -> Dict[str, Any]:
        """Analyze generated signals."""
        ...


class SignalReplayContainer(Container, Protocol):
    """
    Protocol for signal replay containers.
    
    Used for ensemble optimization.
    """
    
    @abstractmethod
    async def replay_signals(
        self,
        signals: Dict[str, List[Any]],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Replay signals with ensemble weights."""
        ...




class ContainerFactory(Protocol):
    """Protocol for container factories."""
    
    @abstractmethod
    def create_instance(self, config: Any) -> Container:
        """Create a container instance from configuration."""
        ...


class ContainerComposition(Protocol):
    """Protocol for composing containers according to patterns."""
    
    @abstractmethod
    def create_container(
        self,
        role: ContainerRole,
        config: Dict[str, Any],
        container_id: Optional[str] = None
    ) -> Container:
        """Create a container of specified role."""
        ...
    
    @abstractmethod
    def compose_pattern(
        self,
        pattern: Dict[str, Any],
        base_config: Dict[str, Any] = None
    ) -> Container:
        """Compose containers according to pattern specification."""
        ...
    
    @abstractmethod
    def validate_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Validate that pattern is valid and composable."""
        ...