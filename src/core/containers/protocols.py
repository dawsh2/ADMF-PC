"""
Container protocols for BACKTEST.MD architecture.

Defines the interfaces that all containers must implement.
"""

from typing import Protocol, Dict, Any, Optional, List, Callable
from abc import abstractmethod

from ..events import EventBus


class Container(Protocol):
    """
    Base container protocol.
    
    All containers must implement this interface to work with
    the coordinator and container lifecycle manager.
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
    def event_bus(self) -> EventBus:
        """Container's scoped event bus."""
        ...
        
    @abstractmethod
    def create_subcontainer(
        self,
        container_id: str,
        container_type: str = "generic"
    ) -> "Container":
        """
        Create a subcontainer within this container.
        
        Args:
            container_id: Unique ID for subcontainer
            container_type: Type of subcontainer
            
        Returns:
            New subcontainer instance
        """
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
    def get_subcontainers(self) -> List["Container"]:
        """Get all subcontainers."""
        ...
        
    @abstractmethod
    def get_subcontainers_by_type(self, container_type: str) -> List["Container"]:
        """Get subcontainers of a specific type."""
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