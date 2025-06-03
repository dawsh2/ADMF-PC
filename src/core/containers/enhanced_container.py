"""
Enhanced container implementation supporting BACKTEST.MD architecture.

Adds subcontainer support and proper event bus scoping.
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging

from .universal import UniversalScopedContainer, ContainerState
from .protocols import Container
from ..events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class EnhancedContainer(UniversalScopedContainer):
    """
    Enhanced container with subcontainer support.
    
    Implements the full Container protocol including:
    - Subcontainer creation and management
    - Scoped event buses
    - Hierarchical component resolution
    """
    
    def __init__(
        self,
        container_id: str,
        container_type: str = "generic",
        parent_container: Optional["EnhancedContainer"] = None
    ):
        """
        Initialize enhanced container.
        
        Args:
            container_id: Unique container ID
            container_type: Type of container
            parent_container: Parent container if this is a subcontainer
        """
        super().__init__(container_id, container_type)
        
        self._parent_container = parent_container
        self._subcontainers: Dict[str, "EnhancedContainer"] = {}
        
        # Create scoped event bus
        if parent_container:
            # Child containers can bubble events up to parent
            self._event_bus = ScopedEventBus(parent_container.event_bus)
        else:
            # Root container has standalone event bus
            self._event_bus = EventBus()
            
        # Communication adapter support
        self._output_event_handlers: List[Callable] = []
        self._expected_input_type: Optional[EventType] = None
            
    @property
    def event_bus(self) -> EventBus:
        """Get container's event bus."""
        return self._event_bus
        
    def create_subcontainer(
        self,
        container_id: str,
        container_type: str = "generic"
    ) -> "EnhancedContainer":
        """
        Create a subcontainer within this container.
        
        Args:
            container_id: Unique ID for subcontainer
            container_type: Type of subcontainer
            
        Returns:
            New subcontainer instance
        """
        # Create full container ID including parent path
        full_id = f"{self.container_id}.{container_id}"
        
        # Create subcontainer
        subcontainer = EnhancedContainer(
            container_id=full_id,
            container_type=container_type,
            parent_container=self
        )
        
        # Register subcontainer
        self._subcontainers[container_id] = subcontainer
        
        # Share parent services
        for key, value in self._shared_services.items():
            subcontainer._shared_services[key] = value
            
        logger.info(f"Created subcontainer {full_id} of type {container_type}")
        return subcontainer
        
    def get_subcontainers(self) -> List["EnhancedContainer"]:
        """Get all subcontainers."""
        return list(self._subcontainers.values())
        
    def get_subcontainers_by_type(self, container_type: str) -> List["EnhancedContainer"]:
        """Get subcontainers of a specific type."""
        return [
            container for container in self._subcontainers.values()
            if container.container_type == container_type
        ]
        
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a component, searching up the container hierarchy.
        
        First looks in this container, then in parent containers.
        """
        # Try local container first
        try:
            return self._dependency_container.resolve(name)
        except:
            # If not found locally and we have a parent, check parent
            if self._parent_container:
                return self._parent_container.get_component(name)
            return None
            
    def register_singleton(
        self,
        name: str,
        factory: Callable[[], Any]
    ) -> None:
        """
        Register a singleton component using a factory.
        
        Args:
            name: Component name
            factory: Factory function to create component
        """
        # Register factory in dependency container
        self._dependency_container.register_factory(name, factory)
        
    async def initialize(self) -> None:
        """Initialize the container and all subcontainers."""
        # Initialize this container
        self.initialize_scope()
        
        # Initialize all subcontainers
        for subcontainer in self._subcontainers.values():
            await subcontainer.initialize()
            
    async def start(self) -> None:
        """Start the container and all subcontainers."""
        # Start this container
        super().start()
        
        # Start all subcontainers
        for subcontainer in self._subcontainers.values():
            await subcontainer.start()
            
    async def stop(self) -> None:
        """Stop all subcontainers then this container."""
        # Stop subcontainers first (in reverse order)
        for subcontainer in reversed(list(self._subcontainers.values())):
            await subcontainer.stop()
            
        # Then stop this container
        super().stop()
        
    async def cleanup(self) -> None:
        """Cleanup all subcontainers then this container."""
        # Cleanup subcontainers first
        for subcontainer in self._subcontainers.values():
            await subcontainer.cleanup()
            
        # Then cleanup this container
        self.dispose()
        
    # Backtest-specific methods (only implemented in backtest containers)
    
    async def prepare_data(self) -> Dict[str, Any]:
        """Prepare data for backtesting."""
        # Get data streamer component
        data_streamer = self.get_component("data_streamer")
        if data_streamer:
            await data_streamer.initialize()
            return {"status": "data_prepared"}
        return {"status": "no_data_streamer"}
        
    async def execute_backtest(self) -> Dict[str, Any]:
        """Execute the backtest."""
        # This would be implemented by the actual backtest engine
        engine = self.get_component("backtest_engine")
        if engine:
            return await engine.run()
        return {"error": "no_backtest_engine"}
        
    async def get_results(self) -> Dict[str, Any]:
        """Get backtest results."""
        engine = self.get_component("backtest_engine")
        if engine:
            return engine.get_results()
        return {}
        
    async def generate_signals(self) -> Dict[str, Any]:
        """Generate signals without execution."""
        # Get signal analysis engine
        analysis_engine = self.get_component("signal_analysis_engine")
        if analysis_engine:
            return await analysis_engine.generate_signals()
        return {"error": "no_signal_analysis_engine"}
        
    async def replay_signals(
        self,
        signals: Dict[str, List[Any]],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Replay signals with ensemble weights."""
        # Get ensemble optimizer
        ensemble = self.get_component("ensemble_optimizer")
        if ensemble:
            return await ensemble.replay_with_weights(signals, weights)
        return {"error": "no_ensemble_optimizer"}
    
    # Communication adapter support methods
    
    def on_output_event(self, handler: Callable) -> None:
        """Register a handler for output events (used by communication adapters).
        
        Args:
            handler: Callable that will receive events emitted by this container
        """
        self._output_event_handlers.append(handler)
        logger.debug(f"Registered output event handler for {self.container_id}")
    
    def remove_output_handler(self, handler: Callable) -> None:
        """Remove an output event handler.
        
        Args:
            handler: Handler to remove
        """
        if handler in self._output_event_handlers:
            self._output_event_handlers.remove(handler)
            logger.debug(f"Removed output event handler from {self.container_id}")
    
    def emit_output_event(self, event: Event) -> None:
        """Emit an event to all registered output handlers.
        
        Args:
            event: Event to emit
        """
        for handler in self._output_event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in output event handler for {self.container_id}: {str(e)}"
                )
    
    async def receive_event(self, event: Event) -> None:
        """Receive an event from a communication adapter.
        
        Args:
            event: Event to process
        """
        # Default implementation publishes to internal event bus
        self._event_bus.publish(event)
        
        # Also process if we have a process_event method
        if hasattr(self, 'process_event'):
            await self.process_event(event)
    
    async def process_event(self, event: Event) -> None:
        """Process a received event. Override in subclasses for custom processing.
        
        Args:
            event: Event to process
        """
        logger.debug(
            f"Container {self.container_id} processing event: {event.event_type}"
        )
    
    @property
    def expected_input_type(self) -> Optional[EventType]:
        """Get the expected input event type for this container."""
        return self._expected_input_type
    
    @expected_input_type.setter
    def expected_input_type(self, event_type: Optional[EventType]) -> None:
        """Set the expected input event type for this container."""
        self._expected_input_type = event_type


class ScopedEventBus(EventBus):
    """
    Event bus scoped to a container with optional parent propagation.
    
    Events can be:
    - Local only (within this container)
    - Bubbled up to parent
    - Broadcast to children
    """
    
    def __init__(self, parent_bus: Optional[EventBus] = None):
        """Initialize scoped event bus."""
        super().__init__()
        self.parent_bus = parent_bus
        self._local_subscribers = {}
        
    def publish(self, event: Event, scope: str = "local") -> None:
        """
        Publish event with scoping.
        
        Args:
            event: Event to publish
            scope: One of "local", "bubble", "broadcast"
        """
        # Always publish locally
        super().publish(event)
        
        # Bubble up to parent if requested
        if scope == "bubble" and self.parent_bus:
            self.parent_bus.publish(event)
            
        # Note: broadcast would be implemented if we tracked child buses