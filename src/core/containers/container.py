"""
Canonical container implementation using composition, not inheritance.

This is the ONE implementation of containers in ADMF-PC, following
Protocol + Composition principles. No inheritance allowed.

Includes structured container naming strategy following BACKTEST.MD.
Implements the naming format:
{container_type}_{phase}_{classifier}_{risk_profile}_{timestamp}
"""

from typing import Dict, Any, Optional, List, Set, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import OrderedDict
import uuid
import logging

from ..events import EventBus
from .protocols import ContainerProtocol, ContainerMetadata, ContainerLimits, ContainerState
from .exceptions import (
    ComponentAlreadyExistsError,
    ComponentNotFoundError,
    ComponentDependencyError,
    InvalidContainerStateError,
    InvalidContainerConfigError,
    ParentContainerNotSetError
)
from .types import ContainerComponent, ContainerConfigDict

logger = logging.getLogger(__name__)


# Container naming enums and strategy

class ContainerType(Enum):
    """Types of containers in the system."""
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    SIGNAL_GEN = "signal_gen"
    SIGNAL_REPLAY = "signal_replay"
    LIVE_TRADING = "live"
    CLASSIFIER = "classifier"
    RISK_PORTFOLIO = "risk_portfolio"
    INDICATOR = "indicator"
    DATA = "data"


# Complex enums removed - these should be configuration-driven metadata instead


class ContainerNamingStrategy:
    """
    Simplified container naming strategy.
    
    Essential components only: role + unique_id
    Complex naming moved to configuration-driven metadata.
    """
    
    @staticmethod
    def generate_container_id(
        container_type: ContainerType,
        name_hint: Optional[str] = None
    ) -> str:
        """
        Generate a simple, unique container ID.
        
        Args:
            container_type: Type of container  
            name_hint: Optional human-readable hint
            
        Returns:
            Simple container ID: {type}[_{hint}]_{uuid}
        """
        parts = [container_type.value]
        
        if name_hint:
            # Sanitize hint: lowercase, replace spaces/special chars with underscores
            clean_hint = ''.join(c if c.isalnum() else '_' for c in name_hint.lower())
            clean_hint = '_'.join(p for p in clean_hint.split('_') if p)  # Remove empty parts
            if clean_hint:
                parts.append(clean_hint)
        
        # Always add unique suffix for guaranteed uniqueness
        parts.append(uuid.uuid4().hex[:8])
        
        return '_'.join(parts)
        
    @staticmethod
    def parse_container_id(container_id: str) -> Dict[str, Any]:
        """
        Parse a simple container ID into basic components.
        
        Args:
            container_id: Container ID to parse (format: type[_hint]_uuid)
            
        Returns:
            Dictionary with basic parsed components
        """
        parts = container_id.split('_')
        
        result = {
            'raw_id': container_id,
            'container_type': None,
            'name_hint': None,
            'uuid': None
        }
        
        if len(parts) >= 2:
            # First part is always container type
            try:
                result['container_type'] = ContainerType(parts[0])
            except ValueError:
                pass
            
            # Last part is always UUID (8 hex chars)
            if len(parts[-1]) == 8:
                try:
                    int(parts[-1], 16)  # Validate it's hex
                    result['uuid'] = parts[-1]
                    
                    # Middle parts are name hint (if any)
                    if len(parts) > 2:
                        result['name_hint'] = '_'.join(parts[1:-1])
                except ValueError:
                    pass
        
        return result
        
    @staticmethod  
    def get_container_family(container_id: str) -> str:
        """
        Get the container type from ID for grouping.
        
        Args:
            container_id: Container ID
            
        Returns:
            Container type or 'unknown'
        """
        parsed = ContainerNamingStrategy.parse_container_id(container_id)
        if parsed['container_type']:
            return parsed['container_type'].value
        return 'unknown'


# Simplified convenience functions - complex naming logic moved to configuration

def create_container_id(container_type: ContainerType, name_hint: Optional[str] = None) -> str:
    """Create a simple container ID. Complex naming moved to metadata."""
    return ContainerNamingStrategy.generate_container_id(container_type, name_hint)




@dataclass
class ContainerConfig:
    """Configuration for a container - role-agnostic, component-driven."""
    name: str
    container_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=set)
    components: List[str] = field(default_factory=list)  # Component types to inject
    
    # Optional metadata (replaces role)
    container_type: Optional[str] = None  # data, strategy, portfolio, etc. - inferred if not set
    

class Container:
    """
    Canonical container implementation using composition.
    
    Implements Container protocol with composition capabilities for the arch-101.md architecture:
    - Protocol + Composition (no inheritance)
    - Isolated event buses for Data, Strategy, Risk, Execution modules
    - Hierarchical composition with parent/child relationships
    - Pluggable route communication
    - Configurable, composable, reliable
    """
    
    def __init__(self, config: ContainerConfig, parent_event_bus: Optional[EventBus] = None):
        """
        Initialize container with configuration.
        
        Args:
            config: Container configuration
            parent_event_bus: Event bus from parent container (None for root container)
        """
        self.config = config
        
        # Infer container type from components if not explicitly set
        if config.container_type:
            self.container_type = config.container_type
        else:
            self.container_type = self._infer_container_type(config.components)
        
        self.container_id = config.container_id or f"{self.container_type}_{uuid.uuid4().hex[:8]}"
        
        # Only root container creates an event bus, others use parent's bus
        if parent_event_bus is None:
            # This is the root container - create the shared event bus
            self.event_bus = EventBus(bus_id=f"root_{self.container_id}")
            self._is_root_container = True
        else:
            # This is a child container - use parent's event bus
            self.event_bus = parent_event_bus
            self._is_root_container = False
        
        # State management
        self._state = ContainerState.UNINITIALIZED
        self._state_history: List[tuple[ContainerState, datetime]] = [
            (ContainerState.UNINITIALIZED, datetime.now())
        ]
        
        # Composition: parent/child relationships
        self._parent_container: Optional['Container'] = None
        self._child_containers: Dict[str, 'Container'] = {}
        
        # Component registry - OrderedDict maintains insertion order for lifecycle management
        self._components: OrderedDict[str, ContainerComponent] = OrderedDict()
        
        # Metadata for composition
        self._metadata = ContainerMetadata(
            container_id=self.container_id,
            role=self.container_type,  # Use inferred type instead of role
            name=config.name,
            config=config.config
        )
        
        # Resource limits
        self._limits: Optional[ContainerLimits] = None
        
        # Metrics tracking
        self._metrics = {
            'events_processed': 0,
            'events_published': 0,
            'start_time': None,
            'last_activity': None
        }
        
        # Setup event bus tracing if enabled (only on root container)
        if self._is_root_container and self._should_enable_tracing():
            self._setup_tracing()
        
        # Streaming metrics support (optional)
        self.streaming_metrics: Optional[Any] = None
        
        # Portfolio containers always setup metrics for event tracing  
        if self.container_type == 'portfolio':
            self._setup_event_tracing_metrics()
        elif self._should_track_metrics():
            self._setup_metrics()
        
        logger.info(f"Created container: {self.name} ({self.container_id})")
    
    def _infer_container_type(self, components: List[str]) -> str:
        """
        Infer container type from component composition.
        
        Args:
            components: List of component types
            
        Returns:
            Inferred container type
        """
        # Simple heuristics based on component types
        component_set = set(components)
        
        if any(comp in component_set for comp in ['portfolio_manager', 'position_manager', 'portfolio']):
            return 'portfolio'
        elif any(comp in component_set for comp in ['strategy', 'signal_generator', 'classifier']):
            return 'strategy'  
        elif any(comp in component_set for comp in ['data_streamer', 'bar_streamer', 'signal_streamer']):
            return 'data'
        elif any(comp in component_set for comp in ['execution_engine', 'order_manager']):
            return 'execution'
        elif any(comp in component_set for comp in ['risk_manager', 'position_sizer']):
            return 'risk'
        elif any(comp in component_set for comp in ['analytics', 'metrics_collector']):
            return 'analytics'
        else:
            return 'generic'  # Default fallback
    
    @property
    def name(self) -> str:
        """Container name."""
        return self.config.name
    
    @property
    def role(self) -> str:
        """Container type (inferred from components)."""
        return self.container_type
    
    @property
    def state(self) -> ContainerState:
        """Current container state."""
        return self._state
    
    @property
    def capabilities(self) -> Set[str]:
        """Container capabilities."""
        return self.config.capabilities.copy()
    
    @property
    def metadata(self) -> ContainerMetadata:
        """Container identification and metadata."""
        # Update metadata with current state
        self._metadata.parent_id = self._parent_container.container_id if self._parent_container else None
        return self._metadata
    
    @property
    def parent_container(self) -> Optional['Container']:
        """Parent container if nested."""
        return self._parent_container
    
    @property
    def child_containers(self) -> List['Container']:
        """Child containers nested within this container."""
        return list(self._child_containers.values())
    
    @property
    def components(self) -> Dict[str, ContainerComponent]:
        """Get all components in this container."""
        return self._components.copy()
    
    def add_component(self, name: str, component: ContainerComponent) -> None:
        """
        Add a component to the container.
        
        Args:
            name: Component name
            component: Component instance
        """
        if name in self._components:
            raise ComponentAlreadyExistsError(name)
        
        self._components[name] = component
        
        # If component needs container reference, provide it
        if hasattr(component, 'set_container'):
            component.set_container(self)
        
        # If component needs event bus access, provide it
        if hasattr(component, 'set_event_bus'):
            component.set_event_bus(self.event_bus)
        
        logger.debug(f"Added component '{name}' to {self.name}")
    
    def get_component(self, name: str) -> Optional[ContainerComponent]:
        """
        Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None
        """
        return self._components.get(name)
    
    def wire_dependencies(self, component_name: str, dependencies: Dict[str, str]) -> None:
        """
        Wire dependencies manually for a component.
        
        This is a helper for manual dependency injection within the container.
        
        Args:
            component_name: Name of component to wire
            dependencies: Dict mapping attribute names to component names
            
        Example:
            container.add_component("data", DataLoader())
            container.add_component("strategy", MomentumStrategy())
            container.wire_dependencies("strategy", {"data_source": "data"})
        """
        component = self.get_component(component_name)
        if not component:
            raise ComponentNotFoundError(component_name)
            
        for attr_name, dep_name in dependencies.items():
            dep_component = self.get_component(dep_name)
            if not dep_component:
                raise ComponentDependencyError(component_name, dep_name)
            setattr(component, attr_name, dep_component)
            logger.debug(f"Wired {dep_name} to {component_name}.{attr_name}")
    
    def register_singleton(self, name: str, factory) -> None:
        """
        Register a singleton component using lazy initialization.
        
        Args:
            name: Component name
            factory: Factory function to create component
        """
        # For now, just create and add the component
        # Could be enhanced with lazy loading if needed
        component = factory()
        self.add_component(name, component)
    
    def initialize(self) -> None:
        """Initialize the container and all components."""
        if self._state not in (ContainerState.UNINITIALIZED, ContainerState.STOPPED):
            logger.warning(f"Container {self.name} already initialized (state: {self._state})")
            return
        
        self._set_state(ContainerState.INITIALIZING)
        
        try:
            # Initialize all components in insertion order
            for name, component in self._components.items():
                if hasattr(component, 'initialize'):
                    logger.debug(f"Initializing component '{name}'")
                    component.initialize()
            
            self._set_state(ContainerState.INITIALIZED)
            logger.info(f"Container {self.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize container {self.name}: {e}")
            self._set_state(ContainerState.ERROR)
            raise
    
    def start(self) -> None:
        """Start the container and begin processing."""
        if self._state not in (ContainerState.INITIALIZED, ContainerState.STOPPED):
            raise InvalidContainerStateError(
                self.name,
                self._state,
                [ContainerState.INITIALIZED, ContainerState.STOPPED]
            )
        
        try:
            # Start all components in insertion order
            for name, component in self._components.items():
                if hasattr(component, 'start'):
                    logger.debug(f"Starting component '{name}'")
                    component.start()
            
            self._metrics['start_time'] = datetime.now()
            self._set_state(ContainerState.RUNNING)
            logger.info(f"Container {self.name} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start container {self.name}: {e}")
            self._set_state(ContainerState.ERROR)
            raise
    
    def stop(self) -> None:
        """Stop the container gracefully."""
        if self._state not in (ContainerState.RUNNING, ContainerState.ERROR):
            logger.warning(f"Container {self.name} not running")
            return
        
        self._set_state(ContainerState.STOPPING)
        
        try:
            # Stop all components in reverse insertion order
            for name, component in reversed(self._components.items()):
                if hasattr(component, 'stop'):
                    logger.debug(f"Stopping component '{name}'")
                    component.stop()
            
            self._set_state(ContainerState.STOPPED)
            logger.info(f"Container {self.name} stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping container {self.name}: {e}")
            self._set_state(ContainerState.ERROR)
            raise
    
    def execute(self) -> None:
        """
        Execution phase - let components do their work.
        
        For event-driven systems, this might just start data streaming.
        Other components react to events naturally.
        """
        if self._state != ContainerState.RUNNING:
            raise InvalidContainerStateError(
                self.name,
                self._state,
                [ContainerState.RUNNING]
            )
        
        logger.info(f"Container {self.container_id} entering execution phase with {len(self._components)} components")
        
        # Let each component execute if it needs to
        # Data streamers will start streaming
        # Other components are event-driven and will react naturally
        for name, component in self._components.items():
            logger.info(f"Checking component {name}: {type(component).__name__}, has_execute: {hasattr(component, 'execute')}")
            if hasattr(component, 'execute'):
                try:
                    logger.info(f"Executing component {name} in {self.container_id}")
                    component.execute()
                except Exception as e:
                    logger.error(f"Error executing component {name}: {e}")
                    # Don't stop other components from executing
    
    def cleanup(self) -> None:
        """Cleanup resources and dispose of components."""
        # Save results before cleanup if disk storage
        if self.config.config.get('results_storage') == 'disk':
            self._save_results_before_cleanup()
        
        # Flush trace storage before cleanup (configurable)
        execution_config = self.config.config.get('execution', {})
        trace_settings = execution_config.get('trace_settings', {})
        auto_flush = trace_settings.get('auto_flush_on_cleanup', True)  # Default: True
        
        if auto_flush:
            self._flush_trace_storage()
        
        # Ensure stopped first
        if self._state == ContainerState.RUNNING:
            self.stop()
        
        # Cleanup all components in reverse insertion order
        for name, component in reversed(self._components.items()):
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up component '{name}': {e}")
        
        # Clear registry
        self._components.clear()
        
        logger.info(f"Container {self.name} cleaned up")
    
    def _flush_trace_storage(self) -> None:
        """Flush trace storage to disk before cleanup."""
        try:
            # Check if event bus has a tracer with storage
            if hasattr(self.event_bus, '_tracer') and self.event_bus._tracer:
                tracer = self.event_bus._tracer
                
                # Check if tracer has storage that can be flushed
                if hasattr(tracer, 'storage') and tracer.storage:
                    storage = tracer.storage
                    
                    # Flush hierarchical storage
                    if hasattr(storage, 'flush_all'):
                        storage.flush_all()
                        logger.info(f"Flushed hierarchical storage for container {self.container_id}")
                    
                    # Also try general flush if available
                    elif hasattr(storage, 'flush'):
                        storage.flush()
                        logger.info(f"Flushed storage for container {self.container_id}")
                
        except Exception as e:
            logger.error(f"Error flushing trace storage for container {self.container_id}: {e}")
    
    def _set_state(self, state: ContainerState) -> None:
        """Update container state and track history."""
        self._state = state
        self._state_history.append((state, datetime.now()))
        logger.debug(f"Container {self.name} state changed to {state.value}")
    
    # Composition Management
    
    def add_child_container(self, child: 'Container') -> None:
        """Add a child container."""
        if child.container_id in self._child_containers:
            raise InvalidContainerConfigError(
                f"Child container {child.container_id} already exists",
                "child_containers"
            )
        
        self._child_containers[child.container_id] = child
        child._parent_container = self
        logger.info(f"Added child container {child.container_id} to {self.container_id}")
    
    def remove_child_container(self, container_id: str) -> bool:
        """Remove a child container by ID."""
        if container_id not in self._child_containers:
            return False
        
        child = self._child_containers.pop(container_id)
        child._parent_container = None
        logger.info(f"Removed child container {container_id} from {self.container_id}")
        return True
    
    def get_child_container(self, container_id: str) -> Optional['Container']:
        """Get child container by ID."""
        return self._child_containers.get(container_id)
    
    def find_containers_by_type(self, container_type: str) -> List['Container']:
        """Find all nested containers with specific type."""
        containers = []
        
        # Check direct children
        for child in self._child_containers.values():
            if child.container_type == container_type:
                containers.append(child)
            
            # Recursively check grandchildren
            containers.extend(child.find_containers_by_type(container_type))
        
        return containers
    
    # Child container creation helper
    
    def create_child(self, config: ContainerConfig) -> 'Container':
        """
        Create and add a child container.
        
        Args:
            config: Configuration for the child container
            
        Returns:
            The newly created child container
        """
        # Pass our event bus to the child container
        child = Container(config, parent_event_bus=self.event_bus)
        self.add_child_container(child)
        return child
    
    # Event handling methods for route integration
    
    def receive_event(self, event: Any) -> None:
        """
        Receive an event for processing by this container.
        
        Since all containers share the root event bus, this method
        primarily handles metrics tracking and container-specific tracing.
        
        Args:
            event: Event to process
        """
        self._metrics['events_processed'] += 1
        self._metrics['last_activity'] = datetime.now()
        
        # Container-specific event tracing (if enabled)
        # Note: This is separate from the main event bus tracing
        if hasattr(self.event_bus, '_tracer') and self.event_bus._tracer:
            # Only trace certain event types for portfolio containers
            if self.container_type == 'portfolio':
                event_type = getattr(event, 'event_type', None)
                if event_type in ['SIGNAL', 'FILL', 'ORDER']:
                    logger.debug(f"Portfolio {self.name} tracing incoming {event_type} event")
                    self.event_bus._tracer.trace_event(event)
            # Other containers can define their own rules or trace everything
            else:
                self.event_bus._tracer.trace_event(event)
        
        # Event is already published to the shared bus by the publisher
        # No need to republish or forward to children since they share the same bus
    
    # Event Processing and Status methods
    
    def process_event(self, event: Any) -> Optional[Any]:
        """Process incoming event and optionally return response."""
        self.receive_event(event)
        # For now, no response processing - can be enhanced later
        return None
    
    def publish_event(self, event: Any, target_scope: str = "global") -> None:
        """Publish event to the shared root event bus.
        
        Since all containers share the same event bus, all events
        are automatically visible to all containers and components.
        
        Args:
            event: Event to publish
            target_scope: Kept for backward compatibility, but all events go to shared bus
        """
        self._metrics['events_published'] += 1
        self._metrics['last_activity'] = datetime.now()
        
        # All events go to the shared root event bus
        # All containers and components will see the event automatically
        self.event_bus.publish(event)
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update container configuration."""
        self._metadata.config.update(config)
        logger.info(f"Updated config for container {self.container_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current container status and metrics."""
        return {
            'container_id': self.container_id,
            'name': self.name,
            'role': self.role.value,
            'state': self._state.value,
            'parent_id': self._parent_container.container_id if self._parent_container else None,
            'child_count': len(self._child_containers),
            'component_count': len(self._components),
            'capabilities': list(self.capabilities),
            'metrics': self._metrics.copy()
        }
    
    def get_capabilities(self) -> Set[str]:
        """Get container capabilities/features."""
        return self.capabilities
    
    def dispose(self) -> None:
        """Clean up container resources."""
        # Dispose children first
        for child in list(self._child_containers.values()):
            child.dispose()
        
        # Remove from parent
        if self._parent_container:
            self._parent_container.remove_child_container(self.container_id)
        
        # Clean up self
        self.cleanup()
    
    # ==================== Event Tracing Methods ====================
    # These methods provide optional event tracing functionality
    # Containers decide whether to trace based on their configuration
    
    def _should_enable_tracing(self) -> bool:
        """
        Check if tracing should be enabled based on config.
        
        Checks both execution-level config and container-specific settings.
        """
        # Check execution config first
        execution_config = self.config.config.get('execution', {})
        if not execution_config.get('enable_event_tracing', False):
            return False
            
        # Check container-specific settings
        trace_settings = execution_config.get('trace_settings', {})
        
        # Check if this specific container should be traced
        container_settings = trace_settings.get('container_settings', {})
        
        # Check wildcard patterns
        for pattern, settings in container_settings.items():
            if '*' in pattern:
                # Simple wildcard matching
                prefix = pattern.replace('*', '')
                if self.container_id.startswith(prefix) or self.name.startswith(prefix):
                    return settings.get('enabled', True)
            elif pattern == self.container_id or pattern == self.name:
                return settings.get('enabled', True)
        
        # Check trace_specific list
        trace_specific = trace_settings.get('trace_specific', [])
        if trace_specific and (self.container_id in trace_specific or self.name in trace_specific):
            return True
        
        # Default to true if tracing is enabled globally
        return True
    
    def _setup_tracing(self):
        """
        Setup event tracing based on container config.
        
        Enables tracing on the container's event bus if configured.
        """
        execution_config = self.config.config.get('execution', {})
        trace_settings = execution_config.get('trace_settings', {})
        
        # Build trace ID using metadata if available
        trace_id_parts = []
        
        # Use metadata from orchestration for context
        metadata = self.config.config.get('metadata', {})
        if metadata.get('workflow_id'):
            trace_id_parts.append(metadata['workflow_id'])
        if metadata.get('phase_name'):
            trace_id_parts.append(metadata['phase_name'])
        
        # Always include container ID
        trace_id_parts.append(self.container_id)
        
        trace_id = '_'.join(trace_id_parts)
        
        # Get container-specific settings
        container_settings = trace_settings.get('container_settings', {})
        max_events = trace_settings.get('max_events', 10000)
        
        # Check for container-specific max_events
        for pattern, settings in container_settings.items():
            if '*' in pattern:
                prefix = pattern.replace('*', '')
                if self.container_id.startswith(prefix) or self.name.startswith(prefix):
                    max_events = settings.get('max_events', max_events)
                    break
            elif pattern == self.container_id or pattern == self.name:
                max_events = settings.get('max_events', max_events)
                break
        
        # Enable tracing on the event bus
        trace_config = {
            'correlation_id': trace_id,
            'max_events': max_events
        }
        
        # Check if hierarchical storage should be used
        storage_backend = trace_settings.get('storage_backend', 'memory')
        if storage_backend == 'hierarchical':
            trace_config['storage_backend'] = 'hierarchical'
            trace_config['workflow_id'] = metadata.get('workflow_id')
            trace_config['phase_name'] = metadata.get('phase_name')
            trace_config['container_id'] = self.container_id
            trace_config['batch_size'] = trace_settings.get('batch_size', 1000)  # Pass through batch_size
            trace_config['auto_flush_on_cleanup'] = trace_settings.get('auto_flush_on_cleanup', True)
            
            logger.info(f"Using hierarchical storage for container {self.container_id}: "
                       f"workspaces/{metadata.get('workflow_id')}/{self.container_id}/")
        
        # Pass through console output settings
        global_trace_settings = execution_config.get('trace_settings', {})
        if global_trace_settings.get('enable_console_output', False):
            trace_config['enable_console_output'] = True
            trace_config['console_filter'] = global_trace_settings.get('console_filter', [])
            logger.info(f"Console output enabled for container {self.container_id} with filter: {trace_config['console_filter']}")
        
        self.event_bus.enable_tracing(trace_config)
        
        logger.info(f"Tracing enabled for container {self.container_id} "
                   f"with trace_id: {trace_id}, max_events: {max_events}")
    
    def get_trace_summary(self) -> Optional[Dict[str, Any]]:
        """Get trace summary if tracing is enabled."""
        return self.event_bus.get_tracer_summary()
    
    def save_trace(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Save trace to file if tracing is enabled.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path where trace was saved, or None if tracing not enabled
        """
        # Check if event bus has a tracer
        if not self.event_bus._tracer:
            return None
        
        execution_config = self.config.config.get('execution', {})
        trace_settings = execution_config.get('trace_settings', {})
        
        if not output_path:
            # Use configured trace directory
            trace_dir = trace_settings.get('trace_dir', './traces')
            metadata = self.config.config.get('metadata', {})
            
            # Build filename
            filename_parts = []
            if metadata.get('workflow_id'):
                filename_parts.append(metadata['workflow_id'])
            if metadata.get('phase_name'):
                filename_parts.append(metadata['phase_name'])
            filename_parts.append(self.container_id)
            filename_parts.append('trace.jsonl')
            
            output_path = f"{trace_dir}/{'_'.join(filename_parts)}"
        
        # Save trace
        if hasattr(self.event_bus._tracer, 'save_to_file'):
            self.event_bus._tracer.save_to_file(output_path)
            logger.info(f"Saved trace for container {self.container_id} to {output_path}")
            return output_path
        else:
            logger.warning(f"EventTracer does not support save_to_file for container {self.container_id}")
            return None
    
    # ==================== Streaming Metrics Methods ====================
    # These methods provide memory-efficient performance tracking
    # Portfolio containers track metrics by default, others based on config
    
    def _should_track_metrics(self) -> bool:
        """
        Check if this container should track performance metrics.
        
        Portfolio containers track metrics by default.
        Other containers check their configuration.
        """
        # Portfolio containers always track metrics
        if self.container_type == 'portfolio':
            return True
        
        # Check container configuration
        metrics_config = self.config.config.get('metrics', {})
        if metrics_config.get('enabled', False):
            return True
        
        # Check execution configuration
        execution_config = self.config.config.get('execution', {})
        if execution_config.get('track_metrics', False):
            return True
        
        # Check if results configuration requests metrics
        results_config = self.config.config.get('results', {})
        if results_config.get('streaming_metrics', False):
            return True
        
        return False
    
    def _setup_metrics(self):
        """
        Setup event-based metrics tracking using MetricsObserver.
        
        Uses trade-complete retention policy for memory efficiency:
        - Only keeps events for open trades
        - Prunes events when trades close
        - Calculates metrics incrementally
        """
        if self.container_type != 'portfolio':
            # Only portfolio containers need metrics
            return
            
        try:
            from ..events.observers.metrics import MetricsObserver, BasicMetricsCalculator
        except ImportError:
            logger.warning(f"MetricsObserver not available for container {self.container_id}")
            return
        
        # Get configuration
        metrics_config = self.config.config.get('metrics', {})
        results_config = self.config.config.get('results', {})
        
        # Create metrics calculator with portfolio configuration
        calculator = BasicMetricsCalculator(
            initial_capital=self.config.config.get('initial_capital', 100000.0),
            annualization_factor=metrics_config.get('annualization_factor', 252.0)
        )
        
        # Create observer with trade-complete retention for memory efficiency
        self.streaming_metrics = MetricsObserver(
            calculator=calculator,
            retention_policy=results_config.get('retention_policy', 'trade_complete'),
            max_events=results_config.get('max_events', 1000)
        )
        
        # Attach observer to event bus
        self.event_bus.attach_observer(self.streaming_metrics)
        
        logger.info(f"MetricsObserver attached to portfolio container {self.container_id} "
                   f"with retention_policy='trade_complete' for memory efficiency")
    
    def _setup_event_tracing_metrics(self):
        """
        Alternative setup for event tracing as the metrics system.
        
        This just calls _setup_metrics() since we now use MetricsObserver
        for both tracing and metrics with unified trade-complete retention.
        """
        # Unified approach - MetricsObserver handles both tracing and metrics
        self._setup_metrics()
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics if available.
        
        This is the method expected by MetricsCollector.
        
        Returns:
            Dictionary of metrics or None if not tracking
        """
        if not self.streaming_metrics:
            # Return basic metrics even if not tracking performance
            return {
                'container_id': self.container_id,
                'container_type': self.container_type,
                'events_processed': self._metrics['events_processed'],
                'events_published': self._metrics['events_published'],
                'uptime_seconds': (datetime.now() - self._metrics['start_time']).total_seconds() if self._metrics['start_time'] else 0
            }
        
        # Get streaming metrics
        if hasattr(self.streaming_metrics, 'get_metrics'):
            metrics = self.streaming_metrics.get_metrics()
        else:
            metrics = {}
        
        # For MetricsObserver, just return what it provides
        # It already includes container tracking in observer_stats
        
        return metrics
    
    def get_performance(self) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics (alias for get_metrics).
        
        Some systems may look for this method name.
        """
        return self.get_metrics()
    
    def get_final_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get final metrics for phase completion.
        
        Returns complete results including metrics, trades, and equity curve
        based on configuration.
        """
        if not self.streaming_metrics:
            return None
        
        # For portfolio containers using MetricsObserver
        if self.container_type == 'portfolio' and hasattr(self.streaming_metrics, 'get_results'):
            return self.streaming_metrics.get_results()
        
        # Fallback to basic metrics
        return self.get_metrics()
    
    def _save_results_before_cleanup(self) -> None:
        """Save results during container destruction."""
        if not self.streaming_metrics:
            return
            
        results = self.streaming_metrics.get_metrics() if hasattr(self.streaming_metrics, 'get_metrics') else None
        if not results:
            return
        
        # Build path from execution metadata
        metadata = self.config.config.get('metadata', {})
        results_dir = metadata.get('results_dir', './results')
        workflow_id = metadata.get('workflow_id', 'unknown')
        phase_name = metadata.get('phase_name', 'unknown')
        window_id = metadata.get('window_id', '')  # For walk-forward, etc.
        
        # Construct path with window ID if present
        if window_id:
            path = f"{results_dir}/{workflow_id}/{phase_name}/{window_id}"
        else:
            path = f"{results_dir}/{workflow_id}/{phase_name}"
        
        import os
        os.makedirs(path, exist_ok=True)
        
        # Write container results
        filepath = f"{path}/{self.container_id}_results.json"
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved results for {self.container_id} to {filepath}")
    
    def __repr__(self) -> str:
        return f"Container(name={self.name}, id={self.container_id}, state={self._state.value})"