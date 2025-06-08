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
import uuid
import logging

from ..events import EventBus
from .protocols import Container as ContainerProtocol, ContainerMetadata, ContainerLimits, ContainerState, ContainerRole
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


class Phase(Enum):
    """Workflow phases for container naming."""
    PHASE1_GRID_SEARCH = "phase1_grid"
    PHASE2_ENSEMBLE = "phase2_ensemble"
    PHASE3_VALIDATION = "phase3_validation"
    PHASE4_WALK_FORWARD = "phase4_walkforward"
    INITIALIZATION = "init"
    DATA_PREPARATION = "data_prep"
    COMPUTATION = "compute"
    VALIDATION = "validate"
    AGGREGATION = "aggregate"
    LIVE = "live"
    ANALYSIS = "analysis"


class ClassifierType(Enum):
    """Types of classifiers."""
    HMM = "hmm"
    PATTERN = "pattern"
    TREND_VOL = "trend_vol"
    MULTI_INDICATOR = "multi_ind"
    ENSEMBLE = "ensemble"
    NONE = "none"


class RiskProfile(Enum):
    """Risk profiles for portfolio management."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"
    NONE = "none"


class ContainerNamingStrategy:
    """
    Structured naming strategy for containers.
    
    Creates names that:
    - Enable easy identification of container purpose
    - Support tracking across optimization phases
    - Facilitate debugging and monitoring
    - Allow result aggregation by type
    """
    
    @staticmethod
    def generate_container_id(
        container_type: ContainerType,
        phase: Optional[Phase] = None,
        classifier: Optional[ClassifierType] = None,
        risk_profile: Optional[RiskProfile] = None,
        timestamp: Optional[datetime] = None,
        unique_suffix: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a structured container ID.
        
        Args:
            container_type: Type of container
            phase: Current workflow phase
            classifier: Classifier type (if applicable)
            risk_profile: Risk profile (if applicable)
            timestamp: Timestamp for the container
            unique_suffix: Whether to add UUID suffix
            metadata: Additional metadata to encode in name
            
        Returns:
            Structured container ID
        """
        parts = [container_type.value]
        
        # Add phase if specified
        if phase:
            parts.append(phase.value)
            
        # Add classifier if specified and not none
        if classifier and classifier != ClassifierType.NONE:
            parts.append(classifier.value)
            
        # Add risk profile if specified and not none
        if risk_profile and risk_profile != RiskProfile.NONE:
            parts.append(risk_profile.value)
            
        # Add metadata elements if provided
        if metadata:
            # Add symbol set identifier
            if 'symbols' in metadata and metadata['symbols']:
                if len(metadata['symbols']) == 1:
                    parts.append(metadata['symbols'][0].lower())
                elif len(metadata['symbols']) <= 3:
                    parts.append('_'.join(s.lower() for s in metadata['symbols']))
                else:
                    parts.append(f"{len(metadata['symbols'])}syms")
                    
            # Add strategy identifier
            if 'strategy' in metadata:
                parts.append(metadata['strategy'].lower().replace(' ', '_'))
                
            # Add optimization trial number
            if 'trial' in metadata:
                parts.append(f"t{metadata['trial']}")
                
        # Add timestamp
        if timestamp is None:
            timestamp = datetime.now()
        parts.append(timestamp.strftime('%Y%m%d_%H%M%S'))
        
        # Add unique suffix if requested
        if unique_suffix:
            parts.append(uuid.uuid4().hex[:8])
            
        return '_'.join(parts)
        
    @staticmethod
    def parse_container_id(container_id: str) -> Dict[str, Any]:
        """
        Parse a structured container ID into components.
        
        Args:
            container_id: Container ID to parse
            
        Returns:
            Dictionary with parsed components
        """
        parts = container_id.split('_')
        result = {
            'raw_id': container_id,
            'container_type': None,
            'phase': None,
            'classifier': None,
            'risk_profile': None,
            'timestamp': None,
            'uuid': None,
            'metadata': {}
        }
        
        if not parts:
            return result
            
        # Parse container type (always first)
        try:
            result['container_type'] = ContainerType(parts[0])
            idx = 1
        except ValueError:
            idx = 0
            
        # Try to parse remaining parts
        while idx < len(parts):
            part = parts[idx]
            
            # Check if it's a phase
            try:
                for phase in Phase:
                    if phase.value == part or phase.value.startswith(part):
                        result['phase'] = phase
                        idx += 1
                        continue
            except:
                pass
                
            # Check if it's a classifier
            try:
                for classifier in ClassifierType:
                    if classifier.value == part:
                        result['classifier'] = classifier
                        idx += 1
                        continue
            except:
                pass
                
            # Check if it's a risk profile
            try:
                for profile in RiskProfile:
                    if profile.value == part:
                        result['risk_profile'] = profile
                        idx += 1
                        continue
            except:
                pass
                
            # Check if it's a timestamp (YYYYMMDD format)
            if len(part) == 8 and part.isdigit():
                # Next part should be time (HHMMSS)
                if idx + 1 < len(parts) and len(parts[idx + 1]) == 6 and parts[idx + 1].isdigit():
                    try:
                        timestamp_str = f"{part}_{parts[idx + 1]}"
                        result['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        idx += 2
                        continue
                    except:
                        pass
                        
            # Check if it's a UUID (8 hex chars at the end)
            if idx == len(parts) - 1 and len(part) == 8:
                try:
                    int(part, 16)
                    result['uuid'] = part
                    idx += 1
                    continue
                except:
                    pass
                    
            # Otherwise, it's metadata
            result['metadata'][f'part_{idx}'] = part
            idx += 1
            
        return result
        
    @staticmethod
    def create_hierarchical_id(
        parent_id: str,
        child_type: str,
        child_descriptor: Optional[str] = None
    ) -> str:
        """
        Create a child container ID based on parent.
        
        Args:
            parent_id: Parent container ID
            child_type: Type of child container
            child_descriptor: Optional descriptor for child
            
        Returns:
            Child container ID
        """
        parts = [parent_id, child_type]
        
        if child_descriptor:
            parts.append(child_descriptor)
            
        # Add short UUID for uniqueness
        parts.append(uuid.uuid4().hex[:6])
        
        return '_'.join(parts)
        
    @staticmethod
    def get_container_family(container_id: str) -> str:
        """
        Get the family/root of a container ID.
        
        Useful for grouping related containers.
        
        Args:
            container_id: Container ID
            
        Returns:
            Container family identifier
        """
        parsed = ContainerNamingStrategy.parse_container_id(container_id)
        
        family_parts = []
        
        if parsed['container_type']:
            family_parts.append(parsed['container_type'].value)
            
        if parsed['phase']:
            family_parts.append(parsed['phase'].value)
            
        if parsed['classifier']:
            family_parts.append(parsed['classifier'].value)
            
        return '_'.join(family_parts) if family_parts else 'unknown'
        
    @staticmethod
    def create_workflow_container_id(
        workflow_id: str,
        container_type: ContainerType,
        phase: Phase,
        iteration: Optional[int] = None
    ) -> str:
        """
        Create container ID for workflow execution.
        
        Args:
            workflow_id: Workflow identifier
            container_type: Type of container
            phase: Current phase
            iteration: Optional iteration number
            
        Returns:
            Workflow container ID
        """
        parts = [
            'wf',
            workflow_id[:8],  # First 8 chars of workflow ID
            container_type.value,
            phase.value
        ]
        
        if iteration is not None:
            parts.append(f'i{iteration}')
            
        parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        return '_'.join(parts)


# Convenience functions for container naming

def create_backtest_container_id(
    phase: Phase,
    classifier: ClassifierType = ClassifierType.NONE,
    risk_profile: RiskProfile = RiskProfile.BALANCED,
    **kwargs
) -> str:
    """Create a backtest container ID."""
    return ContainerNamingStrategy.generate_container_id(
        ContainerType.BACKTEST,
        phase,
        classifier,
        risk_profile,
        **kwargs
    )


def create_optimization_container_id(
    phase: Phase,
    trial_number: int,
    classifier: ClassifierType = ClassifierType.NONE,
    **kwargs
) -> str:
    """Create an optimization container ID."""
    metadata = kwargs.get('metadata', {})
    metadata['trial'] = trial_number
    kwargs['metadata'] = metadata
    
    return ContainerNamingStrategy.generate_container_id(
        ContainerType.OPTIMIZATION,
        phase,
        classifier,
        metadata=metadata,
        **kwargs
    )


def create_signal_analysis_container_id(
    analysis_type: str = "mae_mfe",
    symbols: Optional[list] = None,
    **kwargs
) -> str:
    """Create a signal analysis container ID."""
    metadata = kwargs.get('metadata', {})
    metadata['analysis'] = analysis_type
    if symbols:
        metadata['symbols'] = symbols
    kwargs['metadata'] = metadata
    
    return ContainerNamingStrategy.generate_container_id(
        ContainerType.SIGNAL_GEN,
        Phase.ANALYSIS,
        metadata=metadata,
        **kwargs
    )




@dataclass
class ContainerConfig:
    """Configuration for a container."""
    role: ContainerRole
    name: str
    container_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=set)
    

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
    
    def __init__(self, config: ContainerConfig):
        """
        Initialize container with configuration.
        
        Args:
            config: Container configuration
        """
        self.config = config
        self.container_id = config.container_id or f"{config.role.value}_{uuid.uuid4().hex[:8]}"
        self.container_type = config.role.value
        
        # Each container gets its own isolated event bus
        self.event_bus = EventBus(container_id=self.container_id)
        
        # State management
        self._state = ContainerState.UNINITIALIZED
        self._state_history: List[tuple[ContainerState, datetime]] = [
            (ContainerState.UNINITIALIZED, datetime.now())
        ]
        
        # Composition: parent/child relationships
        self._parent_container: Optional['Container'] = None
        self._child_containers: Dict[str, 'Container'] = {}
        
        # Component registry
        self._components: Dict[str, ContainerComponent] = {}
        self._component_order: List[str] = []
        
        # Metadata for composition
        self._metadata = ContainerMetadata(
            container_id=self.container_id,
            role=config.role,
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
        
        # Setup event bus tracing if enabled
        if self._should_enable_tracing():
            self._setup_tracing()
        
        # Streaming metrics support (optional)
        self.streaming_metrics: Optional[Any] = None
        
        # Portfolio containers always setup metrics for event tracing
        if self.role == ContainerRole.PORTFOLIO:
            self._setup_event_tracing_metrics()
        elif self._should_track_metrics():
            self._setup_metrics()
        
        logger.info(f"Created container: {self.name} ({self.container_id})")
    
    @property
    def name(self) -> str:
        """Container name."""
        return self.config.name
    
    @property
    def role(self) -> ContainerRole:
        """Container role."""
        return self.config.role
    
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
        self._component_order.append(name)
        
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
            # Initialize all components in order
            for name in self._component_order:
                component = self._components[name]
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
            # Start all components
            for name in self._component_order:
                component = self._components[name]
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
            # Stop all components in reverse order
            for name in reversed(self._component_order):
                component = self._components[name]
                if hasattr(component, 'stop'):
                    logger.debug(f"Stopping component '{name}'")
                    component.stop()
            
            self._set_state(ContainerState.STOPPED)
            logger.info(f"Container {self.name} stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping container {self.name}: {e}")
            self._set_state(ContainerState.ERROR)
            raise
    
    def cleanup(self) -> None:
        """Cleanup resources and dispose of components."""
        # Save results before cleanup if disk storage
        if self.config.config.get('results_storage') == 'disk':
            self._save_results_before_cleanup()
        
        # Ensure stopped first
        if self._state == ContainerState.RUNNING:
            self.stop()
        
        # Cleanup all components
        for name in reversed(self._component_order):
            component = self._components[name]
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up component '{name}': {e}")
        
        # Clear registries
        self._components.clear()
        self._component_order.clear()
        
        logger.info(f"Container {self.name} cleaned up")
    
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
    
    def find_containers_by_role(self, role: ContainerRole) -> List['Container']:
        """Find all nested containers with specific role."""
        containers = []
        
        # Check direct children
        for child in self._child_containers.values():
            if child.role == role:
                containers.append(child)
            
            # Recursively check grandchildren
            containers.extend(child.find_containers_by_role(role))
        
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
        child = Container(config)
        self.add_child_container(child)
        return child
    
    # Event handling methods for route integration
    
    def receive_event(self, event: Any) -> None:
        """
        Receive an event from an route.
        
        Args:
            event: Event to process
        """
        self._metrics['events_processed'] += 1
        self._metrics['last_activity'] = datetime.now()
        
        # Publish to internal event bus for components to handle
        self.event_bus.publish(event)
    
    # Event Processing and Status methods
    
    def process_event(self, event: Any) -> Optional[Any]:
        """Process incoming event and optionally return response."""
        self.receive_event(event)
        # For now, no response processing - can be enhanced later
        return None
    
    def publish_event(self, event: Any, target_scope: str = "local") -> None:
        """Publish event to specified scope.
        
        Scopes:
        - local: Only this container's event bus (default)
        - parent: Direct parent container (for upward propagation)
        
        All other communication patterns should use routes.
        """
        self._metrics['events_published'] += 1
        self._metrics['last_activity'] = datetime.now()
        
        if target_scope == "local":
            self.event_bus.publish(event)
        elif target_scope == "parent" and self._parent_container:
            self._parent_container.receive_event(event)
        else:
            # Default to local if invalid scope or no parent
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
        if self.role == ContainerRole.PORTFOLIO:
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
        Setup event-based metrics tracking.
        
        Creates a MetricsEventTracer that processes events to calculate
        metrics with smart retention policies for memory efficiency.
        """
        try:
            from .metrics import MetricsEventTracer
        except ImportError:
            logger.warning(f"MetricsEventTracer not available for container {self.container_id}")
            return
        
        # Get configuration
        metrics_config = self.config.config.get('metrics', {})
        results_config = self.config.config.get('results', {})
        
        if self.role == ContainerRole.PORTFOLIO:
            # Build config for metrics tracer
            tracer_config = {
                'initial_capital': self.config.config.get('initial_capital', 100000.0),
                'retention_policy': results_config.get('retention_policy', 'trade_complete'),
                'max_events': results_config.get('max_events', 1000),
                'collection': results_config.get('collection', {}),
                'annualization_factor': metrics_config.get('annualization_factor', 252.0),
                'min_periods': metrics_config.get('min_periods', 20),
                'objective_function': self.config.config.get('objective_function', {'name': 'sharpe_ratio'}),
                'custom_metrics': self.config.config.get('custom_metrics', [])
            }
            
            # Create metrics event tracer
            self.streaming_metrics = MetricsEventTracer(tracer_config)
            
            # Subscribe to all relevant events
            from ..events import EventType
            event_types = [
                EventType.ORDER_REQUEST,
                EventType.ORDER,
                EventType.FILL,
                EventType.PORTFOLIO_UPDATE,
                EventType.POSITION_UPDATE
            ]
            
            for event_type in event_types:
                self.event_bus.subscribe(
                    event_type,
                    self.streaming_metrics.trace_event
                )
            
            logger.info(f"Event-based metrics tracking enabled for portfolio container {self.container_id}")
        else:
            # Non-portfolio containers don't need metrics by default
            logger.debug(f"No metrics tracking for {self.role.value} container {self.container_id}")
    
    def _setup_event_tracing_metrics(self):
        """
        Setup event tracing as the metrics system for portfolio containers.
        
        This implements the unified approach where event tracing IS the metrics system.
        Configuration comes from workflow-defined settings with user overrides.
        """
        try:
            from .metrics import MetricsEventTracer
        except ImportError:
            logger.warning(f"MetricsEventTracer not available for container {self.container_id}")
            return
        
        # Get configuration from workflow-defined settings  
        event_tracing = self.config.config.get('event_tracing', ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL'])
        retention_policy = self.config.config.get('retention_policy', 'trade_complete')
        sliding_window_size = self.config.config.get('sliding_window_size', 1000)
        
        # Determine if we should store equity curve based on events being traced
        store_equity_curve = 'PORTFOLIO_UPDATE' in event_tracing
        
        # Setup metrics tracking
        metrics_config = {
            'initial_capital': self.config.config.get('initial_capital', 100000.0),
            'retention_policy': retention_policy,
            'max_events': sliding_window_size if retention_policy == 'sliding_window' else 10000,
            'collection': {
                'store_equity_curve': store_equity_curve,
                'store_trades': True
            },
            'objective_function': self.config.config.get('objective_function', {'name': 'sharpe_ratio'})
        }
        
        self.streaming_metrics = MetricsEventTracer(metrics_config)
        
        # Subscribe to specified events
        from ..events import EventType
        for event_type_str in event_tracing:
            try:
                event_type = EventType[event_type_str]
                self.event_bus.subscribe(event_type, self.streaming_metrics.trace_event)
            except KeyError:
                logger.warning(f"Unknown event type: {event_type_str}")
        
        logger.info(f"Event tracing metrics enabled for {self.container_id} - "
                   f"events: {event_tracing}, retention: {retention_policy}")
    
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
        
        # Add container metadata
        metrics['container_id'] = self.container_id
        metrics['container_type'] = self.container_type
        metrics['role'] = self.role.value
        
        # Add basic operational metrics
        metrics['events_processed'] = self._metrics['events_processed']
        metrics['events_published'] = self._metrics['events_published']
        
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
        
        # For portfolio containers using event tracer
        if self.role == ContainerRole.PORTFOLIO and hasattr(self.streaming_metrics, 'get_results'):
            return self.streaming_metrics.get_results()
        
        # Fallback to basic metrics
        return self.get_metrics()
    
    def _save_results_before_cleanup(self) -> None:
        """Save results during container destruction."""
        if not self.streaming_metrics:
            return
            
        results = self.streaming_metrics.get_results()
        
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