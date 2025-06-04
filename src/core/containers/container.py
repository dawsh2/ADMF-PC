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
from .protocols import ComposableContainer, ContainerMetadata, ContainerLimits, ContainerState, ContainerRole

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
    
    Implements ComposableContainer protocol for the arch-101.md architecture:
    - Protocol + Composition (no inheritance)
    - Isolated event buses for Data, Strategy, Risk, Execution modules
    - Hierarchical composition with parent/child relationships
    - Pluggable adapter communication
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
        self._components: Dict[str, Any] = {}
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
    
    def add_component(self, name: str, component: Any) -> None:
        """
        Add a component to the container.
        
        Args:
            name: Component name
            component: Component instance
        """
        if name in self._components:
            raise ValueError(f"Component '{name}' already exists")
        
        self._components[name] = component
        self._component_order.append(name)
        
        # If component needs event bus access, provide it
        if hasattr(component, 'set_event_bus'):
            component.set_event_bus(self.event_bus)
        
        logger.debug(f"Added component '{name}' to {self.name}")
    
    def get_component(self, name: str) -> Optional[Any]:
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
            raise ValueError(f"Component '{component_name}' not found")
            
        for attr_name, dep_name in dependencies.items():
            dep_component = self.get_component(dep_name)
            if not dep_component:
                raise ValueError(f"Dependency '{dep_name}' not found")
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
    
    async def initialize(self) -> None:
        """Initialize the container and all components."""
        if self._state != ContainerState.CREATED:
            logger.warning(f"Container {self.name} already initialized")
            return
        
        self._set_state(ContainerState.INITIALIZING)
        
        try:
            # Initialize all components in order
            for name in self._component_order:
                component = self._components[name]
                if hasattr(component, 'initialize'):
                    logger.debug(f"Initializing component '{name}'")
                    await component.initialize()
            
            self._set_state(ContainerState.INITIALIZED)
            logger.info(f"Container {self.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize container {self.name}: {e}")
            self._set_state(ContainerState.ERROR)
            raise
    
    async def start(self) -> None:
        """Start the container and begin processing."""
        if self._state != ContainerState.INITIALIZED:
            raise RuntimeError(f"Container {self.name} not initialized")
        
        self._set_state(ContainerState.STARTING)
        
        try:
            # Start all components
            for name in self._component_order:
                component = self._components[name]
                if hasattr(component, 'start'):
                    logger.debug(f"Starting component '{name}'")
                    await component.start()
            
            self._metrics['start_time'] = datetime.now()
            self._set_state(ContainerState.RUNNING)
            logger.info(f"Container {self.name} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start container {self.name}: {e}")
            self._set_state(ContainerState.ERROR)
            raise
    
    async def stop(self) -> None:
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
                    await component.stop()
            
            self._set_state(ContainerState.STOPPED)
            logger.info(f"Container {self.name} stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping container {self.name}: {e}")
            self._set_state(ContainerState.ERROR)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup resources and dispose of components."""
        # Ensure stopped first
        if self._state == ContainerState.RUNNING:
            await self.stop()
        
        # Cleanup all components
        for name in reversed(self._component_order):
            component = self._components[name]
            if hasattr(component, 'cleanup'):
                try:
                    await component.cleanup()
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
    
    # Composition Management (ComposableContainer protocol)
    
    def add_child_container(self, child: 'Container') -> None:
        """Add a child container."""
        if child.container_id in self._child_containers:
            raise ValueError(f"Child container {child.container_id} already exists")
        
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
    
    # Protocol methods for compatibility with existing systems
    
    def create_subcontainer(self, container_id: str, container_type: str = "generic") -> 'Container':
        """
        Create a subcontainer (backward compatibility).
        
        For new code, use add_child_container instead.
        """
        logger.warning(
            f"create_subcontainer called on {self.name} - "
            "consider using add_child_container for new code"
        )
        # Create and add as child
        child = Container(ContainerConfig(
            role=ContainerRole.BACKTEST,  # Default role
            name=f"{container_type}_container",
            container_id=container_id
        ))
        self.add_child_container(child)
        return child
    
    def get_subcontainers(self) -> List['Container']:
        """Get subcontainers (backward compatibility)."""
        return self.child_containers
    
    def get_subcontainers_by_type(self, container_type: str) -> List['Container']:
        """Get subcontainers by type (backward compatibility)."""
        return [child for child in self.child_containers if child.container_type == container_type]
    
    # Event handling methods for adapter integration
    
    def receive_event(self, event: Any) -> None:
        """
        Receive an event from an adapter.
        
        Args:
            event: Event to process
        """
        self._metrics['events_processed'] += 1
        self._metrics['last_activity'] = datetime.now()
        
        # Publish to internal event bus for components to handle
        self.event_bus.publish(event)
    
    # ComposableContainer protocol methods
    
    async def process_event(self, event: Any) -> Optional[Any]:
        """Process incoming event and optionally return response."""
        self.receive_event(event)
        # For now, no response processing - can be enhanced later
        return None
    
    def publish_event(self, event: Any, target_scope: str = "local") -> None:
        """Publish event to specified scope (local, parent, children, broadcast)."""
        self._metrics['events_published'] += 1
        self._metrics['last_activity'] = datetime.now()
        
        if target_scope == "local":
            self.event_bus.publish(event)
        elif target_scope == "parent" and self._parent_container:
            self._parent_container.receive_event(event)
        elif target_scope == "children":
            for child in self._child_containers.values():
                child.receive_event(event)
        elif target_scope == "broadcast":
            # Broadcast to all: self, parent, children
            self.event_bus.publish(event)
            if self._parent_container:
                self._parent_container.receive_event(event)
            for child in self._child_containers.values():
                child.receive_event(event)
        else:
            # Default to local
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
    
    async def dispose(self) -> None:
        """Clean up container resources."""
        # Dispose children first
        for child in list(self._child_containers.values()):
            await child.dispose()
        
        # Remove from parent
        if self._parent_container:
            self._parent_container.remove_child_container(self.container_id)
        
        # Clean up self
        await self.cleanup()
    
    def __repr__(self) -> str:
        return f"Container(name={self.name}, id={self.container_id}, state={self._state.value})"