"""
Container protocols for ADMF-PC using typing.Protocol (NO ABC inheritance).

This module defines pure behavioral contracts for container components,
following the Protocol+Composition architecture.
"""

from typing import Protocol, runtime_checkable, Dict, Any, Optional, List, Set, Tuple
from datetime import datetime
from enum import Enum

from .types import ContainerComponent, ContainerConfigDict


class ContainerState(Enum):
    """Container lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DISPOSED = "disposed"


class ContainerRole(Enum):
    """Container roles in the system."""
    PORTFOLIO = "portfolio"
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"
    DATA = "data"
    CLASSIFIER = "classifier"
    INDICATOR = "indicator"
    ANALYTICS = "analytics"


@runtime_checkable
class ContainerLifecycleProtocol(Protocol):
    """Protocol for container lifecycle management - NO INHERITANCE!"""
    
    def initialize(self) -> None:
        """Initialize container and components."""
        ...
    
    def start(self) -> None:
        """Start container and begin processing."""
        ...
    
    def stop(self) -> None:
        """Stop container gracefully."""
        ...
    
    def dispose(self) -> None:
        """Dispose container and cleanup resources."""
        ...
    
    @property
    def state(self) -> ContainerState:
        """Get current container state."""
        ...


@runtime_checkable
class ComponentRegistryProtocol(Protocol):
    """Protocol for component registry management - NO INHERITANCE!"""
    
    def add_component(self, name: str, component: ContainerComponent) -> None:
        """Add component to registry."""
        ...
    
    def get_component(self, name: str) -> Optional[ContainerComponent]:
        """Get component by name."""
        ...
    
    def remove_component(self, name: str) -> Optional[ContainerComponent]:
        """Remove component from registry."""
        ...
    
    def list_components(self) -> List[str]:
        """List all component names."""
        ...
    
    def has_component(self, name: str) -> bool:
        """Check if component exists."""
        ...


@runtime_checkable
class StateManagerProtocol(Protocol):
    """Protocol for container state management - NO INHERITANCE!"""
    
    def transition_to(self, new_state: ContainerState) -> bool:
        """Transition to new state if valid."""
        ...
    
    def can_transition_to(self, new_state: ContainerState) -> bool:
        """Check if transition is valid."""
        ...
    
    @property
    def current_state(self) -> ContainerState:
        """Get current state."""
        ...
    
    @property
    def state_history(self) -> List[tuple[ContainerState, datetime]]:
        """Get state transition history."""
        ...


@runtime_checkable
class ContainerCompositionProtocol(Protocol):
    """Protocol for parent/child container relationships - NO INHERITANCE!"""
    
    def add_child(self, child_id: str, child: 'ContainerProtocol') -> None:
        """Add child container."""
        ...
    
    def remove_child(self, child_id: str) -> Optional['ContainerProtocol']:
        """Remove child container."""
        ...
    
    def get_child(self, child_id: str) -> Optional['ContainerProtocol']:
        """Get child container by ID."""
        ...
    
    def list_children(self) -> List[str]:
        """List all child container IDs."""
        ...
    
    def set_parent(self, parent: Optional['ContainerProtocol']) -> None:
        """Set parent container."""
        ...
    
    @property
    def parent(self) -> Optional['ContainerProtocol']:
        """Get parent container."""
        ...


@runtime_checkable
class ContainerEventProtocol(Protocol):
    """Protocol for container event handling - NO INHERITANCE!"""
    
    def process_event(self, event: Any) -> Optional[Any]:
        """Process incoming event."""
        ...
    
    def publish_event(self, event: Any, scope: str = "local") -> None:
        """Publish event to specified scope."""
        ...
    
    def setup_event_bus(self, bus_id: str) -> None:
        """Setup isolated event bus."""
        ...


@runtime_checkable
class ContainerMetricsProtocol(Protocol):
    """Protocol for container metrics tracking - NO INHERITANCE!"""
    
    def record_event_processed(self) -> None:
        """Record that an event was processed."""
        ...
    
    def record_event_published(self) -> None:
        """Record that an event was published."""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        ...
    
    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        ...


@runtime_checkable
class ContainerTracingProtocol(Protocol):
    """Protocol for container event tracing - NO INHERITANCE!"""
    
    def start_tracing(self) -> None:
        """Start event tracing."""
        ...
    
    def stop_tracing(self) -> None:
        """Stop event tracing."""
        ...
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get trace summary."""
        ...
    
    def save_trace(self, filepath: str) -> None:
        """Save trace to file."""
        ...


@runtime_checkable
class ContainerConfigProtocol(Protocol):
    """Protocol for container configuration management - NO INHERITANCE!"""
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        ...
    
    def get_config(self, key: str = None) -> Any:
        """Get configuration value or entire config."""
        ...
    
    def get_capabilities(self) -> Set[str]:
        """Get container capabilities."""
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get container metadata."""
        ...


# Main container protocol that composes all the above
@runtime_checkable
class ContainerProtocol(Protocol):
    """Main container protocol - composed of specialized protocols."""
    
    # Basic properties
    container_id: str
    name: str
    role: ContainerRole
    
    # Lifecycle
    def initialize(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def dispose(self) -> None: ...
    
    # Component management
    def add_component(self, name: str, component: ContainerComponent) -> None: ...
    def get_component(self, name: str) -> Optional[ContainerComponent]: ...
    
    # Event handling
    def process_event(self, event: Any) -> Optional[Any]: ...
    def publish_event(self, event: Any, scope: str = "local") -> None: ...
    
    # Status and metrics
    def get_status(self) -> Dict[str, Any]: ...
    def get_metrics(self) -> Optional[Dict[str, Any]]: ...
    
    # State access
    @property
    def state(self) -> ContainerState: ...


# Pure composition data structures - NO INHERITANCE, just data holders
from dataclasses import dataclass, field

@dataclass
class ContainerMetadata:
    """Container metadata - pure data holder, no inheritance."""
    container_id: str
    role: ContainerRole
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    capabilities: Set[str] = field(default_factory=set)


@dataclass
class ContainerLimits:
    """Resource limits - pure data holder, no inheritance."""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_events_per_second: Optional[int] = None
    max_components: Optional[int] = None
    timeout_seconds: Optional[int] = None


# Strategy calling protocol

@runtime_checkable
class StrategyCallerProtocol(Protocol):
    """Protocol for containers that execute strategies with features."""
    
    def handle_bar_for_strategies(self, event: Any) -> None:
        """Handle BAR event by updating features and calling strategies."""
        ...
    
    def add_strategy(self, strategy_id: str, strategy_func: Any, parameters: Dict[str, Any]) -> None:
        """Add a strategy function to be called."""
        ...


# Order tracking protocols for duplicate prevention

@runtime_checkable
class OrderTrackingProtocol(Protocol):
    """Protocol for tracking pending orders by symbol/timeframe."""
    
    def has_pending_orders(self, container_id: str, symbol: str, timeframe: str) -> bool:
        """Check if container has pending orders for symbol/timeframe."""
        ...
    
    def register_order(self, container_id: str, symbol: str, timeframe: str, 
                      order_id: str, metadata: Optional[Dict] = None) -> None:
        """Register new pending order."""
        ...
    
    def clear_order(self, container_id: str, symbol: str, timeframe: str, order_id: str) -> bool:
        """Clear completed order."""
        ...
    
    def get_pending_summary(self, container_id: str) -> Dict[str, int]:
        """Get summary of pending orders by symbol_timeframe."""
        ...