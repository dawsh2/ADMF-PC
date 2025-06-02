"""
Event routing protocols for cross-container communication.

This module defines the core protocols and data structures that enable
flexible event routing between containers while maintaining isolation.
"""

from typing import Protocol, Dict, Set, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..types import Event, EventType


class EventQoS(Enum):
    """
    Quality of Service levels for event delivery.
    
    - BEST_EFFORT: Fire and forget, no delivery guarantees
    - GUARANTEED_DELIVERY: Ensure delivery with retries
    - ORDERED_DELIVERY: Maintain event ordering per source
    """
    BEST_EFFORT = "best_effort"
    GUARANTEED_DELIVERY = "guaranteed_delivery"
    ORDERED_DELIVERY = "ordered_delivery"


class EventScope(Enum):
    """
    Event visibility scopes for routing.
    
    - LOCAL: Only within the container's own event bus
    - SIBLINGS: Containers with the same parent
    - PARENT: Direct parent container only
    - CHILDREN: Direct child containers only
    - GLOBAL: All containers in the workflow
    - UPWARD: Parent and all ancestors
    - DOWNWARD: Children and all descendants
    """
    LOCAL = "local"
    SIBLINGS = "siblings"
    PARENT = "parent"
    CHILDREN = "children"
    GLOBAL = "global"
    UPWARD = "upward"
    DOWNWARD = "downward"


@dataclass
class EventFilter:
    """
    Filter criteria for event subscriptions.
    
    Allows subscribers to receive only events matching specific criteria,
    reducing processing overhead and improving performance.
    
    Examples:
        # Filter by symbol
        EventFilter(attributes={"symbols": ["AAPL", "GOOGL"]})
        
        # Filter by threshold
        EventFilter(attributes={"confidence": ">0.7"})
        
        # Complex expression
        EventFilter(expression="price > 100 AND volume > 1000000")
    """
    attributes: Dict[str, Any] = field(default_factory=dict)
    expression: Optional[str] = None
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria."""
        # TODO: Implement filter matching logic
        return True


@dataclass
class BatchingConfig:
    """
    Configuration for event batching.
    
    Enables efficient processing of high-frequency events by
    batching them together for delivery.
    """
    mode: str = "time_window"  # "time_window", "count_based", "hybrid"
    window_ms: int = 100
    max_batch_size: int = 50
    flush_on_priority: bool = True


@dataclass
class EventPublication:
    """
    Declaration of events a container publishes.
    
    Defines what events a container will emit, their scope,
    quality of service requirements, and any pre-filtering.
    """
    events: Set[Union[EventType, str]]
    scope: EventScope = EventScope.PARENT
    qos: EventQoS = EventQoS.BEST_EFFORT
    filters: Optional[EventFilter] = None
    priority: int = 0  # Higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSubscription:
    """
    Declaration of events a container subscribes to.
    
    Defines what events a container wants to receive, from which
    sources, with optional filtering and transformation.
    """
    source: str  # Container ID or pattern (e.g., "indicator_*")
    events: Set[Union[EventType, str]]
    filters: Optional[EventFilter] = None
    transform: Optional[str] = None
    batching: Optional[BatchingConfig] = None
    handler: Optional[Callable[[Event, str], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    Result of topology validation.
    
    Contains validation status, any errors or warnings found,
    and optionally the validated topology graph.
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    topology_graph: Optional[Dict[str, Any]] = None
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)


class EventRouterProtocol(Protocol):
    """
    Protocol for cross-container event routing.
    
    Defines the interface that event routers must implement to enable
    cross-container communication while maintaining isolation.
    """
    
    def register_publisher(
        self, 
        container_id: str, 
        publications: List[EventPublication]
    ) -> None:
        """
        Register container as event publisher.
        
        Args:
            container_id: Unique identifier of the publishing container
            publications: List of event publication declarations
        """
        ...
        
    def register_subscriber(
        self, 
        container_id: str, 
        subscriptions: List[EventSubscription],
        handler: Callable[[Event, str], None]
    ) -> None:
        """
        Register container as event subscriber with handler.
        
        Args:
            container_id: Unique identifier of the subscribing container
            subscriptions: List of event subscription declarations
            handler: Function to call when matching events are received
        """
        ...
        
    def unregister_container(self, container_id: str) -> None:
        """
        Unregister container from all publications and subscriptions.
        
        Args:
            container_id: Container to unregister
        """
        ...
        
    def route_event(
        self, 
        source_id: str, 
        event: Event,
        scope: Optional[EventScope] = None
    ) -> None:
        """
        Route event from source to subscribers.
        
        Args:
            source_id: ID of the container publishing the event
            event: The event to route
            scope: Override the default scope for this event
        """
        ...
        
    def validate_topology(self) -> ValidationResult:
        """
        Validate complete event topology.
        
        Checks for:
        - Subscription cycles that could cause infinite loops
        - Missing publishers for subscribed events
        - Orphaned containers with no connections
        - Conflicting QoS requirements
        
        Returns:
            ValidationResult with status and any issues found
        """
        ...
        
    def calculate_startup_order(
        self, 
        containers: Dict[str, Any]
    ) -> List[str]:
        """
        Calculate container startup order based on event dependencies.
        
        Ensures containers are started in an order that satisfies
        event dependencies (publishers before subscribers).
        
        Args:
            containers: Dictionary of container_id -> container
            
        Returns:
            Ordered list of container IDs for startup
        """
        ...
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get routing metrics for monitoring.
        
        Returns:
            Dictionary containing:
            - Total events routed
            - Events per source/type
            - Delivery failures
            - Average routing latency
            - Active subscriptions
        """
        ...
        
    def get_topology(self) -> Dict[str, Any]:
        """
        Get current event topology.
        
        Returns:
            Dictionary representing the event flow graph
        """
        ...
        
    def enable_debugging(self, enabled: bool = True) -> None:
        """
        Enable or disable debugging mode.
        
        When enabled, the router will:
        - Log all event routing decisions
        - Track event paths for visualization
        - Collect detailed timing metrics
        
        Args:
            enabled: Whether to enable debugging
        """
        ...