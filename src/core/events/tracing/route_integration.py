"""
Integration of event tracing with the route communication architecture.
This module ensures event tracing works seamlessly with isolated containers
and route-based communication patterns.
"""
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from datetime import datetime

from src.core.events.tracing.event_store import EventStore
from src.core.events.tracing.traced_event import TracedEvent
from src.core.events.event_bus import Event
from src.core.containers.protocols import Container


@dataclass
class TracingConfig:
    """Configuration for event tracing in routes."""
    enabled: bool = True
    store_all_events: bool = False  # False = only store important events
    important_event_types: Set[str] = None
    batch_size: int = 100
    batch_timeout_ms: int = 50
    include_route_metadata: bool = True
    
    def __post_init__(self):
        if self.important_event_types is None:
            # Default important events for trading systems
            self.important_event_types = {
                "TradingSignal", "OrderEvent", "FillEvent", 
                "RiskCheckEvent", "PortfolioUpdateEvent",
                "RejectionEvent", "ErrorEvent"
            }


class TracingRouteMixin:
    """
    Mixin for routes to add event tracing capabilities.
    This respects container isolation while enabling cross-container tracing.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracing_config = TracingConfig(**self.config.get('tracing', {}))
        self.event_store: Optional[EventStore] = None
        self._event_batch: List[TracedEvent] = []
        self._batch_timer = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def setup_tracing(self, event_store: EventStore) -> None:
        """Initialize tracing with the provided event store."""
        self.event_store = event_store
        self.logger.info(f"Tracing enabled for route {self.name}")
        
    def trace_event(
        self, 
        event: Event, 
        source: Container,
        target: Optional[Container] = None,
        route_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Trace an event as it flows through the route.
        Maintains container isolation while enabling correlation tracking.
        """
        if not self.tracing_config.enabled or not self.event_store:
            return
            
        # Check if we should trace this event
        if not self.tracing_config.store_all_events:
            event_type = getattr(event, '__class__', type(event)).__name__
            if event_type not in self.tracing_config.important_event_types:
                return
                
        # Create traced event
        traced_event = self._create_traced_event(
            event, source, target, route_metadata
        )
        
        # Add to batch
        self._event_batch.append(traced_event)
        
        # Flush if batch is full
        if len(self._event_batch) >= self.tracing_config.batch_size:
            self._flush_batch()
        elif not self._batch_timer:
            # Start timer for batch timeout
            self._start_batch_timer()
            
    def _create_traced_event(
        self,
        event: Event,
        source: Container,
        target: Optional[Container] = None,
        route_metadata: Optional[Dict[str, Any]] = None
    ) -> TracedEvent:
        """Create a traced event with full context."""
        # Extract event metadata
        event_id = getattr(event, 'event_id', None) or self._generate_event_id()
        correlation_id = getattr(event, 'correlation_id', None) or event_id
        causation_id = getattr(event, 'causation_id', None)
        
        # Build metadata
        metadata = {
            'route_name': self.name,
            'route_type': self.__class__.__name__,
            'source_container_type': source.__class__.__name__,
        }
        
        if target:
            metadata['target_container_type'] = target.__class__.__name__
            
        if self.tracing_config.include_route_metadata and route_metadata:
            metadata['route_context'] = route_metadata
            
        # Add execution context if available
        if hasattr(source, 'execution_context'):
            metadata['execution_mode'] = source.execution_context.mode
            metadata['optimization_run'] = source.execution_context.optimization_run_id
            
        # Create traced event
        return TracedEvent(
            event_id=event_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            source_container=source.name,
            target_container=target.name if target else None,
            event_type=type(event).__name__,
            timestamp=datetime.now(),
            data=self._serialize_event_data(event),
            metadata=metadata
        )
        
    def _serialize_event_data(self, event: Event) -> Dict[str, Any]:
        """Serialize event data for storage."""
        # Handle different event types
        if hasattr(event, 'to_dict'):
            return event.to_dict()
        elif hasattr(event, '__dict__'):
            # Filter out private attributes
            return {
                k: v for k, v in event.__dict__.items() 
                if not k.startswith('_') and self._is_serializable(v)
            }
        else:
            return {'event': str(event)}
            
    def _is_serializable(self, value: Any) -> bool:
        """Check if a value can be serialized."""
        # Basic types that JSON can handle
        return isinstance(value, (str, int, float, bool, list, dict, type(None)))
        
    def _flush_batch(self) -> None:
        """Flush the current batch of events to storage."""
        if self._event_batch and self.event_store:
            try:
                self.event_store.store_batch(self._event_batch)
                self.logger.debug(f"Flushed {len(self._event_batch)} traced events")
                self._event_batch = []
            except Exception as e:
                self.logger.error(f"Failed to flush event batch: {e}")
                
        self._cancel_batch_timer()
        
    def _start_batch_timer(self) -> None:
        """Start timer for batch timeout."""
        # Implementation depends on event loop (asyncio, threading, etc.)
        pass
        
    def _cancel_batch_timer(self) -> None:
        """Cancel batch timer."""
        if self._batch_timer:
            # Cancel timer
            self._batch_timer = None
            
    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        import uuid
        return str(uuid.uuid4())


class TracingPipelineRoute(TracingRouteMixin):
    """
    Example of a pipeline route with integrated tracing.
    Shows how tracing works with the pipeline communication pattern.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        # Initialize both mixins
        TracingRouteMixin.__init__(self, name=name, config=config)
        self.name = name
        self.config = config
        self.containers = config.get('containers', [])
        self.connections = []
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Setup pipeline connections."""
        for i in range(len(self.containers) - 1):
            source_name = self.containers[i]
            target_name = self.containers[i + 1]
            
            if source_name in containers and target_name in containers:
                source = containers[source_name]
                target = containers[target_name]
                self.connections.append((source, target))
                
                # Subscribe to source events
                source.event_bus.subscribe(
                    lambda e, s=source, t=target: self._forward_event(e, s, t)
                )
                
    def _forward_event(self, event: Event, source: Container, target: Container) -> None:
        """Forward event through pipeline with tracing."""
        # Trace before forwarding
        self.trace_event(
            event, 
            source, 
            target,
            route_metadata={
                'pipeline_position': self.containers.index(source.name),
                'pipeline_length': len(self.containers)
            }
        )
        
        # Forward to target
        target.receive_event(event)
        
    def start(self) -> None:
        """Start the route."""
        self.logger.info(f"Started tracing pipeline route: {self.name}")
        
    def stop(self) -> None:
        """Stop the route and flush any pending traces."""
        self._flush_batch()
        self.logger.info(f"Stopped tracing pipeline route: {self.name}")


class TracingBroadcastRoute(TracingRouteMixin):
    """
    Broadcast route with tracing support.
    Shows how tracing handles one-to-many communication patterns.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        TracingRouteMixin.__init__(self, name=name, config=config)
        self.name = name
        self.config = config
        self.source_name = config.get('source')
        self.target_names = config.get('targets', [])
        self.source_container: Optional[Container] = None
        self.target_containers: List[Container] = []
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Setup broadcast connections."""
        if self.source_name in containers:
            self.source_container = containers[self.source_name]
            
            # Setup targets
            for target_name in self.target_names:
                if target_name in containers:
                    self.target_containers.append(containers[target_name])
                    
            # Subscribe to source
            self.source_container.event_bus.subscribe(
                lambda e: self._broadcast_event(e)
            )
            
    def _broadcast_event(self, event: Event) -> None:
        """Broadcast event to all targets with tracing."""
        for i, target in enumerate(self.target_containers):
            # Trace each broadcast
            self.trace_event(
                event,
                self.source_container,
                target,
                route_metadata={
                    'broadcast_index': i,
                    'total_targets': len(self.target_containers),
                    'target_list': self.target_names
                }
            )
            
            # Send to target
            target.receive_event(event)
            
    def start(self) -> None:
        """Start the route."""
        self.logger.info(f"Started tracing broadcast route: {self.name}")
        
    def stop(self) -> None:
        """Stop the route."""
        self._flush_batch()
        self.logger.info(f"Stopped tracing broadcast route: {self.name}")


class ContainerIsolationTracer:
    """
    Ensures event tracing respects container isolation boundaries.
    This is crucial for optimization runs where containers must not interfere.
    """
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.isolation_contexts: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.ContainerIsolationTracer")
        
    def register_isolation_context(
        self, 
        container_name: str,
        context: Dict[str, Any]
    ) -> None:
        """Register an isolation context for a container."""
        self.isolation_contexts[container_name] = context
        self.logger.debug(
            f"Registered isolation context for {container_name}: {context}"
        )
        
    def validate_event_isolation(
        self,
        event: TracedEvent
    ) -> bool:
        """
        Validate that an event respects isolation boundaries.
        Returns True if valid, False if isolation violated.
        """
        source_context = self.isolation_contexts.get(event.source_container, {})
        target_context = self.isolation_contexts.get(event.target_container, {})
        
        # Check if containers are in same isolation group
        source_group = source_context.get('isolation_group')
        target_group = target_context.get('isolation_group')
        
        if source_group and target_group and source_group != target_group:
            # Different isolation groups - check if communication allowed
            allowed_cross_group = source_context.get('allowed_targets', [])
            if event.target_container not in allowed_cross_group:
                self.logger.warning(
                    f"Isolation violation: {event.source_container} "
                    f"(group: {source_group}) -> {event.target_container} "
                    f"(group: {target_group})"
                )
                return False
                
        return True
        
    def get_isolated_events(
        self,
        container_name: str,
        include_cross_boundary: bool = False
    ) -> List[TracedEvent]:
        """
        Get events for a specific container respecting isolation.
        
        Args:
            container_name: The container to get events for
            include_cross_boundary: If True, include events that cross
                                  isolation boundaries (for debugging)
        """
        events = list(self.event_store.query_events(
            source_container=container_name
        ))
        
        if not include_cross_boundary:
            # Filter out events that cross isolation boundaries
            context = self.isolation_contexts.get(container_name, {})
            isolation_group = context.get('isolation_group')
            
            if isolation_group:
                filtered_events = []
                for event in events:
                    if event.target_container:
                        target_context = self.isolation_contexts.get(
                            event.target_container, {}
                        )
                        target_group = target_context.get('isolation_group')
                        
                        # Only include if same group or explicitly allowed
                        if (target_group == isolation_group or 
                            event.target_container in context.get('allowed_targets', [])):
                            filtered_events.append(event)
                    else:
                        # No target - include it
                        filtered_events.append(event)
                        
                events = filtered_events
                
        return events


# Integration with existing route factory
def create_tracing_route(
    route_type: str,
    name: str,
    config: Dict[str, Any],
    event_store: Optional[EventStore] = None
) -> CommunicationRoute:
    """
    Factory function to create routes with tracing support.
    This would be integrated into the main RouteFactory.
    """
    # Map route types to tracing implementations
    tracing_routes = {
        'pipeline': TracingPipelineRoute,
        'broadcast': TracingBroadcastRoute,
        # Add more as needed
    }
    
    route_class = tracing_routes.get(route_type)
    if not route_class:
        raise ValueError(f"Unknown route type for tracing: {route_type}")
        
    # Create route
    route = route_class(name, config)
    
    # Setup tracing if event store provided
    if event_store and config.get('tracing', {}).get('enabled', True):
        route.setup_tracing(event_store)
        
    return route