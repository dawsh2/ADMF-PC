"""
Route composition utilities and standard implementations.

This module provides standard implementations of route protocols and
utility functions for composing routes, following ADMF-PC's composition principle.
"""

import logging
import time
from typing import Dict, Any, List, Callable
from contextlib import contextmanager

from ..types.events import Event, EventType
from .protocols import Container, CommunicationRoute, RouteMetrics, RouteErrorHandler


# Standard Protocol Implementations

class StandardRouteMetrics:
    """Standard implementation of RouteMetrics protocol.
    
    Provides basic metrics collection for any route.
    """
    
    def __init__(self, route_name: str):
        self.route_name = route_name
        self.success_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.event_count = 0
        
    def increment_success(self) -> None:
        """Increment successful event counter."""
        self.success_count += 1
        
    def increment_error(self) -> None:
        """Increment error counter."""
        self.error_count += 1
        
    @contextmanager
    def measure_latency(self):
        """Context manager for measuring latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.total_latency += duration
            self.event_count += 1
    
    def get_average_latency(self) -> float:
        """Get average event processing latency."""
        if self.event_count == 0:
            return 0.0
        return self.total_latency / self.event_count


class StandardRouteErrorHandler:
    """Standard implementation of RouteErrorHandler protocol.
    
    Logs errors and maintains error count.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_count = 0
        
    def handle(self, event: Event, error: Exception) -> None:
        """Handle an error that occurred during event processing."""
        self.error_count += 1
        self.logger.error(
            f"Error processing event {event.event_type}: {error}",
            exc_info=True,
            extra={
                'event_type': event.event_type,
                'event_source': event.source_id,
                'error_type': type(error).__name__
            }
        )


# Route Composition Functions

def compose_route_with_infrastructure(route_class: type, name: str, config: Dict[str, Any]) -> Any:
    """Compose a route with standard infrastructure.
    
    This function creates a route instance and attaches standard
    infrastructure components using composition.
    
    Args:
        route_class: The route class to instantiate
        name: Route name
        config: Route configuration
        
    Returns:
        Route instance with infrastructure attached
    """
    route = route_class(name, config)
    
    # Compose with infrastructure
    route.metrics = StandardRouteMetrics(name)
    route.logger = logging.getLogger(f"route.{name}")
    route.error_handler = StandardRouteErrorHandler(route.logger)
    
    return route


def wrap_with_metrics(handler: Callable, route: Any) -> Callable:
    """Wrap an event handler with metrics and error handling.
    
    Args:
        handler: The handler function to wrap
        route: Route instance with metrics and error_handler
        
    Returns:
        Wrapped handler function
    """
    def wrapped(event: Event, source: Container = None) -> None:
        if not hasattr(route, 'metrics'):
            # No metrics available, call handler directly
            handler(event, source) if source else handler(event)
            return
        
        with route.metrics.measure_latency():
            try:
                handler(event, source) if source else handler(event)
                route.metrics.increment_success()
            except Exception as e:
                route.metrics.increment_error()
                if hasattr(route, 'error_handler'):
                    route.error_handler.handle(event, e)
                else:
                    raise
    
    return wrapped


# Subscription Utilities

def create_subscription(source: Container, event_type: EventType, handler: Callable) -> bool:
    """Create event subscription between container and handler.
    
    Args:
        source: Source container to subscribe to
        event_type: Type of events to subscribe to
        handler: Handler function to call for events
        
    Returns:
        True if subscription was successful
    """
    if hasattr(source, 'event_bus') and hasattr(source.event_bus, 'subscribe'):
        source.event_bus.subscribe(event_type, handler)
        return True
    return False


def create_forwarding_handler(target: Container, logger: logging.Logger = None) -> Callable:
    """Create an event forwarding handler.
    
    Args:
        target: Target container to forward to
        logger: Optional logger for debugging
        
    Returns:
        Handler function that forwards events to target
    """
    def forward_event(event: Event):
        """Forward event to target container."""
        if logger:
            logger.debug(f"Forwarding {event.event_type} to {target.name}")
        target.receive_event(event)
    
    return forward_event


# Configuration Utilities

def validate_config(config: Dict[str, Any], required_fields: List[str], route_type: str) -> None:
    """Validate route configuration.
    
    Args:
        config: Configuration to validate
        required_fields: List of required field names
        route_type: Type of route for error messages
        
    Raises:
        ValueError: If configuration is invalid
    """
    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"{route_type} route requires '{field}' in configuration"
            )


def extract_connections(route_config: Dict[str, Any]) -> List[tuple]:
    """Extract container connections from route configuration.
    
    Supports different route patterns:
    - Pipeline: containers list
    - Broadcast: source and targets
    - Filter: source and filtered targets
    
    Args:
        route_config: Route configuration
        
    Returns:
        List of (source, target) tuples
    """
    connections = []
    route_type = route_config.get('type', '')
    
    if route_type == 'pipeline' and 'containers' in route_config:
        containers = route_config['containers']
        for i in range(len(containers) - 1):
            connections.append((containers[i], containers[i + 1]))
            
    elif route_type == 'broadcast':
        source = route_config.get('source')
        targets = route_config.get('targets', [])
        if source:
            for target in targets:
                connections.append((source, target))
                
    elif route_type == 'filter':
        source = route_config.get('source')
        # Filter targets are dynamic based on requirements
        if source:
            connections.append((source, 'filtered_targets'))
    
    return connections