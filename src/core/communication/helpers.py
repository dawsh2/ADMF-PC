"""
Helper functions for adapter implementations.

These functions provide common functionality that would traditionally
be in a base class, but following ADMF-PC's composition principle,
they are standalone functions.
"""

import logging
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager

from ..events.types import Event, EventType
from .protocols import Container, CommunicationAdapter, AdapterMetrics, AdapterErrorHandler


def create_adapter_with_logging(adapter_class, name: str, config: Dict[str, Any]):
    """Create adapter with standard infrastructure.
    
    This helper function creates an adapter instance and attaches
    standard infrastructure like logging, metrics, and error handling.
    
    Args:
        adapter_class: The adapter class to instantiate
        name: Adapter name
        config: Adapter configuration
        
    Returns:
        Configured adapter instance with infrastructure attached
    """
    adapter = adapter_class(name, config)
    
    # Attach infrastructure as attributes
    adapter.metrics = SimpleAdapterMetrics(name)
    adapter.logger = logging.getLogger(f"adapter.{name}")
    adapter.error_handler = SimpleAdapterErrorHandler(adapter.logger)
    
    return adapter


def handle_event_with_metrics(adapter: CommunicationAdapter, 
                            event: Event, 
                            source: Container) -> None:
    """Standard event handling with metrics and error handling.
    
    This helper wraps the adapter's event routing with standard
    metrics collection and error handling.
    
    Args:
        adapter: Adapter instance (must have metrics, error_handler, and route_event)
        event: Event to process
        source: Source container
    """
    if not hasattr(adapter, 'metrics'):
        # Handle event without metrics if not available
        if hasattr(adapter, 'route_event'):
            adapter.route_event(event, source)
        return
    
    with adapter.metrics.measure_latency():
        try:
            # Call the adapter's actual routing logic
            if hasattr(adapter, 'route_event'):
                adapter.route_event(event, source)
                adapter.metrics.increment_success()
            else:
                # Fallback for adapters without route_event
                adapter.handle_event(event, source)
                adapter.metrics.increment_success()
        except Exception as e:
            adapter.metrics.increment_error()
            if hasattr(adapter, 'error_handler'):
                adapter.error_handler.handle(event, e)
            else:
                raise


def subscribe_to_container_events(adapter: CommunicationAdapter,
                                source: Container,
                                event_type: EventType,
                                handler) -> None:
    """Subscribe adapter to container events.
    
    Helper to establish event subscriptions between containers and adapters.
    
    Args:
        adapter: Adapter instance
        source: Source container to subscribe to
        event_type: Type of events to subscribe to
        handler: Handler function to call for events
    """
    if hasattr(source, 'event_bus') and hasattr(source.event_bus, 'subscribe'):
        source.event_bus.subscribe(event_type, handler)
    else:
        if hasattr(adapter, 'logger'):
            adapter.logger.warning(
                f"Container {source.name} does not support event subscriptions"
            )


def validate_adapter_config(config: Dict[str, Any], 
                          required_fields: list,
                          adapter_type: str) -> None:
    """Validate adapter configuration.
    
    Args:
        config: Configuration to validate
        required_fields: List of required field names
        adapter_type: Type of adapter for error messages
        
    Raises:
        ValueError: If configuration is invalid
    """
    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"{adapter_type} adapter requires '{field}' in configuration"
            )


class SimpleAdapterMetrics:
    """Simple implementation of AdapterMetrics protocol.
    
    This is a basic metrics collector that can be attached to any adapter.
    """
    
    def __init__(self, adapter_name: str):
        self.adapter_name = adapter_name
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


class SimpleAdapterErrorHandler:
    """Simple implementation of AdapterErrorHandler protocol.
    
    This basic error handler logs errors and can be attached to any adapter.
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


def create_forward_handler(adapter, target: Container):
    """Create an event forwarding handler for subscriptions.
    
    This is a helper to create closures for event forwarding.
    
    Args:
        adapter: Adapter instance (for logging)
        target: Target container to forward to
        
    Returns:
        Handler function that forwards events to target
    """
    def forward_event(event: Event):
        """Forward event to target container."""
        if hasattr(adapter, 'logger'):
            adapter.logger.debug(
                f"Forwarding {event.event_type} to {target.name}"
            )
        target.receive_event(event)
    
    return forward_event


def get_container_connections(adapter_config: Dict[str, Any]) -> list:
    """Extract container connections from adapter configuration.
    
    Supports different adapter patterns:
    - Pipeline: containers list
    - Broadcast: source and targets
    - Hierarchical: parent and children
    
    Args:
        adapter_config: Adapter configuration
        
    Returns:
        List of (source, target) tuples
    """
    connections = []
    adapter_type = adapter_config.get('type', '')
    
    if adapter_type == 'pipeline' and 'containers' in adapter_config:
        containers = adapter_config['containers']
        for i in range(len(containers) - 1):
            connections.append((containers[i], containers[i + 1]))
            
    elif adapter_type == 'broadcast':
        source = adapter_config.get('source')
        targets = adapter_config.get('targets', [])
        if source:
            for target in targets:
                connections.append((source, target))
                
    elif adapter_type == 'hierarchical':
        parent = adapter_config.get('parent')
        children = adapter_config.get('children', [])
        if parent:
            for child in children:
                child_name = child['name'] if isinstance(child, dict) else child
                connections.append((parent, child_name))
    
    return connections