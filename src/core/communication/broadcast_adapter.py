"""
Broadcast adapter implementation using protocol-based design.

This module implements the broadcast communication pattern without
inheritance, following ADMF-PC's protocol-based architecture.
"""

from typing import Dict, Any, List, Set, Optional
import logging

from ..events.types import Event, EventType
from .protocols import Container
from .helpers import (
    handle_event_with_metrics,
    subscribe_to_container_events,
    validate_adapter_config,
    create_forward_handler
)


class BroadcastAdapter:
    """Broadcast adapter - no inheritance needed!
    
    Routes events from one source to multiple targets simultaneously.
    All targets receive the same event.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize broadcast adapter.
        
        Args:
            name: Unique adapter name
            config: Configuration with 'source' and 'targets'
        """
        self.name = name
        self.config = config
        self.source_name = config.get('source')
        self.target_names = config.get('targets', [])
        self.source_container: Optional[Container] = None
        self.target_containers: List[Container] = []
        
        # Optional filtering
        self.event_filter = config.get('event_filter')
        self.allowed_types = set(config.get('allowed_types', []))
        
        # Validate configuration
        validate_adapter_config(config, ['source', 'targets'], 'Broadcast')
        
        # Logger will be attached by factory/helper
        self.logger = logging.getLogger(f"adapter.{name}")
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure broadcast connections.
        
        Args:
            containers: Map of container names to instances
        """
        # Get source container
        if self.source_name not in containers:
            raise ValueError(f"Source container '{self.source_name}' not found")
        self.source_container = containers[self.source_name]
        
        # Get target containers
        self.target_containers = []
        for target_name in self.target_names:
            if target_name not in containers:
                raise ValueError(f"Target container '{target_name}' not found")
            self.target_containers.append(containers[target_name])
        
        self.logger.info(
            f"Broadcast adapter '{self.name}' configured: "
            f"{self.source_name} -> {len(self.target_containers)} targets"
        )
        
    def start(self) -> None:
        """Start broadcast operation by setting up subscriptions."""
        if not self.source_container:
            raise RuntimeError("Adapter not configured - call setup() first")
            
        # Subscribe to all events from source
        # In practice, you might subscribe to specific event types
        handler = lambda event: self.handle_event(event, self.source_container)
        
        # Subscribe to source container's events
        if hasattr(self.source_container, 'event_bus'):
            # Subscribe to all event types or specific ones
            if self.allowed_types:
                for event_type in self.allowed_types:
                    self.source_container.event_bus.subscribe(event_type, handler)
            else:
                # Subscribe to all events (implementation specific)
                self.source_container.event_bus.subscribe_all(handler)
        
        self.logger.info(f"Broadcast adapter '{self.name}' started")
        
    def stop(self) -> None:
        """Stop broadcast operation."""
        # In a real implementation, we'd unsubscribe here
        self.logger.info(f"Broadcast adapter '{self.name}' stopped")
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics.
        
        Args:
            event: Event to handle
            source: Source container
        """
        handle_event_with_metrics(self, event, source)
        
    def route_event(self, event: Event, source: Container) -> None:
        """Broadcast event to all target containers.
        
        Args:
            event: Event to broadcast
            source: Source container
        """
        # Apply filtering if configured
        if not self._should_broadcast(event):
            self.logger.debug(
                f"Event {event.event_type} filtered out by broadcast rules"
            )
            return
        
        # Broadcast to all targets
        broadcast_count = 0
        for target in self.target_containers:
            try:
                self.logger.debug(
                    f"Broadcasting {event.event_type} from {source.name} to {target.name}"
                )
                target.receive_event(event)
                broadcast_count += 1
            except Exception as e:
                self.logger.error(
                    f"Failed to broadcast to {target.name}: {e}"
                )
        
        self.logger.debug(
            f"Broadcast complete: {event.event_type} sent to {broadcast_count}/{len(self.target_containers)} targets"
        )
        
    def _should_broadcast(self, event: Event) -> bool:
        """Check if event should be broadcast based on filters.
        
        Args:
            event: Event to check
            
        Returns:
            True if event should be broadcast
        """
        # Check event type filter
        if self.allowed_types and event.event_type not in self.allowed_types:
            return False
        
        # Check custom filter function
        if self.event_filter:
            try:
                return self.event_filter(event)
            except Exception as e:
                self.logger.warning(f"Event filter error: {e}")
                return True  # Default to broadcasting on filter error
        
        return True


def create_filtered_broadcast(name: str, config: Dict[str, Any]):
    """Factory function for broadcast with advanced filtering.
    
    Args:
        name: Adapter name
        config: Configuration with filters
        
    Returns:
        Configured broadcast adapter with filtering
    """
    # Create custom filter based on config
    filter_rules = config.get('filter_rules', [])
    
    def combined_filter(event: Event) -> bool:
        """Apply all filter rules."""
        for rule in filter_rules:
            field = rule.get('field')
            operator = rule.get('operator')
            value = rule.get('value')
            
            # Get field value from event
            if '.' in field:
                # Handle nested fields like 'payload.symbol'
                parts = field.split('.')
                obj = event
                for part in parts:
                    obj = getattr(obj, part, None) if hasattr(obj, part) else obj.get(part)
                    if obj is None:
                        return False
                field_value = obj
            else:
                field_value = getattr(event, field, None)
            
            # Apply operator
            if operator == 'equals' and field_value != value:
                return False
            elif operator == 'contains' and value not in str(field_value):
                return False
            elif operator == 'greater_than' and field_value <= value:
                return False
            elif operator == 'less_than' and field_value >= value:
                return False
        
        return True
    
    # Add filter to config
    config['event_filter'] = combined_filter
    
    return BroadcastAdapter(name, config)


def create_priority_broadcast(name: str, config: Dict[str, Any]):
    """Factory function for broadcast with priority ordering.
    
    Targets are notified in priority order with optional delays.
    
    Args:
        name: Adapter name
        config: Configuration with priorities
        
    Returns:
        Priority-aware broadcast adapter
    """
    import time
    
    # Extract priority configuration
    priorities = config.get('priorities', {})
    delay_ms = config.get('delay_between_targets_ms', 0)
    
    # Create adapter with custom routing
    adapter = BroadcastAdapter(name, config)
    
    # Override route_event to add priority handling
    original_route = adapter.route_event
    
    def priority_route(event: Event, source: Container) -> None:
        """Route with priority ordering."""
        # Sort targets by priority
        sorted_targets = sorted(
            adapter.target_containers,
            key=lambda t: priorities.get(t.name, 999)
        )
        
        # Broadcast in priority order
        for i, target in enumerate(sorted_targets):
            try:
                adapter.logger.debug(
                    f"Broadcasting to {target.name} (priority: {priorities.get(target.name, 999)})"
                )
                target.receive_event(event)
                
                # Add delay between targets if configured
                if delay_ms > 0 and i < len(sorted_targets) - 1:
                    time.sleep(delay_ms / 1000.0)
                    
            except Exception as e:
                adapter.logger.error(f"Failed to broadcast to {target.name}: {e}")
    
    adapter.route_event = priority_route
    return adapter


class FanOutAdapter:
    """Special broadcast variant that transforms events for each target.
    
    Each target can receive a differently transformed version of the event.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize fan-out adapter.
        
        Args:
            name: Unique adapter name
            config: Configuration with transformations
        """
        self.name = name
        self.config = config
        self.source_name = config.get('source')
        self.target_configs = config.get('targets', [])
        self.source_container: Optional[Container] = None
        self.targets: List[Dict[str, Any]] = []
        
        # Validate configuration
        validate_adapter_config(config, ['source', 'targets'], 'FanOut')
        
        self.logger = logging.getLogger(f"adapter.{name}")
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure fan-out connections with transformations."""
        # Get source
        if self.source_name not in containers:
            raise ValueError(f"Source container '{self.source_name}' not found")
        self.source_container = containers[self.source_name]
        
        # Setup targets with their transformations
        self.targets = []
        for target_config in self.target_configs:
            target_name = target_config.get('name')
            if target_name not in containers:
                raise ValueError(f"Target container '{target_name}' not found")
                
            self.targets.append({
                'container': containers[target_name],
                'transform': target_config.get('transform'),
                'filter': target_config.get('filter')
            })
        
        self.logger.info(
            f"Fan-out adapter '{self.name}' configured: "
            f"{self.source_name} -> {len(self.targets)} targets with transformations"
        )
        
    def start(self) -> None:
        """Start fan-out operation."""
        if not self.source_container:
            raise RuntimeError("Adapter not configured - call setup() first")
            
        # Subscribe to source events
        handler = lambda event: self.handle_event(event, self.source_container)
        
        if hasattr(self.source_container, 'event_bus'):
            self.source_container.event_bus.subscribe_all(handler)
        
        self.logger.info(f"Fan-out adapter '{self.name}' started")
        
    def stop(self) -> None:
        """Stop fan-out operation."""
        self.logger.info(f"Fan-out adapter '{self.name}' stopped")
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics."""
        handle_event_with_metrics(self, event, source)
        
    def route_event(self, event: Event, source: Container) -> None:
        """Fan out event with per-target transformations.
        
        Args:
            event: Event to fan out
            source: Source container
        """
        for target_info in self.targets:
            target = target_info['container']
            transform = target_info.get('transform')
            filter_fn = target_info.get('filter')
            
            # Apply filter if present
            if filter_fn and not filter_fn(event):
                self.logger.debug(f"Event filtered out for target {target.name}")
                continue
            
            # Apply transformation if present
            if transform:
                try:
                    transformed_event = transform(event)
                except Exception as e:
                    self.logger.error(f"Transform failed for {target.name}: {e}")
                    continue
            else:
                transformed_event = event
            
            # Send to target
            try:
                self.logger.debug(
                    f"Sending {'transformed' if transform else 'original'} "
                    f"{event.event_type} to {target.name}"
                )
                target.receive_event(transformed_event)
            except Exception as e:
                self.logger.error(f"Failed to send to {target.name}: {e}")