"""
Broadcast route implementation using protocol-based design.

This module implements the broadcast communication pattern without
inheritance, following ADMF-PC's protocol-based architecture.
"""

from typing import Dict, Any, List, Set, Optional, Type
import logging

from ..types.events import Event, EventType
from ..events.semantic import SemanticEvent, validate_semantic_event
from ..events.type_flow_analysis import TypeFlowAnalyzer, EventTypeRegistry, ContainerTypeInferencer
from .protocols import Container
from .composition import (
    wrap_with_metrics,
    create_subscription,
    validate_config,
    create_forwarding_handler
)


class BroadcastRoute:
    """Broadcast route - no inheritance needed!
    
    Routes events from one source to multiple targets simultaneously.
    All targets receive the same event.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize broadcast route.
        
        Args:
            name: Unique route name
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
        
        # Type flow analysis components
        self.enable_type_validation = config.get('enable_type_validation', True)
        self.registry = EventTypeRegistry()
        self.type_analyzer = TypeFlowAnalyzer(self.registry)
        self.type_inferencer = ContainerTypeInferencer(self.registry)
        
        # Validate configuration
        validate_config(config, ['source', 'targets'], 'Broadcast')
        
        # Logger will be attached by factory/helper
        self.logger = logging.getLogger(f"route.{name}")
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure broadcast connections with type flow validation.
        
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
            target = containers[target_name]
            
            # Validate type flow compatibility
            if self.enable_type_validation:
                self._validate_broadcast_target(self.source_container, target)
            
            self.target_containers.append(target)
        
        # Perform full broadcast type flow analysis
        if self.enable_type_validation:
            self._validate_broadcast_flow(containers)
        
        self.logger.info(
            f"Broadcast route '{self.name}' configured: "
            f"{self.source_name} -> {len(self.target_containers)} targets"
        )
        
    def start(self) -> None:
        """Start broadcast operation by setting up subscriptions."""
        if not self.source_container:
            raise RuntimeError("Route not configured - call setup() first")
            
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
                # Subscribe to common event types since subscribe_all doesn't exist
                # We'll subscribe to all known event types
                from ..types.events import EventType
                for event_type in EventType:
                    self.source_container.event_bus.subscribe(event_type, handler)
        
        self.logger.info(f"Broadcast route '{self.name}' started")
        
    def stop(self) -> None:
        """Stop broadcast operation."""
        # In a real implementation, we'd unsubscribe here
        self.logger.info(f"Broadcast route '{self.name}' stopped")
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics.
        
        Args:
            event: Event to handle
            source: Source container
        """
        # Wrap the route_event method with metrics if available
        if hasattr(self, 'metrics'):
            wrapped = wrap_with_metrics(self.route_event, self)
            wrapped(event, source)
        else:
            self.route_event(event, source)
        
    def route_event(self, event: Event, source: Container) -> None:
        """Broadcast event to all target containers with type validation.
        
        Args:
            event: Event to broadcast
            source: Source container
        """
        # Apply filtering if configured
        if not self._should_broadcast(event):
            self.logger.debug(
                f"Event {getattr(event, 'event_type', type(event).__name__)} filtered out by broadcast rules"
            )
            return
        
        # Broadcast to all targets
        broadcast_count = 0
        for target in self.target_containers:
            try:
                # Validate event before broadcasting
                if self.enable_type_validation:
                    self._validate_event_broadcast(event, source, target)
                
                self.logger.debug(
                    f"Broadcasting {getattr(event, 'event_type', type(event).__name__)} from {source.name} to {target.name}"
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
    
    def _validate_broadcast_target(self, source: Container, target: Container) -> None:
        """Validate that target can handle source's event types.
        
        Args:
            source: Source container
            target: Target container
        """
        try:
            # Get expected event types for each container
            source_outputs = self.type_inferencer.get_expected_outputs(source)
            target_inputs = self.type_inferencer.get_expected_inputs(target)
            
            if source_outputs and target_inputs:
                # Check if any source output can be handled by target
                compatible_types = source_outputs & target_inputs
                if not compatible_types:
                    source_type = self.type_inferencer.infer_container_type(source)
                    target_type = self.type_inferencer.infer_container_type(target)
                    
                    self.logger.warning(
                        f"Broadcast type flow warning: {source.name} ({source_type}) outputs "
                        f"{[t.__name__ for t in source_outputs]} but {target.name} ({target_type}) "
                        f"expects {[t.__name__ for t in target_inputs]}"
                    )
                    
                    if self.config.get('strict_type_validation', False):
                        raise TypeError(
                            f"Type flow error: {target.name} cannot handle any events from {source.name}"
                        )
                else:
                    self.logger.debug(
                        f"Broadcast type flow OK: {source.name} → {target.name} "
                        f"(compatible: {[t.__name__ for t in compatible_types]})"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error validating broadcast target {source.name} → {target.name}: {e}")
            if self.config.get('strict_type_validation', False):
                raise
    
    def _validate_broadcast_flow(self, containers: Dict[str, Container]) -> None:
        """Validate the complete broadcast type flow.
        
        Args:
            containers: All available containers
        """
        try:
            # Build flow map for this broadcast
            broadcast_containers = {self.source_name: containers[self.source_name]}
            for target_name in self.target_names:
                if target_name in containers:
                    broadcast_containers[target_name] = containers[target_name]
            
            flow_map = self.type_analyzer.analyze_flow(broadcast_containers, [self])
            
            # Basic validation - check that source produces something targets can consume
            source_node = flow_map.get(self.source_name)
            if source_node:
                source_outputs = self.type_analyzer._compute_produced_types(source_node)
                
                for target_name in self.target_names:
                    target_node = flow_map.get(target_name)
                    if target_node:
                        compatible = source_outputs & target_node.can_receive
                        if not compatible and source_outputs and target_node.can_receive:
                            self.logger.warning(
                                f"Broadcast flow warning: No type compatibility between "
                                f"{self.source_name} and {target_name}"
                            )
            
            self.logger.debug(f"Broadcast type flow validation completed for '{self.name}'")
                
        except Exception as e:
            self.logger.error(f"Error validating broadcast flow: {e}")
            if self.config.get('strict_type_validation', False):
                raise
    
    def _validate_event_broadcast(self, event: Any, source: Container, target: Container) -> None:
        """Validate that a specific event can be broadcast from source to target.
        
        Args:
            event: Event to validate
            source: Source container
            target: Target container
        """
        try:
            # Validate semantic event if applicable
            if isinstance(event, SemanticEvent):
                if not validate_semantic_event(event):
                    self.logger.warning(f"Invalid semantic event: {event}")
                    
                # Check if target can handle this semantic event type
                target_inputs = self.type_inferencer.get_expected_inputs(target)
                event_type = type(event)
                
                if target_inputs and event_type not in target_inputs:
                    self.logger.warning(
                        f"Broadcast event type mismatch: {target.name} expects "
                        f"{[t.__name__ for t in target_inputs]} but got {event_type.__name__}"
                    )
                    
                    if self.config.get('strict_type_validation', False):
                        raise TypeError(
                            f"Broadcast event type error: {target.name} cannot handle {event_type.__name__}"
                        )
            
            # Additional validation for traditional Event objects
            elif hasattr(event, 'event_type'):
                event_type = getattr(event, 'event_type', None)
                if event_type:
                    self.logger.debug(
                        f"Broadcasting {event_type} event from {source.name} to {target.name}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error validating event broadcast: {e}")
            if self.config.get('strict_type_validation', False):
                raise


def create_filtered_broadcast(name: str, config: Dict[str, Any]):
    """Factory function for broadcast with advanced filtering.
    
    Args:
        name: Route name
        config: Configuration with filters
        
    Returns:
        Configured broadcast route with filtering
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
    
    return BroadcastRoute(name, config)


def create_priority_broadcast(name: str, config: Dict[str, Any]):
    """Factory function for broadcast with priority ordering.
    
    Targets are notified in priority order with optional delays.
    
    Args:
        name: Route name
        config: Configuration with priorities
        
    Returns:
        Priority-aware broadcast route
    """
    import time
    
    # Extract priority configuration
    priorities = config.get('priorities', {})
    delay_ms = config.get('delay_between_targets_ms', 0)
    
    # Create route with custom routing
    route = BroadcastRoute(name, config)
    
    # Override route_event to add priority handling
    original_route = route.route_event
    
    def priority_route(event: Event, source: Container) -> None:
        """Route with priority ordering."""
        # Sort targets by priority
        sorted_targets = sorted(
            route.target_containers,
            key=lambda t: priorities.get(t.name, 999)
        )
        
        # Broadcast in priority order
        for i, target in enumerate(sorted_targets):
            try:
                route.logger.debug(
                    f"Broadcasting to {target.name} (priority: {priorities.get(target.name, 999)})"
                )
                target.receive_event(event)
                
                # Add delay between targets if configured
                if delay_ms > 0 and i < len(sorted_targets) - 1:
                    time.sleep(delay_ms / 1000.0)
                    
            except Exception as e:
                route.logger.error(f"Failed to broadcast to {target.name}: {e}")
    
    route.route_event = priority_route
    return route


class FanOutRoute:
    """Special broadcast variant that transforms events for each target.
    
    Each target can receive a differently transformed version of the event.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize fan-out route.
        
        Args:
            name: Unique route name
            config: Configuration with transformations
        """
        self.name = name
        self.config = config
        self.source_name = config.get('source')
        self.target_configs = config.get('targets', [])
        self.source_container: Optional[Container] = None
        self.targets: List[Dict[str, Any]] = []
        
        # Validate configuration
        validate_config(config, ['source', 'targets'], 'FanOut')
        
        self.logger = logging.getLogger(f"route.{name}")
        
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
            f"Fan-out route '{self.name}' configured: "
            f"{self.source_name} -> {len(self.targets)} targets with transformations"
        )
        
    def start(self) -> None:
        """Start fan-out operation."""
        if not self.source_container:
            raise RuntimeError("Route not configured - call setup() first")
            
        # Subscribe to source events
        handler = lambda event: self.handle_event(event, self.source_container)
        
        if hasattr(self.source_container, 'event_bus'):
            self.source_container.event_bus.subscribe_all(handler)
        
        self.logger.info(f"Fan-out route '{self.name}' started")
        
    def stop(self) -> None:
        """Stop fan-out operation."""
        self.logger.info(f"Fan-out route '{self.name}' stopped")
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics."""
        # Wrap the route_event method with metrics if available
        if hasattr(self, 'metrics'):
            wrapped = wrap_with_metrics(self.route_event, self)
            wrapped(event, source)
        else:
            self.route_event(event, source)
        
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