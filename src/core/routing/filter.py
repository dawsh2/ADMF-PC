"""
Generic content-based filtering route.

This route filters event content based on registered requirements,
allowing efficient routing of only needed data to targets.
"""

import logging
from typing import Dict, List, Set, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict

from ..events import Event, EventType
from .protocols import Container

logger = logging.getLogger(__name__)


@dataclass
class FilterRequirements:
    """Defines filtering requirements for a target."""
    target_id: str
    required_keys: Set[str] = field(default_factory=set)
    conditions: List[Callable[[Event], bool]] = field(default_factory=list)
    transform: Optional[Callable[[Event], Event]] = None
    
    def matches(self, event: Event) -> bool:
        """Check if event matches all conditions."""
        return all(cond(event) for cond in self.conditions)


class FilterRoute:
    """
    Routes events with filtered content based on requirements.
    
    This route can:
    1. Filter event payload to only required fields
    2. Route based on conditions
    3. Transform events before routing
    4. Handle any event type, not just features
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize filter route.
        
        Args:
            name: Unique route name
            config: Configuration including:
                - filter_field: Path to data to filter (e.g. 'payload.features')
                - event_types: List of event types to handle (optional)
                - default_transform: Default transformation function (optional)
        """
        self.name = name
        self.config = config
        self.filter_field = config.get('filter_field', 'payload')
        self.event_types = set(config.get('event_types', []))  # Empty = all types
        self.default_transform = config.get('default_transform')
        
        # Requirements registry
        self.requirements: Dict[str, FilterRequirements] = {}
        self.target_containers: Dict[str, Container] = {}
        self._started = False
        
        # Source container for subscriptions
        self.source_container: Optional[Container] = None
        
        logger.info(f"FilterRoute {name} initialized")
    
    def register_requirements(
        self, 
        target_id: str,
        required_keys: Optional[List[str]] = None,
        conditions: Optional[List[Callable]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Register filtering requirements for a target.
        
        Args:
            target_id: Target identifier
            required_keys: Keys to include from filtered field
            conditions: List of conditions that must all be true
            transform: Optional transformation function
        """
        requirements = FilterRequirements(
            target_id=target_id,
            required_keys=set(required_keys or []),
            conditions=conditions or [],
            transform=transform or self.default_transform
        )
        
        self.requirements[target_id] = requirements
        
        logger.debug(
            f"Registered requirements for {target_id}: "
            f"{len(requirements.required_keys)} keys, "
            f"{len(requirements.conditions)} conditions"
        )
    
    def setup(self, containers: Dict[str, Container]) -> None:
        """
        Configure route with containers.
        
        Args:
            containers: Map of container names to instances
        """
        # Get source container if specified
        source_name = self.config.get('source')
        if source_name and source_name in containers:
            self.source_container = containers[source_name]
        
        # Map target containers
        self.target_containers = {}
        for target_id in self.requirements:
            if target_id in containers:
                self.target_containers[target_id] = containers[target_id]
            else:
                # Try to find by prefix match (e.g., 'momentum_strat' matches 'portfolio_momentum_strat')
                for name, container in containers.items():
                    if target_id in name or name.endswith(target_id):
                        self.target_containers[target_id] = container
                        break
        
        logger.info(f"FilterRoute setup with {len(self.target_containers)} targets")
    
    def start(self) -> None:
        """Start the route by subscribing to events."""
        if self._started:
            return
        
        if self.source_container:
            # Subscribe to specific event types or all
            if self.event_types:
                for event_type in self.event_types:
                    self.source_container.event_bus.subscribe(
                        event_type, self.handle_event
                    )
            else:
                # Subscribe to all events
                self.source_container.event_bus.subscribe_all(self.handle_event)
            
            logger.info(f"FilterRoute {self.name} subscribed to source events")
        
        self._started = True
    
    def stop(self) -> None:
        """Stop the route."""
        if not self._started:
            return
        
        if self.source_container:
            if self.event_types:
                for event_type in self.event_types:
                    self.source_container.event_bus.unsubscribe(
                        event_type, self.handle_event
                    )
            else:
                self.source_container.event_bus.unsubscribe_all(self.handle_event)
        
        self._started = False
    
    def handle_event(self, event: Event) -> None:
        """
        Handle incoming event by filtering and routing.
        
        Args:
            event: Event to filter and route
        """
        # Check if we handle this event type
        if self.event_types and event.event_type not in self.event_types:
            return
        
        # Route to each matching target
        routed_count = 0
        
        for target_id, requirements in self.requirements.items():
            # Check conditions
            if not requirements.matches(event):
                continue
            
            # Get target container
            target_container = self.target_containers.get(target_id)
            if not target_container:
                continue
            
            # Filter or transform event
            if requirements.transform:
                # Use custom transform
                filtered_event = requirements.transform(event)
            elif requirements.required_keys:
                # Filter to required keys
                filtered_event = self._filter_event(event, requirements.required_keys)
            else:
                # Pass through unchanged
                filtered_event = event
            
            if filtered_event:
                # Tag with target for tracing
                filtered_event.metadata['target'] = target_id
                filtered_event.metadata['filtered'] = True
                
                # Route to target
                target_container.receive_event(filtered_event)
                routed_count += 1
        
        if routed_count > 0:
            logger.debug(
                f"Routed {event.event_type} to {routed_count} targets"
            )
    
    def _filter_event(self, event: Event, required_keys: Set[str]) -> Optional[Event]:
        """
        Filter event to only include required keys.
        
        Args:
            event: Original event
            required_keys: Keys to include
            
        Returns:
            Filtered event or None if no data matches
        """
        # Get the data to filter
        data = self._get_nested_field(event, self.filter_field)
        if not data or not isinstance(data, dict):
            return None
        
        # Filter to required keys
        filtered_data = {}
        for key in required_keys:
            if key in data:
                filtered_data[key] = data[key]
            else:
                # Check for prefix matches (e.g., 'rsi' matches 'rsi_14')
                for data_key, value in data.items():
                    if data_key.startswith(f"{key}_"):
                        filtered_data[data_key] = value
        
        if not filtered_data:
            return None
        
        # Create filtered event
        filtered_event = Event(
            event_type=event.event_type,
            timestamp=event.timestamp,
            payload=event.payload.copy() if hasattr(event.payload, 'copy') else event.payload,
            metadata=event.metadata.copy() if hasattr(event.metadata, 'copy') else event.metadata
        )
        
        # Update the filtered field
        self._set_nested_field(filtered_event, self.filter_field, filtered_data)
        
        return filtered_event
    
    def _get_nested_field(self, obj: Any, path: str) -> Any:
        """Get nested field using dot notation."""
        parts = path.split('.')
        current = obj
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _set_nested_field(self, obj: Any, path: str, value: Any) -> None:
        """Set nested field using dot notation."""
        parts = path.split('.')
        current = obj
        
        # Navigate to parent
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set final value
        final_key = parts[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        elif isinstance(current, dict):
            current[final_key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'name': self.name,
            'registered_targets': len(self.requirements),
            'active_targets': len(self.target_containers),
            'filter_field': self.filter_field,
            'event_types': list(self.event_types) if self.event_types else 'all'
        }


# Convenience function for creating feature filter (backward compatibility)
def create_feature_filter(name: str, config: Dict[str, Any]) -> FilterRoute:
    """Create a filter specifically for feature routing."""
    config.setdefault('filter_field', 'payload.features')
    config.setdefault('event_types', [EventType.FEATURES])
    return FilterRoute(name, config)