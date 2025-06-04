"""
Hierarchical adapter implementation using protocol-based design.

This module implements the hierarchical communication pattern without
inheritance, following ADMF-PC's protocol-based architecture.
"""

from typing import Dict, Any, List, Set, Optional, Tuple
import logging
from collections import defaultdict

from ..types.events import Event, EventType
from .protocols import Container
from .helpers import (
    handle_event_with_metrics,
    subscribe_to_container_events,
    validate_adapter_config
)


class HierarchicalAdapter:
    """Hierarchical adapter - no inheritance needed!
    
    Routes events through a tree structure where parent containers
    can aggregate or filter events before passing to children.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize hierarchical adapter.
        
        Args:
            name: Unique adapter name
            config: Configuration with hierarchy structure
        """
        self.name = name
        self.config = config
        
        # Tree structure
        self.root_name = config.get('root')
        self.hierarchy = config.get('hierarchy', {})
        self.root_container: Optional[Container] = None
        self.parent_to_children: Dict[str, List[Container]] = defaultdict(list)
        self.child_to_parent: Dict[str, Container] = {}
        self.all_containers: Dict[str, Container] = {}
        
        # Event routing rules
        self.propagation_rules = config.get('propagation_rules', {})
        self.aggregation_window_ms = config.get('aggregation_window_ms', 0)
        self.filter_duplicates = config.get('filter_duplicates', True)
        
        # Validate configuration
        validate_adapter_config(config, ['root', 'hierarchy'], 'Hierarchical')
        
        self.logger = logging.getLogger(f"adapter.{name}")
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure hierarchical connections.
        
        Args:
            containers: Map of container names to instances
        """
        # Validate root exists
        if self.root_name not in containers:
            raise ValueError(f"Root container '{self.root_name}' not found")
        self.root_container = containers[self.root_name]
        self.all_containers[self.root_name] = self.root_container
        
        # Build tree structure
        self._build_tree(self.hierarchy, containers, self.root_name)
        
        # Log tree structure
        self.logger.info(
            f"Hierarchical adapter '{self.name}' configured with root '{self.root_name}'"
        )
        self._log_tree_structure()
        
    def _build_tree(self, node_config: Dict[str, Any], 
                    containers: Dict[str, Container], 
                    parent_name: str) -> None:
        """Recursively build tree structure.
        
        Args:
            node_config: Configuration for this node's children
            containers: Available containers
            parent_name: Name of parent container
        """
        for child_name, child_config in node_config.items():
            # Validate child exists
            if child_name not in containers:
                raise ValueError(f"Child container '{child_name}' not found")
                
            child_container = containers[child_name]
            parent_container = containers[parent_name]
            
            # Store relationships
            self.parent_to_children[parent_name].append(child_container)
            self.child_to_parent[child_name] = parent_container
            self.all_containers[child_name] = child_container
            
            # Recursively process children
            if isinstance(child_config, dict) and child_config:
                self._build_tree(child_config, containers, child_name)
                
    def _log_tree_structure(self, node_name: Optional[str] = None, 
                           level: int = 0) -> None:
        """Log the tree structure for debugging."""
        if node_name is None:
            node_name = self.root_name
            
        indent = "  " * level
        children = self.parent_to_children.get(node_name, [])
        self.logger.debug(
            f"{indent}{node_name} -> {len(children)} children"
        )
        
        for child in children:
            self._log_tree_structure(child.name, level + 1)
            
    def start(self) -> None:
        """Start hierarchical operation by setting up subscriptions."""
        # Subscribe to events at each level
        for container_name, container in self.all_containers.items():
            handler = lambda event, src=container: self.handle_event(event, src)
            
            if hasattr(container, 'event_bus'):
                # Subscribe to all events for now
                # In a real system, you'd be more selective
                container.event_bus.subscribe_all(handler)
        
        self.logger.info(f"Hierarchical adapter '{self.name}' started")
        
    def stop(self) -> None:
        """Stop hierarchical operation."""
        self.logger.info(f"Hierarchical adapter '{self.name}' stopped")
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics."""
        handle_event_with_metrics(self, event, source)
        
    def route_event(self, event: Event, source: Container) -> None:
        """Route event through hierarchy.
        
        Args:
            event: Event to route
            source: Source container
        """
        source_name = source.name
        
        # Determine routing direction based on event type and source
        if self._should_propagate_up(event, source_name):
            self._propagate_up(event, source_name)
        
        if self._should_propagate_down(event, source_name):
            self._propagate_down(event, source_name)
            
        if self._should_broadcast_siblings(event, source_name):
            self._broadcast_to_siblings(event, source_name)
            
    def _should_propagate_up(self, event: Event, source_name: str) -> bool:
        """Check if event should propagate up the hierarchy."""
        # Always propagate errors up
        if event.event_type == EventType.ERROR:
            return source_name in self.child_to_parent
            
        # Check propagation rules
        rules = self.propagation_rules.get(source_name, {})
        return rules.get('propagate_up', False)
        
    def _should_propagate_down(self, event: Event, source_name: str) -> bool:
        """Check if event should propagate down the hierarchy."""
        # Check if source has children
        if source_name not in self.parent_to_children:
            return False
            
        # Check propagation rules
        rules = self.propagation_rules.get(source_name, {})
        return rules.get('propagate_down', True)
        
    def _should_broadcast_siblings(self, event: Event, source_name: str) -> bool:
        """Check if event should be broadcast to siblings."""
        rules = self.propagation_rules.get(source_name, {})
        return rules.get('broadcast_siblings', False)
        
    def _propagate_up(self, event: Event, source_name: str) -> None:
        """Propagate event up to parent."""
        if source_name not in self.child_to_parent:
            return
            
        parent = self.child_to_parent[source_name]
        self.logger.debug(
            f"Propagating {event.event_type} up from {source_name} to {parent.name}"
        )
        
        # Add hierarchy metadata
        if 'hierarchy_path' not in event.metadata:
            event.metadata['hierarchy_path'] = []
        event.metadata['hierarchy_path'].append(source_name)
        
        parent.receive_event(event)
        
    def _propagate_down(self, event: Event, source_name: str) -> None:
        """Propagate event down to children."""
        children = self.parent_to_children.get(source_name, [])
        if not children:
            return
            
        for child in children:
            self.logger.debug(
                f"Propagating {event.event_type} down from {source_name} to {child.name}"
            )
            
            # Clone event for each child to avoid shared state
            child_event = self._clone_event(event)
            child.receive_event(child_event)
            
    def _broadcast_to_siblings(self, event: Event, source_name: str) -> None:
        """Broadcast event to siblings at same level."""
        if source_name not in self.child_to_parent:
            return
            
        parent_name = self.child_to_parent[source_name].name
        siblings = [
            c for c in self.parent_to_children.get(parent_name, [])
            if c.name != source_name
        ]
        
        for sibling in siblings:
            self.logger.debug(
                f"Broadcasting {event.event_type} from {source_name} to sibling {sibling.name}"
            )
            sibling.receive_event(self._clone_event(event))
            
    def _clone_event(self, event: Event) -> Event:
        """Create a copy of an event to avoid shared state."""
        # Simple cloning - in production might need deep copy
        return Event(
            event_type=event.event_type,
            payload=event.payload.copy() if isinstance(event.payload, dict) else event.payload,
            timestamp=event.timestamp,
            source_id=event.source_id,
            container_id=event.container_id,
            metadata=event.metadata.copy() if isinstance(event.metadata, dict) else event.metadata
        )


def create_aggregating_hierarchy(name: str, config: Dict[str, Any]):
    """Factory function for hierarchy with aggregation support.
    
    Parent nodes can aggregate events from children before processing.
    
    Args:
        name: Adapter name
        config: Configuration with aggregation rules
        
    Returns:
        Hierarchical adapter with aggregation
    """
    from collections import deque
    import time
    import threading
    
    # Create base adapter
    adapter = HierarchicalAdapter(name, config)
    
    # Add aggregation state
    adapter.aggregation_buffers = defaultdict(lambda: deque())
    adapter.aggregation_timers = {}
    
    # Override route_event to add aggregation
    original_route = adapter.route_event
    
    def aggregating_route(event: Event, source: Container) -> None:
        """Route with aggregation support."""
        source_name = source.name
        
        # Check if this node should aggregate
        aggregation_rules = config.get('aggregation_rules', {}).get(source_name, {})
        if aggregation_rules.get('aggregate_children'):
            # Buffer events from children
            if source_name in adapter.child_to_parent:
                buffer_key = f"{adapter.child_to_parent[source_name].name}_buffer"
                adapter.aggregation_buffers[buffer_key].append(event)
                
                # Start or reset timer
                if buffer_key not in adapter.aggregation_timers:
                    window_ms = aggregation_rules.get('window_ms', 100)
                    
                    def flush_buffer():
                        """Flush aggregated events."""
                        events = list(adapter.aggregation_buffers[buffer_key])
                        adapter.aggregation_buffers[buffer_key].clear()
                        
                        if events:
                            # Create aggregated event using INFO type
                            aggregated = Event(
                                event_type=EventType.INFO,
                                payload={
                                    'events': events,
                                    'count': len(events),
                                    'source': source_name,
                                    'aggregated': True
                                },
                                timestamp=events[-1].timestamp,
                                source_id=source_name,
                                container_id=adapter.child_to_parent[source_name].name,
                                metadata={'aggregated': True}
                            )
                            
                            # Route aggregated event
                            original_route(aggregated, adapter.child_to_parent[source_name])
                    
                    timer = threading.Timer(window_ms / 1000.0, flush_buffer)
                    timer.start()
                    adapter.aggregation_timers[buffer_key] = timer
                
                return  # Don't route individual events when aggregating
        
        # Normal routing
        original_route(event, source)
    
    adapter.route_event = aggregating_route
    return adapter


def create_filtered_hierarchy(name: str, config: Dict[str, Any]):
    """Factory function for hierarchy with level-based filtering.
    
    Different levels can have different filtering rules.
    
    Args:
        name: Adapter name
        config: Configuration with level filters
        
    Returns:
        Hierarchical adapter with filtering
    """
    # Create base adapter
    adapter = HierarchicalAdapter(name, config)
    
    # Get level filters
    level_filters = config.get('level_filters', {})
    
    # Override propagation methods to add filtering
    original_down = adapter._propagate_down
    original_up = adapter._propagate_up
    
    def filtered_propagate_down(event: Event, source_name: str) -> None:
        """Propagate down with filtering."""
        # Apply level filter
        level = adapter._get_node_level(source_name)
        filter_fn = level_filters.get(f"level_{level}")
        
        if filter_fn and not filter_fn(event):
            adapter.logger.debug(
                f"Event {event.event_type} filtered at level {level}"
            )
            return
            
        original_down(event, source_name)
    
    def filtered_propagate_up(event: Event, source_name: str) -> None:
        """Propagate up with filtering."""
        # Apply upward filter
        filter_fn = level_filters.get('upward')
        
        if filter_fn and not filter_fn(event):
            adapter.logger.debug(
                f"Event {event.event_type} filtered from upward propagation"
            )
            return
            
        original_up(event, source_name)
    
    # Helper to get node level
    def get_node_level(node_name: str) -> int:
        """Get the level of a node in the hierarchy."""
        level = 0
        current = node_name
        while current in adapter.child_to_parent:
            level += 1
            current = adapter.child_to_parent[current].name
        return level
    
    adapter._get_node_level = get_node_level
    adapter._propagate_down = filtered_propagate_down
    adapter._propagate_up = filtered_propagate_up
    
    return adapter