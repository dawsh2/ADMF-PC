"""
Event router implementation for cross-container communication.

This module provides the default implementation of the EventRouterProtocol,
handling event routing between containers with support for filtering,
scoping, and quality of service.
"""

import asyncio
import logging
from typing import Dict, Set, List, Optional, Callable, Any, Tuple
from collections import defaultdict
from datetime import datetime
import traceback

from .protocols import (
    EventRouterProtocol, EventPublication, EventSubscription,
    EventScope, EventQoS, ValidationResult, EventFilter
)
from ..types import Event, EventType


logger = logging.getLogger(__name__)


class RoutingMetrics:
    """Metrics collection for event routing."""
    
    def __init__(self):
        self.total_events = 0
        self.events_by_source = defaultdict(int)
        self.events_by_type = defaultdict(int)
        self.delivery_failures = 0
        self.routing_times = []
        self.active_subscriptions = 0
        
    def record_event(self, source: str, event_type: str, routing_time_ms: float):
        """Record metrics for a routed event."""
        self.total_events += 1
        self.events_by_source[source] += 1
        self.events_by_type[event_type] += 1
        self.routing_times.append(routing_time_ms)
        
    def record_failure(self):
        """Record a delivery failure."""
        self.delivery_failures += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_routing_time = sum(self.routing_times) / len(self.routing_times) if self.routing_times else 0
        return {
            "total_events": self.total_events,
            "events_by_source": dict(self.events_by_source),
            "events_by_type": dict(self.events_by_type),
            "delivery_failures": self.delivery_failures,
            "average_routing_latency_ms": avg_routing_time,
            "active_subscriptions": self.active_subscriptions
        }


class EventRouter(EventRouterProtocol):
    """
    Default implementation of event router.
    
    Provides cross-container event routing with support for:
    - Flexible pub/sub patterns
    - Event filtering and scoping
    - Topology validation
    - Performance optimization
    - Debugging and visualization
    """
    
    def __init__(self, enable_debugging: bool = False):
        """
        Initialize event router.
        
        Args:
            enable_debugging: Enable debug logging and metrics
        """
        # Core routing tables
        self._publishers: Dict[str, List[EventPublication]] = {}
        self._subscribers: Dict[str, List[EventSubscription]] = {}
        self._handlers: Dict[str, Callable[[Event, str], None]] = {}
        
        # Optimized routing lookup: (source_id, event_type) -> Set[subscriber_id]
        self._routing_table: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        
        # Container relationships for scope resolution
        self._container_hierarchy: Dict[str, Dict[str, Any]] = {}
        
        # Metrics and debugging
        self._metrics = RoutingMetrics()
        self._debug_mode = enable_debugging
        self._event_trace: List[Dict[str, Any]] = []
        
        logger.info("EventRouter initialized")
        
    def register_publisher(
        self, 
        container_id: str, 
        publications: List[EventPublication]
    ) -> None:
        """Register container as event publisher."""
        self._publishers[container_id] = publications
        
        # Log publications in debug mode
        if self._debug_mode:
            for pub in publications:
                event_types = [str(e) for e in pub.events]
                logger.debug(
                    f"Container {container_id} publishes {event_types} "
                    f"with scope {pub.scope.value}"
                )
        
        # Rebuild routing table to include new publisher
        self._rebuild_routing_table()
        
    def register_subscriber(
        self, 
        container_id: str, 
        subscriptions: List[EventSubscription],
        handler: Callable[[Event, str], None]
    ) -> None:
        """Register container as event subscriber."""
        self._subscribers[container_id] = subscriptions
        self._handlers[container_id] = handler
        self._metrics.active_subscriptions += len(subscriptions)
        
        # Log subscriptions in debug mode
        if self._debug_mode:
            for sub in subscriptions:
                event_types = [str(e) for e in sub.events]
                logger.debug(
                    f"Container {container_id} subscribes to {event_types} "
                    f"from {sub.source}"
                )
        
        # Rebuild routing table to include new subscriber
        self._rebuild_routing_table()
        
    def unregister_container(self, container_id: str) -> None:
        """Unregister container from all routing."""
        # Remove publications
        if container_id in self._publishers:
            del self._publishers[container_id]
            
        # Remove subscriptions
        if container_id in self._subscribers:
            self._metrics.active_subscriptions -= len(self._subscribers[container_id])
            del self._subscribers[container_id]
            
        # Remove handler
        if container_id in self._handlers:
            del self._handlers[container_id]
            
        # Rebuild routing table
        self._rebuild_routing_table()
        
        logger.info(f"Container {container_id} unregistered from event router")
        
    def route_event(
        self, 
        source_id: str, 
        event: Event,
        scope: Optional[EventScope] = None
    ) -> None:
        """Route event from source to subscribers."""
        start_time = datetime.now()
        
        # Determine event type string
        event_type_str = str(event.event_type)
        
        # Find publication info for scope
        publication = self._find_publication(source_id, event_type_str)
        if not publication:
            if self._debug_mode:
                logger.debug(f"No publication found for {source_id}:{event_type_str}")
            return
            
        # Use publication scope unless overridden
        effective_scope = scope or publication.scope
        
        # Find potential subscribers
        route_key = (source_id, event_type_str)
        potential_subscribers = self._routing_table.get(route_key, set())
        
        # Filter by scope
        subscribers = self._filter_by_scope(
            source_id, potential_subscribers, effective_scope
        )
        
        # Deliver to each subscriber
        delivered_count = 0
        for subscriber_id in subscribers:
            subscription = self._find_subscription(
                subscriber_id, source_id, event_type_str
            )
            if subscription and self._matches_filters(event, subscription.filters):
                if self._deliver_event(subscriber_id, event, source_id, subscription):
                    delivered_count += 1
        
        # Record metrics
        routing_time = (datetime.now() - start_time).total_seconds() * 1000
        self._metrics.record_event(source_id, event_type_str, routing_time)
        
        # Debug logging
        if self._debug_mode:
            logger.debug(
                f"Routed {event_type_str} from {source_id} to "
                f"{delivered_count} subscribers in {routing_time:.2f}ms"
            )
            self._trace_event(source_id, event, subscribers, routing_time)
    
    def validate_topology(self) -> ValidationResult:
        """Validate event topology for issues."""
        result = ValidationResult(is_valid=True)
        
        # Check for cycles
        cycles = self._detect_cycles()
        for cycle in cycles:
            result.add_error(f"Event cycle detected: {' -> '.join(cycle)}")
        
        # Check for missing publishers
        for subscriber_id, subscriptions in self._subscribers.items():
            for sub in subscriptions:
                if sub.source not in self._publishers:
                    # Check if it's a pattern match
                    if not self._matches_publisher_pattern(sub.source):
                        result.add_warning(
                            f"Container {subscriber_id} subscribes to "
                            f"non-existent publisher {sub.source}"
                        )
        
        # Check for orphaned containers
        all_containers = set(self._publishers.keys()) | set(self._subscribers.keys())
        for container_id in all_containers:
            has_connections = False
            
            # Check if publishes to anyone
            if container_id in self._publishers:
                for pub in self._publishers[container_id]:
                    if self._has_subscribers(container_id, pub.events):
                        has_connections = True
                        break
            
            # Check if subscribes to anyone
            if container_id in self._subscribers:
                has_connections = True
                
            if not has_connections:
                result.add_warning(f"Container {container_id} has no active connections")
        
        # Store topology for visualization
        result.topology_graph = self._build_topology_graph()
        
        return result
    
    def calculate_startup_order(self, containers: Dict[str, Any]) -> List[str]:
        """Calculate startup order based on dependencies."""
        # Build dependency graph
        dependencies = defaultdict(set)
        
        for subscriber_id, subscriptions in self._subscribers.items():
            for sub in subscriptions:
                # Handle pattern matching
                publishers = self._find_matching_publishers(sub.source)
                for publisher_id in publishers:
                    dependencies[subscriber_id].add(publisher_id)
        
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node: str):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected at {node}")
            if node in visited:
                return
                
            temp_visited.add(node)
            for dep in dependencies.get(node, []):
                if dep in containers:  # Only visit containers we're managing
                    visit(dep)
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        
        # Visit all containers
        for container_id in containers:
            if container_id not in visited:
                visit(container_id)
        
        return order
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics."""
        return self._metrics.get_metrics()
    
    def get_topology(self) -> Dict[str, Any]:
        """Get current event topology."""
        return self._build_topology_graph()
    
    def enable_debugging(self, enabled: bool = True) -> None:
        """Enable or disable debugging."""
        self._debug_mode = enabled
        logger.info(f"Event router debugging {'enabled' if enabled else 'disabled'}")
    
    def set_container_hierarchy(self, hierarchy: Dict[str, Dict[str, Any]]) -> None:
        """
        Set container hierarchy for scope resolution.
        
        Args:
            hierarchy: Dict mapping container_id to {parent_id, children_ids}
        """
        self._container_hierarchy = hierarchy
    
    # Private helper methods
    
    def _rebuild_routing_table(self) -> None:
        """Rebuild optimized routing lookup table."""
        self._routing_table.clear()
        
        # For each subscriber
        for subscriber_id, subscriptions in self._subscribers.items():
            for sub in subscriptions:
                # Find matching publishers
                publishers = self._find_matching_publishers(sub.source)
                
                # Add routes for each event type
                for publisher_id in publishers:
                    for event_type in sub.events:
                        route_key = (publisher_id, str(event_type))
                        self._routing_table[route_key].add(subscriber_id)
    
    def _find_matching_publishers(self, source_pattern: str) -> Set[str]:
        """Find publishers matching a source pattern."""
        if source_pattern in self._publishers:
            # Exact match
            return {source_pattern}
        elif '*' in source_pattern:
            # Pattern match (simple glob for now)
            import fnmatch
            return {
                pub_id for pub_id in self._publishers
                if fnmatch.fnmatch(pub_id, source_pattern)
            }
        else:
            # No match
            return set()
    
    def _find_publication(
        self, 
        source_id: str, 
        event_type: str
    ) -> Optional[EventPublication]:
        """Find publication declaration for event."""
        publications = self._publishers.get(source_id, [])
        for pub in publications:
            if event_type in [str(e) for e in pub.events]:
                return pub
        return None
    
    def _find_subscription(
        self, 
        subscriber_id: str,
        source_id: str,
        event_type: str
    ) -> Optional[EventSubscription]:
        """Find subscription for specific event."""
        subscriptions = self._subscribers.get(subscriber_id, [])
        for sub in subscriptions:
            # Check if source matches
            if source_id in self._find_matching_publishers(sub.source):
                # Check if event type matches
                if event_type in [str(e) for e in sub.events]:
                    return sub
        return None
    
    def _filter_by_scope(
        self,
        source_id: str,
        subscribers: Set[str],
        scope: EventScope
    ) -> Set[str]:
        """Filter subscribers based on scope rules."""
        if scope == EventScope.GLOBAL:
            return subscribers
        
        hierarchy = self._container_hierarchy.get(source_id, {})
        parent_id = hierarchy.get('parent_id')
        children_ids = set(hierarchy.get('children_ids', []))
        
        filtered = set()
        
        for subscriber_id in subscribers:
            sub_hierarchy = self._container_hierarchy.get(subscriber_id, {})
            sub_parent = sub_hierarchy.get('parent_id')
            
            if scope == EventScope.LOCAL and subscriber_id == source_id:
                filtered.add(subscriber_id)
            elif scope == EventScope.PARENT and subscriber_id == parent_id:
                filtered.add(subscriber_id)
            elif scope == EventScope.CHILDREN and subscriber_id in children_ids:
                filtered.add(subscriber_id)
            elif scope == EventScope.SIBLINGS and sub_parent == parent_id and subscriber_id != source_id:
                filtered.add(subscriber_id)
            elif scope == EventScope.UPWARD:
                # Check if subscriber is ancestor
                if self._is_ancestor(subscriber_id, source_id):
                    filtered.add(subscriber_id)
            elif scope == EventScope.DOWNWARD:
                # Check if subscriber is descendant
                if self._is_ancestor(source_id, subscriber_id):
                    filtered.add(subscriber_id)
        
        return filtered
    
    def _is_ancestor(self, potential_ancestor: str, container_id: str) -> bool:
        """Check if one container is ancestor of another."""
        current = container_id
        while current:
            hierarchy = self._container_hierarchy.get(current, {})
            parent = hierarchy.get('parent_id')
            if parent == potential_ancestor:
                return True
            current = parent
        return False
    
    def _matches_filters(
        self, 
        event: Event, 
        filters: Optional[EventFilter]
    ) -> bool:
        """Check if event matches subscription filters."""
        if not filters:
            return True
        return filters.matches(event)
    
    def _deliver_event(
        self,
        subscriber_id: str,
        event: Event,
        source_id: str,
        subscription: EventSubscription
    ) -> bool:
        """Deliver event to subscriber."""
        handler = self._handlers.get(subscriber_id)
        if not handler:
            logger.warning(f"No handler found for subscriber {subscriber_id}")
            return False
        
        try:
            # Use custom handler if specified
            if subscription.handler:
                subscription.handler(event, source_id)
            else:
                handler(event, source_id)
            return True
        except Exception as e:
            logger.error(
                f"Error delivering event to {subscriber_id}: {e}\n"
                f"{traceback.format_exc()}"
            )
            self._metrics.record_failure()
            return False
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles in event topology."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def visit(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Get all containers this node sends events to
            for subscriber_id in self._get_event_targets(node):
                if subscriber_id not in visited:
                    if visit(subscriber_id, path.copy()):
                        return True
                elif subscriber_id in rec_stack:
                    # Found cycle
                    cycle_start = path.index(subscriber_id)
                    cycles.append(path[cycle_start:] + [subscriber_id])
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        # Check from each publisher
        for publisher_id in self._publishers:
            if publisher_id not in visited:
                visit(publisher_id, [])
        
        return cycles
    
    def _get_event_targets(self, source_id: str) -> Set[str]:
        """Get all containers that receive events from source."""
        targets = set()
        for (pub_id, _), subscribers in self._routing_table.items():
            if pub_id == source_id:
                targets.update(subscribers)
        return targets
    
    def _has_subscribers(self, publisher_id: str, event_types: Set[str]) -> bool:
        """Check if publisher has any subscribers."""
        for event_type in event_types:
            route_key = (publisher_id, str(event_type))
            if self._routing_table.get(route_key):
                return True
        return False
    
    def _matches_publisher_pattern(self, pattern: str) -> bool:
        """Check if pattern matches any publisher."""
        return bool(self._find_matching_publishers(pattern))
    
    def _build_topology_graph(self) -> Dict[str, Any]:
        """Build graph representation of event topology."""
        nodes = []
        edges = []
        
        # Add all containers as nodes
        all_containers = set(self._publishers.keys()) | set(self._subscribers.keys())
        for container_id in all_containers:
            nodes.append({
                "id": container_id,
                "publishes": [str(e) for pub in self._publishers.get(container_id, []) 
                             for e in pub.events],
                "subscribes": [str(e) for sub in self._subscribers.get(container_id, [])
                              for e in sub.events]
            })
        
        # Add edges for event flows
        for (source_id, event_type), subscribers in self._routing_table.items():
            for subscriber_id in subscribers:
                edges.append({
                    "source": source_id,
                    "target": subscriber_id,
                    "event": event_type
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def _trace_event(
        self,
        source_id: str,
        event: Event,
        delivered_to: Set[str],
        routing_time_ms: float
    ) -> None:
        """Trace event for debugging."""
        trace_entry = {
            "timestamp": datetime.now(),
            "source": source_id,
            "event_type": str(event.event_type),
            "delivered_to": list(delivered_to),
            "routing_time_ms": routing_time_ms,
            "event_id": event.metadata.get("event_id", "unknown")
        }
        
        self._event_trace.append(trace_entry)
        
        # Keep trace size bounded
        if len(self._event_trace) > 1000:
            self._event_trace = self._event_trace[-1000:]