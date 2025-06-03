"""
Selective adapter implementation using protocol-based design.

This module implements the selective routing pattern without
inheritance, following ADMF-PC's protocol-based architecture.
"""

from typing import Dict, Any, List, Callable, Optional, Set
import logging

from ..events.types import Event, EventType
from .protocols import Container
from .helpers import (
    handle_event_with_metrics,
    validate_adapter_config
)


class SelectiveAdapter:
    """Selective adapter - no inheritance needed!
    
    Routes events to specific targets based on event properties,
    container capabilities, or custom routing logic.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize selective adapter.
        
        Args:
            name: Unique adapter name
            config: Configuration with routing rules
        """
        self.name = name
        self.config = config
        
        # Routing configuration
        self.source_name = config.get('source')
        self.routing_rules = config.get('routing_rules', [])
        self.default_target = config.get('default_target')
        self.route_by_type = config.get('route_by_type', {})
        
        # Containers
        self.source_container: Optional[Container] = None
        self.target_containers: Dict[str, Container] = {}
        
        # Validate configuration
        validate_adapter_config(config, ['source', 'routing_rules'], 'Selective')
        
        self.logger = logging.getLogger(f"adapter.{name}")
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure selective routing connections.
        
        Args:
            containers: Map of container names to instances
        """
        # Get source container
        if self.source_name not in containers:
            raise ValueError(f"Source container '{self.source_name}' not found")
        self.source_container = containers[self.source_name]
        
        # Collect all possible target containers
        target_names = set()
        
        # From routing rules
        for rule in self.routing_rules:
            target = rule.get('target')
            if target:
                target_names.add(target)
                
        # From type routing
        target_names.update(self.route_by_type.values())
        
        # Default target
        if self.default_target:
            target_names.add(self.default_target)
            
        # Get target containers
        for target_name in target_names:
            if target_name not in containers:
                raise ValueError(f"Target container '{target_name}' not found")
            self.target_containers[target_name] = containers[target_name]
            
        self.logger.info(
            f"Selective adapter '{self.name}' configured: "
            f"{self.source_name} -> {len(self.target_containers)} possible targets"
        )
        
    def start(self) -> None:
        """Start selective routing operation."""
        if not self.source_container:
            raise RuntimeError("Adapter not configured - call setup() first")
            
        # Subscribe to source events
        handler = lambda event: self.handle_event(event, self.source_container)
        
        if hasattr(self.source_container, 'event_bus'):
            self.source_container.event_bus.subscribe_all(handler)
            
        self.logger.info(f"Selective adapter '{self.name}' started")
        
    def stop(self) -> None:
        """Stop selective routing operation."""
        self.logger.info(f"Selective adapter '{self.name}' stopped")
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics."""
        handle_event_with_metrics(self, event, source)
        
    def route_event(self, event: Event, source: Container) -> None:
        """Selectively route event based on rules.
        
        Args:
            event: Event to route
            source: Source container
        """
        # Determine target(s) for this event
        targets = self._select_targets(event)
        
        if not targets:
            self.logger.debug(
                f"No targets selected for {event.event_type} - using default"
            )
            if self.default_target and self.default_target in self.target_containers:
                targets = [self.default_target]
            else:
                self.logger.warning(f"No targets found for event {event.event_type}")
                return
                
        # Route to selected targets
        for target_name in targets:
            target = self.target_containers.get(target_name)
            if target:
                self.logger.debug(
                    f"Routing {event.event_type} from {source.name} to {target_name}"
                )
                target.receive_event(event)
            else:
                self.logger.error(f"Target '{target_name}' not found in containers")
                
    def _select_targets(self, event: Event) -> List[str]:
        """Select targets based on routing rules.
        
        Args:
            event: Event to route
            
        Returns:
            List of target container names
        """
        targets = []
        
        # Check type-based routing first (highest priority)
        if isinstance(event.event_type, EventType) and event.event_type in self.route_by_type:
            type_target = self.route_by_type[event.event_type]
            if type_target:
                return [type_target]
                
        # Check routing rules
        for rule in self.routing_rules:
            if self._matches_rule(event, rule):
                target = rule.get('target')
                if target:
                    targets.append(target)
                    
                # Check if rule is exclusive
                if rule.get('exclusive', False):
                    break
                    
        return targets
        
    def _matches_rule(self, event: Event, rule: Dict[str, Any]) -> bool:
        """Check if event matches a routing rule.
        
        Args:
            event: Event to check
            rule: Routing rule configuration
            
        Returns:
            True if event matches rule
        """
        # Check conditions
        conditions = rule.get('conditions', [])
        match_all = rule.get('match_all', True)
        
        if not conditions:
            return False
            
        matches = []
        for condition in conditions:
            field = condition.get('field')
            operator = condition.get('operator', 'equals')
            value = condition.get('value')
            
            # Get field value from event
            if '.' in field:
                # Handle nested fields
                parts = field.split('.')
                obj = event
                for part in parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    elif isinstance(obj, dict) and part in obj:
                        obj = obj[part]
                    else:
                        obj = None
                        break
                field_value = obj
            else:
                field_value = getattr(event, field, None)
                
            # Apply operator
            match = self._evaluate_condition(field_value, operator, value)
            matches.append(match)
            
        # Apply match mode
        if match_all:
            return all(matches)
        else:
            return any(matches)
            
    def _evaluate_condition(self, field_value: Any, operator: str, value: Any) -> bool:
        """Evaluate a single condition.
        
        Args:
            field_value: Value from event
            operator: Comparison operator
            value: Expected value
            
        Returns:
            True if condition matches
        """
        try:
            if operator == 'equals':
                return field_value == value
            elif operator == 'not_equals':
                return field_value != value
            elif operator == 'contains':
                return value in str(field_value)
            elif operator == 'not_contains':
                return value not in str(field_value)
            elif operator == 'greater_than':
                return field_value > value
            elif operator == 'less_than':
                return field_value < value
            elif operator == 'in':
                return field_value in value
            elif operator == 'not_in':
                return field_value not in value
            elif operator == 'matches':
                # Simple pattern matching
                import re
                return bool(re.match(value, str(field_value)))
            else:
                self.logger.warning(f"Unknown operator: {operator}")
                return False
        except Exception as e:
            self.logger.debug(f"Condition evaluation error: {e}")
            return False


def create_capability_based_router(name: str, config: Dict[str, Any]):
    """Factory function for capability-based routing.
    
    Routes events to containers based on their declared capabilities.
    
    Args:
        name: Adapter name
        config: Configuration with capability mapping
        
    Returns:
        Selective adapter that routes by capabilities
    """
    # Create base adapter
    adapter = SelectiveAdapter(name, config)
    
    # Override target selection to use capabilities
    def select_by_capability(event: Event) -> List[str]:
        """Select targets based on their capabilities."""
        targets = []
        
        # Get required capability for this event type
        capability_map = config.get('capability_map', {})
        required_capability = None
        
        if isinstance(event.event_type, EventType):
            required_capability = capability_map.get(event.event_type.name)
        
        if not required_capability:
            # Fallback to payload inspection
            if 'required_capability' in event.metadata:
                required_capability = event.metadata['required_capability']
                
        if required_capability:
            # Find containers with this capability
            for name, container in adapter.target_containers.items():
                if hasattr(container, 'capabilities'):
                    if required_capability in container.capabilities:
                        targets.append(name)
                elif hasattr(container, 'metadata') and hasattr(container.metadata, 'capabilities'):
                    if required_capability in container.metadata.capabilities:
                        targets.append(name)
                        
        return targets
    
    # Replace the target selection method
    adapter._select_targets = select_by_capability
    
    return adapter


def create_load_balanced_router(name: str, config: Dict[str, Any]):
    """Factory function for load-balanced routing.
    
    Distributes events across targets using various strategies.
    
    Args:
        name: Adapter name
        config: Configuration with load balancing strategy
        
    Returns:
        Selective adapter with load balancing
    """
    import random
    from collections import defaultdict
    
    # Create base adapter
    adapter = SelectiveAdapter(name, config)
    
    # Load balancing state
    adapter.lb_strategy = config.get('strategy', 'round_robin')
    adapter.lb_counters = defaultdict(int)
    adapter.lb_index = 0
    
    # Override target selection for load balancing
    original_select = adapter._select_targets
    
    def load_balanced_select(event: Event) -> List[str]:
        """Select single target using load balancing."""
        # Get candidate targets
        candidates = original_select(event)
        
        if not candidates:
            return []
            
        if len(candidates) == 1:
            return candidates
            
        # Apply load balancing strategy
        if adapter.lb_strategy == 'round_robin':
            target = candidates[adapter.lb_index % len(candidates)]
            adapter.lb_index += 1
            return [target]
            
        elif adapter.lb_strategy == 'random':
            return [random.choice(candidates)]
            
        elif adapter.lb_strategy == 'least_used':
            # Pick the least used target
            counts = [(adapter.lb_counters[t], t) for t in candidates]
            counts.sort()
            target = counts[0][1]
            adapter.lb_counters[target] += 1
            return [target]
            
        elif adapter.lb_strategy == 'weighted':
            # Use weights from config
            weights = config.get('weights', {})
            weighted_targets = []
            for t in candidates:
                weight = weights.get(t, 1)
                weighted_targets.extend([t] * weight)
            return [random.choice(weighted_targets)]
            
        else:
            # Fallback to first candidate
            return [candidates[0]]
    
    adapter._select_targets = load_balanced_select
    
    return adapter


def create_content_based_router(name: str, config: Dict[str, Any]):
    """Factory function for content-based routing.
    
    Routes based on event payload content using custom extractors.
    
    Args:
        name: Adapter name
        config: Configuration with content rules
        
    Returns:
        Selective adapter with content-based routing
    """
    # Get content extractors
    extractors = config.get('content_extractors', {})
    
    # Build routing rules from content rules
    content_rules = config.get('content_rules', [])
    routing_rules = []
    
    for rule in content_rules:
        extractor_name = rule.get('extractor')
        extractor = extractors.get(extractor_name)
        
        if extractor:
            # Create condition based on extractor
            routing_rule = {
                'target': rule.get('target'),
                'exclusive': rule.get('exclusive', False),
                'conditions': [{
                    'field': f"payload.{extractor['field']}",
                    'operator': rule.get('operator', 'equals'),
                    'value': rule.get('value')
                }]
            }
            routing_rules.append(routing_rule)
    
    # Update config with generated rules
    config['routing_rules'] = routing_rules
    
    return SelectiveAdapter(name, config)