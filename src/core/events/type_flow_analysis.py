"""
Type Flow Analysis for ADMF-PC communication adapters.

This module provides sophisticated type flow validation that ensures events
flow correctly through adapter configurations, preventing runtime errors
and ensuring complete audit trails for compliance.
"""

from typing import Dict, Set, List, Tuple, Optional, Type, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

from ..types.events import EventType
from .semantic import (
    SemanticEvent, MarketDataEvent, FeatureEvent, TradingSignal,
    OrderEvent, FillEvent, PortfolioUpdateEvent, EventCategory
)
from ..containers.protocols import Container


@dataclass
class FlowNode:
    """Represents a container's type flow information."""
    name: str
    can_receive: Set[EventType] = field(default_factory=set)  # Types container can handle
    can_produce: Set[EventType] = field(default_factory=set)  # Types container can emit
    will_receive: Set[EventType] = field(default_factory=set) # Types that will reach it
    will_produce: Set[EventType] = field(default_factory=set) # Types it will emit given inputs
    semantic_inputs: Set[Type] = field(default_factory=set)    # Semantic event types it accepts
    semantic_outputs: Set[Type] = field(default_factory=set)   # Semantic event types it produces


@dataclass
class TypeTransition:
    """Represents a type transition between containers."""
    source: str
    target: str
    event_types: Set[EventType] = field(default_factory=set)
    semantic_types: Set[Type] = field(default_factory=set)
    status: str = "OK"  # "OK", "INCOMPATIBLE", "WARNING"
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of type flow validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    flow_map: Optional[Dict[str, FlowNode]] = None
    can_migrate: bool = False
    required_version: Optional[str] = None


class EventTypeRegistry:
    """Registry for semantic event types and their relationships."""
    
    def __init__(self):
        # Map semantic event classes to EventType enum values
        self.event_types: Dict[Type, EventType] = {
            MarketDataEvent: EventType.BAR,
            FeatureEvent: EventType.FEATURE,
            TradingSignal: EventType.SIGNAL,
            OrderEvent: EventType.ORDER,
            FillEvent: EventType.FILL,
            PortfolioUpdateEvent: EventType.PORTFOLIO,
        }
        
        # Reverse mapping
        self.type_to_events: Dict[EventType, Set[Type]] = defaultdict(set)
        for event_class, event_type in self.event_types.items():
            self.type_to_events[event_type].add(event_class)
        
        # Define which events can transform to which (canonical trading flow)
        self.transformations: Dict[EventType, Set[EventType]] = {
            EventType.BAR: {EventType.FEATURE, EventType.SIGNAL},
            EventType.FEATURE: {EventType.SIGNAL},
            EventType.SIGNAL: {EventType.ORDER},
            EventType.ORDER: {EventType.FILL},
            EventType.FILL: {EventType.PORTFOLIO},
            EventType.PORTFOLIO: {EventType.SIGNAL},  # Portfolio updates can trigger new signals
        }
        
        # Container role inference patterns
        self.container_roles = {
            'data': {'produces': {EventType.BAR}, 'consumes': set()},
            'market_data': {'produces': {EventType.BAR}, 'consumes': set()},
            'feature': {'produces': {EventType.FEATURE}, 'consumes': {EventType.BAR}},
            'strategy': {'produces': {EventType.SIGNAL}, 'consumes': {EventType.BAR, EventType.FEATURE, EventType.PORTFOLIO}},
            'risk': {'produces': {EventType.ORDER, EventType.REJECT}, 'consumes': {EventType.SIGNAL, EventType.PORTFOLIO}},
            'execution': {'produces': {EventType.FILL}, 'consumes': {EventType.ORDER}},
            'portfolio': {'produces': {EventType.PORTFOLIO}, 'consumes': {EventType.FILL}},
        }
        
    def register_event_type(self, event_class: Type, event_type: EventType):
        """Register a new semantic event type."""
        self.event_types[event_class] = event_type
        self.type_to_events[event_type].add(event_class)
        
    def get_event_type(self, event: Any) -> Optional[EventType]:
        """Get the EventType for a semantic event instance."""
        return self.event_types.get(type(event))
        
    def get_semantic_types(self, event_type: EventType) -> Set[Type]:
        """Get all semantic event classes for an EventType."""
        return self.type_to_events.get(event_type, set())
        
    def can_transform(self, from_type: EventType, to_type: EventType) -> bool:
        """Check if one event type can transform to another."""
        return to_type in self.transformations.get(from_type, set())
        
    def infer_container_role(self, container_name: str) -> Optional[str]:
        """Infer container role from name."""
        name_lower = container_name.lower()
        for role in self.container_roles:
            if role in name_lower:
                return role
        return None


class TypeFlowAnalyzer:
    """Analyzes and validates event type flow through adapters."""
    
    def __init__(self, registry: Optional[EventTypeRegistry] = None):
        self.registry = registry or EventTypeRegistry()
        self.logger = logging.getLogger(__name__)
        
        # Define execution modes and their requirements
        self.execution_modes = {
            'full_backtest': {
                'required_flows': [
                    (EventType.BAR, 'data', 'strategy'),
                    (EventType.SIGNAL, 'strategy', 'risk'),
                    (EventType.ORDER, 'risk', 'execution'),
                    (EventType.FILL, 'execution', 'portfolio')
                ],
                'must_produce': {EventType.PORTFOLIO}
            },
            'signal_generation': {
                'required_flows': [
                    (EventType.BAR, 'data', 'strategy')
                ],
                'must_produce': {EventType.SIGNAL}
            },
            'signal_replay': {
                'required_flows': [
                    (EventType.SIGNAL, 'signal_source', 'risk'),
                    (EventType.ORDER, 'risk', 'execution')
                ],
                'must_produce': {EventType.PORTFOLIO}
            }
        }
        
    def analyze_flow(self, containers: Dict[str, Container], 
                    adapters: List[Any]) -> Dict[str, FlowNode]:
        """Build complete type flow map from adapters and containers."""
        flow_map = {}
        
        # Initialize flow nodes from containers
        for name, container in containers.items():
            node = FlowNode(name)
            container_role = self._get_container_role(container)
            
            if container_role and container_role in self.registry.container_roles:
                role_config = self.registry.container_roles[container_role]
                node.can_receive = role_config['consumes'].copy()
                node.can_produce = role_config['produces'].copy()
                
                # Map to semantic types
                for event_type in node.can_receive:
                    node.semantic_inputs.update(self.registry.get_semantic_types(event_type))
                for event_type in node.can_produce:
                    node.semantic_outputs.update(self.registry.get_semantic_types(event_type))
            
            flow_map[name] = node
        
        # Propagate types through adapters
        self._propagate_types(flow_map, adapters)
        
        return flow_map
        
    def _get_container_role(self, container: Container) -> Optional[str]:
        """Determine container role from various sources."""
        
        # Method 1: Check if container has explicit role metadata
        if hasattr(container, 'metadata') and hasattr(container.metadata, 'role'):
            return container.metadata.role.value
            
        # Method 2: Check if container has role attribute
        if hasattr(container, 'role'):
            return container.role
            
        # Method 3: Infer from container name
        return self.registry.infer_container_role(container.name)
        
    def _propagate_types(self, flow_map: Dict[str, FlowNode], 
                        adapters: List[Any]) -> None:
        """Propagate event types through adapter connections."""
        
        # Build adjacency list from adapters
        connections = self._build_connections(adapters)
        
        # Iteratively propagate types until no changes
        changed = True
        iterations = 0
        max_iterations = 100  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for source_name, targets in connections.items():
                source_node = flow_map.get(source_name)
                if not source_node:
                    continue
                    
                # Determine what types this source will produce
                produced_types = self._compute_produced_types(source_node)
                
                # Propagate to all targets
                for target_name in targets:
                    target_node = flow_map.get(target_name)
                    if not target_node:
                        continue
                        
                    # Add types that target will receive
                    before_size = len(target_node.will_receive)
                    receivable_types = produced_types & target_node.can_receive
                    target_node.will_receive.update(receivable_types)
                    
                    if len(target_node.will_receive) > before_size:
                        changed = True
                        
        self.logger.debug(f"Type propagation completed in {iterations} iterations")
        
    def _compute_produced_types(self, node: FlowNode) -> Set[EventType]:
        """Compute what types a container will produce given its inputs."""
        produced = set()
        
        # Apply transformations based on what the container will receive
        for input_type in node.will_receive:
            transformable = self.registry.transformations.get(input_type, set())
            # Only produce types this container is capable of producing
            produced.update(transformable & node.can_produce)
            
        # Always include static productions (e.g., data sources)
        if not node.will_receive and node.can_produce:
            produced.update(node.can_produce)
            
        return produced
        
    def _build_connections(self, adapters: List[Any]) -> Dict[str, List[str]]:
        """Build adjacency list from adapter configurations."""
        connections = defaultdict(list)
        
        for adapter in adapters:
            adapter_type = getattr(adapter, 'config', {}).get('type', '')
            
            if hasattr(adapter, 'source') and hasattr(adapter, 'targets'):
                # Broadcast pattern
                source = adapter.source
                targets = adapter.targets if isinstance(adapter.targets, list) else [adapter.targets]
                connections[source].extend(targets)
                
            elif hasattr(adapter, 'containers') and isinstance(adapter.containers, list):
                # Pipeline pattern
                containers = adapter.containers
                for i in range(len(containers) - 1):
                    connections[containers[i]].append(containers[i + 1])
                    
            elif hasattr(adapter, 'config'):
                # Extract from config
                config = adapter.config
                
                if adapter_type == 'broadcast':
                    source = config.get('source')
                    targets = config.get('targets', [])
                    if source and targets:
                        connections[source].extend(targets)
                        
                elif adapter_type == 'pipeline':
                    containers = config.get('containers', [])
                    for i in range(len(containers) - 1):
                        connections[containers[i]].append(containers[i + 1])
                        
                elif adapter_type == 'hierarchical':
                    parent = config.get('parent')
                    children = config.get('children', [])
                    if parent and children:
                        for child in children:
                            child_name = child['name'] if isinstance(child, dict) else child
                            connections[parent].append(child_name)
                            # Bidirectional for hierarchical
                            connections[child_name].append(parent)
                            
        return dict(connections)
        
    def validate_mode(self, flow_map: Dict[str, FlowNode], 
                     mode: str) -> ValidationResult:
        """Validate that type flow meets requirements for execution mode."""
        if mode not in self.execution_modes:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown execution mode: {mode}"]
            )
        
        mode_config = self.execution_modes[mode]
        errors = []
        warnings = []
        
        # Check required flows
        for event_type, source_role, target_role in mode_config['required_flows']:
            if not self._check_flow_exists(flow_map, event_type, source_role, target_role):
                errors.append(
                    f"Missing required flow: {event_type.value} from "
                    f"{source_role} to {target_role}"
                )
        
        # Check required productions
        for required_type in mode_config['must_produce']:
            if not self._check_type_produced(flow_map, required_type):
                errors.append(
                    f"No container produces required event type: {required_type.value}"
                )
        
        # Check for type mismatches
        mismatches = self._check_type_mismatches(flow_map)
        errors.extend(mismatches)
        
        # Check for orphaned events
        orphans = self._check_orphaned_events(flow_map)
        warnings.extend(orphans)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            flow_map=flow_map
        )
        
    def _check_flow_exists(self, flow_map: Dict[str, FlowNode],
                          event_type: EventType,
                          source_role: str, 
                          target_role: str) -> bool:
        """Check if a specific event type flows from source to target role."""
        
        # Find containers matching roles
        sources = [
            node for node in flow_map.values() 
            if self._matches_role(node.name, source_role)
        ]
        targets = [
            node for node in flow_map.values() 
            if self._matches_role(node.name, target_role)
        ]
        
        if not sources or not targets:
            return False
            
        # Check if any source produces the type and any target receives it
        for source in sources:
            produced = self._compute_produced_types(source)
            if event_type in produced:
                for target in targets:
                    if event_type in target.will_receive:
                        return True
        
        return False
        
    def _matches_role(self, container_name: str, role: str) -> bool:
        """Check if container name matches a role."""
        return role.lower() in container_name.lower()
        
    def _check_type_produced(self, flow_map: Dict[str, FlowNode], 
                           event_type: EventType) -> bool:
        """Check if any container produces the required event type."""
        for node in flow_map.values():
            produced = self._compute_produced_types(node)
            if event_type in produced:
                return True
        return False
        
    def _check_type_mismatches(self, flow_map: Dict[str, FlowNode]) -> List[str]:
        """Check for type compatibility issues."""
        errors = []
        
        for node in flow_map.values():
            # Check if container receives types it can't handle
            unhandleable = node.will_receive - node.can_receive
            if unhandleable:
                errors.append(
                    f"Container {node.name} will receive types it cannot handle: "
                    f"{[t.value for t in unhandleable]}"
                )
        
        return errors
        
    def _check_orphaned_events(self, flow_map: Dict[str, FlowNode]) -> List[str]:
        """Check for events that are produced but never consumed."""
        warnings = []
        
        # Collect all produced and consumed types
        all_produced = set()
        all_consumed = set()
        
        for node in flow_map.values():
            produced = self._compute_produced_types(node)
            all_produced.update(produced)
            all_consumed.update(node.will_receive)
        
        # Find orphaned types
        orphaned = all_produced - all_consumed
        if orphaned:
            warnings.append(
                f"Event types are produced but never consumed: "
                f"{[t.value for t in orphaned]}"
            )
        
        return warnings


class ContainerTypeInferencer:
    """Infer container types from semantic events and behavior."""
    
    def __init__(self, registry: EventTypeRegistry):
        self.registry = registry
        
    def infer_container_type(self, container: Container) -> str:
        """Infer container type from its event handlers and name."""
        
        # Method 1: Check explicit semantic event declarations
        if hasattr(container, 'produces_events'):
            produced = container.produces_events()
            
            if MarketDataEvent in produced:
                return 'data_source'
            elif FeatureEvent in produced:
                return 'indicator_engine'
            elif TradingSignal in produced:
                return 'strategy'
            elif OrderEvent in produced:
                return 'risk_manager'
            elif FillEvent in produced:
                return 'execution_engine'
            elif PortfolioUpdateEvent in produced:
                return 'portfolio_manager'
        
        # Method 2: Infer from container role metadata
        role = self._get_container_role(container)
        if role:
            return role
            
        # Method 3: Fallback to name-based inference
        return self._infer_from_name(container.name)
        
    def _get_container_role(self, container: Container) -> Optional[str]:
        """Get explicit container role."""
        if hasattr(container, 'metadata') and hasattr(container.metadata, 'role'):
            return container.metadata.role.value
        if hasattr(container, 'role'):
            return container.role
        return None
        
    def _infer_from_name(self, name: str) -> str:
        """Infer type from container name."""
        name_lower = name.lower()
        
        if any(term in name_lower for term in ['data', 'market', 'feed']):
            return 'data_source'
        elif any(term in name_lower for term in ['indicator', 'technical']):
            return 'indicator_engine'
        elif any(term in name_lower for term in ['strategy', 'signal']):
            return 'strategy'
        elif any(term in name_lower for term in ['risk', 'limit']):
            return 'risk_manager'
        elif any(term in name_lower for term in ['execution', 'broker', 'order']):
            return 'execution_engine'
        elif any(term in name_lower for term in ['portfolio', 'position']):
            return 'portfolio_manager'
        else:
            return 'unknown'
            
    def get_expected_inputs(self, container: Container) -> Set[Type]:
        """Get expected input event types for a container."""
        container_type = self.infer_container_type(container)
        
        # Map container types to expected inputs
        expected_inputs = {
            'indicator_engine': {MarketDataEvent},
            'strategy': {MarketDataEvent, FeatureEvent, PortfolioUpdateEvent},
            'risk_manager': {TradingSignal, PortfolioUpdateEvent},
            'execution_engine': {OrderEvent},
            'portfolio_manager': {FillEvent},
        }
        
        return expected_inputs.get(container_type, set())
        
    def get_expected_outputs(self, container: Container) -> Set[Type]:
        """Get expected output event types for a container."""
        container_type = self.infer_container_type(container)
        
        # Map container types to expected outputs
        expected_outputs = {
            'data_source': {MarketDataEvent},
            'indicator_engine': {FeatureEvent},
            'strategy': {TradingSignal},
            'risk_manager': {OrderEvent},
            'execution_engine': {FillEvent},
            'portfolio_manager': {PortfolioUpdateEvent},
        }
        
        return expected_outputs.get(container_type, set())