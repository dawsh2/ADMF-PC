"""
Generic TopologyBuilder Implementation

Builds topologies from YAML patterns or Python dictionaries.
Completely data-driven - no hardcoded topology logic.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
from pathlib import Path
import hashlib
import json
import yaml
import fnmatch
import itertools

from .config.pattern_loader import PatternLoader
from .config.resolver import ConfigResolver

logger = logging.getLogger(__name__)


class TopologyBuilder:
    """
    Generic topology builder that constructs topologies from patterns.
    
    Instead of hardcoded topology functions, this builder interprets
    declarative patterns that describe what containers, components,
    and routes to create.
    """
    
    def __init__(self, pattern_loader: Optional[PatternLoader] = None,
                 config_resolver: Optional[ConfigResolver] = None):
        """Initialize topology builder."""
        self.logger = logging.getLogger(__name__)
        self.pattern_loader = pattern_loader or PatternLoader()
        self.config_resolver = config_resolver or ConfigResolver()
        self.patterns = self.pattern_loader.load_patterns('topologies')
        self.container_factory = None
        
    
    def build_topology(self, topology_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a topology from a definition.
        
        Args:
            topology_definition: Dict containing:
                - mode: The topology mode/pattern name
                - config: User configuration
                - tracing_config: Optional tracing configuration
                - metadata: Optional metadata
            
        Returns:
            Built topology with containers, routes, components, etc.
        """
        mode = topology_definition.get('mode')
        if not mode:
            raise ValueError("Topology definition must include 'mode'")
            
        config = topology_definition.get('config', {})
        tracing_config = topology_definition.get('tracing_config', {})
        metadata = topology_definition.get('metadata', {})
        
        self.logger.info(f"Building {mode} topology")
        
        # Get the pattern
        pattern = self._get_pattern(mode)
        if not pattern:
            raise ValueError(f"No pattern found for mode: {mode}")
        
        # Initialize factories
        self._initialize_factories()
        
        # Build evaluation context
        context = self._build_context(pattern, config, tracing_config, metadata)
        
        # Infer required features from strategies
        self._infer_and_inject_features(context)
        
        # Build topology
        topology = {
            'name': pattern.get('name', mode),
            'description': pattern.get('description', ''),
            'containers': {},
            'routes': [],
            'components': {},
            'metadata': {}
        }
        
        # 1. Create stateless components
        if 'components' in pattern:
            self.logger.info("Creating stateless components")
            for comp_spec in pattern['components']:
                components = self._create_components(comp_spec, context)
                topology['components'].update(components)
            context['components'] = topology['components']
        
        # 2. Create containers
        if 'containers' in pattern:
            self.logger.info("Creating containers")
            for container_spec in pattern['containers']:
                containers = self._create_containers(container_spec, context)
                topology['containers'].update(containers)
                # Update context with created containers for parent-child relationships
                context['containers'] = topology['containers']
        
        # Hierarchical relationships are now established inline during container creation
        
        # 3. Create routes (only for cross-hierarchy communication)
        if 'routes' in pattern:
            self.logger.info("Creating routes")
            for route_spec in pattern['routes']:
                route = self._create_route(route_spec, context, topology['containers'])
                if route:
                    topology['routes'].append(route)
        
        # 4. Apply behaviors
        if 'behaviors' in pattern:
            self.logger.info("Applying behaviors")
            for behavior_spec in pattern['behaviors']:
                self._apply_behavior(behavior_spec, context, topology)
        
        # Add metadata
        topology['metadata'] = {
            'mode': mode,
            'pattern': pattern.get('name', mode),
            'created_at': str(datetime.now()),
            'config_hash': self._hash_config(config),
            'tracing_enabled': tracing_config.get('enabled', False),
            **(metadata or {})
        }
        
        # 5. Set up event subscriptions based on mode
        self._setup_event_subscriptions(mode, topology, context)
        
        self.logger.info(
            f"Built {mode} topology with {len(topology['containers'])} containers "
            f"and {len(topology['routes'])} routes"
        )
        
        return topology
    
    def _initialize_factories(self):
        """Initialize container factory."""
        if not self.container_factory:
            from ..containers.factory import ContainerFactory
            self.container_factory = ContainerFactory()
    
    def _get_pattern(self, mode: str) -> Optional[Dict[str, Any]]:
        """Get pattern for the given mode."""
        # Check loaded patterns
        if mode in self.patterns:
            return self.patterns[mode]
        
        # Special handling for optimization
        if mode == 'optimization' and 'backtest' in self.patterns:
            return self.patterns['backtest']
        
        return None
    
    def _build_context(self, pattern: Dict[str, Any], config: Dict[str, Any], 
                      tracing_config: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build evaluation context for pattern interpretation."""
        context = {
            'config': config,
            'pattern': pattern,
            'tracing': tracing_config,
            'metadata': metadata,  # Include metadata for container configuration
            'generated': {},
            'root_event_bus': None,  # Still needed for stateless services
            'use_hierarchical_events': True  # Encourage parent-child for containers
        }
        
        # Process tracing configuration with CORRECT structure
        if tracing_config.get('enabled', False):
            if 'execution' not in config:
                config['execution'] = {}
            
            config['execution']['enable_event_tracing'] = True
            config['execution']['trace_settings'] = {
                'trace_id': tracing_config.get('trace_id'),
                'trace_dir': tracing_config.get('trace_dir', './traces'),
                'max_events': tracing_config.get('max_events', 10000),
                'container_settings': tracing_config.get('container_settings', {}),
                # Include console output settings
                'enable_console_output': tracing_config.get('enable_console_output', False),
                'console_filter': tracing_config.get('console_filter', [])
            }
        
        # Create root event bus for stateless services and cross-hierarchy routing
        # This is REQUIRED for strategy services, risk validators, etc.
        if hasattr(self.container_factory, 'root_event_bus'):
            context['root_event_bus'] = self.container_factory.root_event_bus
        else:
            from ..events import EventBus
            context['root_event_bus'] = EventBus()
            self.logger.debug("Created root event bus for stateless services")
        
        # Generate parameter combinations if needed
        if 'strategies' in config and 'risk_profiles' in config:
            context['generated']['parameter_combinations'] = self._generate_parameter_combinations(
                config['strategies'], config['risk_profiles']
            )
        
        return context
    
    def _create_components(self, comp_spec: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Create stateless components from specification."""
        components = {}
        comp_type = comp_spec.get('type')
        
        # Initialize component category
        if comp_type not in context.get('components', {}):
            if comp_type == 'strategies':
                components['strategies'] = {}
            elif comp_type == 'risk_validators':
                components['risk_validators'] = {}
            elif comp_type == 'classifiers':
                components['classifiers'] = {}
            elif comp_type == 'execution_models':
                components['execution_models'] = {}
        
        # Get items from config or spec
        from_config = comp_spec.get('from_config')
        if from_config:
            items = self._resolve_value(from_config, context)
            if not isinstance(items, list):
                items = [items] if items else []
                
            for item in items:
                # Skip if item is not a dict (e.g., if it's a string from bad resolution)
                if not isinstance(item, dict):
                    continue
                component = self._create_single_component(comp_type, item)
                if component:
                    name = item.get('name', item.get('type', comp_type))
                    if comp_type not in components:
                        components[comp_type] = {}
                    components[comp_type][name] = component
        else:
            # Create single component
            component = self._create_single_component(comp_type, comp_spec)
            if component:
                name = comp_spec.get('name', comp_spec.get('type', comp_type))
                if comp_type not in components:
                    components[comp_type] = {}
                components[comp_type][name] = component
        
        return components
    
    def _create_single_component(self, comp_type: str, config: Dict[str, Any]):
        """Create a single stateless component."""
        # TODO: Implement actual component creation
        # For now, return mock components for testing
        
        try:
            if comp_type == 'strategies':
                # Mock strategy for testing
                return {'type': 'strategy', 'name': config.get('name', 'test_strategy'), 'config': config}
            elif comp_type == 'classifiers':
                # Mock classifier
                return {'type': 'classifier', 'name': config.get('name', 'test_classifier'), 'config': config}
            elif comp_type == 'risk_validators':
                # Mock risk validator
                return {'type': 'risk_validator', 'name': config.get('name', 'test_risk'), 'config': config}
            elif comp_type == 'execution_models':
                # Mock execution model
                return {'type': 'execution_model', 'name': config.get('name', 'test_execution'), 'config': config}
            else:
                self.logger.warning(f"Unknown component type: {comp_type}")
                # Return generic mock component
                return {'type': comp_type, 'name': config.get('name', f'test_{comp_type}'), 'config': config}
        except Exception as e:
            self.logger.error(f"Failed to create component {comp_type}: {e}")
            return None
    
    def _create_containers(self, container_spec: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Create containers from specification."""
        containers = {}
        
        if 'foreach' in container_spec:
            # Create multiple containers using foreach
            containers.update(self._expand_container_foreach(container_spec, context))
        else:
            # Create single container
            name = self._resolve_value(container_spec.get('name'), context)
            if name:
                container = self._create_single_container(name, container_spec, context)
                if container:
                    containers[name] = container
                    
                    # If this container has children, collect them too
                    if hasattr(container, '_child_containers') and container._child_containers:
                        # Recursively collect all descendants
                        def collect_descendants(parent):
                            descendants = {}
                            for child_id, child in parent._child_containers.items():
                                descendants[child_id] = child
                                if hasattr(child, '_child_containers') and child._child_containers:
                                    descendants.update(collect_descendants(child))
                            return descendants
                        
                        containers.update(collect_descendants(container))
        
        return containers
    
    def _expand_container_foreach(self, spec: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Expand container specification with foreach loops."""
        containers = {}
        foreach_spec = spec['foreach']
        
        # Resolve foreach variables
        foreach_values = {}
        for var_name, var_spec in foreach_spec.items():
            values = self._resolve_value(var_spec, context)
            if not isinstance(values, list):
                values = [values] if values else []
            foreach_values[var_name] = values
        
        # Special handling for parameter combinations
        if 'combo' in foreach_spec:
            combos = context['generated'].get('parameter_combinations', [])
            for combo in combos:
                iter_context = context.copy()
                iter_context['combo'] = combo
                iter_context['combo_id'] = combo['combo_id']
                
                # Create container
                name_template = spec.get('name_template', '')
                name = self._resolve_value(name_template, iter_context)
                container = self._create_single_container(name, spec, iter_context)
                if container:
                    containers[name] = container
        else:
            # Generate all combinations
            keys = list(foreach_values.keys())
            value_lists = [foreach_values[k] for k in keys]
            
            for values in itertools.product(*value_lists):
                # Create context for this iteration
                iter_context = context.copy()
                for i, key in enumerate(keys):
                    iter_context[key] = values[i]
                
                # Generate container name
                name_template = spec.get('name_template', spec.get('name', ''))
                name = self._resolve_value(name_template, iter_context)
                
                # Create container
                container = self._create_single_container(name, spec, iter_context)
                if container:
                    containers[name] = container
        
        return containers
    
    def _create_single_container(self, name: str, spec: Dict[str, Any], 
                                context: Dict[str, Any]) -> Optional[Any]:
        """Create a single container."""
        # Build container config
        config = {
            'type': spec.get('type'),
            'name': name
        }
        
        # Resolve config values
        if 'config' in spec:
            for key, value_spec in spec['config'].items():
                config[key] = self._resolve_value(value_spec, context)
        
        # Add standard configs
        if 'execution' in context['config']:
            config['execution'] = context['config']['execution']
        
        if 'components' in context:
            config['stateless_components'] = context['components']
        
        # Containers are isolated - no automatic root_event_bus access
        # All cross-container communication goes through parent or routes
        
        # Add combo-specific values
        if 'combo' in context:
            combo = context['combo']
            config['combo_id'] = combo['combo_id']
            if 'strategy_params' in combo:
                config['strategy_type'] = combo['strategy_params'].get('type')
                config['strategy_params'] = combo['strategy_params']
            if 'risk_params' in combo:
                config['risk_type'] = combo['risk_params'].get('type')
                config['risk_params'] = combo['risk_params']
        
        try:
            # Extract components list from config
            components = config.get('components', [])
            container_type = config.get('type')
            
            # Remove components and type from config as they're passed separately
            clean_config = {k: v for k, v in config.items() if k not in ['components', 'type']}
            
            # Add metadata from context for tracing and results organization
            if 'metadata' in context:
                clean_config['metadata'] = context['metadata']
            
            container = self.container_factory.create_container(
                name=name,
                components=components,
                config=clean_config,
                container_type=container_type
            )
            
            # Create child containers if specified inline
            if 'containers' in spec:
                child_containers = self._create_child_containers(container, spec['containers'], context)
                # Add child containers to the topology context
                if 'containers' in context:
                    context['containers'].update(child_containers)
            
            return container
        except Exception as e:
            self.logger.error(f"Failed to create container {name}: {e}")
            return None
    
    def _create_child_containers(self, parent_container: Any, children_specs: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Create child containers inline and establish parent-child relationships."""
        child_containers = {}
        
        for child_spec in children_specs:
            # Handle foreach in child containers too
            if 'foreach' in child_spec:
                children = self._expand_container_foreach(child_spec, context)
                for name, child in children.items():
                    if child:
                        parent_container.add_child_container(child)
                        child_containers[name] = child
                        self.logger.info(f"Created child: {parent_container.name} -> {name}")
            else:
                child_name = self._resolve_value(child_spec.get('name'), context)
                if not child_name:
                    child_name = f"{parent_container.name}_{child_spec.get('type', 'child')}"
                
                # Create child with same context
                child_container = self._create_single_container(child_name, child_spec, context)
                
                if child_container:
                    # Establish parent-child relationship
                    parent_container.add_child_container(child_container)
                    child_containers[child_name] = child_container
                    self.logger.info(f"Created child: {parent_container.name} -> {child_name}")
                    
                    # Recursively create children of children
                    if 'containers' in child_spec:
                        grandchildren = self._create_child_containers(child_container, child_spec['containers'], context)
                        child_containers.update(grandchildren)
        
        return child_containers
    
    def _create_route(self, route_spec: Dict[str, Any], context: Dict[str, Any],
                     containers: Dict[str, Any]) -> Optional[Any]:
        """Create a route from specification."""
        config = {
            'type': route_spec.get('type')
        }
        
        # Routes need root_event_bus for stateless services (strategies, risk validators, etc.)
        # but container-to-container communication should prefer parent-child
        route_type = route_spec.get('type', '')
        
        # These route types typically handle stateless services
        stateless_route_types = ['risk_service', 'strategy_dispatcher', 'filter', 'broadcast']
        
        if route_type in stateless_route_types or route_spec.get('stateless_services', False):
            # Stateless services need root_event_bus
            config['root_event_bus'] = context['root_event_bus']
            self.logger.debug(f"Route {route_spec.get('name', route_type)} using root_event_bus for stateless services")
        elif route_spec.get('cross_hierarchy', False):
            # Cross-hierarchy routing needs root_event_bus
            config['root_event_bus'] = context['root_event_bus']
            self.logger.info(f"Route {route_spec.get('name', 'unnamed')} using root_event_bus for cross-hierarchy communication")
        else:
            # Container-to-container should use parent-child when possible
            self.logger.debug(f"Route {route_spec.get('name', 'unnamed')} not using root_event_bus - "
                            "encouraging parent-child communication")
        
        # Resolve route configuration
        for key in ['source', 'target', 'source_pattern', 'target_pattern', 
                   'processor', 'event_types', 'allowed_types']:
            if key in route_spec:
                config[key] = self._resolve_value(route_spec[key], context)
        
        # Handle target lists with patterns
        if 'targets' in route_spec:
            targets = self._resolve_value(route_spec['targets'], context)
            if isinstance(targets, str) and '*' in targets:
                # Pattern matching
                pattern = targets
                targets = [name for name in containers.keys() 
                          if self._matches_pattern(name, pattern)]
            elif not isinstance(targets, list):
                targets = [targets] if targets else []
            config['targets'] = targets
        
        # Handle source pattern
        if 'source_pattern' in config:
            pattern = config['source_pattern']
            if '*' in pattern:
                # Find matching containers
                sources = [name for name in containers.keys()
                          if self._matches_pattern(name, pattern)]
                config['sources'] = sources
        
        name = route_spec.get('name', f"route_{len(context.get('routes', []))}")
        
        try:
            # Add special components like risk validators
            if config['type'] == 'risk_service' and 'risk_validators' in context['components']:
                config['risk_validators'] = context['components']['risk_validators']
            
            # Routes are being replaced with direct EventBus subscriptions
            # For now, log what would have been created
            self.logger.info(f"Would have created route '{name}' of type '{config['type']}' (now using EventBus)")
            return None
        except Exception as e:
            self.logger.error(f"Failed to create route {name}: {e}")
            return None
    
    def _apply_behavior(self, behavior_spec: Dict[str, Any], context: Dict[str, Any],
                       topology: Dict[str, Any]) -> None:
        """Apply special behaviors to topology."""
        behavior_type = behavior_spec.get('type')
        
        if behavior_type == 'feature_dispatcher':
            self._apply_feature_dispatcher(behavior_spec, context, topology)
        elif behavior_type == 'subscribe_to_root_bus':
            self._apply_root_bus_subscription(behavior_spec, context, topology)
        else:
            self.logger.warning(f"Unknown behavior type: {behavior_type}")
    
    def _apply_feature_dispatcher(self, spec: Dict[str, Any], context: Dict[str, Any],
                                 topology: Dict[str, Any]) -> None:
        """Apply feature filter behavior."""
        from ..events import EventType, Event
        
        # TODO: Implement feature pipeline properly
        # For now, skip feature filtering to test basic topology
        self.logger.info("Skipping feature dispatcher - not implemented yet")
        return
        
        # Register strategies
        if 'parameter_combinations' in context['generated']:
            # Strategies are stateless services that need root_event_bus
            root_event_bus = context['root_event_bus']
            
            for combo in context['generated']['parameter_combinations']:
                strategy_config = combo['strategy_params']
                strategy_type = strategy_config.get('type')
                combo_id = combo['combo_id']
                
                if strategy_type:
                    # Get feature requirements
                    required_features = self._get_strategy_feature_requirements(strategy_type, strategy_config)
                    
                    # Create strategy transform function
                    def create_strategy_transform(sid, stype, sconfig):
                        def transform(event):
                            # Create strategy instance
                            from ..components.factory import create_component
                            # Strategies are stateless services that publish to root_event_bus
                            strategy = create_component(
                                stype,
                                context={
                                    'event_bus': root_event_bus,
                                    'config': sconfig
                                },
                                capabilities=['events']
                            )
                            
                            # Process features
                            if hasattr(strategy, 'handle_features'):
                                result = strategy.handle_features(event)
                                if result:
                                    # Create signal event
                                    return Event(
                                        event_type=EventType.SIGNAL,
                                        timestamp=event.timestamp,
                                        payload=result,
                                        metadata={
                                            'strategy_id': sid,
                                            'strategy_type': stype,
                                            'source_event': event.event_id
                                        }
                                    )
                            return None
                        return transform
                    
                    # Register strategy requirements
                    strategy_id = f"{combo_id}_{strategy_type}"
                    feature_filter.register_requirements(
                        target_id=strategy_id,
                        required_keys=required_features,
                        transform=create_strategy_transform(strategy_id, strategy_type, strategy_config)
                    )
        
        # Set up the filter with containers
        feature_filter.setup(topology['containers'])
        
        # Subscribe feature containers
        source_pattern = spec.get('source_pattern', '*_features')
        for name, container in topology['containers'].items():
            if self._matches_pattern(name, source_pattern):
                container.event_bus.subscribe(EventType.FEATURES, feature_filter.handle_event)
                self.logger.info(f"Connected {name} to feature filter")
        
        # Store the filter in topology routes
        if 'routes' not in topology:
            topology['routes'] = []
        topology['routes'].append(feature_filter)
    
    def _apply_root_bus_subscription(self, spec: Dict[str, Any], context: Dict[str, Any],
                                    topology: Dict[str, Any]) -> None:
        """Apply root event bus subscription.
        
        This is appropriate for:
        - Containers listening to stateless services (strategies, risk validators)
        - Cross-hierarchy communication that can't use parent-child
        - System-wide events (SYSTEM_START, SYSTEM_STOP)
        
        For container-to-container within a hierarchy, use parent-child communication.
        """
        from ..events import EventType
        
        container_pattern = spec.get('containers')
        event_type = EventType[spec.get('event_type')]
        handler_path = spec.get('handler')
        
        root_bus = context['root_event_bus']
        
        # Log appropriately based on use case
        if event_type.name in ['SIGNAL', 'ORDER', 'RISK_APPROVED', 'RISK_REJECTED']:
            # These are typically from stateless services
            log_level = self.logger.debug
            message_suffix = "for stateless service events"
        else:
            # Other uses should consider parent-child
            log_level = self.logger.info
            message_suffix = "- consider parent-child communication for container events"
        
        for name, container in topology['containers'].items():
            if self._matches_pattern(name, container_pattern):
                # Navigate to handler
                handler = container
                for part in handler_path.split('.'):
                    if hasattr(handler, 'get_component'):
                        handler = handler.get_component(part)
                    else:
                        handler = getattr(handler, part, None)
                    if not handler:
                        break
                
                if handler and callable(handler):
                    root_bus.subscribe(event_type, handler)
                    log_level(f"Subscribed {name}.{handler_path} to {event_type.name} events "
                            f"via root_event_bus {message_suffix}")
    
    def _resolve_value(self, value_spec: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value specification."""
        return self.config_resolver.resolve_value(value_spec, context)
    
    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches pattern."""
        return fnmatch.fnmatch(name, pattern)
    
    def _establish_hierarchy(self, hierarchy_spec: Dict[str, Any], 
                           containers: Dict[str, Any]) -> None:
        """Establish parent-child relationships between containers.
        
        This method encourages the use of hierarchical event communication
        by setting up proper parent-child relationships.
        
        Args:
            hierarchy_spec: Hierarchy specification with parent-child mappings
            containers: Dictionary of created containers
        """
        self.logger.info("Establishing container hierarchy for parent-child communication")
        
        for parent_pattern, children_spec in hierarchy_spec.items():
            # Find matching parent containers
            parent_containers = [
                (name, container) for name, container in containers.items()
                if self._matches_pattern(name, parent_pattern)
            ]
            
            if not parent_containers:
                self.logger.warning(f"No containers match parent pattern: {parent_pattern}")
                continue
            
            # Process children specification
            if isinstance(children_spec, str):
                # Simple pattern matching
                for parent_name, parent_container in parent_containers:
                    child_containers = [
                        (name, container) for name, container in containers.items()
                        if self._matches_pattern(name, children_spec) and name != parent_name
                    ]
                    
                    for child_name, child_container in child_containers:
                        parent_container.add_child_container(child_container)
                        self.logger.info(f"Established hierarchy: {parent_name} -> {child_name}")
                        
            elif isinstance(children_spec, list):
                # List of specific children
                for parent_name, parent_container in parent_containers:
                    for child_pattern in children_spec:
                        child_containers = [
                            (name, container) for name, container in containers.items()
                            if self._matches_pattern(name, child_pattern) and name != parent_name
                        ]
                        
                        for child_name, child_container in child_containers:
                            parent_container.add_child_container(child_container)
                            self.logger.info(f"Established hierarchy: {parent_name} -> {child_name}")
                            
            elif isinstance(children_spec, dict):
                # Advanced specification with roles
                for parent_name, parent_container in parent_containers:
                    for role, child_pattern in children_spec.items():
                        child_containers = [
                            (name, container) for name, container in containers.items()
                            if self._matches_pattern(name, child_pattern) and name != parent_name
                        ]
                        
                        for child_name, child_container in child_containers:
                            parent_container.add_child_container(child_container)
                            # Optionally set a role attribute on the child
                            if hasattr(child_container, 'hierarchy_role'):
                                child_container.hierarchy_role = role
                            self.logger.info(f"Established hierarchy: {parent_name} -> {child_name} (role: {role})")
    
    def _generate_parameter_combinations(self, strategies: List[Dict], 
                                       risk_profiles: List[Dict]) -> List[Dict]:
        """Generate parameter combinations."""
        combinations = []
        combo_id = 0
        
        for strategy in strategies:
            for risk in risk_profiles:
                combinations.append({
                    'combo_id': f"c{combo_id:04d}",
                    'strategy_params': strategy,
                    'risk_params': risk
                })
                combo_id += 1
        
        return combinations
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate a hash of the configuration."""
        config_copy = config.copy()
        config_copy.pop('tracing', None)
        config_copy.pop('execution_metadata', None)
        
        config_str = json.dumps(config_copy, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_supported_modes(self) -> List[str]:
        """Get list of supported topology modes."""
        return list(self.patterns.keys())
    
    def get_pattern(self, mode: str) -> Optional[Dict[str, Any]]:
        """Get pattern definition for inspection."""
        return self._get_pattern(mode)
    
    def _setup_event_subscriptions(self, mode: str, topology: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Set up event subscriptions for the topology based on mode.
        
        This replaces the routing setup with direct EventBus subscriptions.
        """
        containers = topology['containers']
        root_bus = context.get('root_event_bus')
        
        if not root_bus:
            self.logger.warning("No root event bus available for subscriptions")
            return
        
        self.logger.info(f"Setting up event subscriptions for {mode} topology")
        
        if mode in ['backtest', 'optimization']:
            self._setup_backtest_subscriptions(containers, root_bus, topology.get('parameter_combinations', []))
        elif mode == 'signal_generation':
            self._setup_signal_generation_subscriptions(containers, root_bus)
        elif mode == 'signal_replay':
            self._setup_signal_replay_subscriptions(containers, root_bus)
        else:
            self.logger.warning(f"No subscription setup defined for mode: {mode}")
    
    def _setup_backtest_subscriptions(self, containers: Dict[str, Any], root_bus: Any, 
                                     parameter_combinations: List[Dict]) -> None:
        """Set up subscriptions for backtest topology."""
        from ..events import EventType
        
        # 1. Portfolio containers subscribe to SIGNAL events
        portfolio_containers = {k: v for k, v in containers.items() if 'portfolio_' in k}
        
        for portfolio_name, portfolio in portfolio_containers.items():
            # Find the corresponding parameter combination
            combo = None
            for param_combo in parameter_combinations:
                if param_combo.get('combo_id') in portfolio_name:
                    combo = param_combo
                    break
            
            if combo:
                strategy_type = combo.get('strategy_params', {}).get('type')
                strategy_id = f"{combo['combo_id']}_{strategy_type}"
                
                # Get signal processor component
                signal_processor = portfolio.get_component('signal_processor')
                if signal_processor and hasattr(signal_processor, 'on_signal'):
                    # Create filter for this portfolio
                    def create_signal_filter(sid):
                        def filter_func(event):
                            return event.metadata.get('strategy_id') == sid
                        return filter_func
                    
                    # Subscribe with filter
                    root_bus.subscribe(
                        EventType.SIGNAL, 
                        signal_processor.on_signal,
                        filter_func=create_signal_filter(strategy_id)
                    )
                    self.logger.info(f"Portfolio {portfolio_name} subscribed to signals from strategy {strategy_id}")
        
        # 2. Execution subscribes to ORDER events
        execution_container = containers.get('execution')
        if execution_container:
            execution_engine = execution_container.get_component('execution_engine')
            if execution_engine and hasattr(execution_engine, 'on_order'):
                root_bus.subscribe(EventType.ORDER, execution_engine.on_order)
                self.logger.info("Execution engine subscribed to ORDER events")
        
        # 3. Portfolios subscribe to FILL events
        for portfolio_name, portfolio in portfolio_containers.items():
            portfolio_state = portfolio.get_component('portfolio_state')
            if portfolio_state and hasattr(portfolio_state, 'on_fill'):
                root_bus.subscribe(EventType.FILL, portfolio_state.on_fill)
                self.logger.info(f"Portfolio {portfolio_name} subscribed to FILL events")
    
    def _setup_signal_generation_subscriptions(self, containers: Dict[str, Any], root_bus: Any) -> None:
        """Set up subscriptions for signal generation topology."""
        # Feature container subscribes to BAR events
        feature_container = containers.get('feature_processor')
        if feature_container:
            root_bus.subscribe('BAR', feature_container.receive_event)
            self.logger.info("Feature processor subscribed to BAR events")
        
        # Signals are captured via event tracing - no additional subscriptions needed
    
    def _setup_signal_replay_subscriptions(self, containers: Dict[str, Any], root_bus: Any) -> None:
        """Set up subscriptions for signal replay topology."""
        from ..events import EventType
        
        # Similar to backtest but without feature processing
        portfolio_containers = {k: v for k, v in containers.items() if 'portfolio_' in k}
        
        # 1. Portfolios subscribe to SIGNAL events (from replay)
        for portfolio_name, portfolio in portfolio_containers.items():
            signal_processor = portfolio.get_component('signal_processor')
            if signal_processor and hasattr(signal_processor, 'on_signal'):
                root_bus.subscribe(EventType.SIGNAL, signal_processor.on_signal)
                self.logger.info(f"Portfolio {portfolio_name} subscribed to replayed signals")
        
        # 2. Execution subscribes to ORDER events
        execution_container = containers.get('execution')
        if execution_container:
            execution_engine = execution_container.get_component('execution_engine')
            if execution_engine and hasattr(execution_engine, 'on_order'):
                root_bus.subscribe(EventType.ORDER, execution_engine.on_order)
                self.logger.info("Execution engine subscribed to ORDER events")
        
        # 3. Portfolios subscribe to FILL events
        for portfolio_name, portfolio in portfolio_containers.items():
            portfolio_state = portfolio.get_component('portfolio_state')
            if portfolio_state and hasattr(portfolio_state, 'on_fill'):
                root_bus.subscribe(EventType.FILL, portfolio_state.on_fill)
                self.logger.info(f"Portfolio {portfolio_name} subscribed to FILL events")
    def _infer_and_inject_features(self, context: Dict[str, Any]) -> None:
        """
        Automatically infer required features from strategy configurations
        and inject them into the context for feature container creation.
        
        This is the key integration point that makes user configs simple:
        - User just lists strategies with parameters
        - System automatically determines needed features
        - Features get added to context for feature containers to use
        """
        # Look for strategies in the user config
        strategies = context.get("strategies", [])
        if not strategies:
            self.logger.info("No strategies found in config, skipping feature inference")
            return
        
        self.logger.info(f"Inferring features from {len(strategies)} strategies")
        
        # Call the feature inference logic
        try:
            required_features = self._infer_features_from_strategies(strategies)
            
            if required_features:
                self.logger.info(f"Inferred features: {sorted(required_features)}")
                
                # Inject the inferred features into context
                # This will be picked up by feature containers
                if "inferred_features" not in context:
                    context["inferred_features"] = list(required_features)
                else:
                    # Merge with any existing features
                    existing = set(context["inferred_features"])
                    combined = existing.union(required_features)
                    context["inferred_features"] = list(combined)
                    
                # Also add to top-level features config for backward compatibility
                if "features" not in context:
                    context["features"] = list(required_features)
                else:
                    existing = set(context.get("features", []))
                    combined = existing.union(required_features)
                    context["features"] = list(combined)
                    
                self.logger.info(f"Injected {len(required_features)} features into context")
            else:
                self.logger.warning("Feature inference returned no features")
                
        except Exception as e:
            self.logger.error(f"Error during feature inference: {e}")
            # Do not fail the topology build, just log the error
    
    def _infer_features_from_strategies(self, strategies: List[Dict[str, Any]]) -> Set[str]:
        """Infer required features from strategy configurations using discovery system.
        
        This uses the discovery registry to automatically determine what features
        are needed based on strategy metadata and parameter values.
        
        Args:
            strategies: List of strategy configuration dictionaries
            
        Returns:
            Set of required feature identifiers
        """
        from ..components.discovery import get_component_registry
        
        required_features = set()
        registry = get_component_registry()
        
        for strategy_config in strategies:
            strategy_type = strategy_config.get('type', strategy_config.get('class'))
            strategy_params = strategy_config.get('parameters', {})
            
            # Get strategy metadata from registry
            strategy_info = registry.get_component(strategy_type)
            
            if strategy_info:
                # Extract feature requirements from metadata
                feature_config = strategy_info.metadata.get('feature_config', {})
                
                # For each feature type the strategy needs
                for feature_name, feature_meta in feature_config.items():
                    param_names = feature_meta.get('params', [])
                    defaults = feature_meta.get('defaults', {})
                    default_value = feature_meta.get('default')
                    
                    # Handle parameter lists (for grid search)
                    if param_names:
                        for param_name in param_names:
                            if param_name in strategy_params:
                                param_values = strategy_params[param_name]
                                # Handle both single values and lists
                                if isinstance(param_values, list):
                                    for value in param_values:
                                        required_features.add(f'{feature_name}_{value}')
                                else:
                                    required_features.add(f'{feature_name}_{param_values}')
                            elif param_name in defaults:
                                # Use specific default for this param
                                required_features.add(f'{feature_name}_{defaults[param_name]}')
                            elif default_value is not None:
                                # Use general default
                                required_features.add(f'{feature_name}_{default_value}')
                    else:
                        # Feature with no params, just add it
                        required_features.add(feature_name)
                
                self.logger.info(f"Strategy '{strategy_type}' requires features: {sorted(required_features)}")
                
            else:
                # Fallback for strategies not in registry
                self.logger.warning(f"Strategy '{strategy_type}' not found in registry, using hardcoded inference")
                
                # Legacy hardcoded logic as fallback
                if strategy_type in ['MomentumStrategy', 'momentum']:
                    lookback_period = strategy_params.get('lookback_period', 20)
                    rsi_period = strategy_params.get('rsi_period', 14)
                    required_features.add(f'sma_{lookback_period}')
                    required_features.add(f'rsi_{rsi_period}')
                elif strategy_type in ['MeanReversionStrategy', 'mean_reversion']:
                    period = strategy_params.get('period', 20)
                    required_features.add(f'bollinger_{period}')
                    required_features.add('rsi_14')
                else:
                    # Default features
                    required_features.update(['sma_20', 'rsi_14'])
        
        # If no strategies found, add default features
        if not required_features:
            self.logger.warning("No strategies found, using default features")
            required_features.update(['sma_20', 'rsi_14'])
            
        return required_features
    
    def _get_strategy_requirements(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive requirements for a single strategy.
        
        Args:
            strategy_config: Strategy configuration dictionary
            
        Returns:
            Dictionary with indicators, dependencies, and other requirements
        """
        strategy_class = strategy_config.get('class', strategy_config.get('type'))
        strategy_params = strategy_config.get('parameters', {})
        
        requirements = {
            'indicators': set(),
            'dependencies': [],
            'risk_requirements': {},
            'data_requirements': {}
        }
        
        if strategy_class in ['MomentumStrategy', 'momentum']:
            lookback_period = strategy_params.get('lookback_period', 20)
            requirements['indicators'].update([f'SMA_{lookback_period}', 'RSI'])
            requirements['data_requirements']['min_history'] = max(lookback_period, 14) + 5
            
        elif strategy_class in ['MeanReversionStrategy', 'mean_reversion']:
            period = strategy_params.get('period', 20)
            requirements['indicators'].update([f'BB_{period}', 'RSI'])
            requirements['data_requirements']['min_history'] = max(period, 14) + 5
            
        elif strategy_class in ['moving_average_crossover', 'momentum_crossover']:
            for param_name, param_value in strategy_params.items():
                if 'fast_period' in param_name:
                    requirements['indicators'].add(f'SMA_{param_value}')
                elif 'slow_period' in param_name:
                    requirements['indicators'].add(f'SMA_{param_value}')
                    requirements['data_requirements']['min_history'] = max(
                        requirements['data_requirements'].get('min_history', 0),
                        param_value + 5
                    )
                elif 'rsi_period' in param_name:
                    requirements['indicators'].add('RSI')
        
        # Convert set to list for JSON serialization
        requirements['indicators'] = list(requirements['indicators'])
        
        return requirements
    
    def _validate_strategy_configuration(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a strategy configuration and return validation results.
        
        Args:
            strategy_config: Strategy configuration to validate
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not strategy_config.get('type') and not strategy_config.get('class'):
            errors.append("Strategy configuration missing 'type' or 'class' field")
        
        strategy_class = strategy_config.get('class', strategy_config.get('type'))
        strategy_params = strategy_config.get('parameters', {})
        
        # Strategy-specific validation
        if strategy_class in ['MomentumStrategy', 'momentum']:
            lookback_period = strategy_params.get('lookback_period', 20)
            if lookback_period < 5:
                warnings.append(f"Momentum lookback period {lookback_period} is very short")
            elif lookback_period > 200:
                warnings.append(f"Momentum lookback period {lookback_period} is very long")
                
        elif strategy_class in ['MeanReversionStrategy', 'mean_reversion']:
            period = strategy_params.get('period', 20)
            if period < 5:
                warnings.append(f"Mean reversion period {period} is very short")
                
        elif strategy_class in ['moving_average_crossover', 'momentum_crossover']:
            fast_period = None
            slow_period = None
            
            for param_name, param_value in strategy_params.items():
                if 'fast_period' in param_name:
                    fast_period = param_value
                elif 'slow_period' in param_name:
                    slow_period = param_value
                    
            if fast_period and slow_period:
                if fast_period >= slow_period:
                    errors.append(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'strategy_class': strategy_class,
            'requirements': self._get_strategy_requirements(strategy_config)
        }
    
    def _get_strategy_feature_requirements(self, strategy_type: str, strategy_config: Dict[str, Any]) -> List[str]:
        """Get feature requirements for a specific strategy type.
        
        This is used by the feature dispatcher behavior to determine what
        features each strategy needs.
        
        Args:
            strategy_type: Type of strategy
            strategy_config: Strategy configuration
            
        Returns:
            List of required feature names
        """
        requirements = self._get_strategy_requirements(strategy_config)
        return requirements.get('indicators', [])

