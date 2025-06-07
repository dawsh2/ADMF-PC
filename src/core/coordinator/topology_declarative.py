"""
Generic TopologyBuilder Implementation

Builds topologies from YAML patterns or Python dictionaries.
Completely data-driven - no hardcoded topology logic.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import hashlib
import json
import yaml
import fnmatch
import itertools

logger = logging.getLogger(__name__)


class TopologyBuilder:
    """
    Generic topology builder that constructs topologies from patterns.
    
    Instead of hardcoded topology functions, this builder interprets
    declarative patterns that describe what containers, components,
    and routes to create.
    """
    
    def __init__(self):
        """Initialize topology builder."""
        self.logger = logging.getLogger(__name__)
        self.patterns = self._load_patterns()
        self.container_factory = None
        self.routing_factory = None
        
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load topology patterns from YAML files."""
        patterns = {}
        pattern_dir = Path(__file__).parent / 'patterns' / 'topologies'
        
        # Create patterns directory if it doesn't exist
        if not pattern_dir.exists():
            pattern_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created topology patterns directory: {pattern_dir}")
        
        # Load YAML patterns
        for pattern_file in pattern_dir.glob('*.yaml'):
            try:
                with open(pattern_file) as f:
                    pattern = yaml.safe_load(f)
                    patterns[pattern_file.stem] = pattern
                    self.logger.info(f"Loaded pattern: {pattern_file.stem}")
            except Exception as e:
                self.logger.error(f"Failed to load pattern {pattern_file}: {e}")
        
        # Also check for Python patterns for backward compatibility
        try:
            from . import patterns as python_patterns
            for name in dir(python_patterns):
                if name.endswith('_PATTERN'):
                    pattern_name = name[:-8].lower()  # Remove _PATTERN suffix
                    patterns[pattern_name] = getattr(python_patterns, name)
                    self.logger.info(f"Loaded Python pattern: {pattern_name}")
        except ImportError:
            pass
        
        return patterns
    
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
        context = self._build_context(pattern, config, tracing_config)
        
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
        
        # 3. Create routes
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
        
        self.logger.info(
            f"Built {mode} topology with {len(topology['containers'])} containers "
            f"and {len(topology['routes'])} routes"
        )
        
        return topology
    
    def _initialize_factories(self):
        """Initialize container and routing factories."""
        if not self.container_factory:
            from ..containers.factory import ContainerFactory
            self.container_factory = ContainerFactory()
            
        if not self.routing_factory:
            from ..routing.factory import RoutingFactory
            self.routing_factory = RoutingFactory()
    
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
                      tracing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build evaluation context for pattern interpretation."""
        context = {
            'config': config,
            'pattern': pattern,
            'tracing': tracing_config,
            'generated': {},
            'root_event_bus': None
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
                'container_settings': tracing_config.get('container_settings', {})
            }
        
        # Create/get root event bus
        if hasattr(self.container_factory, 'root_event_bus'):
            context['root_event_bus'] = self.container_factory.root_event_bus
        else:
            from ..events import EventBus
            context['root_event_bus'] = EventBus()
        
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
        # Import component builders
        from .topologies.helpers.component_builder import (
            create_strategy, create_classifier, create_risk_validator
        )
        
        try:
            if comp_type == 'strategies':
                return create_strategy(config.get('type'), config)
            elif comp_type == 'classifiers':
                return create_classifier(config.get('type'), config)
            elif comp_type == 'risk_validators':
                return create_risk_validator(config.get('type'), config)
            else:
                self.logger.warning(f"Unknown component type: {comp_type}")
                return None
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
            return self.container_factory.create_container(name, config)
        except Exception as e:
            self.logger.error(f"Failed to create container {name}: {e}")
            return None
    
    def _create_route(self, route_spec: Dict[str, Any], context: Dict[str, Any],
                     containers: Dict[str, Any]) -> Optional[Any]:
        """Create a route from specification."""
        config = {
            'type': route_spec.get('type'),
            'root_event_bus': context['root_event_bus']
        }
        
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
            
            route = self.routing_factory.create_route(name, config)
            route.setup(containers)
            route.start()
            return route
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
        from ..routing.factory import RoutingFactory
        from ..events import EventType, Event
        
        # Create feature filter route
        routing_factory = RoutingFactory()
        feature_filter = routing_factory.create_route(
            name='feature_filter',
            config={
                'type': 'filter',
                'filter_field': 'payload.features',
                'event_types': [EventType.FEATURES]
            }
        )
        
        # Register strategies
        if 'parameter_combinations' in context['generated']:
            root_event_bus = context['root_event_bus']
            
            for combo in context['generated']['parameter_combinations']:
                strategy_config = combo['strategy_params']
                strategy_type = strategy_config.get('type')
                combo_id = combo['combo_id']
                
                if strategy_type:
                    # Get feature requirements
                    from .topologies.helpers.component_builder import get_strategy_feature_requirements
                    required_features = get_strategy_feature_requirements(strategy_type, strategy_config)
                    
                    # Create strategy transform function
                    def create_strategy_transform(sid, stype, sconfig):
                        def transform(event):
                            # Create strategy instance
                            from ..components.factory import create_component
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
        """Apply root event bus subscription."""
        from ..events import EventType
        
        container_pattern = spec.get('containers')
        event_type = EventType[spec.get('event_type')]
        handler_path = spec.get('handler')
        
        root_bus = context['root_event_bus']
        
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
                    self.logger.info(f"Subscribed {name}.{handler_path} to {event_type.name} events")
    
    def _resolve_value(self, value_spec: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value specification."""
        if isinstance(value_spec, str):
            # Handle template strings
            if '{' in value_spec and '}' in value_spec:
                # Format with context variables
                try:
                    return value_spec.format(**context)
                except KeyError:
                    # Try with specific sub-contexts
                    format_context = {}
                    if 'combo' in context:
                        format_context.update(context['combo'])
                    format_context.update(context)
                    return value_spec.format(**format_context)
            # Handle context references
            elif value_spec.startswith('$'):
                path = value_spec[1:].split('.')
                value = context
                for part in path:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = getattr(value, part, None)
                    if value is None:
                        break
                return value
        elif isinstance(value_spec, dict):
            if 'from_config' in value_spec:
                path = value_spec['from_config'].split('.')
                value = context['config']
                for part in path:
                    if isinstance(value, dict):
                        value = value.get(part, value_spec.get('default'))
                    else:
                        value = getattr(value, part, value_spec.get('default', None))
                    if value is None:
                        value = value_spec.get('default')
                        break
                return value
            elif 'value' in value_spec:
                return value_spec['value']
        
        return value_spec
    
    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches pattern."""
        return fnmatch.fnmatch(name, pattern)
    
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