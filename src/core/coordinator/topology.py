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
        
        # Auto-extract strategy names from strategies list for portfolio containers
        self._extract_and_inject_strategy_names(context)
        
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
        if 'behaviors' in pattern and pattern['behaviors'] is not None:
            self.logger.info("Applying behaviors")
            for behavior_spec in pattern['behaviors']:
                self._apply_behavior(behavior_spec, context, topology)
        
        # Add metadata including expanded parameter configurations
        topology['metadata'] = {
            'mode': mode,
            'pattern': pattern.get('name', mode),
            'created_at': str(datetime.now()),
            'config_hash': self._hash_config(config),
            'tracing_enabled': tracing_config.get('enabled', False),
            
            # Include expanded configurations for analytics
            'expanded_strategies': config.get('strategies', []),
            'expanded_classifiers': config.get('classifiers', []),
            'original_config': {
                'strategies': context.get('original_strategies', []),
                'classifiers': context.get('original_classifiers', [])
            },
            
            **(metadata or {})
        }
        
        # 5. Set up event subscriptions based on mode
        self._setup_event_subscriptions(mode, topology, context)
        
        # 6. Set up unified signal tracing if enabled
        execution_config = config.get('execution', {})
        trace_enabled = execution_config.get('enable_event_tracing', False)
        use_sparse = execution_config.get('trace_settings', {}).get('use_sparse_storage', False)
        
        # For signal generation, always set up MultiStrategyTracer regardless of container tracing
        # MultiStrategyTracer handles signal storage independently from event tracing
        if mode == 'signal_generation' and use_sparse:
            self._setup_multi_strategy_tracer(topology, context, tracing_config)
        
        self.logger.info(
            f"Built {mode} topology with {len(topology['containers'])} containers "
            f"and {len(topology['routes'])} routes"
        )
        
        return topology
    
    def _setup_multi_strategy_tracer(self, topology: Dict[str, Any], 
                                    context: Dict[str, Any], 
                                    tracing_config: Dict[str, Any]) -> None:
        """Set up unified multi-strategy tracer on the root event bus."""
        # Check if streaming tracer should be used for large runs
        max_bars = context['config'].get('max_bars', 0)
        use_streaming = max_bars > 2000 or context['config'].get('streaming_tracer', False)
        
        if use_streaming:
            from ..events.observers.streaming_multi_strategy_tracer import StreamingMultiStrategyTracer
            self.logger.info(f"Using StreamingMultiStrategyTracer for {max_bars} bars")
        else:
            from ..events.observers.multi_strategy_tracer import MultiStrategyTracer
        from ..events.types import EventType
        
        # Get workspace path from trace settings
        trace_settings = context['config'].get('execution', {}).get('trace_settings', {})
        workspace_path = trace_settings.get('storage', {}).get('base_dir', './workspaces')
        
        # Get study configuration for organized workspace structure
        results_dir = context['config'].get('results_dir')
        wfv_window = context['config'].get('wfv_window')
        phase = context['config'].get('phase')
        
        # Create workspace directory based on study organization
        import os
        import uuid
        
        if results_dir and wfv_window and phase:
            # WFV execution: study_name/window_XX_phase/
            workspace_name = f"window_{wfv_window:02d}_{phase}"
            full_workspace_path = os.path.join(workspace_path, results_dir, workspace_name)
            self.logger.info(f"WFV workspace: {results_dir}/window_{wfv_window:02d}_{phase}")
        elif results_dir:
            # Study execution without WFV: study_name/run_unique_id/
            unique_run_id = str(uuid.uuid4())[:8]
            workspace_name = f"run_{unique_run_id}"
            full_workspace_path = os.path.join(workspace_path, results_dir, workspace_name)
            self.logger.info(f"Study workspace: {results_dir}/{workspace_name}")
        else:
            # Fallback to legacy naming for backwards compatibility
            config_name = context.get('metadata', {}).get('config_name', 'unknown_config')
            if not config_name or config_name == 'unknown_config':
                config_file = context.get('metadata', {}).get('config_file', '')
                if config_file:
                    config_name = Path(config_file).stem
                else:
                    config_name = 'signal_generation'
            
            unique_run_id = str(uuid.uuid4())[:8]
            workspace_name = f"{config_name}_{unique_run_id}"
            full_workspace_path = os.path.join(workspace_path, workspace_name)
            self.logger.info(f"Legacy workspace: {workspace_name}")
        
        # Get all strategy and classifier IDs from expanded configurations
        strategy_ids = []
        classifier_ids = []
        
        # Extract strategy IDs from expanded strategies
        for strategy in context['config'].get('strategies', []):
            strategy_name = strategy.get('name', '')
            if strategy_name:
                # Add symbol prefix if we have symbols
                symbols = context['config'].get('symbols', ['SPY'])
                for symbol in symbols:
                    strategy_ids.append(f"{symbol}_{strategy_name}")
        
        # Extract classifier IDs from expanded classifiers
        for classifier in context['config'].get('classifiers', []):
            classifier_name = classifier.get('name', '')
            if classifier_name:
                # Add symbol prefix if we have symbols
                symbols = context['config'].get('symbols', ['SPY'])
                for symbol in symbols:
                    classifier_ids.append(f"{symbol}_{classifier_name}")
        
        # Extract data source configuration for source metadata
        data_source_config = {
            'data_dir': context['config'].get('data_dir', './data'),
            'data_source': context['config'].get('data_source', 'csv'),
            'symbols': context['config'].get('symbols', []),
            'timeframes': context['config'].get('timeframes', ['1m'])
        }
        
        # Create the multi-strategy tracer
        if use_streaming:
            # Get write settings from config (default to no periodic writes)
            write_interval = trace_settings.get('write_interval', 0)
            write_on_changes = trace_settings.get('write_on_changes', 0)
            
            tracer = StreamingMultiStrategyTracer(
                workspace_path=full_workspace_path,
                workflow_id=config_name,
                managed_strategies=strategy_ids if strategy_ids else None,
                managed_classifiers=classifier_ids if classifier_ids else None,
                data_source_config=data_source_config,
                write_interval=write_interval,
                write_on_changes=write_on_changes
            )
        else:
            tracer = MultiStrategyTracer(
                workspace_path=full_workspace_path,
                workflow_id=config_name,
                managed_strategies=strategy_ids if strategy_ids else None,
                managed_classifiers=classifier_ids if classifier_ids else None,
                data_source_config=data_source_config
            )
        
        # Attach to root event bus - use the actual root container's bus if available
        root_container = topology.get('containers', {}).get('root')
        if root_container and hasattr(root_container, 'event_bus'):
            # Use the actual root container's event bus
            root_bus = root_container.event_bus
            root_bus.attach_observer(tracer)
        else:
            # Fallback to context bus
            root_bus = context.get('root_event_bus')
            if root_bus:
                root_bus.attach_observer(tracer)
            else:
                self.logger.warning("No event bus found for MultiStrategyTracer attachment")
                return
        
        self.logger.info(f"MultiStrategyTracer attached as observer to root event bus")
        self.logger.info(f"Tracing {len(strategy_ids)} strategies and {len(classifier_ids)} classifiers")
        self.logger.info(f"Workspace: {full_workspace_path}")
        
        # Store tracer reference in topology for finalization
        topology['multi_strategy_tracer'] = tracer
    
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
        
        return None
    
    def _build_context(self, pattern: Dict[str, Any], config: Dict[str, Any], 
                      tracing_config: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build evaluation context for pattern interpretation."""
        # Store original configurations before expansion for analytics
        original_strategies = config.get('strategies', []).copy() if 'strategies' in config else []
        original_classifiers = config.get('classifiers', []).copy() if 'classifiers' in config else []
        
        # Expand parameter lists in strategies before building context
        if 'strategies' in config:
            config['strategies'] = self._expand_strategy_parameters(config['strategies'])
        
        # Expand parameter lists in classifiers before building context
        if 'classifiers' in config:
            config['classifiers'] = self._expand_classifier_parameters(config['classifiers'])
        
        context = {
            'config': config,
            'pattern': pattern,
            'tracing': tracing_config,
            'metadata': metadata,  # Include metadata for container configuration
            'generated': {},
            'root_event_bus': None,  # Still needed for stateless services
            'use_hierarchical_events': True,  # Encourage parent-child for containers
            
            # Store original configs for analytics integration
            'original_strategies': original_strategies,
            'original_classifiers': original_classifiers
        }
        
        # Process tracing configuration with CORRECT structure
        # Preserve original execution config if it exists
        original_exec_config = config.get('execution', {})
        original_trace_settings = original_exec_config.get('trace_settings', {})
        
        if tracing_config.get('enabled', False) or original_exec_config.get('enable_event_tracing', False):
            if 'execution' not in config:
                config['execution'] = {}
            
            config['execution']['enable_event_tracing'] = True
            config['execution']['trace_settings'] = {
                'trace_id': tracing_config.get('trace_id'),
                'trace_dir': tracing_config.get('trace_dir', './traces'),
                'max_events': tracing_config.get('max_events', 10000),
                'storage_backend': tracing_config.get('storage_backend', 'memory'),
                'batch_size': tracing_config.get('batch_size', 1000),
                'auto_flush_on_cleanup': tracing_config.get('auto_flush_on_cleanup', True),
                'container_settings': tracing_config.get('container_settings', {}),
                'enable_console_output': tracing_config.get('enable_console_output', False),
                'console_filter': tracing_config.get('console_filter', []),
                # Preserve all original trace settings including use_sparse_storage
                **original_trace_settings
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
            # Direct access to config instead of using broken resolver
            items = context.get('config', {}).get(from_config, [])
            self.logger.info(f"Resolved {from_config} to: {items}")
            if not isinstance(items, list):
                items = [items] if items else []
                
            for item in items:
                self.logger.info(f"Processing {comp_type} item: {item}")
                # Skip if item is not a dict (e.g., if it's a string from bad resolution)
                if not isinstance(item, dict):
                    continue
                component = self._create_single_component(comp_type, item)
                if component:
                    name = item.get('name', item.get('type', comp_type))
                    if comp_type not in components:
                        components[comp_type] = {}
                    components[comp_type][name] = component
                    self.logger.info(f"Added {comp_type} component: {name}")
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
        """Find and return stateless function reference."""
        import importlib
        import importlib.util
        import inspect
        
        try:
            if comp_type == 'strategies':
                # Get strategy function reference
                strategy_type = config.get('type', '')
                
                # Map strategy types to their likely function names
                function_candidates = [
                    f'{strategy_type}_strategy',  # momentum -> momentum_strategy
                    strategy_type,                # momentum -> momentum
                    f'{strategy_type.replace("_strategy", "")}_strategy',  # momentum_strategy -> momentum_strategy
                ]
                
                # Try importing from the appropriate module
                module_name = strategy_type.replace('_strategy', '')  # Remove _strategy suffix if present
                module_path = f'src.strategy.strategies.{module_name}'
                
                # Try multiple import strategies
                module = None
                import_errors = []
                
                # Strategy 1: Try direct module import
                try:
                    module = importlib.import_module(module_path)
                except ImportError as e:
                    import_errors.append(f"Direct import failed: {e}")
                    
                # Strategy 2: Try importing from indicators submodules
                if not module:
                    indicator_modules = {
                        # Crossover strategies
                        'sma_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'ema_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'ema_sma_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'dema_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'dema_sma_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'tema_sma_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'stochastic_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'vortex_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'macd_crossover': 'src.strategy.strategies.indicators.crossovers',
                        'ichimoku_cloud_position': 'src.strategy.strategies.indicators.crossovers',
                        # Oscillator strategies
                        'rsi_threshold': 'src.strategy.strategies.indicators.oscillators',
                        'rsi_bands': 'src.strategy.strategies.indicators.oscillators',
                        'cci_threshold': 'src.strategy.strategies.indicators.oscillators',
                        'cci_bands': 'src.strategy.strategies.indicators.oscillators',
                        'stochastic_rsi': 'src.strategy.strategies.indicators.oscillators',
                        'williams_r': 'src.strategy.strategies.indicators.oscillators',
                        'roc_threshold': 'src.strategy.strategies.indicators.oscillators',
                        'ultimate_oscillator': 'src.strategy.strategies.indicators.oscillators',
                        # Volatility strategies
                        'keltner_breakout': 'src.strategy.strategies.indicators.volatility',
                        'donchian_breakout': 'src.strategy.strategies.indicators.volatility',
                        'bollinger_breakout': 'src.strategy.strategies.indicators.volatility',
                        # Volume strategies
                        'obv_trend': 'src.strategy.strategies.indicators.volume',
                        'mfi_bands': 'src.strategy.strategies.indicators.volume',
                        'vwap_deviation': 'src.strategy.strategies.indicators.volume',
                        'chaikin_money_flow': 'src.strategy.strategies.indicators.volume',
                        'accumulation_distribution': 'src.strategy.strategies.indicators.volume',
                    }
                    
                    if strategy_type in indicator_modules:
                        try:
                            module = importlib.import_module(indicator_modules[strategy_type])
                            module_path = indicator_modules[strategy_type]
                        except ImportError as e:
                            import_errors.append(f"Indicator module import failed: {e}")
                
                # Strategy 3: Try file-based import as last resort
                if not module:
                    try:
                        import sys
                        import os
                        # Import the specific module directly
                        spec = importlib.util.spec_from_file_location(
                            module_path,
                            f"src/strategy/strategies/{module_name}.py"
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_path] = module
                            spec.loader.exec_module(module)
                        else:
                            raise ImportError(f"Could not find module file for {module_name}")
                    except Exception as e:
                        import_errors.append(f"File-based import failed: {e}")
                
                if not module:
                    self.logger.error(f"Failed to import strategy module. Errors: {import_errors}")
                    raise ImportError(f"Could not import strategy module {module_path}")
                
                # Look for the function by name
                for candidate in function_candidates:
                    if hasattr(module, candidate):
                        func = getattr(module, candidate)
                        if callable(func):
                            self.logger.info(f"Found strategy function: {candidate} in {module_path}")
                            return func
                
                self.logger.warning(f"No strategy function found for type '{strategy_type}' in {module_path}")
                return None
                    
            elif comp_type == 'classifiers':
                # Get classifier function reference
                classifier_type = config.get('type', '')
                
                # Map classifier types to their function names
                function_candidates = [
                    classifier_type,  # momentum_regime_classifier -> momentum_regime_classifier
                ]
                
                # Try importing from multiple classifier modules
                module_paths = [
                    'src.strategy.classifiers.classifiers',
                    'src.strategy.classifiers.enhanced_multi_state_classifiers'
                ]
                
                for module_path in module_paths:
                    try:
                        module = importlib.import_module(module_path)
                        
                        # Look for the function by name
                        for candidate in function_candidates:
                            if hasattr(module, candidate):
                                func = getattr(module, candidate)
                                if callable(func):
                                    self.logger.info(f"Found classifier function: {candidate} in {module_path}")
                                    return func
                        
                    except ImportError as e:
                        self.logger.debug(f"Could not import classifier module {module_path}: {e}")
                        continue
                
                self.logger.warning(f"No classifier function found for type '{classifier_type}' in any module")
                return None
                
            else:
                # Other component types not implemented yet
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to find component {comp_type}: {e}")
            return None
    
    def _create_containers(self, container_spec: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Create containers from specification."""
        containers = {}
        
        if 'foreach' in container_spec:
            # Create multiple containers using foreach
            containers.update(self._expand_container_foreach(container_spec, context, None))
        else:
            # Create single container
            name = self._resolve_value(container_spec.get('name'), context)
            if name:
                container = self._create_single_container(name, container_spec, context, None)
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
                                 context: Dict[str, Any],
                                 parent_container: Optional[Any] = None) -> Dict[str, Any]:
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
                container = self._create_single_container(name, spec, iter_context, parent_container)
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
                container = self._create_single_container(name, spec, iter_context, parent_container)
                if container:
                    containers[name] = container
        
        return containers
    
    def _create_single_container(self, name: str, spec: Dict[str, Any], 
                                context: Dict[str, Any], parent_container: Optional[Any] = None) -> Optional[Any]:
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
        
        # Determine parent event bus for child containers
        parent_event_bus = None
        if parent_container:
            parent_event_bus = parent_container.event_bus
        elif name == 'root':
            # Root container creates its own bus
            parent_event_bus = None
        else:
            # Non-root containers without parent should use root bus from context
            parent_event_bus = context.get('root_event_bus')
        
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
            # Extract components list from spec, not config
            components = spec.get('components', [])
            container_type = spec.get('type')
            
            # Remove components and type from config as they're passed separately
            clean_config = {k: v for k, v in config.items() if k not in ['components', 'type']}
            
            # Add metadata from context for tracing and results organization
            if 'metadata' in context:
                clean_config['metadata'] = context['metadata']
            
            container = self.container_factory.create_container(
                name=name,
                components=components,
                config=clean_config,
                container_type=container_type,
                parent_event_bus=parent_event_bus
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
                children = self._expand_container_foreach(child_spec, context, parent_container)
                for name, child in children.items():
                    if child:
                        parent_container.add_child_container(child)
                        child_containers[name] = child
                        self.logger.info(f"Created child: {parent_container.name} -> {name}")
            else:
                child_name = self._resolve_value(child_spec.get('name'), context)
                if not child_name:
                    child_name = f"{parent_container.name}_{child_spec.get('type', 'child')}"
                
                # Create child with same context and parent container
                child_container = self._create_single_container(child_name, child_spec, context, parent_container)
                
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
    
    def _expand_strategy_parameters(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand strategies with list parameters into individual strategy configurations.
        
        For example:
        [{
            'name': 'ma_crossover',
            'type': 'ma_crossover',
            'params': {
                'fast_period': [5, 10],
                'slow_period': [20, 30]
            }
        }]
        
        Becomes:
        [
            {'name': 'ma_5_20', 'type': 'ma_crossover', 'params': {'fast_period': 5, 'slow_period': 20}},
            {'name': 'ma_5_30', 'type': 'ma_crossover', 'params': {'fast_period': 5, 'slow_period': 30}},
            {'name': 'ma_10_20', 'type': 'ma_crossover', 'params': {'fast_period': 10, 'slow_period': 20}},
            {'name': 'ma_10_30', 'type': 'ma_crossover', 'params': {'fast_period': 10, 'slow_period': 30}}
        ]
        """
        expanded_strategies = []
        
        for strategy_config in strategies:
            strategy_type = strategy_config.get('type', 'unknown')
            base_name = strategy_config.get('name', strategy_type)
            params = strategy_config.get('params', {})
            
            # Find parameters that are lists (to be expanded)
            list_params = {}
            scalar_params = {}
            
            for param_name, param_value in params.items():
                if isinstance(param_value, list):
                    list_params[param_name] = param_value
                else:
                    scalar_params[param_name] = param_value
            
            if not list_params:
                # No expansion needed, use as-is
                expanded_strategies.append(strategy_config)
            else:
                # Generate all combinations
                param_names = list(list_params.keys())
                param_values = [list_params[name] for name in param_names]
                
                # Use itertools.product to generate all combinations
                for combination in itertools.product(*param_values):
                    # Create a new strategy config for this combination
                    new_params = scalar_params.copy()
                    
                    # Add the specific values from this combination
                    for i, param_name in enumerate(param_names):
                        new_params[param_name] = combination[i]
                    
                    # Validate parameter combinations for known strategy types
                    if not self._validate_strategy_parameters(strategy_type, new_params):
                        self.logger.warning(f"Skipping invalid parameter combination for {strategy_type}: {new_params}")
                        continue
                    
                    # Generate a descriptive name if using parameter expansion
                    if len(list_params) > 0 and base_name == strategy_type:
                        # Auto-generate name from parameters (e.g., ma_5_20)
                        param_parts = []
                        if 'fast_period' in new_params and 'slow_period' in new_params:
                            param_parts = [str(new_params['fast_period']), str(new_params['slow_period'])]
                        else:
                            # Generic parameter naming
                            param_parts = [str(v) for v in combination]
                        
                        expanded_name = f"{strategy_type}_{'_'.join(param_parts)}"
                    else:
                        # Keep original name with parameter suffix
                        param_suffix = '_'.join(str(v) for v in combination)
                        expanded_name = f"{base_name}_{param_suffix}"
                    
                    expanded_strategy = {
                        'name': expanded_name,
                        'type': strategy_type,
                        'params': new_params
                    }
                    
                    # Copy any other fields from original config
                    for key, value in strategy_config.items():
                        if key not in ['name', 'type', 'params']:
                            expanded_strategy[key] = value
                    
                    expanded_strategies.append(expanded_strategy)
                    
        self.logger.info(f"Expanded {len(strategies)} strategies to {len(expanded_strategies)} configurations")
        
        return expanded_strategies
    
    def _validate_strategy_parameters(self, strategy_type: str, params: Dict[str, Any]) -> bool:
        """
        Validate strategy parameters to ensure they make sense.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        # MA crossover strategies need fast < slow
        if strategy_type in ['ma_crossover', 'moving_average_crossover']:
            fast_period = params.get('fast_period')
            slow_period = params.get('slow_period')
            
            if fast_period is not None and slow_period is not None:
                if fast_period >= slow_period:
                    self.logger.debug(
                        f"Invalid MA crossover: fast_period ({fast_period}) >= slow_period ({slow_period})"
                    )
                    return False
        
        # Bollinger bands need reasonable parameters
        elif strategy_type in ['bollinger_bands', 'mean_reversion']:
            period = params.get('period', params.get('bb_period'))
            std_dev = params.get('std_dev', params.get('num_std'))
            
            if period is not None and period < 2:
                self.logger.debug(f"Invalid Bollinger period: {period} < 2")
                return False
                
            if std_dev is not None and (std_dev <= 0 or std_dev > 5):
                self.logger.debug(f"Invalid Bollinger std_dev: {std_dev}")
                return False
        
        # RSI needs reasonable periods
        elif strategy_type in ['rsi_strategy', 'momentum']:
            rsi_period = params.get('rsi_period', params.get('period'))
            
            if rsi_period is not None and (rsi_period < 2 or rsi_period > 100):
                self.logger.debug(f"Invalid RSI period: {rsi_period}")
                return False
        
        # MACD parameters
        elif strategy_type in ['macd', 'macd_strategy']:
            fast = params.get('fast_period', params.get('fast_ema'))
            slow = params.get('slow_period', params.get('slow_ema'))
            signal = params.get('signal_period', params.get('signal_ema'))
            
            if fast is not None and slow is not None:
                if fast >= slow:
                    self.logger.debug(f"Invalid MACD: fast ({fast}) >= slow ({slow})")
                    return False
                    
            if signal is not None and signal < 1:
                self.logger.debug(f"Invalid MACD signal period: {signal} < 1")
                return False
        
        # If we don't have specific validation rules, assume it's valid
        return True
    
    def _expand_classifier_parameters(self, classifiers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand classifiers with list parameters into individual classifier configurations.
        
        Similar to _expand_strategy_parameters but for classifiers.
        
        For example:
        [{
            'name': 'trend_grid',
            'type': 'trend_classifier',
            'params': {
                'trend_threshold': [0.01, 0.02],
                'fast_ma': [10, 20]
            }
        }]
        
        Becomes:
        [
            {'name': 'trend_0.01_10', 'type': 'trend_classifier', 'params': {'trend_threshold': 0.01, 'fast_ma': 10}},
            {'name': 'trend_0.01_20', 'type': 'trend_classifier', 'params': {'trend_threshold': 0.01, 'fast_ma': 20}},
            {'name': 'trend_0.02_10', 'type': 'trend_classifier', 'params': {'trend_threshold': 0.02, 'fast_ma': 10}},
            {'name': 'trend_0.02_20', 'type': 'trend_classifier', 'params': {'trend_threshold': 0.02, 'fast_ma': 20}}
        ]
        """
        expanded_classifiers = []
        
        for classifier_config in classifiers:
            classifier_type = classifier_config.get('type', 'unknown')
            base_name = classifier_config.get('name', classifier_type)
            params = classifier_config.get('params', {})
            
            # Find parameters that are lists (to be expanded)
            list_params = {}
            scalar_params = {}
            
            for param_name, param_value in params.items():
                if isinstance(param_value, list):
                    list_params[param_name] = param_value
                else:
                    scalar_params[param_name] = param_value
            
            if not list_params:
                # No expansion needed, use as-is
                expanded_classifiers.append(classifier_config)
            else:
                # Generate all combinations
                param_names = list(list_params.keys())
                param_values = [list_params[name] for name in param_names]
                
                # Use itertools.product to generate all combinations
                for combination in itertools.product(*param_values):
                    # Create a new classifier config for this combination
                    new_params = scalar_params.copy()
                    
                    # Add the specific values from this combination
                    for i, param_name in enumerate(param_names):
                        new_params[param_name] = combination[i]
                    
                    # Generate a descriptive name if using parameter expansion
                    if len(list_params) > 0 and base_name == classifier_type:
                        # Auto-generate name from parameters
                        param_parts = [str(v).replace('.', '') for v in combination]
                        expanded_name = f"{classifier_type}_{'_'.join(param_parts)}"
                    else:
                        # Keep original name with parameter suffix
                        param_suffix = '_'.join(str(v).replace('.', '') for v in combination)
                        expanded_name = f"{base_name}_{param_suffix}"
                    
                    expanded_classifier = {
                        'name': expanded_name,
                        'type': classifier_type,
                        'params': new_params
                    }
                    
                    # Copy any other fields from original config
                    for key, value in classifier_config.items():
                        if key not in ['name', 'type', 'params']:
                            expanded_classifier[key] = value
                    
                    expanded_classifiers.append(expanded_classifier)
                    
        self.logger.info(f"Expanded {len(classifiers)} classifiers to {len(expanded_classifiers)} configurations")
        
        return expanded_classifiers
    
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
        from ..events import EventType
        
        # Portfolio containers subscribe to SIGNAL events from root bus
        portfolio_containers = {k: v for k, v in containers.items() if 'portfolio' in k}
        
        for portfolio_name, portfolio in portfolio_containers.items():
            # Get portfolio manager component that can handle signals
            portfolio_manager = portfolio.get_component('portfolio_manager')
            if portfolio_manager and hasattr(portfolio_manager, 'process_event'):
                
                # Subscribe to SIGNAL events on the portfolio's own event bus
                # Signals flow: strategy -> parent (root) -> root forwards to all children including portfolio
                event_type_str = EventType.SIGNAL.value if hasattr(EventType.SIGNAL, 'value') else str(EventType.SIGNAL)
                
                # Get managed strategies for this portfolio from config
                portfolio_config = portfolio.config.config if hasattr(portfolio.config, 'config') else {}
                managed_strategies = portfolio_config.get('managed_strategies', ['default'])
                
                # Create filter function for strategy_id filtering
                def create_strategy_filter(strategies):
                    def filter_func(event):
                        strategy_id = event.payload.get('strategy_id', '') if hasattr(event, 'payload') else ''
                        # For signal generation mode, match any strategy ending with the managed strategy name
                        # e.g., managed_strategies=['default'] matches strategy_id='SPY_momentum'
                        for managed_strategy in strategies:
                            if managed_strategy == 'default' or strategy_id.endswith(f'_{managed_strategy}') or strategy_id == managed_strategy:
                                return True
                        return False
                    return filter_func
                
                # Subscribe directly - Container.receive_event() handles tracing
                portfolio.event_bus.subscribe(
                    event_type_str, 
                    portfolio_manager.process_event,
                    filter_func=create_strategy_filter(managed_strategies)
                )
                self.logger.info(f"Portfolio {portfolio_name} subscribed to '{event_type_str}' events with strategy filter: {managed_strategies}")
            else:
                self.logger.warning(f"Portfolio {portfolio_name} has no portfolio_manager with process_event method")
        
        # Signals are also captured via event tracing for storage and replay
    
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
    def _extract_and_inject_strategy_names(self, context: Dict[str, Any]) -> None:
        """
        Automatically extract strategy names from strategies list and inject
        into context for portfolio container creation.
        
        This allows users to just define strategies with names, and the system
        automatically creates the strategy_names list needed by portfolio containers.
        """
        strategies = context.get('config', {}).get('strategies', [])
        if not strategies:
            self.logger.info("No strategies found in config, skipping strategy name extraction")
            return
        
        # Extract strategy names from the strategies list
        strategy_names = []
        for strategy_config in strategies:
            strategy_name = strategy_config.get('name')
            if strategy_name:
                strategy_names.append(strategy_name)
            else:
                # Fallback to strategy type if no name provided
                strategy_type = strategy_config.get('type', 'default')
                strategy_names.append(strategy_type)
        
        if strategy_names:
            self.logger.info(f"Extracted strategy names from strategies list: {strategy_names}")
            
            # Inject strategy_names into config so portfolio containers can use it
            if 'strategy_names' not in context['config']:
                context['config']['strategy_names'] = strategy_names
                self.logger.info(f"Injected strategy_names into config: {strategy_names}")
            else:
                # Merge with existing strategy_names if present
                existing_names = context['config']['strategy_names']
                if not isinstance(existing_names, list):
                    existing_names = [existing_names]
                
                # Combine and deduplicate
                combined_names = list(set(existing_names + strategy_names))
                context['config']['strategy_names'] = combined_names
                self.logger.info(f"Merged strategy_names: {existing_names} + {strategy_names} = {combined_names}")
        else:
            self.logger.warning("No strategy names could be extracted from strategies list")

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
        strategies = context.get('config', {}).get('strategies', [])
        classifiers = context.get('config', {}).get('classifiers', [])
        
        if not strategies and not classifiers:
            self.logger.info("No strategies or classifiers found in config, skipping feature inference")
            return
        
        self.logger.info(f"Inferring features from {len(strategies)} strategies and {len(classifiers)} classifiers")
        
        # Call the feature inference logic
        try:
            required_features = self._infer_features_from_strategies(strategies)
            
            # Also infer features from classifiers
            classifier_features = self._infer_features_from_classifiers(classifiers)
            required_features = required_features.union(classifier_features)
            
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
                    
                # Convert inferred features to proper feature configs for StrategyState
                feature_configs = {}
                
                # Create feature configs for all inferred features
                for feature_id in required_features:
                    # Parse feature_id like 'sma_5' or 'rsi_14'
                    parts = feature_id.split('_')
                    
                    # Handle special compound features
                    if feature_id.startswith('atr_sma_'):
                        # Pattern: atr_sma_14_20 -> atr_sma with atr_period=14, sma_period=20
                        parts = feature_id.split('_')
                        if len(parts) >= 4:
                            feature_configs[feature_id] = {
                                'feature': 'atr_sma',
                                'atr_period': int(parts[2]),
                                'sma_period': int(parts[3])
                            }
                        else:
                            feature_configs[feature_id] = {'feature': 'atr_sma'}
                    elif feature_id.startswith('volatility_sma_'):
                        # Pattern: volatility_sma_20_20 -> volatility_sma with vol_period=20, sma_period=20
                        parts = feature_id.split('_')
                        if len(parts) >= 4:
                            feature_configs[feature_id] = {
                                'feature': 'volatility_sma',
                                'vol_period': int(parts[2]),
                                'sma_period': int(parts[3])
                            }
                        else:
                            feature_configs[feature_id] = {'feature': 'volatility_sma'}
                    elif len(parts) >= 2:
                        feature_type = parts[0]  # 'sma', 'rsi', etc.
                        try:
                            period_value = int(parts[1])
                            # Special handling for features with non-standard parameters
                            if feature_type == 'macd':
                                # MACD doesn't use 'period', it uses fast/slow/signal
                                # Use the period_value as the signal line period, with defaults for fast/slow
                                feature_configs[feature_id] = {
                                    'feature': 'macd',
                                    'fast': 12,
                                    'slow': 26,
                                    'signal': period_value
                                }
                            elif feature_type == 'stochastic':
                                # Stochastic uses k_period and d_period
                                feature_configs[feature_id] = {
                                    'feature': 'stochastic',
                                    'k_period': period_value,
                                    'd_period': 3  # Default d_period
                                }
                            elif feature_type == 'stochastic_rsi':
                                # Handle stochastic_rsi features with two periods
                                # Pattern: stochastic_rsi_14_14 -> rsi_period=14, stoch_period=14
                                if len(parts) >= 3:
                                    try:
                                        rsi_period = int(parts[1])
                                        stoch_period = int(parts[2])
                                        feature_configs[feature_id] = {
                                            'feature': 'stochastic_rsi',
                                            'rsi_period': rsi_period,
                                            'stoch_period': stoch_period
                                        }
                                    except ValueError:
                                        # Fallback to defaults
                                        feature_configs[feature_id] = {
                                            'feature': 'stochastic_rsi',
                                            'rsi_period': 14,
                                            'stoch_period': 14
                                        }
                                else:
                                    feature_configs[feature_id] = {
                                        'feature': 'stochastic_rsi',
                                        'rsi_period': 14,
                                        'stoch_period': period_value
                                    }
                            elif feature_type == 'ichimoku':
                                # Ichimoku uses multiple periods
                                feature_configs[feature_id] = {
                                    'feature': 'ichimoku',
                                    'conversion_period': 9,
                                    'base_period': 26,
                                    'lead_span_b_period': period_value or 52
                                }
                            else:
                                # Create feature config with 'period' parameter for most features
                                feature_configs[feature_id] = {
                                    'feature': feature_type,
                                    'period': period_value
                                }
                        except ValueError:
                            # Not a numeric period, might be a different feature type
                            if feature_type == 'macd':
                                feature_configs[feature_id] = {'feature': 'macd'}
                            elif feature_type == 'rsi':
                                feature_configs[feature_id] = {'feature': 'rsi'}
                            elif feature_type == 'momentum':
                                feature_configs[feature_id] = {'feature': 'momentum'}
                            elif feature_type == 'keltner':
                                feature_configs[feature_id] = {'feature': 'keltner_channel'}
                            elif feature_type == 'donchian':
                                feature_configs[feature_id] = {'feature': 'donchian_channel'}
                            elif feature_type == 'williams':
                                feature_configs[feature_id] = {'feature': 'williams_r'}
                            elif feature_type == 'bollinger':
                                feature_configs[feature_id] = {'feature': 'bollinger_bands'}
                            elif feature_type == 'ultimate':
                                feature_configs[feature_id] = {'feature': 'ultimate_oscillator'}
                            elif feature_type == 'obv':
                                feature_configs[feature_id] = {'feature': 'obv'}
                            elif feature_type == 'roc':
                                feature_configs[feature_id] = {'feature': 'roc'}
                            elif feature_type == 'cmf':
                                feature_configs[feature_id] = {'feature': 'cmf'}
                            elif feature_type == 'ad':
                                feature_configs[feature_id] = {'feature': 'ad'}
                            elif feature_type == 'aroon':
                                feature_configs[feature_id] = {'feature': 'aroon'}
                            elif feature_type == 'vwap':
                                feature_configs[feature_id] = {'feature': 'vwap'}
                            elif feature_type == 'mfi':
                                feature_configs[feature_id] = {'feature': 'mfi'}
                            elif feature_type == 'supertrend':
                                feature_configs[feature_id] = {'feature': 'supertrend'}
                            elif feature_type == 'psar':
                                feature_configs[feature_id] = {'feature': 'psar'}
                            elif feature_type == 'linear':
                                feature_configs[feature_id] = {'feature': 'linear_regression'}
                            elif feature_type == 'pivot':
                                feature_configs[feature_id] = {'feature': 'pivot_points'}
                            elif feature_type == 'fibonacci':
                                feature_configs[feature_id] = {'feature': 'fibonacci_retracement'}
                            elif feature_type == 'support':
                                feature_configs[feature_id] = {'feature': 'support_resistance'}
                            elif feature_type == 'swing':
                                feature_configs[feature_id] = {'feature': 'swing_points'}
                            elif feature_type == 'di':
                                # DI is part of ADX system, map to ADX
                                feature_configs[feature_id] = {'feature': 'adx'}
                            else:
                                feature_configs[feature_id] = {'feature': feature_type}
                    else:
                        # Simple feature without parameters
                        # Special handling for volume which is raw data, not a calculated feature
                        if feature_id == 'volume':
                            # Volume is provided directly from bar data, not calculated
                            feature_configs[feature_id] = {'feature': 'volume', 'is_raw_data': True}
                        else:
                            feature_configs[feature_id] = {'feature': parts[0] if parts else feature_id}
                
                # Add feature configs to context config for strategy containers  
                if "feature_configs" not in context['config']:
                    context['config']["feature_configs"] = feature_configs
                else:
                    context['config']["feature_configs"].update(feature_configs)
                
                self.logger.info(f"Generated feature configs: {feature_configs}")
                
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
        import importlib
        
        required_features = set()
        registry = get_component_registry()
        
        # Import all indicator modules first to ensure decorators run
        indicator_modules = {
            # Crossover strategies
            'sma_crossover': 'src.strategy.strategies.indicators.crossovers',
            'ema_crossover': 'src.strategy.strategies.indicators.crossovers',
            'ema_sma_crossover': 'src.strategy.strategies.indicators.crossovers',
            'dema_crossover': 'src.strategy.strategies.indicators.crossovers',
            'dema_sma_crossover': 'src.strategy.strategies.indicators.crossovers',
            'tema_sma_crossover': 'src.strategy.strategies.indicators.crossovers',
            'stochastic_crossover': 'src.strategy.strategies.indicators.crossovers',
            'vortex_crossover': 'src.strategy.strategies.indicators.crossovers',
            'ichimoku_cloud_position': 'src.strategy.strategies.indicators.crossovers',
            'macd_crossover': 'src.strategy.strategies.indicators.crossovers',
            # Oscillator strategies
            'rsi_threshold': 'src.strategy.strategies.indicators.oscillators',
            'rsi_bands': 'src.strategy.strategies.indicators.oscillators',
            'cci_threshold': 'src.strategy.strategies.indicators.oscillators',
            'cci_bands': 'src.strategy.strategies.indicators.oscillators',
            'stochastic_rsi': 'src.strategy.strategies.indicators.oscillators',
            'williams_r': 'src.strategy.strategies.indicators.oscillators',
            'roc_threshold': 'src.strategy.strategies.indicators.oscillators',
            'ultimate_oscillator': 'src.strategy.strategies.indicators.oscillators',
            # Volatility strategies
            'keltner_breakout': 'src.strategy.strategies.indicators.volatility',
            'donchian_breakout': 'src.strategy.strategies.indicators.volatility',
            'bollinger_breakout': 'src.strategy.strategies.indicators.volatility',
            # Volume strategies
            'obv_trend': 'src.strategy.strategies.indicators.volume',
            'mfi_bands': 'src.strategy.strategies.indicators.volume',
            'vwap_deviation': 'src.strategy.strategies.indicators.volume',
            'chaikin_money_flow': 'src.strategy.strategies.indicators.volume',
            'accumulation_distribution': 'src.strategy.strategies.indicators.volume',
            # Trend strategies
            'adx_trend_strength': 'src.strategy.strategies.indicators.trend',
            'parabolic_sar': 'src.strategy.strategies.indicators.trend',
            'aroon_crossover': 'src.strategy.strategies.indicators.trend',
            'supertrend': 'src.strategy.strategies.indicators.trend',
            'linear_regression_slope': 'src.strategy.strategies.indicators.trend',
            # Market structure strategies
            'pivot_points': 'src.strategy.strategies.indicators.structure',
            'fibonacci_retracement': 'src.strategy.strategies.indicators.structure',
            'support_resistance_breakout': 'src.strategy.strategies.indicators.structure',
            'atr_channel_breakout': 'src.strategy.strategies.indicators.structure',
            'price_action_swing': 'src.strategy.strategies.indicators.structure',
        }
        
        # Import all unique modules to ensure decorators run
        imported_modules = set()
        for module_path in indicator_modules.values():
            if module_path not in imported_modules:
                try:
                    importlib.import_module(module_path)
                    imported_modules.add(module_path)
                    self.logger.debug(f"Imported module {module_path} for feature inference")
                except ImportError as e:
                    self.logger.debug(f"Could not import {module_path}: {e}")
        
        # Log all registered strategies for debugging
        all_strategies = registry.get_components_by_type('strategy')
        self.logger.info(f"Registry contains {len(all_strategies)} strategies: {[s.name for s in all_strategies]}")
        
        for strategy_config in strategies:
            strategy_type = strategy_config.get('type', strategy_config.get('class'))
            strategy_params = strategy_config.get('params', strategy_config.get('parameters', {}))
            
            self.logger.info(f"Processing strategy {strategy_type} with params: {strategy_params}")
            
            # Import the strategy module to ensure decorator runs
            try:
                # First try the standard location
                module_name = strategy_type.replace('_strategy', '')  
                module_path = f'src.strategy.strategies.{module_name}'
                if module_path not in imported_modules:
                    importlib.import_module(module_path)
                    imported_modules.add(module_path)
            except ImportError:
                # Already imported indicator modules above
                pass
            
            # Get strategy metadata from registry
            # Try different name variations
            strategy_info = None
            for name_variant in [strategy_type, f'{strategy_type}_strategy', strategy_type.replace('_strategy', '')]:
                strategy_info = registry.get_component(name_variant)
                if strategy_info:
                    self.logger.info(f"Found strategy {name_variant} in registry with metadata: {strategy_info.metadata}")
                    break
            
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
                        # Special handling for MA crossover strategies
                        if feature_name == 'sma' and len(param_names) == 2 and 'fast_period' in param_names and 'slow_period' in param_names:
                            # MA crossover needs separate SMA features for fast and slow
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
                                    required_features.add(f'{feature_name}_{defaults[param_name]}')
                        else:
                            # Standard parameter handling
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
                
                self.logger.debug(f"Strategy '{strategy_type}' requires features: {sorted(required_features)}")
                
            else:
                # Fallback for strategies not in registry
                self.logger.debug(f"Strategy '{strategy_type}' not found in registry, using hardcoded inference")
                
                # Comprehensive fallback logic for all indicator strategies
                if strategy_type == 'sma_crossover':
                    # SMA crossover needs fast and slow SMAs
                    fast_periods = strategy_params.get('fast_period', [10])
                    slow_periods = strategy_params.get('slow_period', [20])
                    if not isinstance(fast_periods, list):
                        fast_periods = [fast_periods]
                    if not isinstance(slow_periods, list):
                        slow_periods = [slow_periods]
                    for fast in fast_periods:
                        required_features.add(f'sma_{fast}')
                    for slow in slow_periods:
                        required_features.add(f'sma_{slow}')
                
                elif strategy_type == 'ema_crossover':
                    fast_periods = strategy_params.get('fast_ema_period', [10])
                    slow_periods = strategy_params.get('slow_ema_period', [20])
                    if not isinstance(fast_periods, list):
                        fast_periods = [fast_periods]
                    if not isinstance(slow_periods, list):
                        slow_periods = [slow_periods]
                    for fast in fast_periods:
                        required_features.add(f'ema_{fast}')
                    for slow in slow_periods:
                        required_features.add(f'ema_{slow}')
                
                elif strategy_type == 'rsi_threshold' or strategy_type == 'rsi_bands':
                    rsi_periods = strategy_params.get('rsi_period', [14])
                    if not isinstance(rsi_periods, list):
                        rsi_periods = [rsi_periods]
                    for period in rsi_periods:
                        required_features.add(f'rsi_{period}')
                
                elif strategy_type == 'macd_crossover':
                    # MACD needs specific fast/slow/signal parameters
                    fast_ema = strategy_params.get('fast_ema', 12)
                    slow_ema = strategy_params.get('slow_ema', 26)
                    signal_ema = strategy_params.get('signal_ema', 9)
                    # MACD feature is named by its parameters
                    required_features.add(f'macd_{fast_ema}_{slow_ema}_{signal_ema}')
                
                elif strategy_type == 'bollinger_breakout':
                    periods = strategy_params.get('period', [20])
                    if not isinstance(periods, list):
                        periods = [periods]
                    for period in periods:
                        required_features.add(f'bollinger_{period}')
                
                elif strategy_type == 'keltner_breakout':
                    periods = strategy_params.get('period', [20])
                    if not isinstance(periods, list):
                        periods = [periods]
                    for period in periods:
                        required_features.add(f'keltner_{period}')
                        required_features.add(f'atr_{period}')
                
                elif strategy_type in ['MomentumStrategy', 'momentum']:
                    lookback_period = strategy_params.get('lookback_period', 20)
                    rsi_period = strategy_params.get('rsi_period', 14)
                    required_features.add(f'sma_{lookback_period}')
                    required_features.add(f'rsi_{rsi_period}')
                
                elif strategy_type in ['MeanReversionStrategy', 'mean_reversion']:
                    period = strategy_params.get('period', 20)
                    required_features.add(f'bollinger_{period}')
                    required_features.add('rsi_14')
                
                elif strategy_type in ['breakout_strategy', 'breakout']:
                    lookback_period = strategy_params.get('lookback_period', 20)
                    atr_period = strategy_params.get('atr_period', 14)
                    required_features.add(f'high_{lookback_period}')
                    required_features.add(f'low_{lookback_period}')
                    required_features.add(f'volume_{lookback_period}')
                    required_features.add(f'atr_{atr_period}')
                
                else:
                    # Try to infer from parameters for unknown strategies
                    param_based_features = set()
                    
                    # Check for common parameter patterns
                    for param_name, param_value in strategy_params.items():
                        if 'period' in param_name.lower():
                            # Extract feature type from parameter name
                            if 'rsi' in param_name:
                                if isinstance(param_value, list):
                                    for v in param_value:
                                        param_based_features.add(f'rsi_{v}')
                                else:
                                    param_based_features.add(f'rsi_{param_value}')
                            elif 'sma' in param_name or 'fast' in param_name or 'slow' in param_name:
                                if isinstance(param_value, list):
                                    for v in param_value:
                                        param_based_features.add(f'sma_{v}')
                                else:
                                    param_based_features.add(f'sma_{param_value}')
                            elif 'ema' in param_name:
                                if isinstance(param_value, list):
                                    for v in param_value:
                                        param_based_features.add(f'ema_{v}')
                                else:
                                    param_based_features.add(f'ema_{param_value}')
                    
                    if param_based_features:
                        required_features.update(param_based_features)
                    else:
                        # Absolute default
                        required_features.update(['sma_20', 'rsi_14'])
        
        # If no strategies found, add default features
        if not required_features:
            self.logger.warning("No strategies found, using default features")
            required_features.update(['sma_20', 'rsi_14'])
            
        return required_features
    
    def _infer_features_from_classifiers(self, classifiers: List[Dict[str, Any]]) -> Set[str]:
        """Infer required features from classifier configurations.
        
        Args:
            classifiers: List of classifier configuration dictionaries
            
        Returns:
            Set of required feature identifiers
        """
        from ..components.discovery import get_component_registry
        import importlib
        
        required_features = set()
        registry = get_component_registry()
        
        # Import classifier modules to ensure decorators run
        try:
            importlib.import_module('src.strategy.classifiers.classifiers')
        except ImportError:
            self.logger.debug("Could not import classifiers module")
        
        try:
            importlib.import_module('src.strategy.classifiers.enhanced_multi_state_classifiers')
        except ImportError:
            self.logger.debug("Could not import enhanced classifiers module")
        
        for classifier_config in classifiers:
            classifier_type = classifier_config.get('type', classifier_config.get('name'))
            classifier_params = classifier_config.get('params', classifier_config.get('parameters', {}))
            
            # Get classifier metadata from registry
            classifier_info = registry.get_component(classifier_type)
            
            if classifier_info:
                # Get feature requirements from metadata
                features = classifier_info.metadata.get('features', [])
                
                # Add basic features that classifiers need
                for feature in features:
                    if feature == 'sma_fast' or feature == 'sma_slow':
                        # Add common SMA periods
                        fast_period = classifier_params.get('fast_period', 10)
                        slow_period = classifier_params.get('slow_period', 20)
                        required_features.add(f'sma_{fast_period}')
                        required_features.add(f'sma_{slow_period}')
                    elif feature == 'atr':
                        atr_period = classifier_params.get('atr_period', 14)
                        required_features.add(f'atr_{atr_period}')
                    elif feature == 'atr_sma':
                        atr_period = classifier_params.get('atr_period', 14)
                        sma_period = classifier_params.get('sma_period', 20)
                        required_features.add(f'atr_{atr_period}')
                        required_features.add(f'atr_sma_{atr_period}_{sma_period}')
                    elif feature == 'rsi':
                        rsi_period = classifier_params.get('rsi_period', 14)
                        required_features.add(f'rsi_{rsi_period}')
                    elif feature in ['macd_macd', 'macd']:
                        required_features.add('macd')
                    elif feature.startswith('momentum'):
                        momentum_period = classifier_params.get('momentum_period', 10)
                        required_features.add(f'momentum_{momentum_period}')
                    elif feature == 'volatility':
                        vol_period = classifier_params.get('vol_period', 20)
                        required_features.add(f'volatility_{vol_period}')
                    elif feature == 'volatility_sma':
                        vol_period = classifier_params.get('vol_period', 20)
                        sma_period = classifier_params.get('sma_period', 20)
                        required_features.add(f'volatility_{vol_period}')
                        required_features.add(f'volatility_sma_{vol_period}_{sma_period}')
                    else:
                        # Add the feature as-is
                        required_features.add(feature)
                
                self.logger.debug(f"Classifier '{classifier_type}' requires features: {sorted(features)}")
            else:
                # Fallback for classifiers not in registry
                self.logger.debug(f"Classifier '{classifier_type}' not found in registry, using defaults")
                
                # Add specific features for known classifiers
                if classifier_type == 'volatility_momentum_classifier':
                    required_features.update(['atr_14', 'rsi_14', 'sma_20'])
                elif classifier_type == 'microstructure_classifier':
                    required_features.update(['sma_5', 'sma_20', 'atr_10', 'rsi_7'])
                elif classifier_type == 'hidden_markov_classifier':
                    required_features.update(['volume', 'rsi_14', 'sma_20', 'sma_50', 'atr_14'])
                elif classifier_type == 'enhanced_trend_classifier':
                    required_features.update(['sma_10', 'sma_20', 'sma_50'])
                elif classifier_type == 'market_regime_classifier':
                    required_features.update(['sma_10', 'sma_50', 'atr_20', 'rsi_14'])
                elif 'momentum' in classifier_type:
                    required_features.update(['rsi_14', 'macd', 'momentum_10'])
                elif 'volatility' in classifier_type:
                    required_features.update(['atr_14', 'volatility_20', 'atr_sma_14_20', 'volatility_sma_20_20'])
                elif 'trend' in classifier_type:
                    required_features.update(['sma_10', 'sma_20', 'sma_50'])
                else:
                    # Default classifier features
                    required_features.update(['sma_20', 'rsi_14', 'atr_14'])
        
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

