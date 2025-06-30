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
import uuid
import time

from .config.pattern_loader import PatternLoader
from .config.resolver import ConfigResolver
from ..containers.factory import ContainerFactory

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
        
    def _initialize_factories(self):
        """Initialize factories needed for topology building."""
        if self.container_factory is None:
            self.container_factory = ContainerFactory()
    
    def _get_pattern(self, mode: str) -> Optional[Dict[str, Any]]:
        """Get pattern definition for the specified mode."""
        return self.patterns.get(mode)
    
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
        
        # Set system config on factory for access during component creation
        if hasattr(self.container_factory, 'set_system_config'):
            self.container_factory.set_system_config(config)
        
        # Build evaluation context
        context = self._build_context(pattern, config, tracing_config, metadata, mode)
        
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
        # Use the actual root container's event bus, not the one from context
        root_container = topology['containers'].get('root')
        if root_container and hasattr(root_container, 'event_bus'):
            # Replace context's root_event_bus with the actual root container's bus
            context['root_event_bus'] = root_container.event_bus
            self.logger.info(f"Using root container's event bus: {id(root_container.event_bus)}")
        self._setup_event_subscriptions(mode, topology, context)
        
        # 6. Set up unified signal tracing if enabled
        execution_config = config.get('execution', {})
        trace_enabled = execution_config.get('enable_event_tracing', False)
        
        # For signal generation and universal mode, default to sparse storage unless explicitly disabled
        if mode in ['signal_generation', 'universal']:
            use_sparse = execution_config.get('trace_settings', {}).get('use_sparse_storage', True)
        else:
            use_sparse = execution_config.get('trace_settings', {}).get('use_sparse_storage', False)
        
        # For signal generation and universal mode, always set up MultiStrategyTracer regardless of container tracing
        # MultiStrategyTracer handles signal storage independently from event tracing
        if mode in ['signal_generation', 'universal'] and use_sparse:
            from ..events.tracer_setup import setup_multi_strategy_tracer
            setup_multi_strategy_tracer(topology, context, tracing_config)
        
        self.logger.info(
            f"Built {mode} topology with {len(topology['containers'])} containers "
            f"and {len(topology['routes'])} routes"
        )
        
        return topology
    
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
            # Check for both 'params' and 'param_overrides' (clean syntax uses param_overrides)
            params = strategy_config.get('params', strategy_config.get('param_overrides', {}))
            
            # Special handling for ensemble strategies - don't expand their sub-strategy lists
            if 'ensemble' in strategy_type.lower():
                self.logger.debug(f"Skipping parameter expansion for ensemble strategy: {strategy_type}")
                expanded_strategies.append(strategy_config)
                continue
            
            # Find parameters that are lists (to be expanded)
            list_params = {}
            scalar_params = {}
            
            # Skip certain parameters that should not be expanded
            skip_params = {'baseline_strategies', 'regime_boosters', 'regime_strategies', 'strategies'}
            
            for param_name, param_value in params.items():
                if param_name in skip_params:
                    # These are structural parameters, not grid search parameters
                    scalar_params[param_name] = param_value
                elif isinstance(param_value, list):
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
                    
        self.logger.debug(f"Expanded {len(strategies)} strategies to {len(expanded_strategies)} configurations")
        
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
        elif mode == 'universal':
            self._setup_universal_subscriptions(containers, root_bus)
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
                    self.logger.debug(f"Portfolio {portfolio_name} subscribed to signals from strategy {strategy_id}")
        
        # 2. Execution subscribes to ORDER events
        execution_container = containers.get('execution')
        if execution_container:
            execution_engine = execution_container.get_component('execution_engine')
            if execution_engine and hasattr(execution_engine, 'on_order'):
                root_bus.subscribe(EventType.ORDER, execution_engine.on_order)
                self.logger.info("Execution engine subscribed to ORDER events")
        
        # 3. Portfolios subscribe to FILL events
        # Import filter module
        from ..events.filters import order_ownership_filter
        
        for portfolio_name, portfolio in portfolio_containers.items():
            portfolio_state = portfolio.get_component('portfolio_state')
            if portfolio_state and hasattr(portfolio_state, 'on_fill'):
                # For now, use a permissive filter that accepts all FILL events
                # In production, use order_ownership_filter or container_filter
                all_fills_filter = lambda event: True
                root_bus.subscribe(EventType.FILL.value, portfolio_state.on_fill, filter_func=all_fills_filter)
                self.logger.info(f"Portfolio {portfolio_name} subscribed to FILL events with permissive filter")
    
    def _setup_signal_generation_subscriptions(self, containers: Dict[str, Any], root_bus: Any) -> None:
        """Set up subscriptions for signal generation topology."""
        from ..events import EventType
        
        # 1. Strategy containers subscribe to BAR events for feature calculation and signal generation
        strategy_containers = {k: v for k, v in containers.items() if 'strategy' in k}
        
        for strategy_name, strategy_container in strategy_containers.items():
            # Subscribe FeatureHub component to BAR events
            feature_hub = strategy_container.get_component('feature_hub')
            if feature_hub and hasattr(feature_hub, 'on_bar'):
                root_bus.subscribe(EventType.BAR, feature_hub.on_bar)
                self.logger.info(f"Strategy {strategy_name} FeatureHub subscribed to BAR events")
            
            # Subscribe StrategyState component to BAR events for signal generation
            strategy_state = strategy_container.get_component('strategy_state')
            if strategy_state and hasattr(strategy_state, 'on_bar'):
                root_bus.subscribe(EventType.BAR, strategy_state.on_bar)
                self.logger.info(f"Strategy {strategy_name} StrategyState subscribed to BAR events")
        
        # 2. Portfolio containers subscribe to SIGNAL events from root bus
        portfolio_containers = {k: v for k, v in containers.items() if 'portfolio' in k}
        
        for portfolio_name, portfolio in portfolio_containers.items():
            # Get portfolio state component that can handle signals
            portfolio_state = portfolio.get_component('portfolio_state')
            if portfolio_state and hasattr(portfolio_state, 'process_event'):
                
                # Subscribe to SIGNAL events on the portfolio's own event bus
                # Signals flow: strategy -> parent (root) -> root forwards to all children including portfolio
                event_type_str = EventType.SIGNAL.value if hasattr(EventType.SIGNAL, 'value') else str(EventType.SIGNAL)
                
                # Get managed strategies for this portfolio from config
                portfolio_config = portfolio.config.config if hasattr(portfolio.config, 'config') else {}
                managed_strategies = portfolio_config.get('managed_strategies', ['default'])
                
                # Use structured subscription filter
                from .subscription_helpers import create_portfolio_subscription_filter
                
                # Convert managed_strategies to structured format
                structured_strategies = []
                for strategy in managed_strategies:
                    if isinstance(strategy, str):
                        structured_strategies.append({'type': strategy})
                    else:
                        structured_strategies.append(strategy)
                
                # Create filter that handles both legacy and structured events
                filter_func = create_portfolio_subscription_filter(
                    structured_strategies,
                    symbol=portfolio_config.get('symbol'),
                    timeframe=portfolio_config.get('timeframe')
                )
                
                # Subscribe directly - Container.receive_event() handles tracing
                portfolio.event_bus.subscribe(
                    event_type_str, 
                    portfolio_state.process_event,
                    filter_func=filter_func
                )
                self.logger.info(f"Portfolio {portfolio_name} subscribed to '{event_type_str}' events with strategy filter: {managed_strategies}")
            else:
                self.logger.warning(f"Portfolio {portfolio_name} has no portfolio_state with process_event method")
        
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
        # Import filter module
        from ..events.filters import order_ownership_filter
        
        for portfolio_name, portfolio in portfolio_containers.items():
            portfolio_state = portfolio.get_component('portfolio_state')
            if portfolio_state and hasattr(portfolio_state, 'on_fill'):
                # For now, use a permissive filter that accepts all FILL events
                # In production, use order_ownership_filter or container_filter
                all_fills_filter = lambda event: True
                root_bus.subscribe(EventType.FILL.value, portfolio_state.on_fill, filter_func=all_fills_filter)
                self.logger.info(f"Portfolio {portfolio_name} subscribed to FILL events with permissive filter")
    
    def _setup_universal_subscriptions(self, containers: Dict[str, Any], root_bus: Any) -> None:
        """Set up subscriptions for universal topology.
        
        Universal topology supports the complete trading pipeline:
        BAR -> SIGNAL -> ORDER -> FILL
        """
        from ..events import EventType
        
        # This is essentially the same as signal_generation but with portfolio/execution support
        self._setup_signal_generation_subscriptions(containers, root_bus)
        
        # Additionally set up portfolio and execution subscriptions
        
        # 3. Execution subscribes to ORDER events on root bus
        execution_container = containers.get('execution')
        if execution_container:
            execution_engine = execution_container.get_component('execution_engine')
            if execution_engine:
                self.logger.info(f"Found execution_engine: {type(execution_engine).__name__}")
                if hasattr(execution_engine, 'on_order'):
                    # Subscribe to ORDER events on root bus
                    bus_id = id(root_bus)
                    event_type_str = EventType.ORDER.value if hasattr(EventType.ORDER, 'value') else str(EventType.ORDER)
                    self.logger.info(f"Subscribing execution engine to '{event_type_str}' events on bus {bus_id}")
                    root_bus.subscribe(event_type_str, execution_engine.on_order)
                    self.logger.info(f"Execution engine subscribed to ORDER events on root bus {bus_id}")
                else:
                    self.logger.warning(f"Execution engine {type(execution_engine).__name__} has no on_order method")
            else:
                self.logger.warning("Execution container has no execution_engine component")
        
        # 4. Find all portfolio containers and subscribe to FILL events
        portfolio_containers = {
            name: container for name, container in containers.items()
            if 'portfolio' in name and name != 'portfolio'  # Skip the generic 'portfolio' container
        }
        
        if not portfolio_containers:
            # Fallback to generic portfolio container
            portfolio_container = containers.get('portfolio')
            if portfolio_container:
                portfolio_containers = {'portfolio': portfolio_container}
        
        # Subscribe all portfolios to FILL events on root bus
        for portfolio_name, portfolio in portfolio_containers.items():
            portfolio_state = portfolio.get_component('portfolio_state') or portfolio.get_component('portfolio_manager')
            if portfolio_state and hasattr(portfolio_state, 'on_fill'):
                # For now, use a permissive filter that accepts all FILL events
                # In production, use order_ownership_filter or container_filter
                all_fills_filter = lambda event: True
                self.logger.info(f"📝 Subscribing {portfolio_name} to FILL events on bus {id(root_bus)}")
                try:
                    root_bus.subscribe(EventType.FILL.value, portfolio_state.on_fill, filter_func=all_fills_filter)
                    self.logger.info(f"✅ Portfolio {portfolio_name} subscribed to FILL events on root bus with permissive filter")
                except Exception as e:
                    self.logger.error(f"❌ Failed to subscribe {portfolio_name} to FILL events: {type(e).__name__}: {e}")
            else:
                self.logger.warning(f"Portfolio {portfolio_name} has no on_fill method")
    
    def _extract_and_inject_strategy_names(self, context: Dict[str, Any]) -> None:
        """
        Automatically extract strategy names from strategies list and inject
        into context for portfolio container creation.
        
        This allows users to just define strategies with names, and the system
        automatically creates the strategy_names list needed by portfolio containers.
        """
        config = context.get('config', {})
        strategies = config.get('strategies', [])
        
        # Also check for strategies in parameter_space (from clean syntax parser)
        if not strategies and 'parameter_space' in config:
            parameter_space = config.get('parameter_space', {})
            if isinstance(parameter_space, dict) and 'strategies' in parameter_space:
                strategies = parameter_space['strategies']
                self.logger.info(f"Found {len(strategies)} strategies in parameter_space for name extraction")
        
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
            self.logger.debug(f"Extracted strategy names from strategies list: {strategy_names}")
            
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
        
        Now supports both:
        - New FeatureSpec-based discovery
        - Compositional strategy syntax via StrategyCompiler
        """
        # Check if we have a compositional strategy configuration
        config = context.get('config', {})
        strategy_config = config.get('strategy')
        
        # Also check for strategies in parameter_space (from clean syntax parser)
        strategies = config.get('strategies', [])
        if not strategies and 'parameter_space' in config:
            parameter_space = config.get('parameter_space', {})
            if isinstance(parameter_space, dict) and 'strategies' in parameter_space:
                strategies = parameter_space['strategies']
                self.logger.info(f"Found {len(strategies)} strategies in parameter_space for feature inference")
        
        if strategy_config or 'parameter_combinations' in config or strategies:
            # Check if we have regular strategies list (from parameter_space)
            if strategies and not strategy_config and not 'parameter_combinations' in config:
                # Convert strategies to parameter_combinations for the compiler
                self.logger.info(f"Converting {len(strategies)} strategies to parameter combinations")
                
                # Build parameter combinations from expanded strategies
                param_combinations = []
                for strategy in strategies:
                    combo = {
                        'strategy_type': strategy.get('type'),
                        'strategy_name': strategy.get('name'),
                        'parameters': strategy.get('param_overrides', strategy.get('params', {}))
                    }
                    # Pass through constraints/filter if present
                    if 'constraints' in strategy:
                        combo['constraints'] = strategy['constraints']
                    elif 'threshold' in strategy:
                        combo['threshold'] = strategy['threshold']
                    elif 'filter' in strategy:
                        combo['filter'] = strategy['filter']
                    param_combinations.append(combo)
                
                # Add to config for compiler
                config['parameter_combinations'] = param_combinations
                self.logger.info(f"Created {len(param_combinations)} parameter combinations for compiler")
                
                # Fall through to compiler path
            
            # Use compiler for all cases with parameter_combinations
            if strategy_config or 'parameter_combinations' in config:
                self.logger.info("Using strategy compiler for strategy configuration")
                
                from .compiler import StrategyCompiler
                compiler = StrategyCompiler()
                
                # Build config for compiler - include parameter_combinations if present
                compiler_config = {'strategy': strategy_config} if strategy_config else {}
                if 'parameter_combinations' in config:
                    compiler_config['parameter_combinations'] = config['parameter_combinations']
                    self.logger.info(f"Compiling {len(config['parameter_combinations'])} parameter combinations")
                
                # Extract features from compositional config
                feature_specs_list = compiler.extract_features(compiler_config)
                
                # Convert list to dict keyed by canonical name
                feature_specs = {}
                for spec in feature_specs_list:
                    feature_specs[spec.canonical_name] = spec
                
                # Store compiled strategies in context for later use
                compiled_strategies = compiler.compile_strategies(compiler_config)
                context['compiled_strategies'] = compiled_strategies
            
            # Inject compiled strategies into components for stateless execution
            if 'components' not in context:
                context['components'] = {}
            if 'strategies' not in context['components']:
                context['components']['strategies'] = {}
                
            # Add each compiled strategy to the components (only if we have compiled strategies)
            if 'compiled_strategies' in locals():
                for i, compiled_strategy in enumerate(compiled_strategies):
                    # Use the actual strategy ID from the compiler
                    strategy_id = compiled_strategy.get('id', f"compiled_strategy_{i}")
                    
                    # Always use the clean ID from the compiler (e.g., strategy_0, strategy_1)
                    # This supports the strategy hash pattern from trace-updates.md
                    strategy_name = strategy_id
                    
                    # Extract metadata from compiled strategy
                    metadata = compiled_strategy.get('metadata', {})
                    
                    # Create a wrapper that matches the expected signature
                    def make_strategy_wrapper(compiled_func, strategy_metadata):
                        def strategy_wrapper(features, bar, params):
                            return compiled_func(features, bar, params)
                        # Attach metadata for feature discovery
                        if strategy_metadata:
                            strategy_wrapper._strategy_metadata = strategy_metadata
                            # Also create mock component info for compatibility
                            class MockComponentInfo:
                                def __init__(self, metadata):
                                    self.metadata = metadata
                            strategy_wrapper._component_info = MockComponentInfo(strategy_metadata)
                        # Preserve _compiled_params from the original function
                        if hasattr(compiled_func, '_compiled_params'):
                            strategy_wrapper._compiled_params = compiled_func._compiled_params
                        return strategy_wrapper
                    
                    # Debug: log what metadata we have
                    if 'feature_discovery' in metadata:
                        self.logger.debug(f"Strategy {strategy_name} has feature_discovery")
                    elif 'required_features' in metadata:
                        self.logger.debug(f"Strategy {strategy_name} has required_features")
                    else:
                        self.logger.warning(f"Strategy {strategy_name} missing feature metadata: {list(metadata.keys())}")
                        
                    # Also store the strategy ID in metadata for proper signal publishing
                    metadata['strategy_id'] = strategy_name
                        
                    context['components']['strategies'][strategy_name] = make_strategy_wrapper(
                        compiled_strategy['function'], metadata
                    )
                    self.logger.debug(f"Added compiled strategy: {strategy_name} with id: {metadata.get('strategy_id')}")
                    if 'strategy_type' in metadata:
                        self.logger.debug(f"  Metadata includes strategy_type: {metadata['strategy_type']}")
            
        else:
            # Legacy format - look for strategies list
            strategies = context.get('config', {}).get('strategies', [])
            classifiers = context.get('config', {}).get('classifiers', [])
            
            if not strategies and not classifiers:
                self.logger.info("No strategies or classifiers found in config, skipping feature inference")
                return
            
            self.logger.info(f"Inferring features from {len(strategies)} strategies and {len(classifiers)} classifiers")
            
            # Use ONLY the new feature discovery system - no fallbacks!
            from .feature_discovery import FeatureDiscovery
            discovery = FeatureDiscovery()
            
            # Discover all required features (returns Dict[str, FeatureSpec])
            feature_specs = discovery.discover_all_features(strategies, classifiers)
        
        if not feature_specs:
            self.logger.warning("No features discovered - strategies may not have feature requirements")
            return
            
        self.logger.info(f"Discovered {len(feature_specs)} unique features")
        
        # Store FeatureSpec objects in context
        context["feature_specs"] = feature_specs
        
        # Extract canonical names
        required_features = list(feature_specs.keys())
        context["inferred_features"] = required_features
        
        # Create feature configs for feature computation
        feature_configs = {}
        for canonical_name, spec in feature_specs.items():
            # Merge params directly into config dict as FeatureHub expects
            config = {'type': spec.feature_type}
            config.update(spec.params)  # Add params directly to config
            if spec.output_component:
                config['component'] = spec.output_component
            feature_configs[canonical_name] = config
        
        # MERGE with existing feature_configs instead of overwriting!
        # This allows manual features (e.g., for filters) to be preserved
        existing_feature_configs = context['config'].get('feature_configs', {})
        if existing_feature_configs:
            self.logger.info(f"Merging {len(existing_feature_configs)} manual features with {len(feature_configs)} discovered features")
            # Start with discovered features, then update with manual (manual takes precedence)
            merged_configs = feature_configs.copy()
            merged_configs.update(existing_feature_configs)
            feature_configs = merged_configs
            self.logger.debug(f"Total features after merge: {len(feature_configs)}")
        
        # Add to context for feature containers
        context['config']["feature_configs"] = feature_configs
        
        # Update required features list to include ALL features
        all_features = list(feature_configs.keys())
        context["features"] = all_features
        
        self.logger.debug(f"Injected {len(required_features)} features into context")
        
            

    def _build_context(self, pattern: Dict[str, Any], config: Dict[str, Any], 
                      tracing_config: Dict[str, Any], metadata: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Build evaluation context for pattern interpretation."""
        # Store original configurations before expansion for analytics
        original_strategies = config.get('strategies', []).copy() if 'strategies' in config else []
        original_classifiers = config.get('classifiers', []).copy() if 'classifiers' in config else []
        
        # Parse data field and normalize data configuration
        self.logger.debug(f"Config before data parsing: data={config.get('data')}, symbols={config.get('symbols')}")
        data_config = self.config_resolver.parse_data_field(config)
        self.logger.debug(f"Data config after parsing: {data_config}")
        
        # Update config with parsed data
        if data_config['symbols'] and not config.get('symbols'):
            config['symbols'] = data_config['symbols']
            self.logger.info(f"Extracted symbols from data field: {data_config['symbols']}")
        
        if data_config['timeframes'] and not config.get('timeframes'):
            config['timeframes'] = data_config['timeframes']
            self.logger.info(f"Extracted timeframes from data field: {data_config['timeframes']}")
        
        # If we have a constructed data source, DON'T overwrite the original data field
        # The data field should be preserved as-is for reference
        if 'data_source' in data_config and isinstance(config.get('data'), (str, list)):
            # Don't modify config['data'] - it should stay as the original string/list
            self.logger.info(f"Parsed data source from data field: {config.get('data')}")
        
        # Store parsed data info in metadata for reference
        if data_config['original_data'] is not None:
            if 'metadata' not in config:
                config['metadata'] = {}
            config['metadata']['parsed_data'] = {
                'original': data_config['original_data'],
                'symbols': data_config['symbols'],
                'timeframes': data_config['timeframes'],
                'data_specs': data_config['data_specs']
            }
            
        # If we have data_specs, add them to config for pattern use
        if data_config['data_specs']:
            config['data_specs'] = data_config['data_specs']
            self.logger.info(f"Added {len(data_config['data_specs'])} data specs to config")
        
        # Expand parameter lists in strategies before building context
        if 'strategies' in config:
            config['strategies'] = self._expand_strategy_parameters(config['strategies'])
        
        # Also expand strategies in parameter_space (from clean syntax parser)
        if 'parameter_space' in config and isinstance(config['parameter_space'], dict):
            if 'strategies' in config['parameter_space']:
                self.logger.info(f"Expanding strategies in parameter_space")
                config['parameter_space']['strategies'] = self._expand_strategy_parameters(
                    config['parameter_space']['strategies']
                )
                self.logger.info(f"Expanded to {len(config['parameter_space']['strategies'])} strategy combinations")
        
        # Expand parameter lists in classifiers before building context
        if 'classifiers' in config:
            config['classifiers'] = self._expand_classifier_parameters(config['classifiers'])
        
        context = {
            'config': config,
            'pattern': pattern,
            'tracing': tracing_config,
            'metadata': metadata,
            'generated': {},
            'root_event_bus': None,
            'use_hierarchical_events': True,
            'original_strategies': original_strategies,
            'original_classifiers': original_classifiers,
            'mode': mode  # Add mode for tracer selection
        }
        
        # Create root event bus for stateless services
        from ..events import EventBus
        context['root_event_bus'] = EventBus()
        
        return context
    
    def _create_components(self, comp_spec: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Create stateless components from specification."""
        components = {}
        comp_type = comp_spec.get('type')
        self.logger.info(f"_create_components called for type: {comp_type}, spec: {comp_spec}")
        
        # Initialize component category
        if comp_type == 'strategies':
            # Check if we already have compiled strategies in context
            if 'components' in context and 'strategies' in context['components'] and len(context['components']['strategies']) > 0:
                self.logger.info(f"Using {len(context['components']['strategies'])} pre-compiled strategies")
                components['strategies'] = context['components']['strategies']
                return components
            else:
                components['strategies'] = {}
        elif comp_type == 'risk_validators':
            components['risk_validators'] = {}
        elif comp_type == 'classifiers':
            components['classifiers'] = {}
        elif comp_type == 'execution_models':
            components['execution_models'] = {}
        else:
            components[comp_type] = {}
        
        # Get items from config or spec
        from_config = comp_spec.get('from_config')
        if from_config:
            self.logger.info(f"Looking for {from_config} in config")
            # Direct access to config 
            items = context.get('config', {}).get(from_config, [])
            self.logger.info(f"Direct lookup for {from_config}: found {len(items) if isinstance(items, list) else 0} items")
            
            # Special handling for strategies - check parameter_space if not found at top level
            if from_config == 'strategies' and not items and 'parameter_space' in context.get('config', {}):
                parameter_space = context['config']['parameter_space']
                self.logger.debug(f"Checking parameter_space: {list(parameter_space.keys()) if isinstance(parameter_space, dict) else 'not a dict'}")
                if isinstance(parameter_space, dict) and 'strategies' in parameter_space:
                    items = parameter_space['strategies']
                    self.logger.info(f"Found strategies in parameter_space: {len(items)} items")
            
            self.logger.info(f"Resolved {from_config} to {len(items) if isinstance(items, list) else 'single'} items")
            if not isinstance(items, list):
                items = [items] if items else []
                
            for item in items:
                self.logger.debug(f"Processing {comp_type} item: {item.get('type', 'unknown') if isinstance(item, dict) else item}")
                # Skip if item is not a dict
                if not isinstance(item, dict):
                    continue
                    
                component = self._create_single_component(comp_type, item)
                if component:
                    name = item.get('name', item.get('type', comp_type))
                    if comp_type not in components:
                        components[comp_type] = {}
                    components[comp_type][name] = component
                    self.logger.info(f"Added {comp_type} component: {name} (type: {item.get('type')})")
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
        """Find and return stateless function/component reference with attached config."""
        import importlib
        
        try:
            if comp_type == 'strategies':
                # Get strategy function reference
                strategy_type = config.get('type', '')
                self.logger.debug(f"Creating strategy component for type: {strategy_type}")
                
                # Use the discovery system to find the strategy
                from ...core.components.discovery import get_component_registry
                registry = get_component_registry()
                strategy_info = registry.get_component(strategy_type)
                
                if strategy_info and strategy_info.factory:
                    self.logger.debug(f"Found strategy function: {strategy_info.factory.__name__}")
                    # Return a wrapper that includes the config
                    def strategy_wrapper(features, bar, params):
                        # Merge provided params with config overrides
                        merged_params = params.copy() if params else {}
                        param_overrides = config.get('param_overrides', {})
                        merged_params.update(param_overrides)
                        return strategy_info.factory(features, bar, merged_params)
                    
                    # Attach metadata for feature discovery
                    # Get parameters from various possible sources
                    parameters = config.get('param_overrides', {})
                    if not parameters and 'params' in config:
                        parameters = config.get('params', {})
                    if not parameters and 'parameters' in config:
                        parameters = config.get('parameters', {})
                    
                    strategy_wrapper._strategy_metadata = {
                        'strategy_type': strategy_type,
                        'parameters': parameters,
                        'config': config
                    }
                    strategy_wrapper._component_info = strategy_info
                    
                    return strategy_wrapper
                else:
                    self.logger.warning(f"Strategy '{strategy_type}' not found in discovery registry")
                    return None
                    
            elif comp_type == 'classifiers':
                # Similar logic for classifiers
                classifier_type = config.get('type', '')
                self.logger.debug(f"Creating classifier component for type: {classifier_type}")
                
                from ...core.components.discovery import get_component_registry
                registry = get_component_registry()
                classifier_info = registry.get_component(classifier_type)
                
                if classifier_info and classifier_info.factory:
                    self.logger.debug(f"Found classifier function: {classifier_info.factory.__name__}")
                    return classifier_info.factory
                else:
                    self.logger.warning(f"Classifier '{classifier_type}' not found in discovery registry")
                    return None
                    
            else:
                # For other component types, return a placeholder
                self.logger.debug(f"Creating placeholder component for type: {comp_type}")
                return {"type": comp_type, "config": config}
                
        except Exception as e:
            self.logger.error(f"Error creating {comp_type} component: {e}")
            return None
    
    def _create_containers(self, container_spec: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Create containers from specification, including nested containers."""
        containers = {}
        
        # Handle foreach expansion first if present
        if 'foreach' in container_spec:
            return self._expand_container_foreach(container_spec, context)
        
        # Get container configuration
        container_name = container_spec.get('name', 'unnamed')
        container_type = container_spec.get('type', 'default')
        components = container_spec.get('components', [])
        config = self._resolve_value(container_spec.get('config', {}), context)
        parent_name = container_spec.get('parent')
        
        # Inject stateless components into strategy containers
        if container_type == 'strategy' and 'components' in context:
            self.logger.info(f"Injecting stateless components into {container_name} config")
            config['stateless_components'] = context['components']
            self.logger.info(f"Injected stateless_components: {list(context['components'].keys())}")
            # Debug: show component counts
            for comp_type, comps in context['components'].items():
                self.logger.info(f"  {comp_type}: {len(comps) if isinstance(comps, dict) else 0} components")
        
        # Determine parent event bus
        parent_event_bus = None
        if parent_name:
            # Look for parent in existing containers
            existing_containers = context.get('containers', {})
            parent_container = existing_containers.get(parent_name)
            if parent_container:
                parent_event_bus = parent_container.event_bus
                self.logger.debug(f"Using parent event bus from {parent_name}")
            else:
                self.logger.warning(f"Parent container '{parent_name}' not found for '{container_name}'")
        
        # Create container using factory
        container = self.container_factory.create_container(
            name=container_name,
            components=components,
            config=config,
            container_type=container_type,
            parent_event_bus=parent_event_bus
        )
        containers[container_name] = container
        
        # Add to parent if specified
        if parent_name and parent_name in context.get('containers', {}):
            parent_container = context['containers'][parent_name]
            parent_container.add_child_container(container)
            self.logger.info(f"Added {container_name} as child of {parent_name}")
        
        self.logger.info(f"Created container: {container_name} (type: {container_type})")
        
        return containers
    
    def _expand_container_foreach(self, container_spec: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Expand container specification with foreach loops."""
        containers = {}
        foreach_spec = container_spec['foreach']
        name_template = container_spec.get('name_template', container_spec.get('name', 'container_{index}'))
        
        # Build iteration variables
        iteration_vars = {}
        for var_name, var_spec in foreach_spec.items():
            if isinstance(var_spec, dict) and 'from_config' in var_spec:
                # Resolve from context config
                values = self._resolve_value(var_spec, context)
                if not isinstance(values, list):
                    values = [values]
                iteration_vars[var_name] = values
            else:
                iteration_vars[var_name] = var_spec if isinstance(var_spec, list) else [var_spec]
        
        # Generate all combinations
        var_names = list(iteration_vars.keys())
        var_values = [iteration_vars[name] for name in var_names]
        
        for combination in itertools.product(*var_values):
            # Create iteration context
            iter_context = context.copy()
            for i, var_name in enumerate(var_names):
                iter_context[var_name] = combination[i]
            
            # Resolve container name from template
            container_name = name_template.format(**iter_context)
            
            # Create individual container spec
            individual_spec = container_spec.copy()
            individual_spec['name'] = container_name
            individual_spec.pop('foreach', None)
            individual_spec.pop('name_template', None)
            
            # Resolve config with iteration variables
            if 'config' in individual_spec:
                individual_spec['config'] = self._resolve_value(individual_spec['config'], iter_context)
            
            # Recursively create this container
            created_containers = self._create_containers(individual_spec, iter_context)
            containers.update(created_containers)
        
        return containers

