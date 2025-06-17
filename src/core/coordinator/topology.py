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
            params = strategy_config.get('params', {})
            
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
    
    def _create_feature_config_from_id(self, feature_id: str) -> Dict[str, Any]:
        """
        Parse a feature ID and create the appropriate feature configuration.
        
        Handles compound feature names like 'bollinger_bands_20_2.0' correctly.
        
        Args:
            feature_id: Feature identifier like 'sma_20' or 'bollinger_bands_20_2.0'
            
        Returns:
            Feature configuration dict with 'feature' key and parameter keys
        """
        # Dictionary of compound feature names and their expected parameter patterns
        compound_features = {
            'bollinger_bands': {'params': ['period', 'std_dev'], 'defaults': [20, 2.0]},
            'donchian_channel': {'params': ['period'], 'defaults': [20]},
            'keltner_channel': {'params': ['period', 'multiplier'], 'defaults': [20, 2.0]},
            'linear_regression': {'params': ['period'], 'defaults': [20]},
            'parabolic_sar': {'params': ['af_start', 'af_max'], 'defaults': [0.02, 0.2]},
            'ultimate_oscillator': {'params': ['period1', 'period2', 'period3'], 'defaults': [7, 14, 28]},
            'ichimoku': {'params': ['conversion_period', 'base_period'], 'defaults': [9, 26]},
            'support_resistance': {'params': ['lookback', 'min_touches'], 'defaults': [50, 2]},
            'fibonacci_retracement': {'params': [], 'defaults': []},  # No parameters
            'swing_points': {'params': ['lookback'], 'defaults': [5]},
            'pivot_points': {'params': [], 'defaults': []},  # No parameters
            'stochastic_rsi': {'params': ['rsi_period', 'stoch_period'], 'defaults': [14, 14]},
            'atr_sma': {'params': ['atr_period', 'sma_period'], 'defaults': [14, 20]},
            'volatility_sma': {'params': ['vol_period', 'sma_period'], 'defaults': [20, 20]},
            'trendlines': {'params': ['pivot_lookback', 'min_touches', 'tolerance'], 'defaults': [20, 2, 0.002]},
        }
        
        # First check if it's a compound feature
        for compound_name, info in compound_features.items():
            if feature_id.startswith(compound_name + '_') or feature_id == compound_name:
                # Extract parameter values
                if feature_id == compound_name:
                    # No parameters provided, use defaults
                    config = {'type': compound_name}
                    for i, param_name in enumerate(info['params']):
                        config[param_name] = info['defaults'][i]
                    return config
                
                # Parse parameters after the compound name
                param_str = feature_id[len(compound_name) + 1:]  # +1 for underscore
                param_parts = param_str.split('_')
                
                config = {'type': compound_name}
                
                # Map parameters to their names
                for i, param_name in enumerate(info['params']):
                    if i < len(param_parts):
                        try:
                            # Try to convert to appropriate type
                            if '.' in param_parts[i]:
                                value = float(param_parts[i])
                            else:
                                value = int(param_parts[i])
                            config[param_name] = value
                        except ValueError:
                            # Use default if conversion fails
                            config[param_name] = info['defaults'][i] if i < len(info['defaults']) else None
                    else:
                        # Use default if not enough parameters
                        config[param_name] = info['defaults'][i] if i < len(info['defaults']) else None
                
                return config
        
        # Not a compound feature, parse as simple feature
        parts = feature_id.split('_')
        
        if len(parts) == 1:
            # Simple feature without parameters (e.g., 'vwap', 'ad')
            return {'type': feature_id}
        
        # Handle multi-part feature names like 'williams_r'
        if len(parts) >= 3 and parts[0] == 'williams' and parts[1] == 'r':
            feature_type = 'williams_r'
            # Use remaining parts for period extraction
            parts = ['williams_r'] + parts[2:]
        else:
            feature_type = parts[0]
        
        # Handle MACD specially (3 parameters)
        if feature_type == 'macd' and len(parts) >= 4:
            try:
                return {
                    'type': 'macd',
                    'fast_period': int(parts[1]),
                    'slow_period': int(parts[2]),
                    'signal_period': int(parts[3])
                }
            except ValueError:
                return {'type': 'macd', 'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        
        # Handle stochastic (2 parameters: k_period, d_period)
        elif feature_type == 'stochastic' and len(parts) >= 3:
            try:
                return {
                    'type': 'stochastic',
                    'k_period': int(parts[1]),
                    'd_period': int(parts[2])
                }
            except ValueError:
                return {'type': 'stochastic', 'k_period': 14, 'd_period': 3}
        
        # Handle features with standard 'period' parameter
        elif len(parts) >= 2:
            # Special case: VWAP doesn't take period parameters, the suffix is for other purposes
            if feature_type == 'vwap':
                return {'type': 'vwap'}
            
            try:
                period = int(parts[1])
                return {'type': feature_type, 'period': period}
            except ValueError:
                # Default period for most indicators that actually use periods
                if feature_type in ['sma', 'ema', 'rsi', 'cci', 'stochastic', 'roc', 'macd']:
                    return {'type': feature_type, 'period': 20}
                else:
                    # For unknown features, don't assume they need periods
                    return {'type': feature_type}
        
        # Fallback
        return {'type': feature_type}
    
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
        for portfolio_name, portfolio in portfolio_containers.items():
            portfolio_state = portfolio.get_component('portfolio_state')
            if portfolio_state and hasattr(portfolio_state, 'on_fill'):
                root_bus.subscribe(EventType.FILL, portfolio_state.on_fill)
                self.logger.info(f"Portfolio {portfolio_name} subscribed to FILL events")
    
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
    
    def _setup_universal_subscriptions(self, containers: Dict[str, Any], root_bus: Any) -> None:
        """Set up subscriptions for universal topology.
        
        Universal topology supports the complete trading pipeline:
        BAR -> SIGNAL -> ORDER -> FILL
        """
        from ..events import EventType
        
        # This is essentially the same as signal_generation but with portfolio/execution support
        self._setup_signal_generation_subscriptions(containers, root_bus)
        
        # Additionally set up portfolio and execution subscriptions if those containers exist
        
        # 3. Execution container subscribes to ORDER events only
        execution_container = containers.get('execution')
        if execution_container:
            execution_engine = execution_container.get_component('execution_engine')
            if execution_engine and hasattr(execution_engine, 'on_order'):
                # Subscribe to ORDER events on execution's event bus
                event_type_str = EventType.ORDER.value if hasattr(EventType.ORDER, 'value') else str(EventType.ORDER)
                execution_container.event_bus.subscribe(event_type_str, execution_engine.on_order)
                self.logger.info(f"Execution engine subscribed to '{event_type_str}' events")
            else:
                self.logger.warning("Execution container has no execution_engine with on_order method")
        
        # 4. Portfolio subscribes to FILL events from execution
        portfolio_container = containers.get('portfolio')
        if portfolio_container:
            portfolio_manager = portfolio_container.get_component('portfolio_manager')
            if portfolio_manager:
                # Get portfolio state which has on_fill method
                portfolio_state = portfolio_container.get_component('portfolio_state') or portfolio_manager
                if hasattr(portfolio_state, 'on_fill'):
                    # Subscribe to FILL events with proper filter
                    event_type_str = EventType.FILL.value if hasattr(EventType.FILL, 'value') else str(EventType.FILL)
                    
                    # Create filter function to only receive fills for this portfolio
                    def fill_filter(event):
                        # Accept all fills for now - in production would filter by portfolio ID
                        return True
                    
                    portfolio_container.event_bus.subscribe(
                        event_type_str, 
                        portfolio_state.on_fill,
                        filter_func=fill_filter
                    )
                    self.logger.info(f"Portfolio subscribed to '{event_type_str}' events with filter")
                else:
                    self.logger.warning("Portfolio has no on_fill method")
    
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
                self.logger.info(f"Inferred {len(required_features)} unique features from {len(strategies)} strategies and {len(classifiers)} classifiers")
                self.logger.debug(f"Inferred features: {sorted(required_features)}")
                
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
                    feature_configs[feature_id] = self._create_feature_config_from_id(feature_id)

                # Add feature configs to context config for strategy containers  
                if "feature_configs" not in context['config']:
                    context['config']["feature_configs"] = feature_configs
                else:
                    context['config']["feature_configs"].update(feature_configs)
                
                self.logger.debug(f"Generated feature configs: {feature_configs}")
                
                # Also add to top-level features config for backward compatibility
                if "features" not in context:
                    context["features"] = list(required_features)
                else:
                    existing = set(context.get("features", []))
                    combined = existing.union(required_features)
                    context["features"] = list(combined)
                    
                self.logger.debug(f"Injected {len(required_features)} features into context")
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
        
        # Dynamically import all indicator strategy modules to ensure decorators run
        indicator_modules = [
            'src.strategy.strategies.indicators.crossovers',
            'src.strategy.strategies.indicators.oscillators', 
            'src.strategy.strategies.indicators.momentum',  # Add momentum module
            'src.strategy.strategies.indicators.trend',
            'src.strategy.strategies.indicators.volatility',
            'src.strategy.strategies.indicators.volume',
            'src.strategy.strategies.indicators.structure',
            'src.strategy.strategies.ensemble.duckdb_ensemble'  # Add ensemble module
        ]
        
        # Import all indicator modules to ensure decorators run
        imported_modules = set()
        for module_path in indicator_modules:
            try:
                importlib.import_module(module_path)
                imported_modules.add(module_path)
                self.logger.debug(f"Imported indicator module {module_path}")
            except ImportError as e:
                self.logger.debug(f"Could not import {module_path}: {e}")
        
        # Log all registered strategies for debugging
        all_strategies = registry.get_components_by_type('strategy')
        self.logger.info(f"Registry contains {len(all_strategies)} strategies: {[s.name for s in all_strategies]}")
        
        for strategy_config in strategies:
            strategy_type = strategy_config.get('type', strategy_config.get('class'))
            strategy_params = strategy_config.get('params', strategy_config.get('parameters', {}))
            
            self.logger.debug(f"Processing strategy {strategy_type} with params: {strategy_params}")
            
            # Get strategy metadata from registry (modules already imported above)
            strategy_info = None
            for name_variant in [strategy_type, f'{strategy_type}_strategy', strategy_type.replace('_strategy', '')]:
                strategy_info = registry.get_component(name_variant)
                if strategy_info:
                    self.logger.debug(f"Found strategy {name_variant} in registry with metadata: {strategy_info.metadata}")
                    break
            
            if strategy_info:
                # Extract feature requirements from metadata
                feature_config = strategy_info.metadata.get('feature_config', {})
                
                # Handle both old dict format and new list format
                if isinstance(feature_config, list):
                    # New simplified format: ['sma', 'rsi', 'bollinger_bands']
                    self.logger.debug(f"Strategy '{strategy_type}' uses simplified feature config: {feature_config}")
                    
                    # Check for strategy-specific parameter mapping (best practice from strategy-interface.md)
                    strategy_param_mapping = strategy_info.metadata.get('param_feature_mapping', {})
                    if strategy_param_mapping:
                        self.logger.debug(f"Strategy '{strategy_type}' has custom param_feature_mapping: {strategy_param_mapping}")
                    
                    # All strategies should now use param_feature_mapping (strategy-interface.md best practice)
                    
                    # Use strategy-specific param_feature_mapping for all feature inference
                    if strategy_param_mapping:
                        # Generate features using custom parameter mapping
                        feature_names_generated = set()
                        for param_name, param_values in strategy_params.items():
                            if param_name in strategy_param_mapping:
                                # Get the feature name template
                                feature_template = strategy_param_mapping[param_name]
                                
                                # Substitute parameter values into template
                                if isinstance(param_values, list):
                                    for value in param_values:
                                        # Create a context for template substitution
                                        template_context = {param_name: value}
                                        template_context.update({k: v if not isinstance(v, list) else v[0] 
                                                                for k, v in strategy_params.items() 
                                                                if k != param_name})
                                        feature_name_generated = feature_template.format(**template_context)
                                        feature_names_generated.add(feature_name_generated)
                                else:
                                    # Single value
                                    template_context = dict(strategy_params)
                                    feature_name_generated = feature_template.format(**template_context)
                                    feature_names_generated.add(feature_name_generated)
                        
                        # Add all generated feature names
                        required_features.update(feature_names_generated)
                        self.logger.debug(f"Generated features for '{strategy_type}' using param_feature_mapping: {feature_names_generated}")
                        
                        # IMPORTANT: Also add base features from feature_config that don't need parameters
                        # For example, obv_trend needs both 'obv' (base) and 'sma_{period}' (parameterized)
                        for feature_name in feature_config:
                            # Check if this feature is covered by param_feature_mapping by checking if 
                            # any template starts with the feature name followed by underscore or equals
                            is_parameterized = any(
                                template.startswith(feature_name + '_') or template == feature_name
                                for template in strategy_param_mapping.values()
                            )
                            if not is_parameterized:
                                # This is a base feature without parameters - add it directly
                                required_features.add(feature_name)
                                self.logger.debug(f"Added base feature '{feature_name}' for strategy '{strategy_type}'")
                    else:
                        # Check if strategy has parameters that would affect feature names
                        has_feature_affecting_params = any(
                            'period' in param_name.lower() or 'lookback' in param_name.lower()
                            for param_name in strategy_params.keys()
                        )
                        
                        if has_feature_affecting_params:
                            # Only warn if strategy has parameters that should affect feature names
                            self.logger.debug(f"Strategy '{strategy_type}' missing param_feature_mapping - using simple feature names")
                        else:
                            # Strategy uses hardcoded periods, no warning needed
                            self.logger.debug(f"Strategy '{strategy_type}' uses hardcoded periods - no param_feature_mapping needed")
                        for feature_name in feature_config:
                            # Special case for VWAP (no parameters)
                            if feature_name == 'vwap':
                                required_features.add('vwap')
                            else:
                                # Try to find period parameters for simple inference
                                for param_name, param_values in strategy_params.items():
                                    if 'period' in param_name.lower():
                                        if isinstance(param_values, list):
                                            for value in param_values:
                                                required_features.add(f'{feature_name}_{value}')
                                        else:
                                            required_features.add(f'{feature_name}_{param_values}')
                                        break
                                else:
                                    # No period found, just add feature name
                                    required_features.add(feature_name)
                
                self.logger.debug(f"Strategy '{strategy_type}' requires features: {sorted(required_features)}")
                
                # RECURSIVE ANALYSIS: Check if this is an ensemble strategy with sub-strategies
                if strategy_type in ['duckdb_ensemble', 'ensemble'] or 'ensemble' in strategy_type.lower():
                    ensemble_features = self._infer_features_from_ensemble_substrategies(strategy_params, registry)
                    if ensemble_features:
                        required_features.update(ensemble_features)
                        self.logger.info(f"Ensemble '{strategy_type}' recursive analysis added {len(ensemble_features)} sub-strategy features")
                
            else:
                # Strategy not found in registry - this shouldn't happen with proper imports
                self.logger.warning(f"Strategy '{strategy_type}' not found in registry after importing all modules")
                # Add basic default features as fallback
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
            importlib.import_module('src.strategy.classifiers.multi_state_classifiers')
        except ImportError:
            self.logger.debug("Could not import multi_state_classifiers module")
        
        for classifier_config in classifiers:
            classifier_type = classifier_config.get('type', classifier_config.get('name'))
            classifier_params = classifier_config.get('params', classifier_config.get('parameters', {}))
            
            # Get classifier metadata from registry
            classifier_info = registry.get_component(classifier_type)
            
            if classifier_info:
                # Extract feature requirements from metadata using same pattern as strategies
                feature_config = classifier_info.metadata.get('feature_config', [])
                
                # Handle both old dict format and new list format
                if isinstance(feature_config, list):
                    # New simplified format: ['sma', 'rsi', 'atr']
                    self.logger.debug(f"Classifier '{classifier_type}' uses simplified feature config: {feature_config}")
                    
                    # Check for classifier-specific parameter mapping (same pattern as strategies)
                    classifier_param_mapping = classifier_info.metadata.get('param_feature_mapping', {})
                    if classifier_param_mapping:
                        self.logger.debug(f"Classifier '{classifier_type}' has custom param_feature_mapping: {classifier_param_mapping}")
                        
                        # Generate features using custom parameter mapping
                        feature_names_generated = set()
                        for param_name, param_values in classifier_params.items():
                            if param_name in classifier_param_mapping:
                                # Get the feature name template
                                feature_template = classifier_param_mapping[param_name]
                                
                                # Substitute parameter values into template
                                if isinstance(param_values, list):
                                    for value in param_values:
                                        template_context = {param_name: value}
                                        template_context.update({k: v if not isinstance(v, list) else v[0] 
                                                                for k, v in classifier_params.items() 
                                                                if k != param_name})
                                        feature_name_generated = feature_template.format(**template_context)
                                        feature_names_generated.add(feature_name_generated)
                                else:
                                    # Single value
                                    template_context = dict(classifier_params)
                                    feature_name_generated = feature_template.format(**template_context)
                                    feature_names_generated.add(feature_name_generated)
                        
                        # Add all generated feature names
                        required_features.update(feature_names_generated)
                        self.logger.debug(f"Generated features for classifier '{classifier_type}' using param_feature_mapping: {feature_names_generated}")
                    else:
                        # Check if classifier has parameters that would affect feature names
                        has_feature_affecting_params = any(
                            'period' in param_name.lower() or 'lookback' in param_name.lower()
                            for param_name in classifier_params.keys()
                        )
                        
                        if has_feature_affecting_params:
                            # Only warn if classifier has parameters that should affect feature names
                            self.logger.debug(f"Classifier '{classifier_type}' missing param_feature_mapping - using simple feature names")
                        else:
                            # Classifier uses hardcoded periods, no warning needed
                            self.logger.debug(f"Classifier '{classifier_type}' uses hardcoded periods - no param_feature_mapping needed")
                        for feature_name in feature_config:
                            # Try to find period parameters for simple inference
                            for param_name, param_values in classifier_params.items():
                                if 'period' in param_name.lower():
                                    if isinstance(param_values, list):
                                        for value in param_values:
                                            required_features.add(f'{feature_name}_{value}')
                                    else:
                                        required_features.add(f'{feature_name}_{param_values}')
                                    break
                            else:
                                # No period found, add default or just feature name
                                if feature_name in ['sma', 'rsi', 'atr']:
                                    default_periods = {'sma': 20, 'rsi': 14, 'atr': 14}
                                    required_features.add(f'{feature_name}_{default_periods[feature_name]}')
                                else:
                                    required_features.add(feature_name)
                
                self.logger.debug(f"Classifier '{classifier_type}' requires features: {sorted(required_features)}")
            else:
                # Classifier not found in registry - add minimal default features
                self.logger.warning(f"Classifier '{classifier_type}' not found in registry - using minimal defaults")
                required_features.update(['sma_20', 'rsi_14', 'atr_14'])
        
        return required_features

    def _infer_features_from_ensemble_substrategies(self, ensemble_params: Dict[str, Any], registry) -> Set[str]:
        """
        Recursively analyze ensemble sub-strategies to infer their feature requirements.
        
        Args:
            ensemble_params: Parameters of the ensemble strategy containing regime_strategies
            registry: Component registry for looking up sub-strategies
            
        Returns:
            Set of required feature identifiers for all sub-strategies
        """
        required_features = set()
        
        # Look for regime_strategies in params (duckdb_ensemble format)
        regime_strategies = ensemble_params.get('regime_strategies')
        if not regime_strategies:
            # If not specified, ensemble will use DEFAULT_REGIME_STRATEGIES
            # Import the module to get the default strategies
            try:
                from ...strategy.strategies.ensemble.duckdb_ensemble import DEFAULT_REGIME_STRATEGIES
                regime_strategies = DEFAULT_REGIME_STRATEGIES
                self.logger.debug("Using DEFAULT_REGIME_STRATEGIES from duckdb_ensemble")
            except ImportError as e:
                self.logger.warning(f"Could not import DEFAULT_REGIME_STRATEGIES: {e}")
                return required_features
        
        self.logger.debug(f"Analyzing ensemble with {len(regime_strategies)} regimes")
        
        # Iterate through all regimes and their sub-strategies
        for regime_name, sub_strategies in regime_strategies.items():
            if not isinstance(sub_strategies, list):
                continue
                
            self.logger.debug(f"Regime '{regime_name}' has {len(sub_strategies)} sub-strategies")
            
            for sub_strategy_config in sub_strategies:
                if not isinstance(sub_strategy_config, dict):
                    continue
                    
                sub_strategy_name = sub_strategy_config.get('name')
                sub_strategy_params = sub_strategy_config.get('params', {})
                
                if not sub_strategy_name:
                    continue
                
                self.logger.debug(f"Analyzing sub-strategy '{sub_strategy_name}' with params: {sub_strategy_params}")
                
                # Get sub-strategy metadata from registry
                sub_strategy_info = registry.get_component(sub_strategy_name)
                if not sub_strategy_info:
                    self.logger.warning(f"Sub-strategy '{sub_strategy_name}' not found in registry")
                    continue
                
                # Extract feature requirements using same logic as main inference
                feature_config = sub_strategy_info.metadata.get('feature_config', [])
                
                if isinstance(feature_config, list):
                    # Use parameter mapping if available
                    param_mapping = sub_strategy_info.metadata.get('param_feature_mapping', {})
                    
                    if param_mapping:
                        # Generate features using parameter mapping
                        for param_name, param_value in sub_strategy_params.items():
                            if param_name in param_mapping:
                                feature_template = param_mapping[param_name]
                                
                                # Simple template substitution
                                if isinstance(param_value, list):
                                    for value in param_value:
                                        context = {param_name: value}
                                        context.update({k: v if not isinstance(v, list) else v[0] 
                                                      for k, v in sub_strategy_params.items() if k != param_name})
                                        feature_name = feature_template.format(**context)
                                        required_features.add(feature_name)
                                else:
                                    feature_name = feature_template.format(**sub_strategy_params)
                                    required_features.add(feature_name)
                        
                        # Add base features not covered by param mapping
                        for feature_name in feature_config:
                            is_parameterized = any(
                                template.startswith(feature_name + '_') or template == feature_name
                                for template in param_mapping.values()
                            )
                            if not is_parameterized:
                                required_features.add(feature_name)
                    else:
                        # Fallback: simple period-based inference
                        for feature_name in feature_config:
                            if feature_name == 'vwap':
                                required_features.add('vwap')
                            else:
                                # Look for period parameters
                                period_found = False
                                for param_name, param_value in sub_strategy_params.items():
                                    if 'period' in param_name.lower():
                                        if isinstance(param_value, list):
                                            for value in param_value:
                                                required_features.add(f'{feature_name}_{value}')
                                        else:
                                            required_features.add(f'{feature_name}_{param_value}')
                                        period_found = True
                                        break
                                
                                if not period_found:
                                    # Use default period for common features
                                    defaults = {'sma': 20, 'ema': 20, 'rsi': 14, 'atr': 14, 'cci': 20}
                                    default_period = defaults.get(feature_name, 14)
                                    required_features.add(f'{feature_name}_{default_period}')
                
                self.logger.debug(f"Sub-strategy '{sub_strategy_name}' requires features: {sorted(required_features)}")
        
        return required_features

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
            'metadata': metadata,
            'generated': {},
            'root_event_bus': None,
            'use_hierarchical_events': True,
            'original_strategies': original_strategies,
            'original_classifiers': original_classifiers
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
        
        # Initialize component category
        if comp_type == 'strategies':
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
            # Direct access to config 
            items = context.get('config', {}).get(from_config, [])
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
        """Find and return stateless function/component reference."""
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
                    return strategy_info.factory
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

