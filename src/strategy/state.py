"""
Generic component state management - Replaces StrategyState with ComponentState.

Manages feature state and executes any stateless component functions (strategies, classifiers, etc.)
This is the canonical implementation that deprecates the old StrategyState.
"""

from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import logging
from datetime import datetime

from .protocols import FeatureProvider
from .types import Signal, SignalType, SignalDirection
from .classification_types import Classification, create_classification_event
from ..core.events.types import Event, EventType
from .components.config_filter import ConfigSignalFilter, create_filter_from_config

logger = logging.getLogger(__name__)


class ComponentState:
    """
    Generic component execution engine - NO INHERITANCE!
    
    Manages feature state and executes any stateless component functions:
    - Strategies (existing behavior)
    - Classifiers (new behavior) 
    - Future component types (extensible)
    
    Similar to PortfolioState but for any component type:
    - PortfolioState maintains positions and calls risk validators
    - ComponentState maintains features and calls any stateless functions
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        feature_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        verbose_signals: bool = False
    ):
        """
        Initialize component state.
        
        Args:
            symbols: Initial symbols to track
            feature_configs: Feature configurations (used for readiness checks only)
            verbose_signals: Whether to print signal generation to console
        """
        self._symbols = symbols or []
        self._verbose_signals = verbose_signals
        
        # Feature configurations - only used for readiness checks now
        self._feature_configs = feature_configs or {}
        
        # Bar count per symbol
        self._bar_count: Dict[str, int] = defaultdict(int)
        
        # Track actual bar indices from data stream
        self._current_bar_indices: Dict[str, Dict[str, int]] = defaultdict(lambda: {'original': 0, 'split': 0})
        
        # Component registry - populated by container/topology
        # Each component has: function, parameters, component_type, last_output, filter
        self._components: Dict[str, Dict[str, Any]] = {}
        
        # Signal filters from configuration
        self._component_filters: Dict[str, ConfigSignalFilter] = {}
        
        # Container reference for event publishing
        self._container = None
        
        # Feature Hub reference for centralized computation
        self._feature_hub = None
        self._deferred_feature_hub_name = None
        
        # Performance optimization: cache ready components after warmup
        self._ready_components_cache: Dict[str, List[tuple]] = {}  # symbol -> [(id, info)]
        self._warmup_complete: Dict[str, bool] = {}  # symbol -> bool
        self._warmup_bars = 200  # After this many bars, cache ready components - increased for classifiers
        
        # Metrics
        self._bars_processed = 0
        self._outputs_generated = 0
        self._last_update = datetime.now()
        
        # Component name for identification
        self.name = "component_state"
        
        # Default timeframe
        self._timeframe = "1m"
    
    def _calculate_max_lookback(self, feature_configs: Optional[Dict[str, Dict[str, Any]]]) -> int:
        """Calculate maximum lookback period needed based on features."""
        if not feature_configs:
            return 100  # Conservative default if no features configured
        
        max_lookback = 50  # Minimum for basic warmup
        
        for feature_name, config in feature_configs.items():
            feature_type = config.get('feature', '')
            
            # Extract period parameters
            if 'period' in config:
                max_lookback = max(max_lookback, config['period'] + 10)
            elif 'slow_period' in config:
                max_lookback = max(max_lookback, config['slow_period'] + 10)
            elif 'bb_period' in config:
                max_lookback = max(max_lookback, config['bb_period'] + 10)
            elif 'lookback_period' in config:
                max_lookback = max(max_lookback, config['lookback_period'] + 10)
                
            # Handle specific feature types
            if feature_type == 'macd':
                slow = config.get('slow', 26)
                signal = config.get('signal', 9)
                max_lookback = max(max_lookback, slow + signal + 10)
            elif feature_type == 'atr':
                period = config.get('period', 14)
                max_lookback = max(max_lookback, period + 10)
            elif feature_type in ['bollinger', 'bollinger_bands']:
                period = config.get('period', 20)
                max_lookback = max(max_lookback, period + 10)
        
        logger.debug(f"ComponentState configured with max_lookback={max_lookback} based on features")
        return max_lookback
    
    def _get_container_config(self, container=None) -> Dict[str, Any]:
        """Safely get container config, returning empty dict if not available."""
        container = container or self._container
        if not container:
            return {}
        config_obj = getattr(container, 'config', None)
        if not config_obj:
            return {}
        config_dict = getattr(config_obj, 'config', None)
        if config_dict is None:
            return {}
        return config_dict
    
    def set_container(self, container) -> None:
        """Set container reference and subscribe to events."""
        self._container = container
        
        # Check if FeatureHub reference is provided
        config_dict = self._get_container_config(container)
        feature_hub_name = config_dict.get('feature_hub_name')
        logger.info(f"ComponentState checking for FeatureHub: feature_hub_name={feature_hub_name}")
        if feature_hub_name:
            # First try parent container
            parent = container.parent_container
            logger.info(f"Parent container: {parent.name if parent else 'None'}")
            
            # If no parent yet, this might be during initialization
            # We'll set up a deferred connection
            if not parent:
                logger.info("Parent not set yet, deferring FeatureHub connection")
                self._deferred_feature_hub_name = feature_hub_name
            else:
                self._connect_to_feature_hub(parent, feature_hub_name)
        
        # Subscribe to BAR events
        logger.debug(f"ComponentState subscribing to event_bus: {container.event_bus}")
        container.event_bus.subscribe(EventType.BAR.value, self.on_bar)
        logger.debug(f"ComponentState subscribed to BAR events in container {container.name}")
        
        # Load components from container config
        self._load_components_from_config(container)
    
    def _connect_to_feature_hub(self, parent_container, feature_hub_name: str) -> None:
        """Connect to FeatureHub in parent container."""
        # Find the feature hub container
        logger.info(f"Child containers in parent: {list(parent_container._child_containers.keys())}")
        for child_name, child_container in parent_container._child_containers.items():
            logger.debug(f"Checking child container: {child_name} (name={child_container.name})")
            if child_container.name == feature_hub_name:
                # Get the feature_hub component from the container
                feature_hub_component = child_container.get_component('feature_hub')
                if feature_hub_component:
                    self._feature_hub = feature_hub_component.get_feature_hub()
                    logger.info(f"✅ ComponentState connected to FeatureHub from container {feature_hub_name}")
                    break
                else:
                    logger.warning(f"No feature_hub component in container {feature_hub_name}")
        
        if not self._feature_hub:
            logger.warning(f"FeatureHub container '{feature_hub_name}' not found in parent")
    
    def complete_deferred_connections(self) -> None:
        """Complete any deferred connections after container hierarchy is established."""
        if self._deferred_feature_hub_name and self._container:
            parent = self._container.parent_container
            if parent:
                logger.info(f"Completing deferred FeatureHub connection to '{self._deferred_feature_hub_name}'")
                self._connect_to_feature_hub(parent, self._deferred_feature_hub_name)
                self._deferred_feature_hub_name = None
    
    # Implement FeatureProvider protocol directly
    def update_bar(self, symbol: str, bar: Dict[str, float]) -> None:
        """Update with new bar data."""
        # With FeatureHub, we only track bar count for readiness checks
        self._bar_count[symbol] += 1
    
    def get_features(self, symbol: str) -> Dict[str, Any]:
        """Get current features for symbol."""
        # Always get features from FeatureHub
        if self._feature_hub:
            features = self._feature_hub.get_features(symbol).copy()
            if self._bar_count.get(symbol, 0) == 25:  # Debug at bar 25
                logger.debug(f"FeatureHub returned {len(features)} features for {symbol}")
                logger.debug(f"  Feature keys: {list(features.keys())}")
                if 'bollinger_bands_20_2.0_upper' in features:
                    logger.debug(f"  bollinger upper: {features['bollinger_bands_20_2.0_upper']}")
        else:
            # FeatureHub not connected yet
            features = {}
            if self._bar_count.get(symbol, 0) == 25:
                logger.debug(f"FeatureHub not connected for {symbol}")
        
        # Add state features that strategies may need
        features['bar_count'] = self._bar_count.get(symbol, 0)
        
        # Also add actual bar indices if available
        if hasattr(self, '_current_bar_indices') and symbol in self._current_bar_indices:
            features['original_bar_index'] = self._current_bar_indices[symbol].get('original', features['bar_count'])
            features['split_bar_index'] = self._current_bar_indices[symbol].get('split', features['bar_count'])
            features['actual_bar_count'] = self._current_bar_indices[symbol].get('original', features['bar_count'])
        
        # Ensure basic price data is available as features
        if self._feature_hub and not features:
            # If FeatureHub returns empty, we might not have enough data yet
            logger.debug(f"FeatureHub returned empty features for {symbol} at bar {features['bar_count']}")
        
        return features
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]]) -> None:
        """Configure which features to calculate."""
        self._feature_configs = feature_configs
        logger.debug(f"Configured {len(feature_configs)} features")
    
    def has_sufficient_data(self, symbol: str, min_bars: int = None) -> bool:
        """Check if sufficient data available.
        
        The minimum bars required is determined by:
        1. Explicit min_bars parameter
        2. Maximum lookback period from all configured features
        3. Default of 50 bars for warmup
        """
        # If we have a FeatureHub, use its check
        if self._feature_hub:
            return self._feature_hub.has_sufficient_data(symbol, min_bars or 50)
        
        if min_bars is None:
            # Calculate minimum bars needed based on configured features
            min_bars = 50  # Default warmup period
            
            # Check feature requirements
            for feature_name, config in self._feature_configs.items():
                feature_type = config.get('feature', '')
                
                # Extract period parameters
                if 'period' in config:
                    min_bars = max(min_bars, config['period'] + 5)
                elif 'slow_period' in config:
                    min_bars = max(min_bars, config['slow_period'] + 5)
                elif 'bb_period' in config:
                    min_bars = max(min_bars, config['bb_period'] + 5)
                elif 'lookback_period' in config:
                    min_bars = max(min_bars, config['lookback_period'] + 5)
                    
                # Handle specific feature types
                if feature_type == 'macd':
                    slow = config.get('slow', 26)
                    signal = config.get('signal', 9)
                    min_bars = max(min_bars, slow + signal + 5)
                elif feature_type == 'atr':
                    period = config.get('period', 14)
                    min_bars = max(min_bars, period + 5)
                elif feature_type in ['bollinger', 'bollinger_bands']:
                    period = config.get('period', 20)
                    min_bars = max(min_bars, period + 5)
        
        current_bars = self._bar_count.get(symbol, 0)
        return current_bars >= min_bars
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset state for symbol or all symbols."""
        if symbol:
            # Reset specific symbol
            self._bar_count[symbol] = 0
            self._warmup_complete[symbol] = False
            if symbol in self._ready_components_cache:
                del self._ready_components_cache[symbol]
            if symbol in self._current_bar_indices:
                self._current_bar_indices[symbol] = {'original': 0, 'split': 0}
        else:
            # Full reset
            self._bar_count.clear()
            self._warmup_complete.clear()
            self._ready_components_cache.clear()
            self._current_bar_indices.clear()
            for component in self._components.values():
                component['last_output'] = None
    
    
    
    # Component management (generalized from strategy management)
    def add_component(self, component_id: str, component_func: Any, 
                     component_type: str = "strategy",
                     parameters: Optional[Dict[str, Any]] = None,
                     filter_config: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a component function to execute.
        
        Args:
            component_id: Unique identifier
            component_func: Pure component function (stateless)
            component_type: Type of component (strategy, classifier, etc.)
            parameters: Component parameters
            filter_config: Filter configuration with 'threshold' (or deprecated 'filter') and 'filter_params' keys
        """
        self._components[component_id] = {
            'function': component_func,
            'component_type': component_type,
            'parameters': parameters or {},
            'last_output': None,
            'metadata': metadata or {}
        }
        
        # Create filter if configured
        if filter_config and component_type == "strategy":
            filter_obj = create_filter_from_config(filter_config)
            if filter_obj:
                self._component_filters[component_id] = filter_obj
                logger.debug(f"Added filter for {component_id}: {filter_config.get('threshold') or filter_config.get('filter')}")
        
        logger.info(f"[COMPONENT_ADDED] Added {component_type}: {component_id}")
        logger.info(f"[COMPONENT_COUNT] Total components: {len(self._components)}")
    
    def _load_components_from_config(self, container) -> None:
        """Load components from container configuration."""
        # Check for components in container config
        config = container.config.config
        
        # Load feature configurations first
        features = config.get('features', {})
        logger.debug(f"ComponentState received features config: {features}")
        if features:
            self.configure_features(features)
            logger.debug(f"Configured {len(features)} features from container config")
        else:
            logger.warning("No features config received by ComponentState")
        
        # Load components - supports legacy strategies, legacy classifiers, and new components config
        strategies = config.get('strategies', [])
        classifiers = config.get('classifiers', [])
        components_config = config.get('components', [])
        
        logger.debug(f"ComponentState config sections: strategies={len(strategies)}, classifiers={len(classifiers)}, components={len(components_config)}")
        
        # Check if stateless components were passed from topology
        stateless_components = config.get('stateless_components', {})
        logger.info(f"ComponentState received stateless_components: {list(stateless_components.keys())}")
        for comp_type, components in stateless_components.items():
            if isinstance(components, dict):
                logger.info(f"  {comp_type}: {len(components)} components")
                # Show first few component names
                for i, name in enumerate(list(components.keys())[:5]):
                    logger.info(f"    - {name}")
                if len(components) > 5:
                    logger.info(f"    ... and {len(components) - 5} more")
            elif isinstance(components, list):
                logger.info(f"  {comp_type}: {len(components)} items")
                # For signal generation mode, strategies are passed as list
                # Convert to dict format expected by load methods
                if comp_type == 'strategies' and components:
                    # Convert list to dict with compiled strategies
                    strategies_dict = {}
                    for i, item in enumerate(components):
                        logger.debug(f"  Strategy item {i}: type={type(item)}, hasattr name={hasattr(item, '__name__')}")
                        if hasattr(item, '__name__'):
                            strategies_dict[item.__name__] = item
                            logger.debug(f"    Added as {item.__name__}")
                        elif isinstance(item, dict) and 'name' in item:
                            # This might be a dict with name and function
                            strategies_dict[item['name']] = item.get('function', item)
                        elif isinstance(item, str):
                            # It's just a name, use a placeholder
                            strategies_dict[item] = item
                            logger.debug(f"    Added string as {item}")
                    stateless_components[comp_type] = strategies_dict
                    logger.info(f"  Converted {len(strategies_dict)} strategies to dict format")
            else:
                logger.info(f"  {comp_type}: {components}")
        
        # Handle legacy strategies (backwards compatibility)
        if strategies or stateless_components.get('strategies'):
            self._load_strategies_from_config(strategies, stateless_components, container)
        
        # Handle legacy classifiers (backwards compatibility)
        if classifiers or stateless_components.get('classifiers'):
            self._load_classifiers_from_config(classifiers, stateless_components, container)
        
        # Handle new components configuration
        if components_config:
            self._load_components_from_new_config(components_config, stateless_components, container)
        
        if not strategies and not classifiers and not components_config and not stateless_components:
            logger.debug("No components configured")
    
    def _load_strategies_from_config(self, strategies: List[Dict[str, Any]], 
                                   stateless_components: Dict[str, Any],
                                   container) -> None:
        """Load strategies (backwards compatibility)."""
        logger.info(f"[LOAD_DEBUG] _load_strategies_from_config called")
        logger.info(f"  strategies count: {len(strategies) if strategies else 0}")
        logger.info(f"  stateless_components: {type(stateless_components)}")
        logger.info(f"  Keys: {list(stateless_components.keys()) if isinstance(stateless_components, dict) else 'Not a dict'}")
        
        strategy_components = stateless_components.get('strategies', {})
        
        logger.info(f"STRATEGY LOADING: Found {len(strategies)} strategy configs")
        logger.info(f"STRATEGY LOADING: Found {type(strategy_components)} for strategy_components")
        logger.info(f"STRATEGY LOADING: strategy_components = {strategy_components}")
        
        if strategy_components:
            logger.debug(f"Using {len(strategy_components)} strategies from topology")
            
            # If no strategy configs provided but we have compiled strategies, 
            # create minimal configs for them
            if not strategies and isinstance(strategy_components, dict):
                logger.info("No strategy configs provided, using compiled strategies directly")
                logger.info(f"Total strategies to process: {len(strategy_components)}")
                for idx, (strategy_name, strategy_func) in enumerate(strategy_components.items()):
                    logger.info(f"Processing compiled strategy {idx+1}/{len(strategy_components)}: {strategy_name}")
                    
                    # Create component_id
                    symbols = container.config.config.get('symbols', ['unknown'])
                    symbol = symbols[0] if symbols else 'unknown'
                    component_id = f"{symbol}_{strategy_name}"
                    
                    # Extract metadata from strategy function if available
                    strategy_metadata = {}
                    if hasattr(strategy_func, '_strategy_metadata'):
                        strategy_metadata = strategy_func._strategy_metadata
                        logger.info(f"Compiled strategy {strategy_name} has metadata: {list(strategy_metadata.keys())}")
                        if 'strategy_type' in strategy_metadata:
                            logger.info(f"  strategy_type: {strategy_metadata['strategy_type']}")
                        if 'composite_strategies' in strategy_metadata:
                            logger.info(f"  composite_strategies: {strategy_metadata['composite_strategies']}")
                        if 'parameters' in strategy_metadata:
                            logger.info(f"  direct parameters in metadata: {strategy_metadata['parameters']}")
                    
                    # Also check for _compiled_params directly
                    if hasattr(strategy_func, '_compiled_params'):
                        logger.info(f"  _compiled_params on function: {strategy_func._compiled_params}")
                    else:
                        logger.warning(f"  NO _compiled_params on function {strategy_name}")
                    
                    # Extract parameters from compiled strategy metadata config
                    params_with_type = {'_strategy_type': strategy_name}
                    
                    # Also check for parameter_combinations if config extraction fails
                    if strategy_metadata.get('strategy_type'):
                        params_with_type['_actual_strategy_type'] = strategy_metadata['strategy_type']
                    
                    # For composite strategies, extract sub-strategy info
                    if strategy_metadata.get('composite_strategies'):
                        params_with_type['composite_strategies'] = strategy_metadata['composite_strategies']
                        logger.info(f"Found composite strategy with {len(strategy_metadata['composite_strategies'])} sub-strategies")
                    
                    # First try direct parameters from metadata (new approach)
                    if strategy_metadata.get('parameters'):
                        params_with_type.update(strategy_metadata['parameters'])
                        logger.info(f"Extracted parameters directly from metadata: {params_with_type}")
                    elif strategy_metadata.get('params'):
                        # Try 'params' key as well (some strategies use this)
                        params_with_type.update(strategy_metadata['params'])
                        logger.info(f"Extracted parameters from metadata.params: {params_with_type}")
                    else:
                        # For compiled strategies in signal generation mode, parameters are often 
                        # already baked into the function and we just need the metadata to know
                        # what they are for tracing purposes
                        logger.warning(f"No 'parameters' or 'params' key in strategy metadata for {strategy_name}")
                        logger.warning(f"Available metadata keys: {list(strategy_metadata.keys())}")
                    
                    # If we still don't have real parameters, check if this is a compiled strategy
                    # Compiled strategies have their parameters in the function metadata
                    if len(params_with_type) <= 2 and hasattr(strategy_func, '_compiled_params'):
                        # This is set by the compiler
                        params_with_type.update(strategy_func._compiled_params)
                        logger.info(f"Extracted parameters from _compiled_params: {params_with_type}")
                    
                    if not strategy_metadata.get('parameters') and not strategy_metadata.get('params') and strategy_metadata.get('config'):
                        # The config contains the full strategy configuration
                        config = strategy_metadata['config']
                        logger.info(f"Compiled strategy config: {config}")
                        logger.info(f"Config type: {type(config)}, keys: {list(config.keys()) if isinstance(config, dict) else 'not a dict'}")
                        
                        # For atomic strategies, the config is {strategy_type: {params}}
                        if isinstance(config, dict) and len(config) == 1:
                            strategy_type_key = list(config.keys())[0]
                            strategy_params = config[strategy_type_key]
                            
                            # Handle both old format {params: {...}} and new format {...params...}
                            if isinstance(strategy_params, dict):
                                if 'params' in strategy_params:
                                    # Old format: {strategy_name: {params: {...}}}
                                    actual_params = strategy_params['params']
                                else:
                                    # New format: {strategy_name: {...params...}}
                                    # Extract all non-metadata keys as params
                                    actual_params = {k: v for k, v in strategy_params.items() 
                                                   if k not in {'weight', 'condition', 'metadata'}}
                                
                                # Merge with params_with_type to keep _strategy_type
                                params_with_type.update(actual_params)
                                logger.info(f"Extracted parameters from compiled strategy: {params_with_type}")
                    else:
                        logger.warning(f"No config found in compiled strategy metadata for {strategy_name}")
                    
                    # Debug: Log what we're actually passing
                    logger.info(f"[PARAMS_DEBUG] Adding component {component_id} with parameters: {params_with_type}")
                    
                    self.add_component(
                        component_id=component_id,
                        component_func=strategy_func,
                        component_type="strategy",
                        parameters=params_with_type,
                        metadata=strategy_metadata
                    )
                    logger.info(f"Loaded compiled strategy: {strategy_name}")
                # End of for loop - continue to next strategy
                
                # After all strategies processed, return to avoid double-processing
                return
            
            # Original logic for matching configs (only runs if strategies list is provided)
            for strategy_name, strategy_func in strategy_components.items():
                logger.debug(f"Processing strategy component: {strategy_name}")
                # Find matching config by name or type
                # For grid search, strategy_name might be expanded (e.g., 'sma_crossover_grid_3_15')
                # but we need to match against the config with the same name
                strategy_config = None
                for s in strategies:
                    if s.get('name') == strategy_name:
                        strategy_config = s
                        break
                    # Also check if this is a base strategy type match
                    if s.get('type') == strategy_name:
                        strategy_config = s
                        break
                
                if not strategy_config:
                    # For grid search: the strategy name in components matches the expanded name
                    # but we need the config which has the same expanded name
                    logger.warning(f"No config found for strategy {strategy_name}. Available configs: {[s.get('name', s.get('type')) for s in strategies][:5]}...")
                    # Try to find by partial match - grid search names include parameters
                    for s in strategies:
                        config_name = s.get('name', '')
                        if strategy_name == config_name:
                            strategy_config = s
                            logger.debug(f"Found config by exact name match: {config_name}")
                            break
                    if not strategy_config:
                        strategy_config = {}
                
                # Create component_id with symbol pattern for proper filtering
                symbols = container.config.config.get('symbols', ['unknown'])
                symbol = symbols[0] if symbols else 'unknown'
                component_id = f"{symbol}_{strategy_name}"
                
                # Add strategy type to parameters for better feature resolution
                params_with_type = strategy_config.get('params', {}).copy()
                params_with_type['_strategy_type'] = strategy_config.get('type', strategy_name)
                
                # Include risk parameters if present
                if 'risk' in strategy_config:
                    params_with_type['_risk'] = strategy_config['risk']
                    logger.info(f"Added risk parameters to strategy: {strategy_config['risk']}")
                
                # If no params from config, try to extract from function metadata
                if len(params_with_type) <= 1 and hasattr(strategy_func, '_strategy_metadata'):
                    strategy_metadata = strategy_func._strategy_metadata
                    if strategy_metadata.get('parameters'):
                        params_with_type.update(strategy_metadata['parameters'])
                        logger.info(f"Extracted parameters from function metadata: {params_with_type}")
                
                logger.info(f"Loading strategy {strategy_name} with params: {params_with_type}")
                logger.info(f"Strategy function metadata: {getattr(strategy_func, '_strategy_metadata', 'No metadata')}")
                if 'ensemble' in strategy_name.lower():
                    logger.info(f"ENSEMBLE STRATEGY LOADED: {component_id}")
                
                # Extract filter configuration
                filter_config = None
                if 'filter' in strategy_config:
                    filter_config = {
                        'filter': strategy_config['filter'],
                        'filter_params': strategy_config.get('filter_params', {})
                    }
                
                # Apply EOD close filter if enabled
                # TEMPORARILY DISABLED to debug strategy loading issue
                eod_close_enabled = False
                # if container and hasattr(container, 'config') and container.config:
                #     if hasattr(container.config, 'config') and container.config.config:
                #         eod_close_enabled = container.config.config.get('execution', {}).get('close_eod', False)
                
                if eod_close_enabled:
                    logger.info("[COMPONENT_STATE] Applying EOD close filter")
                    # Import helper to detect timeframe
                    from .components.eod_timeframe_helper import (
                        create_eod_filter_for_timeframe, 
                        detect_timeframe_from_config
                    )
                    
                    # Detect timeframe from config
                    timeframe_minutes = detect_timeframe_from_config(container.config.config)
                    
                    # Get existing filter if any
                    existing_filter = filter_config.get('filter') if filter_config else None
                    
                    # Create EOD filter for the detected timeframe
                    eod_filter_expr = create_eod_filter_for_timeframe(timeframe_minutes, existing_filter)
                    logger.info(f"[COMPONENT_STATE] EOD filter expression: {eod_filter_expr}")
                    
                    if filter_config:
                        filter_config['filter'] = eod_filter_expr
                    else:
                        filter_config = {
                            'filter': eod_filter_expr,
                            'filter_params': {}
                        }
                    
                    logger.info(f"Added EOD filter for {strategy_name} ({timeframe_minutes}m bars): {filter_config['filter']}")
                
                # Extract metadata from strategy function if available
                strategy_metadata = {}
                if hasattr(strategy_func, '_strategy_metadata'):
                    strategy_metadata = strategy_func._strategy_metadata
                    logger.info(f"Strategy {strategy_name} has metadata: {list(strategy_metadata.keys())}")
                    if 'strategy_type' in strategy_metadata:
                        logger.info(f"  strategy_type: {strategy_metadata['strategy_type']}")
                
                self.add_component(
                    component_id=component_id,
                    component_func=strategy_func,
                    component_type="strategy",
                    parameters=params_with_type,
                    filter_config=filter_config,
                    metadata=strategy_metadata
                )
                logger.debug(f"Loaded strategy: {strategy_name} from topology")
        else:
            logger.warning("No strategies found in stateless_components")
    
    def _load_classifiers_from_config(self, classifiers: List[Dict[str, Any]], 
                                    stateless_components: Dict[str, Any],
                                    container) -> None:
        """Load classifiers (backwards compatibility)."""
        classifier_components = stateless_components.get('classifiers', {})
        
        if classifier_components:
            logger.debug(f"Using {len(classifier_components)} classifiers from topology")
            
            # Get expanded parameters from context if available
            expanded_params = container.config.config.get('stateless_components', {}).get('parameters', {}).get('classifiers', {})
            
            for classifier_name, classifier_func in classifier_components.items():
                # Try to get parameters from expanded_params first
                classifier_params = expanded_params.get(classifier_name, {})
                
                # If not found, try to find matching config by base name
                if not classifier_params:
                    # Extract base name (before first underscore with digits)
                    base_name = classifier_name
                    for c in classifiers:
                        if c.get('name') in classifier_name or c.get('type') in classifier_name:
                            classifier_params = c.get('params', {})
                            break
                
                # Create component_id with symbol pattern for proper filtering
                symbols = container.config.config.get('symbols', ['unknown'])
                symbol = symbols[0] if symbols else 'unknown'
                component_id = f"{symbol}_{classifier_name}"
                
                self.add_component(
                    component_id=component_id,
                    component_func=classifier_func,
                    component_type="classifier",
                    parameters=classifier_params
                )
                logger.debug(f"Loaded classifier: {classifier_name} with params: {list(classifier_params.keys())}")
        else:
            logger.warning("No classifiers found in stateless_components")
    
    def _load_components_from_new_config(self, components_config: List[Dict[str, Any]], 
                                       stateless_components: Dict[str, Any],
                                       container) -> None:
        """Load components from new configuration format."""
        for comp_config in components_config:
            comp_type = comp_config.get('type')
            if not comp_type:
                logger.warning("Component config missing 'type': %s", comp_config)
                continue
            
            # Get functions for this component type
            type_components = stateless_components.get(comp_type, {})
            
            if not type_components:
                logger.warning(f"No {comp_type} found in stateless_components")
                continue
            
            logger.debug(f"Using {len(type_components)} {comp_type} from topology")
            
            for comp_name, comp_func in type_components.items():
                # Find matching parameters
                comp_params = comp_config.get('parameters', {}).get(comp_name, {})
                
                # Create component_id with symbol pattern
                symbols = container.config.config.get('symbols', ['unknown'])
                symbol = symbols[0] if symbols else 'unknown'
                component_id = f"{symbol}_{comp_name}"
                
                self.add_component(
                    component_id=component_id,
                    component_func=comp_func,
                    component_type=comp_type,
                    parameters=comp_params
                )
                logger.debug(f"Loaded {comp_type}: {comp_name} from topology")
    
    def remove_component(self, component_id: str) -> bool:
        """Remove a component."""
        if component_id in self._components:
            del self._components[component_id]
            logger.debug(f"Removed component: {component_id}")
            return True
        return False
    
    def get_component_ids(self) -> List[str]:
        """Get list of active component IDs."""
        component_ids = list(self._components.keys())
        logger.info(f"[GET_COMPONENT_IDS] Returning {len(component_ids)} component IDs: {component_ids[:5]}...")
        return component_ids
    
    # Event handling
    def on_bar(self, event: Event) -> None:
        """
        Handle BAR event - update features and execute components.
        
        This is the main event handler that coordinates:
        1. Feature calculation
        2. Component execution (strategies, classifiers, etc.)
        3. Output publishing as SIGNAL events
        """
        import time
        start_time = time.time()
        
        logger.debug(f"on_bar called with event type: {event.event_type}")
        
        if event.event_type != EventType.BAR.value:
            return
        
        # Complete deferred connections on first bar if needed
        if self._deferred_feature_hub_name:
            self.complete_deferred_connections()
        
        payload = event.payload
        symbol = payload.get('symbol')
        bar = payload.get('bar')
        
        if not symbol or not bar:
            logger.warning(f"Invalid BAR event: symbol={symbol}, bar={bar}")
            return
        
        self._bars_processed += 1
        self._last_update = datetime.now()
        
        # Convert bar object to dict for feature computation
        bar_dict = {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'symbol': symbol,
            'timestamp': bar.timestamp if hasattr(bar, 'timestamp') else None
        }
        
        current_bar = self._bar_count.get(symbol, 0) + 1
        logger.debug(f"Processing BAR for {symbol}: close={bar_dict['close']}, bar_count={current_bar}")
        
        # Get actual bar indices from payload
        original_bar_index = payload.get('original_bar_index', current_bar)
        split_bar_index = payload.get('split_bar_index', current_bar)
        
        # Store actual bar indices for later use
        self._current_bar_indices[symbol]['original'] = original_bar_index
        self._current_bar_indices[symbol]['split'] = split_bar_index
        
        # Update features
        update_start = time.time()
        try:
            self.update_bar(symbol, bar_dict)
        except Exception as e:
            logger.error(f"Error updating bar for {symbol}: {e}", exc_info=True)
            return
        update_time = time.time() - update_start
        
        # Get current features
        features_start = time.time()
        try:
            features = self.get_features(symbol)
            # Only log occasionally to reduce overhead
            if self._bar_count.get(symbol, 0) % 100 == 1:
                logger.debug(f"Features for {symbol}: {len(features)} features computed")
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}", exc_info=True)
            return
        features_time = time.time() - features_start
        
        # Execute components individually based on their readiness
        execute_start = time.time()
        try:
            # Use bar timestamp from payload instead of event execution timestamp
            bar_timestamp = payload.get('timestamp', event.timestamp)
            self._execute_components_individually(symbol, features, bar_dict, bar_timestamp)
        except Exception as e:
            logger.error(f"Error executing components for {symbol}: {e}", exc_info=True)
        execute_time = time.time() - execute_start
        
        total_time = time.time() - start_time
        if current_bar % 20 == 0 or total_time > 0.1:  # Log every 20 bars or if slow
            logger.debug(f"Bar {original_bar_index} (relative: {current_bar}) timing: total={total_time*1000:.1f}ms, update={update_time*1000:.1f}ms, features={features_time*1000:.1f}ms, execute={execute_time*1000:.1f}ms")
    
    def _execute_components_individually(self, symbol: str, features: Dict[str, Any], 
                                       bar: Dict[str, float], timestamp: datetime) -> None:
        """Execute components individually, each checking its own readiness."""
        current_bars = self._bar_count.get(symbol, 0)
        
        # Get actual bar index for logging
        actual_bar_index = self._current_bar_indices.get(symbol, {}).get('original', current_bars)
        
        # Performance optimization: Use cached ready components after warmup
        if self._warmup_complete.get(symbol, False):
            # Use cached ready components (much faster)
            cached = self._ready_components_cache.get(symbol, {})
            ready_classifiers = cached.get('classifiers', [])
            ready_strategies = cached.get('strategies', [])
            
            if current_bars % 100 == 0:  # Log occasionally
                logger.debug(f"Bar {actual_bar_index} (relative: {current_bars}): Using cached ready components - {len(ready_classifiers)} classifiers, {len(ready_strategies)} strategies")
                # Check for ready target strategies
                target_strategies_ready = sum(1 for c_id, _ in ready_strategies if any(name in c_id for name in ['parabolic_sar', 'supertrend', 'adx_trend']))
                logger.debug(f"  Target strategies (parabolic_sar/supertrend/adx_trend) ready: {target_strategies_ready}")
        else:
            # During warmup, check readiness
            components_snapshot = list(self._components.items())
            
            # Debug: count total strategies
            total_strategies = sum(1 for _, info in components_snapshot if info['component_type'] == 'strategy')
            # Log warmup only every 100 bars
            if current_bars % 100 == 0:
                logger.info(f"[WARMUP] Bar {current_bars}: Total strategies: {total_strategies}")
            
            # Track which components are ready
            ready_classifiers = []
            ready_strategies = []
            
            if current_bars == 1 or current_bars % 20 == 0:  # Log occasionally for debugging
                logger.debug(f"Bar {actual_bar_index} (relative: {current_bars}): Checking {len(components_snapshot)} components for readiness. Available features: {len(features)} features")
            
            for component_id, component_info in components_snapshot:
                component_type = component_info['component_type']
                
                # Log component checks at first bar
                if current_bars == 1:
                    logger.debug(f"Checking component: {component_id} (type: {component_type})")
                
                # Check if this specific component has the features it needs
                is_ready = self._is_component_ready(component_id, component_info, features, current_bars)
                
                # Debug ensemble strategies specifically (disabled - interferes with execution)
                # if 'ensemble' in component_id and current_bars <= 50:
                #     logger.error(f"🔍 ENSEMBLE DEBUG bar {current_bars}: {component_id} ready={is_ready}")
                #     if not is_ready:
                #         required_features = self._get_component_required_features(component_id, component_info)
                #         missing_features = [f for f in required_features if f not in features or features[f] is None]
                #         logger.error(f"🔍 ENSEMBLE MISSING: {missing_features[:5]}")
                
                if is_ready:
                    if component_type == 'classifier':
                        ready_classifiers.append((component_id, component_info))
                    else:
                        ready_strategies.append((component_id, component_info))
                        if current_bars == 20:  # Log when strategies become ready
                            logger.debug(f"Strategy {component_id} is READY at bar {current_bars}")
                else:
                    # Log why this component isn't ready yet
                    required_features = self._get_component_required_features(component_id, component_info)
                    missing_features = [f for f in required_features if f not in features or features[f] is None]
                    if missing_features and current_bars % 100 == 0:  # Log very rarely
                        logger.debug(f"Bar {current_bars}: Component {component_id} waiting for features: {missing_features[:3]}...")
            
            # Cache ready components after warmup period
            if current_bars >= self._warmup_bars and not self._warmup_complete.get(symbol, False):
                self._ready_components_cache[symbol] = {
                    'classifiers': ready_classifiers,
                    'strategies': ready_strategies
                }
                self._warmup_complete[symbol] = True
                
                # Detailed logging for target strategies
                target_strategies_ready = [c_id for c_id, _ in ready_strategies if any(name in c_id for name in ['parabolic_sar', 'supertrend', 'adx_trend'])]
                target_strategies_not_ready = [c_id for c_id, _ in components_snapshot if any(name in c_id for name in ['parabolic_sar', 'supertrend', 'adx_trend']) and c_id not in [r_id for r_id, _ in ready_strategies]]
                
                logger.debug(f"Warmup complete for {symbol} at bar {actual_bar_index} (relative: {current_bars}). Cached {len(ready_classifiers)} classifiers and {len(ready_strategies)} strategies")
                logger.debug(f"Target strategies ready: {len(target_strategies_ready)}")
                logger.debug(f"Target strategies NOT ready: {len(target_strategies_not_ready)}")
                if target_strategies_not_ready:
                    logger.info(f"NOT ready strategies: {target_strategies_not_ready[:5]}...")  # Show first 5
        
        # Only log ready check occasionally
        if actual_bar_index == 1 or actual_bar_index % 100 == 0:
            logger.debug(f"[READY_CHECK] Bar {actual_bar_index}: Total components: {len(self._components)}, Ready classifiers: {len(ready_classifiers)}, Ready strategies: {len(ready_strategies)}")
        if ready_classifiers or ready_strategies:
            if current_bars <= self._warmup_bars or current_bars % 20 == 0:
                logger.debug(f"Bar {actual_bar_index} (relative: {current_bars}): Executing {len(ready_classifiers)} classifiers and {len(ready_strategies)} strategies")
        
        # Execute ready classifiers first
        current_classifications = {}
        outputs_to_update = {}
        
        for component_id, component_info in ready_classifiers:
            try:
                result = component_info['function'](
                    features=features,
                    params=component_info['parameters']
                )
                
                if result:
                    outputs_to_update[component_id] = result
                    # Ensure metadata is included in component_info
                    component_info_with_metadata = component_info.copy()
                    if component_id in self._components and 'metadata' in self._components[component_id]:
                        component_info_with_metadata['metadata'] = self._components[component_id]['metadata']
                    self._process_component_output(
                        component_id=component_id,
                        component_type=component_info['component_type'],
                        result=result,
                        symbol=symbol,
                        timestamp=timestamp,
                        component_info=component_info_with_metadata,
                        bar=bar,
                        features=features
                    )
                    
                    # Store classification result for strategies to use
                    if result.get('regime'):
                        classifier_name = component_id.split('_', 1)[1] if '_' in component_id else component_id
                        current_classifications[f'regime_{classifier_name}'] = result['regime']
                        current_classifications[f'regime_confidence_{classifier_name}'] = result.get('confidence', 0.0)
                        
            except Exception as e:
                logger.error(f"Error executing classifier {component_id}: {e}")
        
        # Add current classifications to features for strategies to use
        if current_classifications:
            features = features.copy()  # Don't modify original features
            features.update(current_classifications)
        
        # Execute ready strategies
        # Only log execution count occasionally  
        if actual_bar_index == 1 or actual_bar_index % 500 == 0:
            logger.debug(f"[EXECUTING] About to execute {len(ready_strategies)} strategies")
        for idx, (component_id, component_info) in enumerate(ready_strategies):
            try:
                # Debug log execution
                if idx < 5 or idx % 50 == 0:  # Log first 5 and every 50th
                    # Debug logging commented out for production
                    # logger.info(f"[EXECUTING] Strategy {idx+1}/{len(ready_strategies)}: {component_id}")
                    pass
                # Debug logging for ensemble strategies
                if 'ensemble' in component_id.lower():
                    logger.debug(f"EXECUTING ENSEMBLE COMPONENT: {component_id} at bar {current_bars}")
                # Debug logging for missing strategies
                if any(name in component_id for name in ['parabolic_sar', 'supertrend', 'adx_trend']):
                    if current_bars % 50 == 0:  # Log every 50 bars
                        logger.debug(f"Executing {component_id} with {len(features)} features")
                
                result = component_info['function'](
                    features=features,
                    bar=bar,
                    params=component_info['parameters']
                )
                
                # Debug log the result for compiled strategies
                if 'compiled' in component_id and current_bars <= 30:
                    logger.debug(f"Compiled strategy {component_id} at bar {current_bars} returned: {result}")
                
                # Debug log for compiled strategies  
                if 'compiled_strategy' in component_id:
                    logger.debug(f"[STATE] Strategy {component_id} returned: {result}")
                
                if result:
                    outputs_to_update[component_id] = result
                    # Ensure metadata is included in component_info
                    component_info_with_metadata = component_info.copy()
                    if component_id in self._components and 'metadata' in self._components[component_id]:
                        component_info_with_metadata['metadata'] = self._components[component_id]['metadata']
                    # Debug log to check parameters
                    if 'parameters' in component_info_with_metadata:
                        params = component_info_with_metadata['parameters']
                        if params and len(params) > 1:  # More than just _strategy_type
                            logger.debug(f"[EXEC_DEBUG] Passing parameters to _process_component_output: {list(params.keys())}")
                    self._process_component_output(
                        component_id=component_id,
                        component_type=component_info['component_type'],
                        result=result,
                        symbol=symbol,
                        timestamp=timestamp,
                        component_info=component_info_with_metadata,
                        bar=bar,
                        features=features
                    )
                    
            except Exception as e:
                logger.error(f"Error executing strategy {component_id}: {e}", exc_info=True)
        
        # Update all outputs after iteration is complete
        for component_id, result in outputs_to_update.items():
            if component_id in self._components:  # Check if still exists
                self._components[component_id]['last_output'] = result
        
        # Check for EOD closure if enabled
        try:
            config_dict = self._get_container_config()
            if config_dict and config_dict.get('execution', {}).get('close_eod'):
                self._check_and_force_eod_closure(symbol, timestamp, bar, features, ready_strategies)
        except Exception as e:
            logger.debug(f"Error checking EOD closure config: {e}")
            # Continue without EOD closure
    
    def _is_component_ready(self, component_id: str, component_info: Dict[str, Any], 
                           features: Dict[str, Any], current_bars: int) -> bool:
        """Check if a specific component has the features it needs to execute."""
        component_type = component_info['component_type']
        params = component_info['parameters']
        
        if component_type == 'classifier':
            # Classifiers typically need basic features, ready quickly
            required_features = self._get_classifier_required_features(component_id, params)
            min_bars_needed = self._get_classifier_min_bars(component_id, params, required_features)
        else:
            # Strategies need specific features based on their parameters
            required_features = self._get_strategy_required_features(component_id, params)
            min_bars_needed = self._get_strategy_min_bars(component_id, params)
        
        # Check if we have enough bars
        if current_bars < min_bars_needed:
            # Debug ensemble bar requirements
            if 'ensemble' in component_id:
                logger.error(f"🔍 ENSEMBLE BARS: {component_id} needs {min_bars_needed} bars, has {current_bars}")
            elif current_bars == 1:
                logger.debug(f"Component {component_id} needs {min_bars_needed} bars, has {current_bars}")
            return False
        
        # Check if all required features are available and not None
        missing_features = []
        for feature_name in required_features:
            # Handle multi-value features (e.g., supertrend_10_3.0 -> supertrend_10_3.0_supertrend)
            found = False
            
            # First check exact match
            if feature_name in features:
                if features[feature_name] is not None:
                    found = True
                else:
                    missing_features.append(f"{feature_name} (is None)")
            else:
                # Check for sub-key features (multi-value features)
                # For example: supertrend_10_3.0 might be stored as supertrend_10_3.0_supertrend
                prefix = feature_name + '_'
                matching_features = [k for k in features.keys() if k.startswith(prefix)]
                
                if matching_features:
                    # Found sub-keys, check if at least one has a value
                    has_value = any(features[k] is not None for k in matching_features)
                    if has_value:
                        found = True
                    else:
                        missing_features.append(f"{feature_name} (all sub-keys are None)")
                else:
                    missing_features.append(f"{feature_name} (not in dict)")
            
            if not found:
                continue  # Will be added to missing_features already
        
        if missing_features and (current_bars == 20 or current_bars == 1):  # Log at bar 20 and 1 for debugging
            logger.debug(f"Component {component_id} at bar {current_bars}: Missing {len(missing_features)}/{len(required_features)} features")
            if current_bars == 1:
                logger.debug(f"  Required: {required_features[:3]}...")
                logger.debug(f"  Available feature keys: {list(features.keys())[:10]}...")
            # Special logging for target strategies
            if any(name in component_id for name in ['parabolic_sar', 'supertrend', 'adx_trend']):
                logger.info(f"NOT READY STRATEGY: {component_id} at bar 20")
                logger.info(f"  Required: {required_features}")
                logger.info(f"  Missing: {missing_features}")
                # Check for partial matches in available features
                partial_matches = [k for k in features.keys() if any(req.split('_')[0] in k for req in required_features)]
                logger.info(f"  Partial feature matches: {partial_matches[:10]}")
                logger.info(f"  Total available features: {len(features)}")
            else:
                logger.debug(f"Component {component_id} at bar 20 - Required: {required_features}, Missing: {missing_features}, Available: {list(features.keys())[:10]}...")
        
        return len(missing_features) == 0
        
        return True
    
    def _get_strategy_required_features(self, component_id: str, params: Dict[str, Any]) -> List[str]:
        """Get the specific features this strategy needs by extracting from function metadata."""
        # Add debug logging for target strategies
        is_target_strategy = any(name in component_id for name in ['parabolic_sar', 'supertrend', 'adx_trend'])
        if is_target_strategy:
            logger.debug(f"Getting required features for TARGET STRATEGY {component_id} with params: {params}")
        else:
            logger.debug(f"Getting required features for {component_id} with params: {params}")
        
        # First, try to get the strategy function and extract its feature config
        component_info = self._components.get(component_id)
        if component_info and 'function' in component_info:
            strategy_func = component_info['function']
            
            # Extract feature config from the strategy function's metadata
            # Check both _component_info (from original strategy) and _strategy_metadata (from compiler)
            if hasattr(strategy_func, '_component_info'):
                logger.debug(f"Found _component_info for {component_id}")
                metadata = strategy_func._component_info.metadata
            elif hasattr(strategy_func, '_strategy_metadata'):
                metadata = strategy_func._strategy_metadata
                logger.debug(f"Found _strategy_metadata for {component_id}")
            else:
                metadata = None
                
            if metadata:
                required_features = []
                
                logger.debug(f"Strategy {component_id} metadata keys: {list(metadata.keys())}")
                
                # Check for feature_specs from compiler
                if metadata.get('feature_specs'):
                    logger.debug(f"Found feature_specs: {len(metadata['feature_specs'])} specs")
                    for spec in metadata['feature_specs']:
                        if hasattr(spec, 'canonical_name'):
                            required_features.append(spec.canonical_name)
                            logger.debug(f"  - {spec.canonical_name}")
                        else:
                            logger.debug(f"  - spec without canonical_name: {spec}")
                    if required_features:
                        logger.debug(f"Extracted features from feature_specs: {required_features}")
                        return required_features
                
                # Check if strategy has new-style feature_discovery (all migrated strategies)
                elif metadata.get('feature_discovery'):
                    logger.debug(f"Strategy {component_id} has feature_discovery, using it for feature extraction")
                    discovery_func = metadata['feature_discovery']
                    try:
                        # Call the discovery function with parameters to get FeatureSpec objects
                        feature_specs = discovery_func(params)
                        # Convert FeatureSpec objects to canonical names
                        for spec in feature_specs:
                            required_features.append(spec.canonical_name)
                            logger.debug(f"Discovered feature: {spec.canonical_name}")
                    except Exception as e:
                        logger.error(f"Feature discovery failed for {component_id}: {e}")
                
                # Check if strategy has static required_features
                elif metadata.get('required_features'):
                    logger.debug(f"Strategy {component_id} has static required_features")
                    for spec in metadata['required_features']:
                        required_features.append(spec.canonical_name)
                        logger.debug(f"Required feature: {spec.canonical_name}")
                
                # All strategies must use new-style feature_discovery or required_features
                else:
                    logger.error(f"Strategy {component_id} does not use feature_discovery or required_features. Must be migrated.")
                    return []
                
                if required_features:
                    if is_target_strategy:
                        logger.debug(f"Extracted features from TARGET STRATEGY {component_id} metadata: {required_features}")
                    else:
                        logger.debug(f"Extracted features from {component_id} metadata: {required_features}")
                    return required_features
            else:
                logger.debug(f"No _strategy_metadata found for {component_id}")
        
        # No fallback - all strategies must be migrated
        logger.error(f"Strategy {component_id} has no feature discovery metadata. It must be migrated to use feature_discovery or required_features.")
        return []
    
    def _get_classifier_required_features(self, component_id: str, params: Dict[str, Any]) -> List[str]:
        """Get the specific features this classifier needs by extracting from function metadata."""
        logger.debug(f"Getting required features for classifier {component_id} with params: {params}")
        
        # First, try to get the classifier function and extract its feature config
        component_info = self._components.get(component_id)
        if component_info and 'function' in component_info:
            classifier_func = component_info['function']
            
            # Extract feature config from the classifier function's metadata
            if hasattr(classifier_func, '_component_info'):
                metadata = classifier_func._component_info.metadata
                feature_config = metadata.get('feature_config', [])
                # Check for new-style feature_discovery or required_features
                feature_discovery = metadata.get('feature_discovery')
                required_features_specs = metadata.get('required_features', [])
                
                logger.debug(f"Found classifier metadata for {component_id}: has_discovery={bool(feature_discovery)}, static_features={len(required_features_specs)}")
                
                required_features = []
                
                # Handle new-style feature_discovery (dynamic)
                if feature_discovery:
                    try:
                        # Call the discovery function with parameters to get FeatureSpec objects
                        feature_specs = feature_discovery(params)
                        # Convert FeatureSpec objects to canonical names
                        for spec in feature_specs:
                            required_features.append(spec.canonical_name)
                            logger.debug(f"Discovered feature: {spec.canonical_name}")
                    except Exception as e:
                        logger.error(f"Feature discovery failed for classifier {component_id}: {e}")
                
                # Handle static required_features
                elif required_features_specs:
                    for spec in required_features_specs:
                        required_features.append(spec.canonical_name)
                        logger.debug(f"Required feature: {spec.canonical_name}")
                
                if required_features:
                    logger.debug(f"Classifier {component_id} requires features: {required_features}")
                    return required_features
        
        # Fallback to old hardcoded approach if metadata not found
        logger.warning(f"No metadata found for classifier {component_id}, using fallback")
        if 'enhanced_trend' in component_id or 'multi_timeframe_trend' in component_id:
            # Use parameterized features based on params
            sma_short = params.get('sma_short', 10)
            sma_medium = params.get('sma_medium', 20)
            sma_long = params.get('sma_long', 50)
            return [f'sma_{sma_short}', f'sma_{sma_medium}', f'sma_{sma_long}', 'close']
        elif 'market_regime' in component_id:
            sma_short = params.get('sma_short', 10)
            sma_long = params.get('sma_long', 50)
            atr_period = params.get('atr_period', 20)
            rsi_period = params.get('rsi_period', 14)
            return [f'sma_{sma_short}', f'sma_{sma_long}', f'atr_{atr_period}', f'rsi_{rsi_period}', 'close']
        elif 'hidden_markov' in component_id:
            rsi_period = params.get('rsi_period', 14)
            sma_short = params.get('sma_short', 20)
            sma_long = params.get('sma_long', 50)
            atr_period = params.get('atr_period', 14)
            return ['volume', f'rsi_{rsi_period}', f'sma_{sma_short}', f'sma_{sma_long}', f'atr_{atr_period}', 'close']
        elif 'volatility_momentum' in component_id:
            atr_period = params.get('atr_period', 14)
            rsi_period = params.get('rsi_period', 14)
            sma_period = params.get('sma_period', 20)
            return [f'atr_{atr_period}', f'rsi_{rsi_period}', f'sma_{sma_period}', 'close']
        elif 'microstructure' in component_id:
            sma_fast = params.get('sma_fast', 5)
            sma_slow = params.get('sma_slow', 20)
            atr_period = params.get('atr_period', 10)
            rsi_period = params.get('rsi_period', 7)
            return [f'sma_{sma_fast}', f'sma_{sma_slow}', f'atr_{atr_period}', f'rsi_{rsi_period}', 'close']
        else:
            return ['sma_20', 'rsi_14']  # Basic features
    
    def _get_classifier_min_bars(self, component_id: str, params: Dict[str, Any], required_features: List[str]) -> int:
        """Get minimum bars needed for this classifier based on its features."""
        min_bars = 20  # Default for basic classifiers
        
        # Extract maximum period from required features
        for feature in required_features:
            parts = feature.split('_')
            if len(parts) >= 2:
                try:
                    period = int(parts[1])
                    min_bars = max(min_bars, period + 5)  # Feature period + buffer
                except ValueError:
                    continue
        
        return min_bars
    
    def _get_strategy_min_bars(self, component_id: str, params: Dict[str, Any]) -> int:
        """Get minimum bars needed for this specific strategy."""
        # Try to get from required features first
        required_features = self._get_strategy_required_features(component_id, params)
        if required_features:
            max_period = 0
            for feature in required_features:
                # Extract period from feature name (e.g., 'rsi_14' -> 14, 'bollinger_bands_20_2.0_upper' -> 20)
                parts = feature.split('_')
                # Try each part as a potential period
                for part in parts[1:]:  # Skip the feature type prefix
                    try:
                        # Handle float periods by taking the integer part
                        if '.' in part:
                            period = int(float(part))
                        else:
                            period = int(part)
                        # Reasonable period range check (ignore very large numbers that might be IDs)
                        if 1 <= period <= 500:
                            max_period = max(max_period, period)
                            break  # Found a valid period, no need to check more parts
                    except ValueError:
                        continue
            if max_period > 0:
                return max_period + 5  # Add buffer
        
        # Debug logging
        if 'compiled' in component_id and max_period == 0:
            logger.debug(f"Compiled strategy {component_id} min_bars calculation: required_features={required_features}, max_period={max_period}")
        
        # Fallback to pattern matching
        if 'ma_crossover' in component_id:
            fast_period = params.get('fast_period', 10)
            slow_period = params.get('slow_period', 20)
            return max(fast_period, slow_period) + 5
        elif 'rsi_strategy' in component_id or 'rsi_grid' in component_id:
            rsi_period = params.get('rsi_period', 14)
            return rsi_period + 5
        elif 'macd_strategy' in component_id or 'macd_grid' in component_id:
            slow_ema = params.get('slow_ema', 26)
            signal_ema = params.get('signal_ema', 9)
            return slow_ema + signal_ema + 5
        elif 'breakout_strategy' in component_id or 'breakout_grid' in component_id:
            lookback_period = params.get('lookback_period', 20)
            return lookback_period + 14 + 5  # lookback + ATR period + buffer
        else:
            return 30  # Conservative default
    
    def _get_component_required_features(self, component_id: str, component_info: Dict[str, Any]) -> List[str]:
        """Get required features for any component type."""
        component_type = component_info['component_type']
        params = component_info['parameters']
        
        if component_type == 'classifier':
            return self._get_classifier_required_features(component_id, params)
        else:
            return self._get_strategy_required_features(component_id, params)
    
    def _execute_components(self, symbol: str, features: Dict[str, Any], 
                          bar: Dict[str, float], timestamp: datetime) -> None:
        """Execute all components with current features.
        
        Execution order:
        1. Classifiers first - to determine market regime
        2. Strategies second - can use classification in features
        """
        # Create a snapshot of components to avoid dictionary modification errors
        components_snapshot = list(self._components.items())
        logger.debug(f"Executing {len(components_snapshot)} components for {symbol}")
        
        # Track outputs to update after iteration
        outputs_to_update = {}
        
        # First pass: Execute classifiers and update features with classification
        current_classifications = {}
        for component_id, component_info in components_snapshot:
            if component_info['component_type'] != 'classifier':
                continue
                
            try:
                # Classifiers expect only features and params
                result = component_info['function'](
                    features=features,
                    params=component_info['parameters']
                )
                
                # Process result and potentially publish CLASSIFICATION event
                if result:
                    outputs_to_update[component_id] = result
                    # Pass the component_info from snapshot to avoid dictionary access
                    self._process_component_output(
                        component_id=component_id,
                        component_type=component_info['component_type'],
                        result=result,
                        symbol=symbol,
                        timestamp=timestamp,
                        component_info=component_info,
                        bar=bar,
                        features=features
                    )
                    
                    # Store classification result for strategies to use
                    if result.get('regime'):
                        classifier_name = component_id.split('_', 1)[1] if '_' in component_id else component_id
                        current_classifications[f'regime_{classifier_name}'] = result['regime']
                        current_classifications[f'regime_confidence_{classifier_name}'] = result.get('confidence', 0.0)
                        
            except Exception as e:
                logger.error(f"Error executing classifier {component_id}: {e}")
        
        # Add current classifications to features for strategies to use
        if current_classifications:
            features = features.copy()  # Don't modify original features
            features.update(current_classifications)
            logger.debug(f"Added classifications to features: {list(current_classifications.keys())}")
        
        # Second pass: Execute strategies and other components with classification-aware features
        for component_id, component_info in components_snapshot:
            if component_info['component_type'] == 'classifier':
                continue  # Already processed
                
            try:
                # Strategies and other components expect features, bar, and params
                result = component_info['function'](
                    features=features,
                    bar=bar,
                    params=component_info['parameters']
                )
                
                # Process result and convert to SIGNAL event format
                if result:
                    outputs_to_update[component_id] = result
                    # Pass the component_info from snapshot to avoid dictionary access
                    self._process_component_output(
                        component_id=component_id,
                        component_type=component_info['component_type'],
                        result=result,
                        symbol=symbol,
                        timestamp=timestamp,
                        component_info=component_info,
                        bar=bar,
                        features=features
                    )
                    
            except Exception as e:
                logger.error(f"Error executing component {component_id}: {e}")
        
        # Update all outputs after iteration is complete
        for component_id, result in outputs_to_update.items():
            self._components[component_id]['last_output'] = result
    
    def _process_component_output(self, component_id: str, component_type: str,
                                result: Dict[str, Any], symbol: str, timestamp: datetime,
                                component_info: Optional[Dict[str, Any]] = None, bar: Optional[Dict[str, Any]] = None,
                                features: Optional[Dict[str, Any]] = None) -> None:
        """Process component output and publish as SIGNAL event."""
        
        # Get previous output for comparison
        # Use component_info if provided (from snapshot), otherwise access the dict
        if component_info:
            previous_output = component_info.get('last_output')
        else:
            previous_output = self._components[component_id].get('last_output')
        
        # Don't update the dictionary during iteration - caller will handle it
        
        # Convert component output to SIGNAL event format
        if component_type == "strategy":
            # Handle both signal_type format and signal_value format
            if result.get('signal_type'):
                # Original format with signal_type
                # Extract strategy type from metadata for tracing
                comp_metadata = {}
                if component_info and 'metadata' in component_info:
                    comp_metadata = component_info['metadata']
                else:
                    # Fallback to getting from stored components
                    component_data = self._components.get(component_id, {})
                    comp_metadata = component_data.get('metadata', {})
                
                strategy_type = comp_metadata.get('strategy_type', '')
                
                signal = Signal(
                    symbol=symbol,
                    direction=result.get('direction', SignalDirection.FLAT),
                    strength=result.get('strength', 1.0),
                    timestamp=timestamp,
                    strategy_id=component_id,
                    signal_type=SignalType(result.get('signal_type', 'entry')),
                    metadata={
                        **result.get('metadata', {}),
                        'component_type': component_type,
                        'price': result.get('price', bar.get('close', 0) if bar else 0)  # Add price from result or bar
                    }
                )
                self._publish_signal(signal, strategy_type=strategy_type)
            elif 'signal_value' in result:
                # Indicator strategy format with signal_value (-1, 0, 1)
                signal_value = result.get('signal_value', 0)
                
                # Debug logging for compiled strategies with signals
                if 'compiled_strategy' in component_id and signal_value != 0:
                    logger.debug(f"[STATE] Processing {component_id} signal_value={signal_value}")
                
                # Debug: Log component info for first few signals
                if signal_value != 0 and component_id.endswith(('_0', '_1', '_2')):
                    logger.debug(f"[SIGNAL_DEBUG] Processing signal from {component_id}")
                    logger.debug(f"[SIGNAL_DEBUG] component_info available: {component_info is not None}")
                    if component_info:
                        logger.debug(f"[SIGNAL_DEBUG] component_info keys: {list(component_info.keys())}")
                        logger.debug(f"[SIGNAL_DEBUG] component_info['parameters']: {component_info.get('parameters', 'NOT FOUND')}")
                
                # Convert signal_value to direction
                if signal_value > 0:
                    direction = SignalDirection.LONG
                elif signal_value < 0:
                    direction = SignalDirection.SHORT
                else:
                    direction = SignalDirection.FLAT
                
                # Log ALL signals including FLAT for debugging (commented out for production)
                # if 'bollinger' in component_id:
                #     logger.info(f"[SIGNAL_DEBUG] {component_id}: signal_value={signal_value}, direction={direction}")
                
                # Apply signal filter if configured
                should_publish = True
                
                # Check if we have a filter for this component
                if component_id in self._component_filters:
                    filter_obj = self._component_filters[component_id]
                    
                    # Use passed features or empty dict if not provided
                    filter_features = features if features else {}
                    
                    # Evaluate filter
                    filter_params = component_info.get('filter_params', {}) if component_info else {}
                    should_publish = filter_obj.evaluate_filter(
                        signal=result,
                        features=filter_features,
                        bar=bar,
                        filter_params=filter_params
                    )
                    
                    if not should_publish:
                        logger.debug(f"Signal from {component_id} rejected by filter: signal_value={signal_value}")
                
                # Publish signal if it passes filter (or no filter configured)
                # TEMPORARILY: Always publish to ensure all strategies appear in index
                if should_publish or True:  # Force publish all signals
                    # Debug logging commented out for production
                    # if direction != SignalDirection.FLAT:
                    #     logger.info(f"📡 Publishing NON-FLAT signal from {component_id}: direction={direction}, value={signal_value}")
                    
                    # Handle timestamp - strategy returns None, use the bar timestamp
                    signal_timestamp = result.get('timestamp')
                    if signal_timestamp is None:
                        signal_timestamp = timestamp
                    
                    # Extract strategy type from metadata or result
                    # First check if metadata was passed in component_info
                    comp_metadata = {}
                    component_data = {}
                    if component_info and 'metadata' in component_info:
                        comp_metadata = component_info['metadata']
                    else:
                        # Fallback to getting from stored components
                        component_data = self._components.get(component_id, {})
                        comp_metadata = component_data.get('metadata', {})
                    
                    strategy_type = comp_metadata.get('strategy_type', '')
                    strategy_hash = comp_metadata.get('strategy_hash', '')  # Get hash from metadata
                    
                    if strategy_type:
                        logger.debug(f"Got strategy_type from metadata: {strategy_type}")
                    if strategy_hash:
                        logger.debug(f"Got strategy_hash from metadata: {strategy_hash}")
                    
                    # Fallback to result if not in metadata
                    if not strategy_type:
                        strategy_type = result.get('strategy_id', '')
                        if strategy_type:
                            logger.debug(f"Got strategy_type from result: {strategy_type}")
                    
                    # If still not found, extract from component_id
                    if not strategy_type and '_' in component_id:
                        # For compiled strategies, component_id is like "SPY_compiled_strategy_0"
                        # but we want the actual strategy type from metadata
                        strategy_type = 'unknown'
                        logger.warning(f"No strategy_type found for {component_id}, using 'unknown'")
                    
                    # Get parameters from component info if available, otherwise from stored component
                    if component_info:
                        strategy_params = component_info.get('parameters', {})
                    else:
                        strategy_params = component_data.get('parameters', {})
                    
                    # Extract risk parameters from strategy_params if present
                    risk_params = None
                    if '_risk' in strategy_params:
                        risk_params = strategy_params['_risk']
                        logger.info(f"Found risk parameters in strategy: {risk_params}")
                    
                    # Debug logging for parameter extraction
                    if not strategy_params or strategy_params == {'_strategy_type': strategy_type}:
                        logger.warning(f"[SIGNAL] Limited parameters for {component_id}: {strategy_params}")
                        logger.debug(f"[SIGNAL] component_info keys: {list(component_info.keys()) if component_info else 'None'}")
                        logger.debug(f"[SIGNAL] component_data keys: {list(component_data.keys()) if component_data else 'None'}")
                        # Log the full component_data to understand what's available
                        if component_data:
                            logger.debug(f"[SIGNAL] Full component_data: {component_data}")
                    else:
                        # Log successful parameter extraction (temporarily)
                        logger.debug(f"[SIGNAL] Got parameters for {component_id}: {strategy_params}")
                    
                    signal_metadata = {
                        **result.get('metadata', {}),
                        'component_type': component_type,
                        'signal_value': signal_value,
                        'symbol_timeframe': result.get('symbol_timeframe', f"{symbol}_1m"),
                        'base_strategy_id': strategy_type,  # Clean strategy type
                        'price': result.get('metadata', {}).get('price', bar.get('close', 0) if bar else 0),
                        'parameters': strategy_params,
                        'timeframe': self._timeframe or '1m'
                    }
                    
                    # Add risk parameters to metadata if present
                    if risk_params:
                        signal_metadata['risk'] = risk_params
                    
                    # Add strategy hash if available
                    if strategy_hash:
                        signal_metadata['strategy_hash'] = strategy_hash
                    
                    # For ensemble strategies, include composite strategy info
                    if comp_metadata.get('composite_strategies'):
                        signal_metadata['composite_strategies'] = comp_metadata['composite_strategies']
                        logger.debug(f"Including composite_strategies in signal metadata: {len(comp_metadata['composite_strategies'])} sub-strategies")
                    
                    signal = Signal(
                        symbol=symbol,
                        direction=direction,
                        strength=abs(signal_value),  # Use absolute value as strength
                        timestamp=signal_timestamp,
                        strategy_id=component_id,  # Keep for backward compatibility
                        signal_type=SignalType.ENTRY,  # Indicator signals are continuous entry signals
                        metadata=signal_metadata
                    )
                    
                    # Add strategy_type AND parameters to the signal value for MultiStrategyTracer
                    signal.value = {
                        'strategy_type': strategy_type,
                        **strategy_params  # Include all strategy parameters
                    }
                    # Debug log for parameter propagation
                    if strategy_params and len(strategy_params) > 1:  # More than just _strategy_type
                        logger.debug(f"[SIGNAL] Including parameters in signal.value: {list(strategy_params.keys())}")
                    self._publish_signal(signal, strategy_type=strategy_type)
                    if 'compiled_strategy' in component_id:
                        logger.debug(f"[STATE] Published signal for {component_id}: direction={direction}, signal_value={signal_value}")
                    else:
                        logger.debug(f"Published signal for {component_id}: direction={direction}, signal_value={signal_value}")
                else:
                    if direction == SignalDirection.FLAT:
                        logger.debug(f"Not publishing flat signal for {component_id}: signal_value={signal_value}")
                    elif not should_publish:
                        logger.debug(f"Signal rejected by filter for {component_id}: signal_value={signal_value}")
                
        elif component_type == "classifier":
            # Classifier outputs - only publish on classification change
            regime = result.get('regime')
            confidence = result.get('confidence', 0.0)
            
            logger.debug(f"🔍 Classifier {component_id} output: regime={regime}, confidence={confidence:.3f}, previous_output={previous_output}")
            
            if regime:
                # Check if classification changed
                previous_regime = previous_output.get('regime') if previous_output else None
                
                logger.debug(f"🔍 Checking regime change: previous={previous_regime}, current={regime}")
                
                if previous_regime != regime:
                    # Classification changed - publish CLASSIFICATION event
                    # Get actual bar index for this symbol
                    symbol_from_id = component_id.split('_')[0] if '_' in component_id else symbol
                    actual_bar_index = self._current_bar_indices.get(symbol_from_id, {}).get('original', 0)
                    
                    logger.debug(f"📊 Classifier {component_id} regime CHANGED at bar {actual_bar_index}: {previous_regime} → {regime} (confidence: {confidence:.3f})")
                    
                    # Create Classification object without full features
                    classification = Classification(
                        symbol=symbol,
                        regime=regime,
                        confidence=confidence,
                        timestamp=timestamp,
                        classifier_id=component_id,
                        previous_regime=previous_regime,
                        features={},  # Empty features dict to reduce payload size
                        metadata={
                            **result.get('metadata', {}),
                            'bar_index': actual_bar_index,  # Add actual bar index
                            'original_bar_index': actual_bar_index
                        }
                    )
                    
                    self._publish_classification(classification)
                else:
                    # Same classification - don't publish (sparse storage)
                    logger.debug(f"🔍 Classifier {component_id} regime UNCHANGED: {regime} (confidence: {confidence:.3f})")
        
        else:
            # Future component types - generic handling
            logger.debug(f"Unsupported component type: {component_type}")
    
    def _publish_signal(self, signal: Signal, strategy_type: Optional[str] = None) -> None:
        """Publish signal event to parent container."""
        if not self._container:
            logger.warning("No container set, cannot publish signal")
            return
        
        # Create payload from signal
        payload = signal.to_dict()
        
        # Add strategy_type at the top level of payload if provided
        if strategy_type:
            payload['strategy_type'] = strategy_type
            logger.debug(f"Added strategy_type to payload: {strategy_type}")
        
        # Extract parameters from metadata and add to top level for MultiStrategyTracer
        if 'metadata' in payload and 'parameters' in payload['metadata']:
            payload['parameters'] = payload['metadata']['parameters']
            logger.debug(f"Extracted parameters to top level: {payload['parameters']}")
            
            # Extract risk parameters if present
            if '_risk' in payload['parameters']:
                payload['risk'] = payload['parameters']['_risk']
                logger.info(f"Added risk parameters to signal payload: {payload['risk']}")
        
        # Extract strategy_hash from metadata if available
        if 'metadata' in payload and 'strategy_hash' in payload['metadata']:
            payload['strategy_hash'] = payload['metadata']['strategy_hash']
            logger.debug(f"Extracted strategy_hash to top level: {payload['strategy_hash']}")
        
        # Add full strategy configuration if available
        if signal.strategy_id in self._components:
            component_info = self._components[signal.strategy_id]
            # Build complete strategy config
            strategy_config = {
                'type': strategy_type or component_info.get('component_type', 'unknown'),
                'parameters': component_info.get('parameters', {})
            }
            # Add constraints if present in filter config
            if signal.strategy_id in self._component_filters:
                # Get the filter expression from the component
                filter_config = getattr(self._component_filters[signal.strategy_id], 'filter_expr', None)
                if filter_config:
                    strategy_config['constraints'] = filter_config
            
            payload['strategy_config'] = strategy_config
            logger.debug(f"Added full strategy config to payload")
        
        # Create SIGNAL event
        signal_event = Event(
            event_type=EventType.SIGNAL.value,
            timestamp=signal.timestamp,
            payload=payload,
            source_id=self.name,
            container_id=self._container.container_id
        )
        
        logger.debug(f"📡 Publishing SIGNAL event: {signal_event.event_type} with payload: {signal_event.payload}")
        
        # Console output for signal visibility (removed spam)
        if hasattr(signal.direction, 'name'):
            signal_type = signal.direction.name
        else:
            signal_type = str(signal.direction).upper()
        if self._verbose_signals:
            logger.debug(f"📡 SIGNAL: {signal.strategy_id} → {signal_type} @ {signal.timestamp}")
        
        # Publish to parent for cross-container visibility
        self._container.publish_event(signal_event, target_scope="parent")
        self._outputs_generated += 1
        
        logger.debug(f"Published {signal.signal_type.value} signal: "
                    f"{signal.direction} {signal.symbol} from {signal.strategy_id}")
    
    def _publish_classification(self, classification: Classification) -> None:
        """Publish classification event to parent container."""
        if not self._container:
            logger.warning("No container set, cannot publish classification")
            return
        
        # Create CLASSIFICATION event
        classification_event = create_classification_event(
            classification=classification,
            source_id=self.name,
            container_id=self._container.container_id
        )
        
        logger.debug(f"👑 Publishing CLASSIFICATION event: {classification.classifier_id} → {classification.regime} (from {classification.previous_regime})")
        logger.debug(f"Full payload: {classification_event.payload}")
        
        # Console output for classifier visibility  
        if self._verbose_signals:
            logger.debug(f"👑 REGIME: {classification.classifier_id} → {classification.regime} @ {classification.timestamp}")
        
        # Publish to parent for cross-container visibility
        self._container.publish_event(classification_event, target_scope="parent")
        self._outputs_generated += 1
        
        logger.debug(f"Published classification: "
                    f"{classification.regime} for {classification.symbol} from {classification.classifier_id}")
    
    
    # Status and metrics
    def get_metrics(self) -> Dict[str, Any]:
        """Get state metrics."""
        return {
            'bars_processed': self._bars_processed,
            'outputs_generated': self._outputs_generated,
            'active_components': len(self._components),
            'last_update': self._last_update,
            'feature_summary': {
                "symbols": len(self._symbols),
                "configured_features": len(self._feature_configs),
                "bar_counts": dict(self._bar_count),
                "uses_feature_hub": self._feature_hub is not None
            },
            'component_summary': {
                component_id: {
                    'type': info['component_type'],
                    'has_output': info['last_output'] is not None
                }
                for component_id, info in self._components.items()
            }
        }
    
    # Legacy methods for backward compatibility with StrategyState
    def add_strategy(self, strategy_id: str, strategy_func: Any, 
                    parameters: Optional[Dict[str, Any]] = None) -> None:
        """Legacy method for backward compatibility."""
        self.add_component(strategy_id, strategy_func, "strategy", parameters)
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Legacy method for backward compatibility."""
        return self.remove_component(strategy_id)
    
    def get_strategy_ids(self) -> List[str]:
        """Legacy method for backward compatibility."""
        return [comp_id for comp_id, comp_info in self._components.items() 
                if comp_info['component_type'] == 'strategy']
    
    def get_classifier_ids(self) -> List[str]:
        """Get all classifier component IDs."""
        return [comp_id for comp_id, comp_info in self._components.items() 
                if comp_info['component_type'] == 'classifier']
    
    def get_last_signal(self, strategy_id: str) -> Optional[Signal]:
        """Legacy method for backward compatibility."""
        component = self._components.get(strategy_id)
        if component and component['component_type'] == 'strategy':
            return component.get('last_output')
        return None
    
    def _check_and_force_eod_closure(self, symbol: str, timestamp: datetime, 
                                   bar: Dict[str, Any], features: Dict[str, Any],
                                   ready_strategies: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Check if we need to force EOD closure and inject flat signals.
        
        This is called after all strategies have executed to ensure we close
        any open positions at the end of day.
        """
        # Extract time from timestamp
        hour = timestamp.hour
        minute = timestamp.minute
        time_hhmm = hour * 100 + minute
        
        # Check if we're at or after EOD cutoff (3:50 PM = 1550)
        eod_cutoff = 1550
        if time_hhmm >= eod_cutoff:
            # Check each strategy for open positions
            for component_id, component_info in ready_strategies:
                # Get the last output for this strategy
                last_output = component_info.get('last_output')
                if not last_output:
                    # Also check in the main components dict
                    if component_id in self._components:
                        last_output = self._components[component_id].get('last_output')
                
                # If strategy has a non-zero signal, force it to flat
                if last_output and last_output.get('signal_value', 0) != 0:
                    logger.info(f"[EOD] Forcing strategy {component_id} to flat at {timestamp} (time={time_hhmm})")
                    
                    # Create a flat signal
                    flat_result = {
                        'signal_value': 0,
                        'timestamp': timestamp,
                        'metadata': {
                            'forced_eod_closure': True,
                            'closure_time': time_hhmm,
                            'previous_signal': last_output.get('signal_value', 0)
                        }
                    }
                    
                    # Process this as a normal signal output
                    component_info_with_metadata = component_info.copy()
                    if component_id in self._components and 'metadata' in self._components[component_id]:
                        component_info_with_metadata['metadata'] = self._components[component_id]['metadata']
                    
                    self._process_component_output(
                        component_id=component_id,
                        component_type='strategy',
                        result=flat_result,
                        symbol=symbol,
                        timestamp=timestamp,
                        component_info=component_info_with_metadata,
                        bar=bar,
                        features=features
                    )
                    
                    # Update the last output to prevent repeated closures
                    if component_id in self._components:
                        self._components[component_id]['last_output'] = flat_result


# Legacy alias for backward compatibility
StrategyState = ComponentState