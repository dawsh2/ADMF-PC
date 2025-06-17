"""
Generic component state management - Replaces StrategyState with ComponentState.

Manages feature state and executes any stateless component functions (strategies, classifiers, etc.)
This is the canonical implementation that deprecates the old StrategyState.
"""

from typing import Dict, Any, Optional, List
from collections import defaultdict
import logging
from datetime import datetime

from .protocols import FeatureProvider
from .types import Signal, SignalType, SignalDirection
from .classification_types import Classification, create_classification_event
from ..core.events.types import Event, EventType

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
        # Each component has: function, parameters, component_type, last_output
        self._components: Dict[str, Dict[str, Any]] = {}
        
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
    
    def set_container(self, container) -> None:
        """Set container reference and subscribe to events."""
        self._container = container
        
        # Check if FeatureHub reference is provided
        feature_hub_name = container.config.config.get('feature_hub_name')
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
        else:
            # FeatureHub not connected yet
            features = {}
        
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
                     parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a component function to execute.
        
        Args:
            component_id: Unique identifier
            component_func: Pure component function (stateless)
            component_type: Type of component (strategy, classifier, etc.)
            parameters: Component parameters
        """
        self._components[component_id] = {
            'function': component_func,
            'component_type': component_type,
            'parameters': parameters or {},
            'last_output': None
        }
        logger.debug(f"Added {component_type}: {component_id}")
    
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
            else:
                logger.info(f"  {comp_type}: {components}")
        
        # Handle legacy strategies (backwards compatibility)
        if strategies:
            self._load_strategies_from_config(strategies, stateless_components, container)
        
        # Handle legacy classifiers (backwards compatibility)
        if classifiers:
            self._load_classifiers_from_config(classifiers, stateless_components, container)
        
        # Handle new components configuration
        if components_config:
            self._load_components_from_new_config(components_config, stateless_components, container)
        
        if not strategies and not classifiers and not components_config:
            logger.debug("No components configured")
    
    def _load_strategies_from_config(self, strategies: List[Dict[str, Any]], 
                                   stateless_components: Dict[str, Any],
                                   container) -> None:
        """Load strategies (backwards compatibility)."""
        strategy_components = stateless_components.get('strategies', {})
        
        logger.info(f"STRATEGY LOADING: Found {len(strategies)} strategy configs")
        logger.info(f"STRATEGY LOADING: Found {len(strategy_components)} strategy functions from topology")
        
        if strategy_components:
            logger.debug(f"Using {len(strategy_components)} strategies from topology")
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
                
                logger.info(f"Loading strategy {strategy_name} with params: {params_with_type}")
                logger.info(f"Strategy function metadata: {getattr(strategy_func, '_strategy_metadata', 'No metadata')}")
                if 'ensemble' in strategy_name.lower():
                    logger.info(f"ENSEMBLE STRATEGY LOADED: {component_id}")
                
                self.add_component(
                    component_id=component_id,
                    component_func=strategy_func,
                    component_type="strategy",
                    parameters=params_with_type
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
        return list(self._components.keys())
    
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
            'symbol': symbol
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
            logger.debug(f"Bar {current_bars}: Total strategies: {total_strategies}")
            
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
                    self._process_component_output(
                        component_id=component_id,
                        component_type=component_info['component_type'],
                        result=result,
                        symbol=symbol,
                        timestamp=timestamp,
                        component_info=component_info,
                        bar=bar
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
        for component_id, component_info in ready_strategies:
            try:
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
                
                if result:
                    outputs_to_update[component_id] = result
                    self._process_component_output(
                        component_id=component_id,
                        component_type=component_info['component_type'],
                        result=result,
                        symbol=symbol,
                        timestamp=timestamp,
                        component_info=component_info,
                        bar=bar
                    )
                    
            except Exception as e:
                logger.error(f"Error executing strategy {component_id}: {e}", exc_info=True)
        
        # Update all outputs after iteration is complete
        for component_id, result in outputs_to_update.items():
            if component_id in self._components:  # Check if still exists
                self._components[component_id]['last_output'] = result
    
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
        
        if missing_features and current_bars == 20:  # Log at bar 20 for debugging
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
            if hasattr(strategy_func, '_strategy_metadata'):
                metadata = strategy_func._strategy_metadata
                feature_config = metadata.get('feature_config', {})
                param_feature_mapping = metadata.get('param_feature_mapping', {})
                logger.debug(f"Found strategy metadata for {component_id}: {feature_config}")
                
                required_features = []
                
                # Check if strategy has param_feature_mapping (for ensemble strategies)
                if param_feature_mapping:
                    logger.debug(f"Strategy {component_id} has param_feature_mapping, using it for feature extraction")
                    # Use the explicit parameter-to-feature mapping
                    processed_templates = set()
                    for param_name, feature_template in param_feature_mapping.items():
                        if param_name in params and feature_template not in processed_templates:
                            # For templates with multiple parameters, we need all parameters to be available
                            try:
                                # Try to format the template with all available parameters
                                feature_name = feature_template.format(**params)
                                required_features.append(feature_name)
                                processed_templates.add(feature_template)
                                logger.debug(f"Generated feature {feature_name} from template {feature_template}")
                            except KeyError as e:
                                # Template needs parameters we don't have - skip it
                                logger.debug(f"Skipping template {feature_template} - missing parameter: {e}")
                                continue
                    
                    # Also add any base features from feature_config that don't have parameters
                    if isinstance(feature_config, list):
                        for feature_name in feature_config:
                            if feature_name in ['close', 'open', 'high', 'low', 'volume']:
                                required_features.append(feature_name)
                                
                # Handle both old dict format and new list format (fallback)
                elif isinstance(feature_config, list):
                    # New simplified format: ['sma', 'rsi', 'bollinger_bands']
                    for feature_name in feature_config:
                        # Try to infer required features from strategy parameters
                        if feature_name == 'sma':
                            # For SMA, check for period parameters
                            if 'fast_period' in params and 'slow_period' in params:
                                required_features.append(f"sma_{params['fast_period']}")
                                required_features.append(f"sma_{params['slow_period']}")
                            elif 'sma_period' in params:
                                required_features.append(f"sma_{params['sma_period']}")
                            elif 'period' in params:
                                required_features.append(f"sma_{params['period']}")
                        elif feature_name == 'ema':
                            # For EMA, check for period parameters
                            if 'fast_ema_period' in params and 'slow_ema_period' in params:
                                required_features.append(f"ema_{params['fast_ema_period']}")
                                required_features.append(f"ema_{params['slow_ema_period']}")
                            elif 'ema_period' in params:
                                required_features.append(f"ema_{params['ema_period']}")
                        elif feature_name == 'rsi':
                            rsi_period = params.get('rsi_period', 14)
                            required_features.append(f"rsi_{rsi_period}")
                        elif feature_name == 'bollinger_bands':
                            period = params.get('period', 20)
                            std_dev = params.get('std_dev', 2.0)
                            required_features.extend([
                                f"bollinger_bands_{period}_{std_dev}_upper",
                                f"bollinger_bands_{period}_{std_dev}_lower",
                                f"bollinger_bands_{period}_{std_dev}_middle"
                            ])
                        else:
                            # For other features, use generic parameter inference
                            period = params.get('period') or params.get(f'{feature_name}_period')
                            if period:
                                required_features.append(f"{feature_name}_{period}")
                            else:
                                required_features.append(feature_name)
                    
                elif isinstance(feature_config, dict):
                    # Old complex format: {'sma': {'params': [...], 'defaults': {...}}}
                    for feature_base, config in feature_config.items():
                        param_names = config.get('params', [])
                        defaults = config.get('defaults', {})
                        
                        if param_names:
                            # For MA crossover: sma with ['fast_period', 'slow_period'] should generate sma_5 and sma_100 
                            if feature_base == 'sma' and len(param_names) == 2 and 'fast_period' in param_names and 'slow_period' in param_names:
                                # Special case for MA crossover - create separate SMA features
                                fast_period = params.get('fast_period') or defaults.get('fast_period')
                                slow_period = params.get('slow_period') or defaults.get('slow_period')
                                if fast_period is not None:
                                    required_features.append(f"{feature_base}_{fast_period}")
                                if slow_period is not None:
                                    required_features.append(f"{feature_base}_{slow_period}")
                            else:
                                # Standard case: create separate features for each parameter
                                # For example, EMA crossover with fast/slow should create ema_3 and ema_15, not ema_3_15
                                for param_name in param_names:
                                    param_value = params.get(param_name) or defaults.get(param_name) or config.get('default')
                                    if param_value is not None:
                                        # Create individual feature for this parameter
                                        required_features.append(f'{feature_base}_{param_value}')
                        else:
                            # Simple feature without parameters
                            required_features.append(feature_base)
                
                if required_features:
                    if is_target_strategy:
                        logger.debug(f"Extracted features from TARGET STRATEGY {component_id} metadata: {required_features}")
                    else:
                        logger.debug(f"Extracted features from {component_id} metadata: {required_features}")
                    return required_features
            else:
                logger.debug(f"No _strategy_metadata found for {component_id}")
        
        # Fallback: try to infer from parameters and component type
        strategy_type = params.get('_strategy_type', '')
        if is_target_strategy:
            logger.info(f"FALLBACK: Using fallback for TARGET STRATEGY {component_id}, strategy_type: {strategy_type}, params: {params}")
        else:
            logger.debug(f"FALLBACK: Using fallback for {component_id}, strategy_type: {strategy_type}, params: {params}")
        
        # Handle specific strategy types
        if 'momentum' in component_id.lower() or strategy_type == 'simple_momentum':
            # Momentum strategies need SMA and RSI
            sma_period = params.get('sma_period', 20)
            rsi_period = params.get('rsi_period', 14)
            result = [f"sma_{sma_period}", f"rsi_{rsi_period}"]
            logger.debug(f"FALLBACK: Momentum strategy {component_id} requires: {result}")
            return result
        elif 'ma_crossover' in component_id.lower() or strategy_type == 'ma_crossover':
            # MA crossover needs fast and slow SMAs - use actual parameters
            fast_period = params.get('fast_period', 10)
            slow_period = params.get('slow_period', 20)
            result = [f"sma_{fast_period}", f"sma_{slow_period}"]
            logger.debug(f"FALLBACK: MA crossover strategy {component_id} using periods: fast={fast_period}, slow={slow_period} from params: {params}")
            logger.debug(f"FALLBACK: MA crossover strategy {component_id} requires: {result}")
            return result
        elif 'breakout' in component_id.lower() or strategy_type in ['breakout', 'breakout_strategy']:
            # Breakout strategies need high/low/volume/atr
            lookback_period = params.get('lookback_period', 20)
            atr_period = params.get('atr_period', 14)
            return [f"high_{lookback_period}", f"low_{lookback_period}", f"volume_{lookback_period}_volume_ma", f"atr_{atr_period}"]
        elif 'mean_reversion' in component_id.lower() or strategy_type in ['mean_reversion', 'mean_reversion_simple']:
            # Mean reversion needs RSI and Bollinger Bands
            rsi_period = params.get('rsi_period', 14)
            bb_period = params.get('bb_period', 20)
            return [f"rsi_{rsi_period}", f"bollinger_{bb_period}_upper", f"bollinger_{bb_period}_lower"]
        
        # Basic parameter-based inference for common patterns
        if 'rsi_period' in params:
            return [f"rsi_{params['rsi_period']}"]
        elif 'entry_rsi_period' in params:
            return [f"rsi_{params['entry_rsi_period']}"]
        elif 'fast_period' in params and 'slow_period' in params:
            return [f"sma_{params['fast_period']}", f"sma_{params['slow_period']}"]
        else:
            # Last resort - minimal default features
            logger.warning(f"Could not determine required features for {component_id} (type: {strategy_type}), using minimal defaults")
            return ['rsi_14']
    
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
                param_feature_mapping = metadata.get('param_feature_mapping', {})
                
                logger.debug(f"Found classifier metadata for {component_id}: feature_config={feature_config}, param_mapping={param_feature_mapping}")
                
                required_features = []
                
                # Handle new list format with param_feature_mapping
                if isinstance(feature_config, list) and param_feature_mapping:
                    # Process param_feature_mapping to generate actual feature names
                    for param_name, feature_template in param_feature_mapping.items():
                        param_value = params.get(param_name)
                        if param_value is not None:
                            # Substitute parameter value into template
                            feature_name = feature_template.format(**{param_name: param_value})
                            required_features.append(feature_name)
                            logger.debug(f"Generated feature {feature_name} from param {param_name}={param_value}")
                    
                    # Add base features that don't have parameter mappings
                    for feature_name in feature_config:
                        # Check if this feature is covered by param_feature_mapping
                        is_parameterized = any(
                            template.startswith(feature_name + '_') or template == feature_name
                            for template in param_feature_mapping.values()
                        )
                        if not is_parameterized:
                            # This is a base feature without parameters
                            required_features.append(feature_name)
                            logger.debug(f"Added base feature {feature_name}")
                
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
                        bar=bar
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
                        bar=bar
                    )
                    
            except Exception as e:
                logger.error(f"Error executing component {component_id}: {e}")
        
        # Update all outputs after iteration is complete
        for component_id, result in outputs_to_update.items():
            self._components[component_id]['last_output'] = result
    
    def _process_component_output(self, component_id: str, component_type: str,
                                result: Dict[str, Any], symbol: str, timestamp: datetime,
                                component_info: Optional[Dict[str, Any]] = None, bar: Optional[Dict[str, Any]] = None) -> None:
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
                self._publish_signal(signal)
            elif 'signal_value' in result:
                # Indicator strategy format with signal_value (-1, 0, 1)
                signal_value = result.get('signal_value', 0)
                
                # Convert signal_value to direction
                if signal_value > 0:
                    direction = SignalDirection.LONG
                elif signal_value < 0:
                    direction = SignalDirection.SHORT
                else:
                    direction = SignalDirection.FLAT
                
                # Publish all signals including flat ones for debugging
                # if direction != SignalDirection.FLAT:
                if True:  # Temporarily publish all signals
                    
                    # Handle timestamp - strategy returns None, use the bar timestamp
                    signal_timestamp = result.get('timestamp')
                    if signal_timestamp is None:
                        signal_timestamp = timestamp
                    
                    signal = Signal(
                        symbol=symbol,
                        direction=direction,
                        strength=abs(signal_value),  # Use absolute value as strength
                        timestamp=signal_timestamp,
                        strategy_id=component_id,  # Use component_id which includes the full expanded name
                        signal_type=SignalType.ENTRY,  # Indicator signals are continuous entry signals
                        metadata={
                            **result.get('metadata', {}),
                            'component_type': component_type,
                            'signal_value': signal_value,
                            'symbol_timeframe': result.get('symbol_timeframe', f"{symbol}_1m"),
                            'base_strategy_id': result.get('strategy_id', ''),  # Keep original strategy type for reference
                            'price': result.get('metadata', {}).get('price', bar.get('close', 0) if bar else 0)  # Ensure price is included
                        }
                    )
                    self._publish_signal(signal)
                    logger.debug(f"Published signal for {component_id}: direction={direction}, signal_value={signal_value}")
                else:
                    logger.debug(f"Not publishing flat signal for {component_id}: signal_value={signal_value}")
                
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
    
    def _publish_signal(self, signal: Signal) -> None:
        """Publish signal event to parent container."""
        if not self._container:
            logger.warning("No container set, cannot publish signal")
            return
        
        # Create SIGNAL event
        signal_event = Event(
            event_type=EventType.SIGNAL.value,
            timestamp=signal.timestamp,
            payload=signal.to_dict(),
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


# Legacy alias for backward compatibility
StrategyState = ComponentState