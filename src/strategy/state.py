"""
Generic component state management - Replaces StrategyState with ComponentState.

Manages feature state and executes any stateless component functions (strategies, classifiers, etc.)
This is the canonical implementation that deprecates the old StrategyState.
"""

from typing import Dict, Any, Optional, List, Deque, Union
from collections import deque, defaultdict
import logging
from datetime import datetime
import pandas as pd

from .protocols import FeatureProvider
from .types import Signal, SignalType, SignalDirection
from .classification_types import Classification, create_classification_event
from .components.features import compute_feature, FEATURE_REGISTRY
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
        feature_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize component state.
        
        Args:
            symbols: Initial symbols to track
            feature_configs: Feature configurations
        """
        self._symbols = symbols or []
        
        # Calculate max lookback needed from features
        self._max_lookback = self._calculate_max_lookback(feature_configs)
        
        # Price data storage - rolling windows per symbol
        # Size based on actual feature requirements + small buffer
        self._price_data: Dict[str, Dict[str, Deque[float]]] = defaultdict(lambda: {
            'open': deque(maxlen=self._max_lookback),
            'high': deque(maxlen=self._max_lookback),
            'low': deque(maxlen=self._max_lookback),
            'close': deque(maxlen=self._max_lookback),
            'volume': deque(maxlen=self._max_lookback)
        })
        
        # Computed features per symbol
        self._features: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Feature configurations
        self._feature_configs = feature_configs or {}
        
        # Bar count per symbol
        self._bar_count: Dict[str, int] = defaultdict(int)
        
        # Component registry - populated by container/topology
        # Each component has: function, parameters, component_type, last_output
        self._components: Dict[str, Dict[str, Any]] = {}
        
        # Container reference for event publishing
        self._container = None
        
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
        
        logger.info(f"ComponentState configured with max_lookback={max_lookback} based on features")
        return max_lookback
    
    def set_container(self, container) -> None:
        """Set container reference and subscribe to events."""
        self._container = container
        # Subscribe to BAR events
        logger.info(f"ComponentState subscribing to event_bus: {container.event_bus}")
        container.event_bus.subscribe(EventType.BAR.value, self.on_bar)
        logger.info(f"ComponentState subscribed to BAR events in container {container.name}")
        
        # Load components from container config
        self._load_components_from_config(container)
    
    # Implement FeatureProvider protocol directly
    def update_bar(self, symbol: str, bar: Dict[str, float]) -> None:
        """Update with new bar data and recalculate features."""
        # Store price data
        for field in ['open', 'high', 'low', 'close', 'volume']:
            if field in bar:
                self._price_data[symbol][field].append(bar[field])
        
        self._bar_count[symbol] += 1
        
        # Recalculate features if we have enough data
        if self._bar_count[symbol] >= 2:  # Need at least 2 bars for some features
            self._update_features(symbol)
    
    def get_features(self, symbol: str) -> Dict[str, Any]:
        """Get current features for symbol."""
        features = self._features.get(symbol, {}).copy()
        
        # Add state features that strategies may need
        features['bar_count'] = self._bar_count.get(symbol, 0)
        
        return features
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]]) -> None:
        """Configure which features to calculate."""
        self._feature_configs = feature_configs
        logger.info(f"Configured {len(feature_configs)} features")
    
    def has_sufficient_data(self, symbol: str, min_bars: int = None) -> bool:
        """Check if sufficient data available.
        
        The minimum bars required is determined by:
        1. Explicit min_bars parameter
        2. Maximum lookback period from all configured features
        3. Default of 50 bars for warmup
        """
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
            for field in self._price_data[symbol]:
                self._price_data[symbol][field].clear()
            self._features[symbol].clear()
            self._bar_count[symbol] = 0
        else:
            # Full reset
            self._price_data.clear()
            self._features.clear()
            self._bar_count.clear()
            for component in self._components.values():
                component['last_output'] = None
    
    def _update_features(self, symbol: str) -> None:
        """
        Update features for a symbol using current data.
        
        This converts the rolling window data to pandas and computes
        features using the stateless feature functions.
        """
        # Convert deques to pandas DataFrame for feature computation
        data_dict = {}
        for field, deque_data in self._price_data[symbol].items():
            if len(deque_data) > 0:
                data_dict[field] = list(deque_data)
        
        if not data_dict or len(data_dict['close']) < 2:
            logger.debug(f"Not enough data for features: {len(data_dict.get('close', []))} bars")
            return
        
        # Create DataFrame for feature computation
        df = pd.DataFrame(data_dict)
        logger.debug(f"Computing features with {len(df)} bars, feature configs: {len(self._feature_configs)}")
        
        # Compute all configured features
        symbol_features = {}
        
        for feature_name, config in self._feature_configs.items():
            try:
                feature_type = config.get('feature')
                
                # Check if this is a raw data feature that doesn't need calculation
                if config.get('is_raw_data', False) or feature_type in ['volume', 'close', 'open', 'high', 'low']:
                    # Raw data features come directly from price data
                    if feature_type in self._price_data[symbol]:
                        raw_data = self._price_data[symbol][feature_type]
                        if len(raw_data) > 0:
                            symbol_features[feature_name] = float(raw_data[-1])
                    continue
                
                if feature_type not in FEATURE_REGISTRY:
                    logger.warning("Unknown feature type: %s", feature_type)
                    continue
                
                # Create config copy without 'feature' key
                feature_params = {k: v for k, v in config.items() if k != 'feature' and k != 'is_raw_data'}
                
                # Compute feature using stateless function
                result = compute_feature(feature_type, df, **feature_params)
                
                # Store latest feature value(s)
                if isinstance(result, dict):
                    # Multi-value features (e.g., MACD, Bollinger Bands)
                    for sub_name, series in result.items():
                        if len(series) > 0 and not pd.isna(series.iloc[-1]):
                            symbol_features[f"{feature_name}_{sub_name}"] = float(series.iloc[-1])
                else:
                    # Single-value features (e.g., SMA, RSI)
                    if len(result) > 0 and not pd.isna(result.iloc[-1]):
                        symbol_features[feature_name] = float(result.iloc[-1])
                        
            except Exception as e:
                logger.error("Error computing feature %s for %s: %s", feature_name, symbol, e)
        
        # Also ensure close price is always available as a feature
        if 'close' not in symbol_features:
            close_data = self._price_data[symbol].get('close')
            if close_data and len(close_data) > 0:
                symbol_features['close'] = float(close_data[-1])
        
        # Add feature aliases for classifiers that expect generic names
        # Map specific feature names to generic ones
        if 'sma_10' in symbol_features and 'sma_fast' not in symbol_features:
            symbol_features['sma_fast'] = symbol_features['sma_10']
        if 'sma_20' in symbol_features and 'sma_slow' not in symbol_features:
            symbol_features['sma_slow'] = symbol_features['sma_20']
        if 'atr_14' in symbol_features and 'atr' not in symbol_features:
            symbol_features['atr'] = symbol_features['atr_14']
        if 'rsi_14' in symbol_features and 'rsi' not in symbol_features:
            symbol_features['rsi'] = symbol_features['rsi_14']
        if 'volatility_20' in symbol_features and 'volatility' not in symbol_features:
            symbol_features['volatility'] = symbol_features['volatility_20']
        if 'momentum_10' in symbol_features and 'momentum_momentum_10' not in symbol_features:
            symbol_features['momentum_momentum_10'] = symbol_features['momentum_10']
        
        # Create generic feature aliases for classifiers
        self._create_generic_feature_aliases(symbol, symbol_features)
        
        # Update cache
        self._features[symbol].update(symbol_features)
        
        logger.debug("Updated %d features for %s", len(symbol_features), symbol)
    
    def _create_generic_feature_aliases(self, symbol: str, features: Dict[str, Any]) -> None:
        """Create generic feature aliases that classifiers expect.
        
        This maps strategy-specific feature names to generic names that classifiers use.
        For example: 'ma_crossover_grid_5_20_1.0_sma_5' -> 'sma_fast'
        """
        # Store aliases to add after iteration
        aliases_to_add = {}
        
        # Find all SMA features and create sma_fast/sma_slow aliases
        sma_features = {}
        for name, value in features.items():
            if 'sma_' in name and not name.endswith(('_fast', '_slow')):
                # Extract period from various naming patterns
                try:
                    # Try pattern: xxx_sma_N
                    if '_sma_' in name:
                        period = int(name.split('_sma_')[1].split('_')[0])
                        sma_features[period] = value
                    # Try pattern: sma_N  
                    elif name.startswith('sma_') and name[4:].isdigit():
                        period = int(name[4:])
                        sma_features[period] = value
                except (ValueError, IndexError):
                    continue
        
        # Assign fast/slow based on periods
        if sma_features:
            sorted_periods = sorted(sma_features.keys())
            if len(sorted_periods) >= 1:
                # Smallest period is fast
                if 'sma_fast' not in features:
                    aliases_to_add['sma_fast'] = sma_features[sorted_periods[0]]
            if len(sorted_periods) >= 2:
                # Larger period is slow
                if 'sma_slow' not in features:
                    aliases_to_add['sma_slow'] = sma_features[sorted_periods[-1]]
        
        # Find ATR features and create generic alias
        for name, value in features.items():
            if 'atr_' in name and 'atr' not in features and not name.endswith('_sma'):
                aliases_to_add['atr'] = value
                break
        
        # Find RSI features and create generic alias
        for name, value in features.items():
            if 'rsi_' in name and 'rsi' not in features:
                aliases_to_add['rsi'] = value
                break
        
        # Find MACD features and create proper aliases
        for name, value in features.items():
            if 'macd_macd' in name and 'macd_macd' not in features:
                aliases_to_add['macd_macd'] = value
            elif name.endswith('_macd') and 'macd' not in features:
                aliases_to_add['macd'] = value
                if 'macd_macd' not in features:
                    aliases_to_add['macd_macd'] = value
        
        # Find momentum features
        for name, value in features.items():
            if 'momentum_' in name and 'momentum' not in features:
                aliases_to_add['momentum'] = value
                # Also create momentum_momentum_N pattern if needed
                if '_momentum_' in name:
                    try:
                        period = name.split('_momentum_')[1].split('_')[0]
                        if period.isdigit() and f'momentum_momentum_{period}' not in features:
                            aliases_to_add[f'momentum_momentum_{period}'] = value
                    except:
                        pass
        
        # Find volatility features
        for name, value in features.items():
            if 'volatility_' in name and 'volatility' not in features and not name.endswith('_sma'):
                aliases_to_add['volatility'] = value
                break
        
        # Now add all aliases after iteration is complete
        for alias_name, alias_value in aliases_to_add.items():
            features[alias_name] = alias_value
        
        # Create ATR SMA and volatility SMA if base features exist
        # Check both original features and newly added aliases
        if ('atr' in features or 'atr' in aliases_to_add) and 'atr_sma' not in features:
            # For now, use the same value (proper SMA would require history)
            features['atr_sma'] = features.get('atr', aliases_to_add.get('atr'))
        
        if ('volatility' in features or 'volatility' in aliases_to_add) and 'volatility_sma' not in features:
            # For now, use the same value
            features['volatility_sma'] = features.get('volatility', aliases_to_add.get('volatility'))
        
        logger.debug(f"Created feature aliases for {symbol}: {list(features.keys())}")
    
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
        logger.info(f"ComponentState received features config: {features}")
        if features:
            self.configure_features(features)
            logger.info(f"Configured {len(features)} features from container config")
        else:
            logger.warning("No features config received by ComponentState")
        
        # Load components - supports legacy strategies, legacy classifiers, and new components config
        strategies = config.get('strategies', [])
        classifiers = config.get('classifiers', [])
        components_config = config.get('components', [])
        
        logger.info(f"ComponentState config sections: strategies={len(strategies)}, classifiers={len(classifiers)}, components={len(components_config)}")
        
        # Check if stateless components were passed from topology
        stateless_components = config.get('stateless_components', {})
        logger.info(f"ComponentState received stateless_components: {list(stateless_components.keys())}")
        for comp_type, components in stateless_components.items():
            logger.info(f"  {comp_type}: {list(components.keys()) if isinstance(components, dict) else components}")
        
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
            logger.info("No components configured")
    
    def _load_strategies_from_config(self, strategies: List[Dict[str, Any]], 
                                   stateless_components: Dict[str, Any],
                                   container) -> None:
        """Load strategies (backwards compatibility)."""
        strategy_components = stateless_components.get('strategies', {})
        
        logger.info(f"STRATEGY LOADING: Found {len(strategies)} strategy configs")
        logger.info(f"STRATEGY LOADING: Found {len(strategy_components)} strategy functions from topology")
        
        if strategy_components:
            logger.info(f"Using {len(strategy_components)} strategies from topology")
            for strategy_name, strategy_func in strategy_components.items():
                logger.info(f"Processing strategy component: {strategy_name}")
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
                            logger.info(f"Found config by exact name match: {config_name}")
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
                
                self.add_component(
                    component_id=component_id,
                    component_func=strategy_func,
                    component_type="strategy",
                    parameters=params_with_type
                )
                logger.info(f"Loaded strategy: {strategy_name} from topology")
        else:
            logger.warning("No strategies found in stateless_components")
    
    def _load_classifiers_from_config(self, classifiers: List[Dict[str, Any]], 
                                    stateless_components: Dict[str, Any],
                                    container) -> None:
        """Load classifiers (backwards compatibility)."""
        classifier_components = stateless_components.get('classifiers', {})
        
        if classifier_components:
            logger.info(f"Using {len(classifier_components)} classifiers from topology")
            for classifier_name, classifier_func in classifier_components.items():
                # Find matching config by name or type
                classifier_config = next(
                    (c for c in classifiers if c.get('name') == classifier_name or c.get('type') == classifier_name),
                    {}
                )
                
                # Create component_id with symbol pattern for proper filtering
                symbols = container.config.config.get('symbols', ['unknown'])
                symbol = symbols[0] if symbols else 'unknown'
                component_id = f"{symbol}_{classifier_name}"
                
                self.add_component(
                    component_id=component_id,
                    component_func=classifier_func,
                    component_type="classifier",
                    parameters=classifier_config.get('params', {})
                )
                logger.info(f"Loaded classifier: {classifier_name} from topology")
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
            
            logger.info(f"Using {len(type_components)} {comp_type} from topology")
            
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
                logger.info(f"Loaded {comp_type}: {comp_name} from topology")
    
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
        logger.info(f"on_bar called with event type: {event.event_type}")
        
        if event.event_type != EventType.BAR.value:
            return
        
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
        
        logger.info(f"Processing BAR for {symbol}: close={bar_dict['close']}, bar_count={self._bar_count.get(symbol, 0) + 1}")
        
        # Update features
        try:
            self.update_bar(symbol, bar_dict)
        except Exception as e:
            logger.error(f"Error updating bar for {symbol}: {e}", exc_info=True)
            return
        
        # Get current features
        try:
            features = self.get_features(symbol)
            logger.debug(f"Features for {symbol}: {len(features)} features computed")
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}", exc_info=True)
            return
        
        # Execute components individually based on their readiness
        try:
            self._execute_components_individually(symbol, features, bar_dict, event.timestamp)
        except Exception as e:
            logger.error(f"Error executing components for {symbol}: {e}", exc_info=True)
    
    def _execute_components_individually(self, symbol: str, features: Dict[str, Any], 
                                       bar: Dict[str, float], timestamp: datetime) -> None:
        """Execute components individually, each checking its own readiness."""
        current_bars = self._bar_count.get(symbol, 0)
        components_snapshot = list(self._components.items())
        
        # Track which components are ready
        ready_classifiers = []
        ready_strategies = []
        
        if current_bars == 1 or current_bars % 20 == 0:  # Log occasionally for debugging
            logger.info(f"Bar {current_bars}: Checking {len(components_snapshot)} components for readiness. Available features: {list(features.keys())}")
        
        for component_id, component_info in components_snapshot:
            component_type = component_info['component_type']
            
            # Log component checks at first bar
            if current_bars == 1:
                logger.info(f"Checking component: {component_id} (type: {component_type})")
            
            # Check if this specific component has the features it needs
            if self._is_component_ready(component_id, component_info, features, current_bars):
                if component_type == 'classifier':
                    ready_classifiers.append((component_id, component_info))
                else:
                    ready_strategies.append((component_id, component_info))
                    if current_bars == 20:  # Log when strategies become ready
                        logger.info(f"Strategy {component_id} is READY at bar {current_bars}")
            else:
                # Log why this component isn't ready yet
                required_features = self._get_component_required_features(component_id, component_info)
                missing_features = [f for f in required_features if f not in features or features[f] is None]
                if current_bars == 1 or (missing_features and current_bars % 20 == 0):  # Log occasionally
                    logger.info(f"Bar {current_bars}: Component {component_id} waiting for features: {missing_features[:3]}")
                if missing_features and current_bars % 10 == 1:  # Log every 10 bars
                    logger.debug(f"Component {component_id} waiting for features: {missing_features[:3]}...")
        
        if ready_classifiers or ready_strategies:
            logger.info(f"Bar {current_bars}: Executing {len(ready_classifiers)} classifiers and {len(ready_strategies)} strategies")
        
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
                        component_info=component_info
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
                        component_info=component_info
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
            return False
        
        # Check if all required features are available and not None
        missing_features = []
        for feature_name in required_features:
            if feature_name not in features:
                missing_features.append(f"{feature_name} (not in dict)")
            elif features[feature_name] is None:
                missing_features.append(f"{feature_name} (is None)")
        
        if missing_features and current_bars == 20:  # Log at bar 20 for debugging
            logger.info(f"Component {component_id} at bar 20 - Required: {required_features}, Missing: {missing_features}, Available: {list(features.keys())[:10]}...")
        
        return len(missing_features) == 0
        
        return True
    
    def _get_strategy_required_features(self, component_id: str, params: Dict[str, Any]) -> List[str]:
        """Get the specific features this strategy needs by extracting from function metadata."""
        logger.debug(f"Getting required features for {component_id} with params: {params}")
        
        # First, try to get the strategy function and extract its feature config
        component_info = self._components.get(component_id)
        if component_info and 'function' in component_info:
            strategy_func = component_info['function']
            
            # Extract feature config from the strategy function's metadata
            if hasattr(strategy_func, '_strategy_metadata'):
                feature_config = strategy_func._strategy_metadata.get('feature_config', {})
                logger.debug(f"Found strategy metadata for {component_id}: {feature_config}")
                
                required_features = []
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
                    logger.debug(f"Extracted features from {component_id} metadata: {required_features}")
                    return required_features
            else:
                logger.debug(f"No _strategy_metadata found for {component_id}")
        
        # Fallback: try to infer from parameters and component type
        strategy_type = params.get('_strategy_type', '')
        logger.info(f"FALLBACK: Using fallback for {component_id}, strategy_type: {strategy_type}, params: {params}")
        
        # Handle specific strategy types
        if 'momentum' in component_id.lower() or strategy_type == 'simple_momentum':
            # Momentum strategies need SMA and RSI
            sma_period = params.get('sma_period', 20)
            rsi_period = params.get('rsi_period', 14)
            result = [f"sma_{sma_period}", f"rsi_{rsi_period}"]
            logger.info(f"FALLBACK: Momentum strategy {component_id} requires: {result}")
            return result
        elif 'ma_crossover' in component_id.lower() or strategy_type == 'ma_crossover':
            # MA crossover needs fast and slow SMAs - use actual parameters
            fast_period = params.get('fast_period', 10)
            slow_period = params.get('slow_period', 20)
            result = [f"sma_{fast_period}", f"sma_{slow_period}"]
            logger.info(f"FALLBACK: MA crossover strategy {component_id} using periods: fast={fast_period}, slow={slow_period} from params: {params}")
            logger.info(f"FALLBACK: MA crossover strategy {component_id} requires: {result}")
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
        """Get the specific features this classifier needs."""
        # Extract classifier type from component_id
        if 'enhanced_trend' in component_id:
            return ['sma_10', 'sma_20', 'sma_50']
        elif 'market_regime' in component_id:
            return ['sma_10', 'sma_50', 'atr_20', 'rsi_14']
        elif 'hidden_markov' in component_id:
            return ['volume', 'rsi_14', 'sma_20', 'sma_50', 'atr_14']
        elif 'volatility_momentum' in component_id:
            return ['atr_14', 'rsi_14', 'sma_20']
        elif 'microstructure' in component_id:
            return ['sma_5', 'sma_20', 'atr_10', 'rsi_7']
        elif 'momentum' in component_id:
            return ['rsi_14', 'macd', 'momentum_10']
        elif 'volatility' in component_id:
            return ['atr_14', 'volatility_20']
        elif 'trend' in component_id:
            return ['sma_10', 'sma_20', 'sma_50']
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
                        component_info=component_info
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
                        component_info=component_info
                    )
                    
            except Exception as e:
                logger.error(f"Error executing component {component_id}: {e}")
        
        # Update all outputs after iteration is complete
        for component_id, result in outputs_to_update.items():
            self._components[component_id]['last_output'] = result
    
    def _process_component_output(self, component_id: str, component_type: str,
                                result: Dict[str, Any], symbol: str, timestamp: datetime,
                                component_info: Optional[Dict[str, Any]] = None) -> None:
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
                        'component_type': component_type
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
                
                # Only publish non-flat signals
                if direction != SignalDirection.FLAT:
                    
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
                            'base_strategy_id': result.get('strategy_id', '')  # Keep original strategy type for reference
                        }
                    )
                    self._publish_signal(signal)
                else:
                    logger.debug(f"Not publishing flat signal for {component_id}: signal_value={signal_value}")
                
        elif component_type == "classifier":
            # Classifier outputs - only publish on classification change
            regime = result.get('regime')
            confidence = result.get('confidence', 0.0)
            
            if regime:
                # Check if classification changed
                previous_regime = previous_output.get('regime') if previous_output else None
                
                if previous_regime != regime:
                    # Classification changed - publish CLASSIFICATION event
                    logger.info(f"Classifier {component_id} regime change: {previous_regime} → {regime} (confidence: {confidence:.3f})")
                    
                    # Create Classification object without full features
                    classification = Classification(
                        symbol=symbol,
                        regime=regime,
                        confidence=confidence,
                        timestamp=timestamp,
                        classifier_id=component_id,
                        previous_regime=previous_regime,
                        features={},  # Empty features dict to reduce payload size
                        metadata=result.get('metadata', {})
                    )
                    
                    self._publish_classification(classification)
                else:
                    # Same classification - don't publish (sparse storage)
                    logger.debug(f"Classifier {component_id} regime unchanged: {regime} (confidence: {confidence:.3f})")
        
        else:
            # Future component types - generic handling
            logger.info(f"Unsupported component type: {component_type}")
    
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
        
        logger.info(f"📡 Publishing SIGNAL event: {signal_event.event_type} with payload: {signal_event.payload}")
        
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
        
        logger.info(f"👑 Publishing CLASSIFICATION event: {classification_event.event_type} with payload: {classification_event.payload}")
        
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
                "feature_counts": {
                    symbol: len(features)
                    for symbol, features in self._features.items()
                }
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