"""
Strategy state management - NO INHERITANCE!

Maintains feature state and executes strategies, similar to how
portfolio/state.py maintains portfolio state and manages positions.
"""

from typing import Dict, Any, Optional, List, Deque
from collections import deque, defaultdict
import logging
from datetime import datetime
import pandas as pd

from .protocols import FeatureProvider
from .types import Signal, SignalType, SignalDirection
from .components.features import compute_feature, FEATURE_REGISTRY
from ..core.events.types import Event, EventType

logger = logging.getLogger(__name__)


class StrategyState:
    """
    Manage feature state and strategy execution - NO INHERITANCE!
    
    This directly maintains feature state (no separate FeatureHub needed).
    The strategy container IS the feature hub.
    
    Similar to PortfolioState but for features/strategies:
    - PortfolioState maintains positions and calls risk validators
    - StrategyState maintains features and calls strategy functions
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        feature_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize strategy state.
        
        Args:
            symbols: Initial symbols to track
            feature_configs: Feature configurations
        """
        self._symbols = symbols or []
        
        # Price data storage - rolling windows per symbol
        self._price_data: Dict[str, Dict[str, Deque[float]]] = defaultdict(lambda: {
            'open': deque(maxlen=1000),
            'high': deque(maxlen=1000),
            'low': deque(maxlen=1000),
            'close': deque(maxlen=1000),
            'volume': deque(maxlen=1000)
        })
        
        # Computed features per symbol
        self._features: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Feature configurations
        self._feature_configs = feature_configs or {}
        
        # Bar count per symbol
        self._bar_count: Dict[str, int] = defaultdict(int)
        
        # Strategy registry - populated by container/topology
        self._strategies: Dict[str, Dict[str, Any]] = {}
        
        # Container reference for event publishing
        self._container = None
        
        # Metrics
        self._bars_processed = 0
        self._signals_generated = 0
        self._last_update = datetime.now()
        
        # Component name for identification
        self.name = "strategy_state"
    
    def set_container(self, container) -> None:
        """Set container reference and subscribe to events."""
        self._container = container
        # Subscribe to BAR events
        container.event_bus.subscribe(EventType.BAR.value, self.on_bar)
        logger.info(f"StrategyState subscribed to BAR events in container {container.name}")
        
        # Load strategies from container config
        self._load_strategies_from_config(container)
    
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
        return self._features.get(symbol, {})
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]]) -> None:
        """Configure which features to calculate."""
        self._feature_configs = feature_configs
        logger.info(f"Configured {len(feature_configs)} features")
    
    def has_sufficient_data(self, symbol: str, min_bars: int = 5) -> bool:
        """Check if sufficient data available."""
        return self._bar_count.get(symbol, 0) >= min_bars
    
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
            for strategy in self._strategies.values():
                strategy['last_signal'] = None
    
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
            return
        
        # Create DataFrame for feature computation
        df = pd.DataFrame(data_dict)
        
        # Compute all configured features
        symbol_features = {}
        
        for feature_name, config in self._feature_configs.items():
            try:
                feature_type = config.get('feature')
                if feature_type not in FEATURE_REGISTRY:
                    logger.warning("Unknown feature type: %s", feature_type)
                    continue
                
                # Create config copy without 'feature' key
                feature_params = {k: v for k, v in config.items() if k != 'feature'}
                
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
        
        # Update cache
        self._features[symbol].update(symbol_features)
        
        logger.debug("Updated %d features for %s", len(symbol_features), symbol)
    
    # Strategy management
    def add_strategy(self, strategy_id: str, strategy_func: Any, 
                    parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a strategy function to execute.
        
        Args:
            strategy_id: Unique identifier
            strategy_func: Pure strategy function (stateless)
            parameters: Strategy parameters
        """
        self._strategies[strategy_id] = {
            'function': strategy_func,
            'parameters': parameters or {},
            'last_signal': None
        }
        logger.debug(f"Added strategy: {strategy_id}")
    
    def _load_strategies_from_config(self, container) -> None:
        """Load strategies from container configuration."""
        # Check for strategies in container config
        config = container.config.config
        strategies = config.get('strategies', [])
        
        # Load feature configurations first
        features = config.get('features', {})
        logger.info(f"StrategyState received features config: {features}")
        if features:
            self.configure_features(features)
            logger.info(f"Configured {len(features)} features from container config")
        else:
            logger.warning("No features config received by StrategyState")
        
        if not strategies:
            logger.info("No strategies configured")
            return
            
        logger.info(f"Loading {len(strategies)} strategies from config")
        
        # Check if stateless components were passed from topology
        stateless_components = config.get('stateless_components', {})
        strategy_components = stateless_components.get('strategies', {})
        
        if strategy_components:
            # Use pre-loaded strategy components from topology
            logger.info(f"Using {len(strategy_components)} strategies from topology")
            for strategy_name, strategy_func in strategy_components.items():
                # Find matching config by name or type
                strategy_config = next(
                    (s for s in strategies if s.get('name') == strategy_name or s.get('type') == strategy_name),
                    {}
                )
                logger.info(f"Strategy {strategy_name} config: {strategy_config}")
                logger.info(f"Strategy {strategy_name} params: {strategy_config.get('params', {})}")
                
                # Create strategy_id with symbol_timeframe pattern for proper filtering
                # Extract symbol from container config if available
                symbols = container.config.config.get('symbols', ['unknown'])
                symbol = symbols[0] if symbols else 'unknown'
                strategy_id = f"{symbol}_{strategy_name}"  # e.g., "SPY_momentum"
                
                self.add_strategy(
                    strategy_id=strategy_id,
                    strategy_func=strategy_func,
                    parameters=strategy_config.get('params', {})
                )
                logger.info(f"Loaded strategy: {strategy_name} from topology")
        else:
            # Fallback: strategies should have been loaded by topology
            logger.warning("No strategies found in stateless_components. " +
                         "Strategies should be loaded by the topology builder.")
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy."""
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            logger.debug(f"Removed strategy: {strategy_id}")
            return True
        return False
    
    def get_strategy_ids(self) -> List[str]:
        """Get list of active strategy IDs."""
        return list(self._strategies.keys())
    
    # Event handling
    def on_bar(self, event: Event) -> None:
        """
        Handle BAR event - update features and execute strategies.
        
        This is the main event handler that coordinates:
        1. Feature calculation via FeatureHub
        2. Strategy execution
        3. Signal publishing
        """
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
        
        # Convert bar object to dict for FeatureHub
        bar_dict = {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        
        logger.info(f"Processing BAR for {symbol}: close={bar_dict['close']}, bar_count={self._bar_count.get(symbol, 0) + 1}")
        
        # Update features
        self.update_bar(symbol, bar_dict)
        
        # Get current features
        features = self.get_features(symbol)
        logger.info(f"Features for {symbol}: {list(features.keys())} -> {features}")
        logger.info(f"Feature configs: {list(self._feature_configs.keys())}")
        
        # Execute strategies if we have sufficient data
        if self.has_sufficient_data(symbol):
            logger.info(f"Sufficient data for {symbol}, executing {len(self._strategies)} strategies")
            logger.info(f"Available features: {list(features.keys())}")
            for strategy_id, strategy_info in self._strategies.items():
                logger.info(f"Strategy {strategy_id} params: {strategy_info.get('parameters', {})}")
            self._execute_strategies(symbol, features, bar_dict, event.timestamp)
        else:
            logger.info(f"Insufficient data for {symbol}: {self._bar_count.get(symbol, 0)} bars (need {5})")
    
    def _execute_strategies(self, symbol: str, features: Dict[str, Any], 
                          bar: Dict[str, float], timestamp: datetime) -> None:
        """Execute all strategies with current features."""
        for strategy_id, strategy_info in self._strategies.items():
            try:
                # Call pure strategy function
                result = strategy_info['function'](
                    features=features,
                    bar=bar,
                    params=strategy_info['parameters']
                )
                
                # Process result if we got a signal
                if result and result.get('signal_type'):
                    # Create Signal object
                    signal = Signal(
                        symbol=symbol,
                        direction=result.get('direction', SignalDirection.FLAT),
                        strength=result.get('strength', 1.0),
                        timestamp=timestamp,
                        strategy_id=strategy_id,
                        signal_type=SignalType(result.get('signal_type', 'entry')),
                        metadata=result.get('metadata', {})
                    )
                    
                    # Store last signal
                    strategy_info['last_signal'] = signal
                    
                    # Publish signal
                    self._publish_signal(signal)
                    
            except Exception as e:
                logger.error(f"Error executing strategy {strategy_id}: {e}")
    
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
        self._signals_generated += 1
        
        logger.debug(f"Published {signal.signal_type.value} signal: "
                    f"{signal.direction} {signal.symbol} from {signal.strategy_id}")
    
    # Status and metrics
    def get_metrics(self) -> Dict[str, Any]:
        """Get state metrics."""
        return {
            'bars_processed': self._bars_processed,
            'signals_generated': self._signals_generated,
            'active_strategies': len(self._strategies),
            'last_update': self._last_update,
            'feature_summary': {
                "symbols": len(self._symbols),
                "configured_features": len(self._feature_configs),
                "bar_counts": dict(self._bar_count),
                "feature_counts": {
                    symbol: len(features)
                    for symbol, features in self._features.items()
                }
            }
        }
    
    def get_last_signal(self, strategy_id: str) -> Optional[Signal]:
        """Get last signal from a specific strategy."""
        strategy = self._strategies.get(strategy_id)
        return strategy.get('last_signal') if strategy else None