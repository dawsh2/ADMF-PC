"""
Feature Hub Container Component.

This component wraps the FeatureHub to provide centralized feature computation
for all strategies in the system. It subscribes to BAR events and maintains
feature state, allowing strategies to remain stateless.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...events.types import Event, EventType
from ....strategy.components.features.hub import FeatureHub

logger = logging.getLogger(__name__)


class FeatureHubComponent:
    """
    Container component that manages centralized feature computation.
    
    This component:
    1. Subscribes to BAR events from the event bus
    2. Updates the FeatureHub with new bar data
    3. Provides feature access to strategies via reference
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize Feature Hub Component.
        
        Args:
            symbols: List of symbols to track
        """
        self.name = "feature_hub"
        # Use incremental mode by default for O(1) feature updates
        self._feature_hub = FeatureHub(symbols)  # Defaults to use_incremental=True
        self._container = None
        self._bars_processed = 0
        
        logger.info(f"FeatureHubComponent initialized for symbols: {symbols}")
    
    def set_container(self, container) -> None:
        """Set container reference and subscribe to events."""
        self._container = container
        
        # Subscribe to BAR events
        container.event_bus.subscribe(EventType.BAR.value, self.on_bar)
        logger.info(f"FeatureHubComponent subscribed to BAR events in container {container.name}")
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]], 
                          required_features: Optional[List[str]] = None) -> None:
        """
        Configure which features to compute.
        
        Args:
            feature_configs: Dict mapping feature names to their configurations
            required_features: Optional list of features actually needed (for optimization)
        """
        self._feature_hub.configure_features(feature_configs, required_features)
        
        # Log configuration details
        if required_features:
            logger.info(f"FeatureHubComponent configured with {len(self._feature_hub._feature_configs)} "
                       f"features (optimized from {len(feature_configs)})")
        else:
            logger.info(f"FeatureHubComponent configured with {len(feature_configs)} features")
    
    def on_bar(self, event: Event) -> None:
        """
        Handle BAR event - update features for the symbol.
        
        Args:
            event: BAR event containing symbol and bar data
        """
        if event.event_type != EventType.BAR.value:
            return
        
        payload = event.payload
        symbol = payload.get('symbol')
        bar = payload.get('bar')
        
        if not symbol or not bar:
            logger.warning(f"Invalid BAR event: symbol={symbol}, bar keys={bar.keys() if bar else None}")
            return
        
        self._bars_processed += 1
        
        # Convert bar object to dict for FeatureHub
        bar_dict = {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        
        # Update FeatureHub with new bar
        import time
        start_time = time.time()
        try:
            self._feature_hub.update_bar(symbol, bar_dict)
            update_time = time.time() - start_time
            
            # Log progress occasionally
            if self._bars_processed % 20 == 0 or update_time > 0.01:
                features_count = len(self._feature_hub.get_features(symbol))
                logger.debug(f"FeatureHub bar {self._bars_processed}: update_bar took {update_time*1000:.1f}ms for {features_count} features")
            
            if self._bars_processed % 100 == 0:
                summary = self._feature_hub.get_feature_summary()
                logger.debug(f"FeatureHub processed {self._bars_processed} bars. "
                          f"Features computed: {summary['feature_counts']}")
        except Exception as e:
            logger.error(f"Error updating FeatureHub for {symbol}: {e}", exc_info=True)
    
    def get_features(self, symbol: str) -> Dict[str, Any]:
        """
        Get current features for a symbol.
        
        Args:
            symbol: Symbol to get features for
            
        Returns:
            Dict of feature_name -> feature_value
        """
        return self._feature_hub.get_features(symbol)
    
    def get_all_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all features for all symbols.
        
        Returns:
            Dict mapping symbol -> feature_dict
        """
        return self._feature_hub.get_all_features()
    
    def has_sufficient_data(self, symbol: str, min_bars: int = 50) -> bool:
        """
        Check if symbol has sufficient data for feature calculation.
        
        Args:
            symbol: Symbol to check
            min_bars: Minimum number of bars required
            
        Returns:
            True if sufficient data available
        """
        return self._feature_hub.has_sufficient_data(symbol, min_bars)
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        self._feature_hub.add_symbol(symbol)
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset feature computation state.
        
        Args:
            symbol: Symbol to reset (None for all symbols)
        """
        self._feature_hub.reset(symbol)
        if symbol is None:
            self._bars_processed = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics."""
        summary = self._feature_hub.get_feature_summary()
        return {
            'bars_processed': self._bars_processed,
            'feature_hub_summary': summary,
            'name': self.name
        }
    
    def get_feature_hub(self) -> FeatureHub:
        """
        Get the underlying FeatureHub instance.
        
        This allows direct access for strategies that need it.
        
        Returns:
            The FeatureHub instance
        """
        return self._feature_hub


def create_feature_hub_component(container_config: Dict[str, Any]) -> FeatureHubComponent:
    """
    Factory function to create FeatureHubComponent from container config.
    
    Args:
        container_config: Container configuration dict
        
    Returns:
        Configured FeatureHubComponent instance
    """
    # Extract symbols from config
    symbols = container_config.get('symbols', [])
    
    # Create component
    component = FeatureHubComponent(symbols)
    
    # Configure features if provided
    features = container_config.get('features', {})
    if features:
        # Check if we have required_features from pre-flight check
        required_features = container_config.get('required_features')
        component.configure_features(features, required_features)
    
    return component