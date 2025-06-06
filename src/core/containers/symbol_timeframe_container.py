"""
Symbol-Timeframe Container Implementation

This container encapsulates both market data and feature computation for a 
specific symbol-timeframe combination, implementing the new EVENT_FLOW_ARCHITECTURE.

Key responsibilities:
- Stream market data (BAR events)
- Compute and cache indicators/features
- Broadcast FEATURES events to portfolios
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
import logging
import pandas as pd
from datetime import datetime

from ..events import EventBus, Event, EventType
from ...data.models import Bar
from .protocols import Container as ContainerProtocol, ContainerMetadata, ContainerState, ContainerRole
from .container import Container, ContainerConfig
from ..tracing import trace, TracePoint

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    indicators: List[Dict[str, Any]]  # List of indicator configs
    lookback_periods: Dict[str, int]  # Lookback for each indicator
    update_frequency: str = "bar"  # When to update: "bar" or "batch"


class SymbolTimeframeContainer(Container):
    """
    Container that combines data streaming and feature computation for a 
    specific symbol-timeframe combination.
    
    This is the standard unit in the EVENT_FLOW_ARCHITECTURE, broadcasting
    FEATURES events that include both market data and computed indicators.
    """
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 data_config: Dict[str, Any],
                 feature_config: Optional[Dict[str, Any]] = None,
                 container_id: Optional[str] = None):
        """
        Initialize Symbol-Timeframe container.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY')
            timeframe: Time granularity (e.g., '1m', '5m', '1d')
            data_config: Configuration for data source
            feature_config: Configuration for feature computation
            container_id: Optional container ID override
        """
        # Create container ID if not provided
        if container_id is None:
            container_id = f"{symbol}_{timeframe}"
            
        # Initialize base container with DATA role
        super().__init__(ContainerConfig(
            role=ContainerRole.DATA,  # Primary role is data
            name=container_id,
            container_id=container_id,
            config={
                'symbol': symbol,
                'timeframe': timeframe,
                'data_config': data_config,
                'feature_config': feature_config or {}
            },
            capabilities={'data.streaming', 'feature.computation', 'feature.broadcast'}
        ))
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_config = data_config
        self.feature_config = feature_config or {}
        
        # Data components
        self.data_source = None
        self.current_bar = None
        self.bar_history = []  # Rolling window of bars
        self.max_history = feature_config.get('max_history', 1000) if feature_config else 1000
        
        # Feature computation components
        self.feature_calculators = {}
        self.feature_cache = {}
        self.indicators_initialized = False
        
        # Event tracking
        self.bars_processed = 0
        self.features_broadcasted = 0
        
        logger.info(f"Created SymbolTimeframeContainer: {container_id}")
    
    async def initialize(self) -> None:
        """Initialize the container and its components."""
        await super().initialize()
        
        # Initialize data source only if we have data config
        if self.data_config and self.data_config.get('source'):
            logger.info(f"{self.container_id} creating data source with config: {self.data_config}")
            self.data_source = await self._create_data_source()
        else:
            logger.info(f"{self.container_id} NOT creating data source (no config)")
            self.data_source = None
        
        # Initialize feature calculators
        self._initialize_feature_calculators()
        
        # Subscribe to internal BAR events for feature computation
        self.event_bus.subscribe(EventType.BAR, self._on_bar_received)
        
        logger.info(f"Initialized {self.container_id} with {len(self.feature_calculators)} features: {list(self.feature_calculators.keys())}")
        if len(self.feature_calculators) == 0:
            logger.info(f"{self.container_id} has NO feature calculators")
    
    async def start(self) -> None:
        """Start streaming data and computing features."""
        await super().start()
        
        # Start data streaming
        if self.data_source:
            await self.data_source.start_streaming()
            logger.info(f"Started data streaming for {self.symbol} {self.timeframe}")
    
    async def stop(self) -> None:
        """Stop data streaming and cleanup."""
        if self.data_source:
            await self.data_source.stop_streaming()
            
        await super().stop()
    
    async def _create_data_source(self):
        """Create appropriate data source based on configuration."""
        source_type = self.data_config.get('source', 'csv')
        
        if source_type == 'csv':
            # Import data handler
            from ...data.csv_handler import CSVDataHandler
            
            # Determine file path - use configured path or infer from symbol/timeframe
            file_path = self.data_config.get('file_path')
            
            if not file_path:
                # Use path resolver to infer path
                from ...data.path_resolver import resolve_data_path
                
                # Get data directory from config or use default
                data_dir = self.data_config.get('data_dir', './data')
                
                # Try to resolve path based on symbol and timeframe
                resolved_path = resolve_data_path(self.symbol, self.timeframe, data_dir)
                
                if resolved_path:
                    file_path = str(resolved_path)
                    logger.info(f"Auto-resolved data path: {self.symbol} {self.timeframe} -> {file_path}")
                else:
                    # Fallback to simple pattern
                    file_path = f'{data_dir}/{self.symbol}.csv'
                    logger.warning(f"Could not resolve specific path for {self.symbol} {self.timeframe}, "
                                 f"falling back to: {file_path}")
            
            return CSVDataHandler(
                file_path=file_path,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_handler=self._handle_market_data,
                max_bars=self.data_config.get('max_bars')
            )
        elif source_type == 'live':
            # Future: Live data handler
            raise NotImplementedError("Live data source not yet implemented")
        else:
            raise ValueError(f"Unknown data source type: {source_type}")
    
    def _initialize_feature_calculators(self):
        """Initialize indicator/feature calculators based on configuration."""
        logger.debug(f"{self.container_id} feature_config: {self.feature_config}")
        
        # Get required features from config
        features = self.feature_config.get('indicators', [])
        
        # Add some default features ONLY if:
        # 1. No features specified in indicators
        # 2. Feature config exists and is not an empty dict
        # Empty dict {} means no features should be computed
        if not features and self.feature_config and 'indicators' not in self.feature_config:
            # Only add defaults if this looks like a feature container without explicit config
            features = [
                {'name': 'sma_20', 'type': 'sma', 'period': 20},
                {'name': 'sma_50', 'type': 'sma', 'period': 50},
                {'name': 'rsi', 'type': 'rsi', 'period': 14}
            ]
        
        # Create calculator for each feature
        for feature_conf in features:
            feature_name = feature_conf['name']
            feature_type = feature_conf['type']
            
            if feature_type == 'sma':
                self.feature_calculators[feature_name] = self._create_sma_calculator(
                    period=feature_conf['period']
                )
            elif feature_type == 'rsi':
                self.feature_calculators[feature_name] = self._create_rsi_calculator(
                    period=feature_conf.get('period', 14)
                )
            # Add more indicator types as needed
    
    def _create_sma_calculator(self, period: int):
        """Create SMA calculator function."""
        def calculate_sma(bars: List[Bar]) -> Optional[float]:
            if len(bars) < period:
                return None
            prices = [bar.close for bar in bars[-period:]]
            return sum(prices) / len(prices)
        return calculate_sma
    
    def _create_rsi_calculator(self, period: int):
        """Create RSI calculator function."""
        def calculate_rsi(bars: List[Bar]) -> Optional[float]:
            if len(bars) < period + 1:
                return None
            
            # Simple RSI calculation
            prices = [bar.close for bar in bars[-(period+1):]]
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        return calculate_rsi
    
    def _handle_market_data(self, bar: Bar):
        """Handle incoming market data."""
        # Store in history
        self.bar_history.append(bar)
        if len(self.bar_history) > self.max_history:
            self.bar_history.pop(0)
        
        self.current_bar = bar
        self.bars_processed += 1
        
        # Emit BAR event internally
        bar_event = Event(
            event_type=EventType.BAR,
            payload={'bar': bar, 'symbol': self.symbol, 'timeframe': self.timeframe},
            source_id=self.container_id
        )
        self.event_bus.publish(bar_event)
        logger.debug(f"{self.container_id} published BAR event")
    
    def _on_bar_received(self, event: Event):
        """Process bar and compute features."""
        logger.debug(f"{self.container_id} received BAR event, has {len(self.feature_calculators)} calculators")
        
        # Only compute and broadcast features if we have feature calculators
        if not self.feature_calculators:
            return
            
        # Compute all features
        features = self._compute_features()
        
        # Broadcast FEATURES event
        if features:
            # Trace feature calculation
            correlation_id = trace(TracePoint.FEATURE_CALC, "symbol_timeframe_container.py", {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'features': list(features.keys()),
                'price': self.current_bar.close if self.current_bar else None
            })
            
            feature_event = Event(
                event_type=EventType.FEATURES,
                payload={
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'bar': self.current_bar,
                    'features': features,
                    'correlation_id': correlation_id,  # Pass correlation through
                    'timestamp': datetime.now()
                },
                source_id=self.container_id
            )
            logger.debug(f"{self.container_id} publishing to event bus id: {id(self.event_bus)}")
            self.event_bus.publish(feature_event)
            self.features_broadcasted += 1
            logger.debug(f"{self.container_id} published FEATURES event with {len(features)} features")
    
    def _compute_features(self) -> Dict[str, Any]:
        """Compute all configured features."""
        if not self.bar_history:
            return {}
        
        features = {
            # Always include current price data
            'open': self.current_bar.open if self.current_bar else None,
            'high': self.current_bar.high if self.current_bar else None,
            'low': self.current_bar.low if self.current_bar else None,
            'close': self.current_bar.close if self.current_bar else None,
            'volume': self.current_bar.volume if self.current_bar else None,
        }
        
        # Compute each configured feature
        for feature_name, calculator in self.feature_calculators.items():
            try:
                value = calculator(self.bar_history)
                if value is not None:
                    features[feature_name] = value
                    logger.debug(f"Computed {feature_name}: {value}")
            except Exception as e:
                logger.error(f"Error computing {feature_name}: {e}")
        
        return features
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get container state information."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'bars_processed': self.bars_processed,
            'features_broadcasted': self.features_broadcasted,
            'history_size': len(self.bar_history),
            'active_features': list(self.feature_calculators.keys())
        }


def create_symbol_timeframe_container(
    symbol: str,
    timeframe: str,
    data_config: Dict[str, Any],
    feature_config: Optional[Dict[str, Any]] = None
) -> SymbolTimeframeContainer:
    """
    Factory function to create a SymbolTimeframeContainer.
    
    This is the preferred way to create symbol-timeframe containers
    in the EVENT_FLOW_ARCHITECTURE.
    """
    return SymbolTimeframeContainer(
        symbol=symbol,
        timeframe=timeframe,
        data_config=data_config,
        feature_config=feature_config
    )