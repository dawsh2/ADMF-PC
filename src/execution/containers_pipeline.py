"""
Execution-specific container implementations using pipeline communication adapters.

This version removes the hybrid event interface and relies on the pipeline
adapter for cross-container communication, eliminating circular dependencies.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from decimal import Decimal
import asyncio
import logging

# No global state - use proper container communication instead

# TEMPORARY: Simple module-level registry for position closing during workflow shutdown
# This enables ExecutionContainer to call PortfolioContainer directly when async 
# event routing fails due to workflow completion timing. Should be replaced with
# proper dependency injection in future architecture refactoring.
_PORTFOLIO_CONTAINER_REGISTRY = None

from ..core.containers.composable import (
    BaseComposableContainer, ComposableContainerProtocol, ContainerRole,
    ContainerState, ContainerLimits
)
from ..core.events.types import Event, EventType
from ..data.protocols import DataLoader
from ..strategy.protocols import Strategy
from ..risk.protocols import Signal
from ..core.logging.event_logger import (
    log_bar_event, log_indicator_event, log_signal_event, 
    log_order_event, log_fill_event, log_portfolio_event,
    get_event_logger
)


logger = get_event_logger(__name__)

# Import signal aggregation for ensemble container
try:
    from ..strategy.signal_aggregation import (
        WeightedVotingAggregator, Direction, TradingSignal, 
        AggregatedSignal, ConsensusSignal
    )
except ImportError:
    # Fallback if signal aggregation not available
    WeightedVotingAggregator = None


class BacktestContainer(BaseComposableContainer):
    """Backtest container that manages peer containers for backtesting."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.BACKTEST,
            name="BacktestContainer",
            config=config,
            container_id=container_id
        )
        
    async def _initialize_self(self) -> None:
        """Initialize backtest container - mainly just coordinates child containers."""
        logger.info("BacktestContainer initialized")
    
    async def start(self) -> None:
        """Start all child containers in proper order."""
        await super().start()
        logger.info("BacktestContainer started - managing peer containers")
    
    async def _stop_self(self) -> None:
        """Stop backtest container - children are stopped by base class."""
        logger.info("BacktestContainer stopped")
    
    def get_capabilities(self) -> Set[str]:
        """Backtest container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("backtest.coordination")
        capabilities.add("backtest.peer_management")
        return capabilities


class DataContainer(BaseComposableContainer):
    """Container for data streaming and management."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.DATA,
            name="DataContainer",
            config=config,
            container_id=container_id
        )
        self.data_loader: Optional[DataLoader] = None
        self._streaming_task: Optional[asyncio.Task] = None
        
    def on_output_event(self, handler):
        """Register handler for output events (used by pipeline adapter)."""
        self.event_bus.subscribe(EventType.BAR, handler)
        self.event_bus.subscribe(EventType.SYSTEM, handler)
        
    @property
    def expected_input_type(self):
        """DataContainer doesn't expect input events."""
        return None
        
    async def _initialize_self(self) -> None:
        """Initialize data container."""
        # Initialize data loader based on config
        logger.info(f"DataContainer config: {self._metadata.config}")
        data_source = self._metadata.config.get('source', 'historical')
        
        if data_source in ['historical', 'csv']:
            from ..data.loaders import SimpleCSVLoader
            
            # Check if specific file_path is provided
            file_path = self._metadata.config.get('file_path')
            if file_path:
                # Use parent directory as data_dir for file_path config
                from pathlib import Path
                file_path_obj = Path(file_path)
                loader_config = {
                    'data_dir': str(file_path_obj.parent),
                    'date_column': self._metadata.config.get('date_column', 'Date'),
                    'date_format': self._metadata.config.get('date_format')
                }
                self.data_loader = SimpleCSVLoader(**loader_config)
                # Store the actual filename for use in streaming
                self._specific_file = file_path_obj.name
                logger.info(f"Using specific file path: {file_path}")
            else:
                # Fallback to pattern-based loading
                loader_config = {k: v for k, v in self._metadata.config.items() 
                               if k in ['data_dir', 'date_column', 'date_format']}
                if 'data_dir' not in loader_config:
                    loader_config['data_dir'] = 'data'
                self.data_loader = SimpleCSVLoader(**loader_config)
                self._specific_file = None
        elif data_source == 'live':
            # For now, use CSV loader as placeholder - live data loader not implemented
            from ..data.loaders import SimpleCSVLoader
            loader_config = {k: v for k, v in self._metadata.config.items() 
                           if k in ['data_dir', 'date_column', 'date_format']}
            if 'data_dir' not in loader_config:
                loader_config['data_dir'] = 'data'
            self.data_loader = SimpleCSVLoader(**loader_config)
            self._specific_file = None
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        logger.info(f"DataContainer initialized with {data_source} data source")
        
        # Track last market data for END_OF_DATA closing
        self._last_market_data = {}
    
    async def start(self) -> None:
        """Start data streaming."""
        await super().start()
        
        # Start data streaming
        self._streaming_task = asyncio.create_task(self._stream_data())
    
    async def _stop_self(self) -> None:
        """Stop data streaming."""
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Data streaming stopped")
    
    async def _stream_data(self) -> None:
        """Stream data from loader."""
        logger.info("Starting data streaming")
        symbols = self._metadata.config.get('symbols', ['SPY'])
        max_bars = self._metadata.config.get('max_bars')
        
        try:
            # Load data for all symbols
            for symbol in symbols:
                logger.info(f"Loading data for {symbol}")
                # SimpleCSVLoader.load() always requires symbol parameter
                # When specific file is configured, the loader will find it based on symbol and directory
                data = self.data_loader.load(symbol=symbol)
                
                # Convert to streaming format
                if data is not None:
                    # Log key data points
                    logger.info(f"Loaded {len(data)} rows for {symbol}")
                    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
                    
                    # Limit bars if requested
                    if max_bars and max_bars > 0:
                        data = data.iloc[:max_bars]
                        logger.info(f"Limited to {max_bars} bars as requested")
                    
                    # Stream each bar
                    for timestamp, row in data.iterrows():
                        if self._state != ContainerState.RUNNING:
                            break
                        
                        bar_event = Event(
                            event_type=EventType.BAR,
                            payload={
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'data': row.to_dict()
                            },
                            timestamp=timestamp
                        )
                        
                        # Track last market data
                        self._last_market_data[symbol] = row.to_dict()
                        
                        # Publish via event bus (pipeline adapter will pick this up)
                        self.event_bus.publish(bar_event)
                        log_bar_event(logger, symbol, timestamp, row.get('Close', row.get('close', 0)))
                        
                        # Brief pause to simulate real-time streaming
                        await asyncio.sleep(0.001)
            
            # Send end of data signal
            if self._state == ContainerState.RUNNING:
                end_event = Event(
                    event_type=EventType.SYSTEM,
                    payload={
                        'message': 'END_OF_DATA',
                        'last_market_data': self._last_market_data
                    },
                    timestamp=datetime.now()
                )
                
                self.event_bus.publish(end_event)
                logger.info("📢 Published END_OF_DATA event")
                
        except Exception as e:
            logger.error(f"Error streaming data: {e}")
            self._state = ContainerState.ERROR
    
    def get_capabilities(self) -> Set[str]:
        """Data container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("data.streaming")
        capabilities.add("data.historical")
        capabilities.add("data.csv")
        return capabilities


class IndicatorContainer(BaseComposableContainer):
    """Container for indicator calculation and distribution."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.INDICATOR,
            name="IndicatorContainer",
            config=config,
            container_id=container_id
        )
        self.indicators = {}
        self.indicator_hub = None
        
    def on_output_event(self, handler):
        """Register handler for output events (used by pipeline adapter)."""
        self.event_bus.subscribe(EventType.INDICATORS, handler)
        
    @property
    def expected_input_type(self):
        """IndicatorContainer expects BAR events."""
        return EventType.BAR
        
    def receive_event(self, event: Event):
        """Receive and process events from pipeline adapter."""
        logger.info(f"🔍 IndicatorContainer received event: {event.event_type}")
        if event.event_type == EventType.BAR:
            logger.info(f"📊 Processing BAR event for symbol: {event.payload.get('symbol')}")
            asyncio.create_task(self._handle_bar_event(event))
    
    async def _initialize_self(self) -> None:
        """Initialize indicator container with specified indicators."""
        from ..strategy.components.indicator_hub import IndicatorHub, IndicatorConfig, IndicatorType
        
        # Debug: Log the full config received
        logger.info(f"IndicatorContainer config: {self._metadata.config}")
        
        # Get indicator configs - check both 'indicators' and 'required_indicators'
        indicators_to_create = (
            self._metadata.config.get('indicators', []) or 
            self._metadata.config.get('required_indicators', [])
        )
        
        # Handle config-inferred indicators
        if indicators_to_create and isinstance(indicators_to_create[0], str):
            logger.info(f"Initializing IndicatorContainer with config-inferred indicators: {indicators_to_create}")
            
            # Convert string indicators to configs
            indicator_configs = []
            for indicator_str in indicators_to_create:
                # Parse indicator strings like 'SMA_20', 'BB_20', 'RSI'
                if indicator_str.startswith('SMA_'):
                    period = int(indicator_str.split('_')[1])
                    config = IndicatorConfig(
                        name=indicator_str,
                        indicator_type=IndicatorType.TREND,
                        parameters={'period': period}
                    )
                elif indicator_str.startswith('BB_'):
                    period = int(indicator_str.split('_')[1])
                    config = IndicatorConfig(
                        name=indicator_str,
                        indicator_type=IndicatorType.VOLATILITY,
                        parameters={'period': period, 'std_dev': 2}  # Changed from 'num_std' to 'std_dev' to match IndicatorHub
                    )
                elif indicator_str == 'RSI':
                    config = IndicatorConfig(
                        name='RSI',
                        indicator_type=IndicatorType.MOMENTUM,
                        parameters={'period': 14}
                    )
                else:
                    logger.warning(f"Unknown indicator format: {indicator_str}")
                    continue
                
                indicator_configs.append(config)
                logger.info(f"Created indicator config: {config.name} ({config.indicator_type})")
        else:
            # Use provided configs
            indicator_configs = indicators_to_create
        
        # Store indicator configs for use in calculations
        self.indicator_configs = indicator_configs
        
        # Initialize indicator hub with configs
        self.indicator_hub = IndicatorHub(
            indicators=indicator_configs,
            event_bus=self.event_bus
        )
        
        logger.info(f"IndicatorContainer initialized with {len(self.indicator_configs)} indicator configs")
    
    async def _handle_bar_event(self, event: Event) -> None:
        """Process BAR event and update indicators."""
        try:
            # Transform event to match IndicatorHub format
            symbol = event.payload.get('symbol')
            data = event.payload.get('data', {})
            
            # Create market data in the format IndicatorHub expects
            market_data = {
                symbol: type('MarketData', (), {
                    'open': data.get('Open', data.get('open', 0)),
                    'high': data.get('High', data.get('high', 0)),
                    'low': data.get('Low', data.get('low', 0)),
                    'close': data.get('Close', data.get('close', 0)),
                    'volume': data.get('Volume', data.get('volume', 0))
                })()
            }
            
            # Create adapted event for IndicatorHub
            adapted_event = Event(
                event_type=event.event_type,
                payload={
                    'timestamp': event.timestamp,
                    'market_data': market_data
                },
                timestamp=event.timestamp
            )
            
            # Process market data event
            self.indicator_hub.process_market_data(adapted_event)
            
            # Get latest indicator values
            all_indicators = {}
            
            if symbol:
                # Collect all indicator values for this symbol
                for indicator_config in self.indicator_configs:
                    indicator_name = indicator_config.name
                    latest_value = self.indicator_hub.get_latest_value(indicator_name, symbol)
                    
                    if latest_value:
                        if symbol not in all_indicators:
                            all_indicators[symbol] = {}
                        all_indicators[symbol][indicator_name] = latest_value.value
                
                # Only send indicator event if we have indicators
                if all_indicators:
                    # Create indicator event with all calculations AND market data
                    indicator_event = Event(
                        event_type=EventType.INDICATORS,
                        payload={
                            'indicators': all_indicators,
                            'timestamp': event.timestamp,
                            'market_data': {symbol: data}  # Include the market data that triggered this
                        },
                        timestamp=event.timestamp
                    )
                    
                    # Publish via event bus (pipeline adapter will pick this up)
                    self.event_bus.publish(indicator_event)
                    
                    # Log each indicator value
                    for symbol_key, indicators_dict in all_indicators.items():
                        for indicator_name, indicator_value in indicators_dict.items():
                            log_indicator_event(logger, symbol_key, indicator_name, indicator_value)
                
        except Exception as e:
            logger.error(f"Error processing BAR event: {e}")
    
    def get_capabilities(self) -> Set[str]:
        """Indicator container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("indicators.calculation")
        capabilities.add("indicators.streaming")
        return capabilities


class StrategyContainer(BaseComposableContainer):
    """Container for strategy execution and signal generation."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.STRATEGY,
            name="StrategyContainer",
            config=config,
            container_id=container_id
        )
        self.strategy: Optional[Strategy] = None
        self.signal_aggregator = None
        self.strategies_config = config.get('strategies', [])
        self.multi_strategy = len(self.strategies_config) > 1
        self.last_indicators = {}
        
        
    def on_output_event(self, handler):
        """Register handler for output events (used by pipeline adapter)."""
        self.event_bus.subscribe(EventType.SIGNAL, handler)
        
    @property 
    def expected_input_type(self):
        """StrategyContainer expects INDICATORS events."""
        return EventType.INDICATORS
        
    def receive_event(self, event: Event):
        """Receive and process events from pipeline adapter."""
        logger.info(f"🎯 StrategyContainer received event: {event.event_type}")
        if event.event_type == EventType.INDICATORS:
            indicators = event.payload.get('indicators', {})
            market_data = event.payload.get('market_data', {})
            logger.info(f"📈 Processing INDICATORS event with {len(indicators)} symbols and {len(market_data)} market data entries")
            self.last_indicators = indicators
            
            # Store market data from the INDICATORS event
            if not hasattr(self, '_market_data'):
                self._market_data = {}
            self._market_data.update(market_data)
            logger.info(f"📊 Updated market data, now have {len(self._market_data)} symbols")
            
            # Process signals now that we have both indicators and market data
            asyncio.create_task(self._process_signals(event.timestamp))
        elif event.event_type == EventType.BAR:
            # Store market data
            logger.info(f"📊 StrategyContainer received BAR event for symbol: {event.payload.get('symbol')}")
            asyncio.create_task(self._handle_bar_event(event))
    
    async def _initialize_self(self) -> None:
        """Initialize strategy container."""
        if self.multi_strategy:
            await self._initialize_multi_strategy()
        else:
            await self._initialize_single_strategy()
    
    async def _initialize_single_strategy(self) -> None:
        """Initialize single strategy mode."""
        # Try to get strategy config from different locations
        strategy_config = {}
        
        # First, check if the config has the single strategy format
        if 'type' in self._metadata.config:
            strategy_config = self._metadata.config
        # Second, check strategies list
        elif self.strategies_config:
            strategy_config = self.strategies_config[0]
        # Third, check if it's wrapped in 'strategy' key
        elif 'strategy' in self._metadata.config:
            strategy_config = self._metadata.config['strategy']
        else:
            logger.warning("No strategy config found, using defaults")
        
        strategy_type = strategy_config.get('type', 'momentum')
        
        # Import and create strategy
        if strategy_type == 'momentum':
            from ..strategy.strategies.momentum import MomentumStrategy
            # Unpack parameters dictionary as keyword arguments
            strategy_params = strategy_config.get('parameters', {})
            self.strategy = MomentumStrategy(**strategy_params)
        elif strategy_type == 'mean_reversion':
            from ..strategy.strategies.mean_reversion import MeanReversionStrategy
            strategy_params = strategy_config.get('parameters', {})
            self.strategy = MeanReversionStrategy(**strategy_params)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        logger.info(f"StrategyContainer initialized with {strategy_type} strategy")
    
    async def _initialize_multi_strategy(self) -> None:
        """Initialize multiple strategies as sub-containers."""
        logger.info(f"Initializing {len(self.strategies_config)} strategies as sub-containers")
        
        # Initialize signal aggregator
        if WeightedVotingAggregator:
            aggregation_config = self._metadata.config.get('signal_aggregation', {})
            self.signal_aggregator = WeightedVotingAggregator(
                min_confidence=aggregation_config.get('min_confidence', 0.5)
            )
        
        # Create sub-containers for each strategy
        for i, strategy_config in enumerate(self.strategies_config):
            strategy_name = strategy_config.get('name', f'strategy_{i}')
            
            # Create sub-container config - ensure it's in single-strategy format
            sub_container_config = {
                'type': strategy_config.get('type', 'momentum'),
                'parameters': strategy_config.get('parameters', {}),
                'strategies': []  # Empty to force single-strategy mode
            }
            
            # Create sub-container
            sub_container = StrategyContainer(
                config=sub_container_config,
                container_id=f"{self.metadata.container_id}_{strategy_name}"
            )
            sub_container._metadata.name = f"SubStrategy_{strategy_name}"
            
            # Initialize the sub-container
            await sub_container._initialize_self()
            
            # Add as child - this will set up internal communication
            self.add_child_container(sub_container)
            
            logger.info(f"Created strategy sub-container: {strategy_name} with type: {strategy_config.get('type')}")
    
    async def _handle_bar_event(self, event: Event) -> None:
        """Handle market data events."""
        # Store market data for signal generation
        if not hasattr(self, '_market_data'):
            self._market_data = {}
        
        symbol = event.payload.get('symbol')
        if symbol:
            self._market_data[symbol] = event.payload.get('data', {})
    
    async def _process_signals(self, timestamp) -> None:
        """Process signals when we have both market data and indicators."""
        logger.info(f"🔍 _process_signals called with timestamp: {timestamp}")
        logger.info(f"📊 Market data available: {hasattr(self, '_market_data') and bool(self._market_data)}")
        logger.info(f"📈 Indicators available: {bool(self.last_indicators)}")
        logger.info(f"🏭 Strategy available: {self.strategy is not None}")
        
        if not hasattr(self, '_market_data') or not self._market_data:
            logger.warning("❌ No market data available for signal processing")
            return
            
        if self.multi_strategy:
            # Multi-strategy mode - directly call sub-containers and collect signals
            logger.info("🔀 Multi-strategy mode - processing with sub-containers")
            
            # Create INDICATORS event to process with sub-containers
            indicator_event = Event(
                event_type=EventType.INDICATORS,
                payload={
                    'indicators': self.last_indicators,
                    'market_data': self._market_data,
                    'timestamp': timestamp
                },
                timestamp=timestamp
            )
            
            # Collect all signals from sub-containers directly
            all_sub_signals = []
            
            # Process each sub-container directly
            for child in self.child_containers:
                if hasattr(child, 'receive_event') and hasattr(child, 'strategy'):
                    # Ensure sub-container has the indicator data
                    child.last_indicators = self.last_indicators
                    child._market_data = self._market_data
                    
                    # Call the sub-container's signal processing directly
                    if child.strategy and child.last_indicators:
                        logger.info(f"🚀 Processing signals with sub-strategy: {type(child.strategy).__name__}")
                        
                        # Create strategy input for the sub-container
                        strategy_input = {
                            'market_data': self._market_data,
                            'indicators': self.last_indicators,
                            'timestamp': timestamp
                        }
                        
                        # Generate signals from the sub-strategy
                        sub_signals = child.strategy.generate_signals(strategy_input)
                        
                        if sub_signals:
                            logger.info(f"📥 Got {len(sub_signals)} signals from {child.metadata.name}")
                            # Add metadata to track source
                            for signal in sub_signals:
                                if hasattr(signal, 'metadata'):
                                    signal.metadata['source_container'] = child.metadata.name
                            all_sub_signals.extend(sub_signals)
                        else:
                            logger.info(f"📭 No signals from {child.metadata.name}")
            
            # If we have signals, aggregate or pass through
            if all_sub_signals:
                if self.signal_aggregator and WeightedVotingAggregator:
                    # TODO: Implement proper signal aggregation
                    # For now, just pass all signals through
                    aggregated_signals = all_sub_signals
                    logger.info(f"🔄 Aggregated {len(aggregated_signals)} signals from sub-containers")
                else:
                    # No aggregator, just pass all signals
                    aggregated_signals = all_sub_signals
                    logger.info(f"📦 Passing through {len(aggregated_signals)} signals without aggregation")
                
                # Emit the aggregated signals
                await self._emit_signals(aggregated_signals, timestamp, self._market_data)
            else:
                logger.info("📭 No signals generated by any sub-container strategy")
            
            return
        
        # Single strategy mode
        if self.strategy and self.last_indicators:
            logger.info(f"🚀 Generating signals with strategy: {type(self.strategy).__name__}")
            logger.info(f"📊 Market data symbols: {list(self._market_data.keys())}")
            logger.info(f"📈 Indicator symbols: {list(self.last_indicators.keys())}")
            
            # Create strategy input dictionary that MomentumStrategy expects
            strategy_input = {
                'market_data': self._market_data,
                'indicators': self.last_indicators,
                'timestamp': timestamp
            }
            signals = self.strategy.generate_signals(strategy_input)
            
            logger.info(f"✨ Generated {len(signals) if signals else 0} signals")
            if signals:
                await self._emit_signals(signals, timestamp, self._market_data)
            else:
                logger.info("No signals generated at this timestamp")
        else:
            if not self.strategy:
                logger.error("❌ No strategy available")
            if not self.last_indicators:
                logger.error("❌ No indicators available")
    
    async def _emit_signals(self, signals: List, timestamp, market_data: Dict[str, Any]) -> None:
        """Emit signals via event bus."""
        if signals:
            signal_event = Event(
                event_type=EventType.SIGNAL,
                payload={
                    'timestamp': timestamp,
                    'signals': signals,
                    'market_data': market_data,
                    'source': self.metadata.container_id
                },
                timestamp=timestamp
            )
            
            logger.info(f"🚀 StrategyContainer publishing SIGNAL via event bus")
            self.event_bus.publish(signal_event)
            # Log each signal individually
            for signal in signals:
                log_signal_event(logger, signal)
    
    def get_capabilities(self) -> Set[str]:
        """Strategy container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("strategy.execution")
        capabilities.add("strategy.signal_generation")
        if self.multi_strategy:
            capabilities.add("strategy.multi_strategy")
            capabilities.add("strategy.signal_aggregation")
        return capabilities


class RiskContainer(BaseComposableContainer):
    """Container for risk management."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.RISK,
            name="RiskContainer",
            config=config,
            container_id=container_id
        )
        self.risk_manager = None
        self._cached_portfolio_state = None
        
    def on_output_event(self, handler):
        """Register handler for output events (used by pipeline adapter)."""
        self.event_bus.subscribe(EventType.ORDER, handler)
        
    @property
    def expected_input_type(self):
        """RiskContainer expects SIGNAL events."""
        return EventType.SIGNAL
        
    def receive_event(self, event: Event):
        """Receive and process events from pipeline adapter."""
        if event.event_type == EventType.SIGNAL:
            asyncio.create_task(self._handle_signal_event(event))
        elif event.event_type == EventType.FILL:
            asyncio.create_task(self._handle_fill_event(event))
        elif event.event_type == EventType.SYSTEM:
            asyncio.create_task(self._handle_system_event(event))
        elif event.event_type == EventType.PORTFOLIO:
            asyncio.create_task(self._handle_portfolio_event(event))
    
    async def _initialize_self(self) -> None:
        """Initialize risk management components (separate from portfolio)."""
        # Initialize position sizer
        position_sizers = self._metadata.config.get('position_sizers', [])
        if position_sizers:
            sizer_config = position_sizers[0]
            if sizer_config.get('type') == 'fixed':
                from ..risk.position_sizing import FixedPositionSizer
                size = sizer_config.get('size', '5000')
                self.position_sizer = FixedPositionSizer(
                    size=Decimal(str(size))
                )
            else:
                # Default position sizer
                from ..risk.position_sizing import FixedPositionSizer
                self.position_sizer = FixedPositionSizer(size=Decimal('5000'))
        else:
            # Default position sizer
            from ..risk.position_sizing import FixedPositionSizer
            self.position_sizer = FixedPositionSizer(size=Decimal('5000'))
        
        # Initialize risk limits
        self.risk_limits = []
        limits = self._metadata.config.get('limits', [])
        for limit_config in limits:
            if limit_config.get('type') == 'position':
                from ..risk.risk_limits import MaxPositionLimit
                max_position = limit_config.get('max_position', '5000')
                risk_limit = MaxPositionLimit(
                    max_position_value=Decimal(str(max_position))
                )
                self.risk_limits.append(risk_limit)
            elif limit_config.get('type') == 'exposure':
                from ..risk.risk_limits import MaxExposureLimit
                max_exposure_pct = limit_config.get('max_exposure_pct', '80')
                risk_limit = MaxExposureLimit(
                    max_exposure_pct=Decimal(str(max_exposure_pct))
                )
                self.risk_limits.append(risk_limit)
        
        logger.info("RiskContainer initialized with position sizing and risk limits")
    
    async def _handle_signal_event(self, event: Event) -> None:
        """Process signals through risk management."""
        signals = event.payload.get('signals', [])
        
        if not signals:
            return
        
        market_data = event.payload.get('market_data', {})
        approved_orders = []
        
        # Process each signal through risk management
        for signal in signals:
            # Apply risk limits
            if self._check_risk_limits(signal):
                # Generate order with position sizing
                order = self._create_order_from_signal(signal, market_data)
                if order:
                    approved_orders.append(order)
        
        if approved_orders:
            # Emit approved orders
            for order in approved_orders:
                order_event = Event(
                    event_type=EventType.ORDER,
                    payload={
                        'order': order,
                        'market_data': market_data,  # Include market data for execution
                        'source': self.metadata.container_id
                    },
                    timestamp=event.timestamp
                )
                
                self.event_bus.publish(order_event)
                log_order_event(logger, order)
                
            logger.info(f"RiskContainer approved and emitted {len(approved_orders)} orders")
    
    def _check_risk_limits(self, signal) -> bool:
        """Check if signal passes risk limits."""
        symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol', 'UNKNOWN')
        side = signal.side if hasattr(signal, 'side') else signal.get('side')
        
        # Get current portfolio state from PortfolioContainer via cross-container communication
        portfolio_state = self._get_portfolio_state()
        if not portfolio_state:
            logger.warning("No portfolio state available for risk checks")
            return False
        
        # Check each risk limit
        for risk_limit in self.risk_limits:
            limit_type = type(risk_limit).__name__
            
            if limit_type == 'MaxPositionLimit':
                # Check if we already have a position in this symbol
                positions = portfolio_state.get_all_positions()
                if symbol in positions:
                    position = positions[symbol]
                    current_value = abs(position.quantity * position.current_price) if hasattr(position, 'current_price') else 0
                    
                    # Check if adding to position would exceed limit
                    # For now, prevent any additional positions if we already have one
                    if current_value > 0:
                        # Check if this is a signal in the same direction (adding to position)
                        position_side = 'BUY' if position.quantity > 0 else 'SELL'
                        signal_side = str(side) if side else 'UNKNOWN'
                        
                        if position_side == signal_side or 'BUY' in signal_side and position_side == 'BUY' or 'SELL' in signal_side and position_side == 'SELL':
                            logger.warning(f"❌ Risk limit rejected signal for {symbol}: Already have position worth ${current_value:.2f}, not adding more")
                            return False
                    
                    if current_value >= float(risk_limit.max_position_value):
                        logger.warning(f"❌ Risk limit rejected signal for {symbol}: Position value ${current_value:.2f} >= limit ${risk_limit.max_position_value}")
                        return False
                        
            elif limit_type == 'MaxExposureLimit':
                # Check total exposure as percentage of capital
                cash_balance = portfolio_state.get_cash_balance()
                total_capital = portfolio_state.get_total_value()
                
                if total_capital > 0:
                    # Calculate current exposure
                    total_position_value = 0
                    for sym, pos in portfolio_state.get_all_positions().items():
                        if hasattr(pos, 'market_value'):
                            total_position_value += abs(float(pos.market_value))
                        elif hasattr(pos, 'quantity') and hasattr(pos, 'current_price'):
                            total_position_value += abs(float(pos.quantity * pos.current_price))
                    
                    current_exposure_pct = (total_position_value / float(total_capital)) * 100
                    
                    if current_exposure_pct >= float(risk_limit.max_exposure_pct):
                        logger.warning(f"❌ Risk limit rejected signal for {symbol}: Exposure {current_exposure_pct:.1f}% >= limit {risk_limit.max_exposure_pct}%")
                        return False
        
        # Check if we have enough cash for the trade
        cash_available = float(portfolio_state.get_cash_balance())
        
        # Estimate required cash for this trade
        estimated_trade_cost = 1000  # This matches our fixed position sizer
        
        if cash_available < estimated_trade_cost + 1000:  # Need trade cost + buffer
            logger.warning(f"❌ Risk limit rejected signal for {symbol}: Insufficient cash ${cash_available:.2f} < ${estimated_trade_cost + 1000}")
            return False
        
        logger.info(f"✅ Risk limits passed for {symbol} {side} signal")
        return True
    
    def _get_portfolio_state(self):
        """Get current portfolio state from cache or create temporary one."""
        if self._cached_portfolio_state:
            return self._cached_portfolio_state
            
        # If not cached yet, create a temporary one
        logger.warning("No cached portfolio state available yet, using temporary state")
        from ..risk.portfolio_state import PortfolioState
        return PortfolioState(initial_capital=100000)
    
    def _create_order_from_signal(self, signal, market_data: Dict[str, Any]):
        """Create order from signal using position sizer."""
        from ..execution.protocols import Order, OrderType
        import uuid
        from datetime import datetime
        
        # Get position size from position sizer
        symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol', 'UNKNOWN')
        side = signal.side if hasattr(signal, 'side') else signal.get('side')
        
        # Use position sizer to determine quantity
        current_price = None
        if 'prices' in market_data:
            current_price = market_data['prices'].get(symbol)
        elif symbol in market_data:
            current_price = market_data[symbol].get('close')
        
        if not current_price:
            current_price = 100.0  # Default price
        
        # Calculate position size using the correct method name
        # Get actual portfolio state
        portfolio_state = self._get_portfolio_state()
        
        # Convert signal to proper Signal object if needed
        if hasattr(signal, 'symbol'):
            # Already a proper Signal object
            signal_obj = signal
        else:
            # Convert dict to Signal object
            from ..risk.protocols import Signal, SignalType
            from ..execution.protocols import OrderSide
            signal_obj = Signal(
                signal_id=f"signal_{symbol}_{datetime.now().timestamp()}",
                strategy_id="momentum_strategy",
                symbol=symbol,
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                strength=signal.get('strength', 1.0),
                timestamp=datetime.now(),
                metadata={}
            )
        
        # Check if this is a close/exit signal
        from ..risk.protocols import SignalType
        if hasattr(signal_obj, 'signal_type') and signal_obj.signal_type == SignalType.EXIT:
            # For close orders, use the actual position quantity
            positions = portfolio_state.get_all_positions()
            if symbol in positions:
                position = positions[symbol]
                quantity = abs(int(position.quantity))  # Use actual position quantity
                logger.info(f"🔄 EXIT signal: Using actual position quantity {quantity} for {symbol}")
            else:
                logger.warning(f"⚠️ EXIT signal for {symbol} but no position found")
                quantity = 0
        else:
            # For entry signals, use position sizer
            # But first check if we already have a position
            positions = portfolio_state.get_all_positions()
            if symbol in positions and positions[symbol].quantity != 0:
                # We already have a position, check if this is a reversal signal
                existing_side = 'BUY' if positions[symbol].quantity > 0 else 'SELL'
                new_side = str(side) if side else str(signal_obj.side)
                
                if ('BUY' in existing_side and 'SELL' in new_side) or ('SELL' in existing_side and 'BUY' in new_side):
                    # This is a reversal signal, close existing position first
                    logger.info(f"🔄 Reversal signal for {symbol}: closing existing {existing_side} position")
                    quantity = 0  # Risk limit will handle whether to allow new position
                else:
                    # Same direction, don't add to position
                    logger.info(f"📊 Already have {existing_side} position in {symbol}, ignoring {new_side} signal")
                    quantity = 0
            else:
                # No existing position, calculate new position size
                # Prepare market data in expected format
                market_data_formatted = {
                    'prices': {symbol: current_price}
                }
                
                position_value = self.position_sizer.calculate_size(
                    signal=signal_obj,
                    portfolio_state=portfolio_state,
                    market_data=market_data_formatted
                )
                
                # Convert dollar amount to shares
                quantity = position_value / Decimal(str(current_price))
            # Round down to whole shares
            quantity = int(quantity)
        
        # Don't create order if quantity is 0
        if quantity == 0:
            logger.info(f"RiskContainer skipping order creation for {symbol} - quantity is 0")
            return None
        
        # Create order
        order = Order(
            order_id=f"ORD-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=None,  # Market order
            created_at=datetime.now()
        )
        
        logger.info(f"RiskContainer created order: {order.symbol} {order.side} {order.quantity}")
        return order
    
    async def _handle_fill_event(self, event: Event) -> None:
        """RiskContainer doesn't handle fills - they should go to PortfolioContainer."""
        logger.info("🔄 RiskContainer received FILL event - forwarding to PortfolioContainer")
        # Fills should be handled by PortfolioContainer, not RiskContainer
        # This is just logging for debugging
        pass
    
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system events - RiskContainer doesn't need END_OF_DATA."""
        message = event.payload.get('message')
        if message == 'END_OF_DATA':
            logger.info("🏁 RiskContainer received END_OF_DATA event - ignoring (ExecutionContainer handles position closing)")
        else:
            logger.info(f"📋 RiskContainer received system event: {message}")
    
    async def _handle_portfolio_event(self, event: Event) -> None:
        """Handle portfolio update events to cache the latest state."""
        portfolio_state = event.payload.get('portfolio_state')
        if portfolio_state:
            self._cached_portfolio_state = portfolio_state
            logger.info("📊 RiskContainer cached updated portfolio state")
            
            # Log some basic info about the portfolio
            cash = portfolio_state.get_cash_balance()
            positions = portfolio_state.get_all_positions()
            logger.info(f"   💰 Cash: ${cash:.2f}, Positions: {len(positions)}")
            for symbol, pos in positions.items():
                logger.info(f"   📍 {symbol}: {pos.quantity:.2f} shares")
    
    def get_capabilities(self) -> Set[str]:
        """Risk container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("risk.management")
        capabilities.add("risk.position_sizing")
        capabilities.add("risk.portfolio_tracking")
        return capabilities


class PortfolioContainer(BaseComposableContainer):
    """Container for portfolio tracking and management."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.PORTFOLIO,
            name="PortfolioContainer",
            config=config,
            container_id=container_id
        )
        self.portfolio_state = None
        
    def on_output_event(self, handler):
        """Register handler for output events (used by pipeline adapter)."""
        # Portfolio only outputs PORTFOLIO events for state synchronization
        self.event_bus.subscribe(EventType.PORTFOLIO, handler)
        
    @property
    def expected_input_type(self):
        """PortfolioContainer expects FILL events (via reverse routing)."""
        return EventType.FILL
        
    def receive_event(self, event: Event):
        """Receive and process events from pipeline adapter."""
        if event.event_type == EventType.FILL:
            asyncio.create_task(self._handle_fill_event(event))
        elif event.event_type == EventType.SYSTEM:
            asyncio.create_task(self._handle_system_event(event))
        elif event.event_type == EventType.BAR:
            # Track market data for position pricing
            self._update_market_data(event)
    
    async def _initialize_self(self) -> None:
        """Initialize portfolio tracking."""
        from ..risk.portfolio_state import PortfolioState
        
        # Initialize market data tracking
        self._last_market_data = {}
        
        initial_capital = self._metadata.config.get('initial_capital', '100000')
        if isinstance(initial_capital, str):
            initial_capital = Decimal(initial_capital)
        self.portfolio_state = PortfolioState(initial_capital=initial_capital)
        
        logger.info(f"PortfolioContainer initialized with capital: ${initial_capital}")
        
        # Register this container for position closing
        global _PORTFOLIO_CONTAINER_REGISTRY
        _PORTFOLIO_CONTAINER_REGISTRY = self
        logger.info("📋 PortfolioContainer registered for position closing")
        
        # Publish initial portfolio state
        await self._publish_portfolio_update()
    
    async def _handle_fill_event(self, event: Event) -> None:
        """Update portfolio state with fills."""
        fill = event.payload.get('fill')
        if not fill or not self.portfolio_state:
            logger.warning("PortfolioContainer: No fill or portfolio state available for update")
            return
        
        try:
            # Determine quantity delta (positive for buys, negative for sells)
            # Ensure all fill values are Decimal for consistency
            quantity_delta = Decimal(str(fill.quantity))
            
            # Handle different fill.side types robustly
            if hasattr(fill, 'side'):
                side_value = fill.side
                
                # Handle OrderSide enum with .value attribute
                if hasattr(side_value, 'value'):
                    # OrderSide enum values: BUY=1, SELL=-1
                    if side_value.value == 1:
                        side = 'BUY'
                    elif side_value.value == -1:
                        side = 'SELL'
                    else:
                        # Fallback: check the name
                        side = str(side_value.name).upper() if hasattr(side_value, 'name') else str(side_value).upper()
                # Handle integer side values (1=BUY, -1=SELL)
                elif isinstance(side_value, int):
                    side = 'BUY' if side_value == 1 else 'SELL'
                # Handle string side values
                elif isinstance(side_value, str):
                    side = side_value.upper()
                # Handle any other type by converting to string
                else:
                    side = str(side_value).upper()
                    
                logger.info(f"🔍 Converted fill.side {side_value} (type: {type(side_value)}) to '{side}'")
            else:
                side = 'BUY'  # Default
                logger.warning("Fill object has no 'side' attribute, defaulting to BUY")
            
            logger.debug(f"🔍 Before adjustment: side={side}, quantity_delta={quantity_delta}")
            if side == 'SELL':
                quantity_delta = -quantity_delta
            logger.debug(f"🔍 After adjustment: quantity_delta={quantity_delta}")
            
            # Calculate commission and include it in the effective price
            commission = Decimal(str(getattr(fill, 'commission', 0)))
            
            # For cash flow calculation, we need to include commission
            # BUY: pay price + commission per share
            # SELL: receive price - commission per share  
            effective_price = Decimal(str(fill.price))
            if side == 'BUY':
                # When buying, commission increases the effective cost per share
                effective_price += commission / Decimal(str(fill.quantity))
            else:
                # When selling, commission reduces the effective proceeds per share
                effective_price -= commission / Decimal(str(fill.quantity))
            
            # Update portfolio position with effective price that includes commission impact
            position = self.portfolio_state.update_position(
                symbol=fill.symbol,
                quantity_delta=quantity_delta,
                price=effective_price,
                timestamp=fill.executed_at
            )
            
            # Calculate cash change for logging (should match the actual cash flow from effective price)
            effective_trade_value = Decimal(str(fill.quantity)) * effective_price
            cash_change = -effective_trade_value if side == 'BUY' else effective_trade_value
            
            # Commission is now properly included in the effective price
            
            logger.info(f"💼 PortfolioContainer updated: {fill.symbol} {side} {fill.quantity} @ {fill.price}")
            logger.info(f"   📈 Position: {position.quantity if position else 'None'} (quantity_delta was {quantity_delta})")
            logger.info(f"   💰 Cash change: {cash_change:.2f}, New balance: {self.portfolio_state._cash_balance:.2f}")
            
            # Publish updated portfolio state
            await self._publish_portfolio_update()
            
        except Exception as e:
            logger.error(f"Error updating portfolio with fill: {e}")
            logger.error(f"Fill details: symbol={getattr(fill, 'symbol', 'N/A')}, quantity={getattr(fill, 'quantity', 'N/A')}, price={getattr(fill, 'price', 'N/A')}")
    
    async def _publish_portfolio_update(self) -> None:
        """Publish portfolio state update for other containers."""
        if self.portfolio_state:
            portfolio_event = Event(
                event_type=EventType.PORTFOLIO,
                payload={
                    'portfolio_state': self.portfolio_state,
                    'cash_balance': float(self.portfolio_state.get_cash_balance()),
                    'total_value': float(self.portfolio_state.get_total_value()),
                    'positions': len(self.portfolio_state.get_all_positions()),
                    'timestamp': datetime.now()
                },
                timestamp=datetime.now()
            )
            
            # Publish to event bus
            self.event_bus.publish(portfolio_event)
            logger.info("📊 PortfolioContainer published PORTFOLIO update")
    
    def _update_market_data(self, event: Event):
        """Update last known market prices from BAR events."""
        if event.payload:
            symbol = event.payload.get('symbol')
            data = event.payload.get('data', {})
            if symbol and data:
                self._last_market_data[symbol] = data
                
                # Also update position current prices if we have positions
                if self.portfolio_state:
                    positions = self.portfolio_state.get_all_positions()
                    if symbol in positions:
                        price = data.get('close', data.get('price'))
                        if price:
                            positions[symbol].current_price = Decimal(str(price))
    
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system events."""
        message = event.payload.get('message')
        
        if message == 'END_OF_DATA':
            logger.info("🏁 PortfolioContainer received END_OF_DATA event")
            # PortfolioContainer doesn't handle position closing anymore
            # Position closing is handled by ExecutionContainer via proper order execution
        else:
            logger.info(f"📋 PortfolioContainer received system event: {event.payload}")
    
    
    def get_capabilities(self) -> Set[str]:
        """Portfolio container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("portfolio.tracking")
        capabilities.add("portfolio.performance")
        return capabilities


class ExecutionContainer(BaseComposableContainer):
    """Container for order execution and fill generation."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.EXECUTION,
            name="ExecutionContainer",
            config=config,
            container_id=container_id
        )
        self.execution_engine = None
        self.broker = None
        
    def on_output_event(self, handler):
        """Register handler for output events (used by pipeline adapter)."""
        self.event_bus.subscribe(EventType.FILL, handler)
        # Note: SYSTEM events for position closing use direct container communication
        
    @property
    def expected_input_type(self):
        """ExecutionContainer expects ORDER events."""
        return EventType.ORDER
        
    def receive_event(self, event: Event):
        """Receive and process events from pipeline adapter."""
        if event.event_type == EventType.ORDER:
            asyncio.create_task(self._handle_order_event(event))
        elif event.event_type == EventType.SYSTEM:
            asyncio.create_task(self._handle_system_event(event))
    
    async def _initialize_self(self) -> None:
        """Initialize execution engine."""
        try:
            from ..execution.backtest_engine import UnifiedBacktestEngine
            from ..execution.backtest_broker_refactored import BacktestBrokerRefactored
            # Create market simulator using the regular MarketSimulator with deterministic fills
            from ..execution.market_simulation import MarketSimulator, FixedSlippageModel, PerShareCommissionModel
            
            # Use explicit defaults to avoid conversion issues
            slippage_bps = 5
            commission = 0.005  
            fill_probability = 1.0  # Deterministic fills for testing
            initial_capital = 100000
            
            slippage_model = FixedSlippageModel(
                slippage_percent=slippage_bps / 10000  # Convert basis points to decimal
            )
            commission_model = PerShareCommissionModel(
                commission_per_share=commission
            )
            
            market_sim = MarketSimulator(
                slippage_model=slippage_model,
                commission_model=commission_model,
                fill_probability=fill_probability,
                partial_fill_enabled=False  # Disable partial fills for deterministic results
            )
            
            # Create a simple portfolio state for the broker
            from ..risk.portfolio_state import PortfolioState
            
            portfolio_state = PortfolioState(
                initial_capital=Decimal(str(initial_capital))
            )
            
            # Create broker
            self.broker = BacktestBrokerRefactored(
                portfolio_state=portfolio_state,
                market_simulator=market_sim
            )
            
            # Create execution engine directly instead of using UnifiedBacktestEngine
            # This ensures our ImprovedMarketSimulator with fill_probability=1.0 is used
            from ..execution.execution_engine import DefaultExecutionEngine
            from ..execution.order_manager import OrderManager
            
            order_manager = OrderManager()
            
            self.execution_engine = DefaultExecutionEngine(
                broker=self.broker,
                order_manager=order_manager,
                market_simulator=market_sim  # Use our configured market simulator
            )
            
            logger.info("ExecutionContainer initialized with backtest engine")
            
        except Exception as e:
            logger.error(f"Error initializing ExecutionContainer: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    async def _handle_order_event(self, event: Event) -> None:
        """Process orders through execution engine."""
        order = event.payload.get('order')
        if not order:
            return
        
        # Extract market data from event
        market_data = event.payload.get('market_data', {})
        
        # Update execution engine's market data if available
        if market_data and hasattr(order, 'symbol') and order.symbol in market_data:
            symbol_data = market_data[order.symbol]
            if hasattr(self.execution_engine, '_market_data'):
                # Update the execution engine's market data
                self.execution_engine._market_data[order.symbol] = {
                    'price': symbol_data.get('close', 100.0),
                    'close': symbol_data.get('close', 100.0),
                    'open': symbol_data.get('open', 100.0),
                    'high': symbol_data.get('high', 100.0),
                    'low': symbol_data.get('low', 100.0),
                    'volume': symbol_data.get('volume', 1000000),
                    'bid': symbol_data.get('close', 100.0) - 0.01,
                    'ask': symbol_data.get('close', 100.0) + 0.01,
                    'timestamp': event.timestamp
                }
        
        # Execute order through the execution engine
        fill = self.execution_engine.execute_order(order)
        
        if fill:
            # Emit fill event
            fill_event = Event(
                event_type=EventType.FILL,
                payload={
                    'fill': fill,
                    'order': order,
                    'source': self.metadata.container_id
                },
                timestamp=fill.executed_at
            )
            
            self.event_bus.publish(fill_event)
            log_fill_event(logger, fill)
            
            logger.info(f"ExecutionContainer generated fill for order {order.order_id}")
    
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system events like END_OF_DATA, END_OF_DAY, etc."""
        message = event.payload.get('message')
        close_time = event.payload.get('close_time')  # For scheduled closes
        
        if message in ['END_OF_DATA', 'END_OF_DAY', 'FORCE_CLOSE']:
            logger.info(f"🏁 ExecutionContainer received {message} - closing all positions directly")
            
            # Extract market data if available
            market_data = event.payload.get('last_market_data', {})
            if market_data:
                logger.info(f"📊 Received market data for {list(market_data.keys())} symbols")
            
            await self._close_all_positions_directly(reason=message, market_data=market_data)
        elif message == 'SCHEDULED_CLOSE' and close_time:
            logger.info(f"⏰ ExecutionContainer received SCHEDULED_CLOSE for {close_time}")
            await self._close_all_positions_directly(reason=f"scheduled_close_{close_time}")
        else:
            logger.info(f"🏁 ExecutionContainer received {message} event")
            # ExecutionContainer can continue processing any remaining orders
    
    async def _close_all_positions_directly(self, reason: str = "system_close", market_data: Dict[str, Any] = None) -> None:
        """Generate close orders for all positions through the normal execution engine."""
        logger.info(f"🏁 Closing all positions via execution engine for {reason}")
        
        # Get current positions from registered PortfolioContainer
        global _PORTFOLIO_CONTAINER_REGISTRY
        portfolio_container = _PORTFOLIO_CONTAINER_REGISTRY
        
        if not portfolio_container or not hasattr(portfolio_container, 'portfolio_state'):
            logger.warning("⚠️ Could not find registered PortfolioContainer for position closing")
            return
            
        if not portfolio_container.portfolio_state:
            logger.warning("⚠️ No portfolio state available for position closing")
            return
            
        # Get all current positions
        positions = portfolio_container.portfolio_state.get_all_positions()
        
        if not positions:
            logger.info("📊 No open positions to close")
            return
            
        logger.info(f"📊 Found {len(positions)} positions to close via execution engine")
        
        # Update market data in execution engine if available
        if market_data and hasattr(self.execution_engine, '_market_data'):
            for symbol, data in market_data.items():
                self.execution_engine._market_data[symbol] = {
                    'price': data.get('close', data.get('price', 100.0)),
                    'close': data.get('close', data.get('price', 100.0)),
                    'open': data.get('open', data.get('price', 100.0)),
                    'high': data.get('high', data.get('price', 100.0)),
                    'low': data.get('low', data.get('price', 100.0)),
                    'volume': data.get('volume', 1000000),
                    'bid': data.get('close', data.get('price', 100.0)) - 0.01,
                    'ask': data.get('close', data.get('price', 100.0)) + 0.01,
                    'timestamp': datetime.now()
                }
                logger.info(f"📊 Updated execution engine market data for {len(market_data)} symbols")
        
        # Generate close orders for each position
        close_orders = []
        for symbol, position in positions.items():
            if hasattr(position, 'quantity') and position.quantity != 0:
                # Determine close side (opposite of current position)
                current_qty = position.quantity
                close_side = 'SELL' if current_qty > 0 else 'BUY'
                close_qty = abs(float(current_qty))
                
                # Create close order
                close_order = self._create_close_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=close_qty,
                    reason=reason
                )
                
                if close_order:
                    close_orders.append(close_order)
                    logger.info(f"📋 Created close order: {symbol} {close_side} {close_qty} shares")
        
        # Execute all close orders through the normal execution engine
        logger.info(f"🚀 Executing {len(close_orders)} close orders through execution engine")
        
        for order in close_orders:
            try:
                # Execute order through the execution engine (with slippage and commission)
                fill = self.execution_engine.execute_order(order)
                
                if fill:
                    # Emit fill event for portfolio container to process
                    fill_event = Event(
                        event_type=EventType.FILL,
                        payload={
                            'fill': fill,
                            'order': order,
                            'source': self.metadata.container_id,
                            'close_reason': reason
                        },
                        timestamp=fill.executed_at
                    )
                    
                    # Publish fill event via event bus (pipeline adapter will route to portfolio)
                    self.event_bus.publish(fill_event)
                    log_fill_event(logger, fill)
                    
                    logger.info(f"✅ Executed close order: {order.symbol} {order.side} {order.quantity} @ ${fill.price:.4f}")
                    logger.info(f"   💸 Commission: ${fill.commission:.2f}, Slippage: ${fill.price - (market_data.get(order.symbol, {}).get('close', fill.price)):.4f}")
                else:
                    logger.warning(f"⚠️ Close order execution failed for {order.symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error executing close order for {order.symbol}: {e}")
        
        # Give time for fill events to be processed
        await asyncio.sleep(0.1)
        
        logger.info(f"🎯 Position closing via execution engine completed for {reason}")
    
    
    def _create_close_order(self, symbol: str, side: str, quantity: float, reason: str):
        """Create a market order to close a position."""
        from ..execution.protocols import Order, OrderType, OrderSide
        import uuid
        
        try:
            # Convert string side to OrderSide enum
            if isinstance(side, str):
                side_enum = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
            else:
                side_enum = side
            
            order = Order(
                order_id=f"CLOSE-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side_enum,  # Use enum instead of string
                order_type=OrderType.MARKET,
                quantity=quantity,  # Keep fractional shares for close orders
                price=None,  # Market order
                created_at=datetime.now(),
                metadata={'close_reason': reason}
            )
            return order
        except Exception as e:
            logger.error(f"Error creating close order for {symbol}: {e}")
            return None
    

    
    def get_capabilities(self) -> Set[str]:
        """Execution container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("execution.backtest")
        capabilities.add("execution.order_management")
        capabilities.add("execution.fill_generation")
        return capabilities


def register_execution_containers():
    """Register all execution containers with the composition engine."""
    from ..core.containers.composition_engine import get_global_registry
    from ..core.containers.composable import ContainerRole
    
    registry = get_global_registry()
    
    # Register container types with capabilities
    registry.register_container_type(
        ContainerRole.BACKTEST, 
        BacktestContainer,
        capabilities={'backtest.coordination', 'backtest.peer_management'}
    )
    registry.register_container_type(
        ContainerRole.DATA, 
        DataContainer,
        capabilities={'data.streaming', 'data.historical', 'data.csv'}
    )
    registry.register_container_type(
        ContainerRole.INDICATOR, 
        IndicatorContainer,
        capabilities={'indicators.calculation', 'indicators.streaming', 'indicator.computation'}
    )
    registry.register_container_type(
        ContainerRole.STRATEGY, 
        StrategyContainer,
        capabilities={'strategy.execution', 'strategy.signal_generation'}
    )
    registry.register_container_type(
        ContainerRole.EXECUTION, 
        ExecutionContainer,
        capabilities={'execution.backtest', 'execution.order_management', 'execution.fill_generation'}
    )
    registry.register_container_type(
        ContainerRole.RISK, 
        RiskContainer,
        capabilities={'risk.management', 'risk.position_sizing', 'risk.portfolio_tracking'}
    )
    registry.register_container_type(
        ContainerRole.PORTFOLIO, 
        PortfolioContainer,
        capabilities={'portfolio.tracking', 'portfolio.performance'}
    )
    
    # Skip classifier container registration for now - it needs refactoring to inherit from BaseComposableContainer
    # to work properly with the composition engine
    
    logger.info("Execution containers registered with composition engine")