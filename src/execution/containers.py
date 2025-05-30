"""
Execution-specific container implementations using the composable container protocol.

This demonstrates how different container types can be implemented while
maintaining the protocol-based composable architecture.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import asyncio
import logging

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
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        logger.info(f"DataContainer initialized with {data_source} data source")
    
    async def start(self) -> None:
        """Start data streaming."""
        await super().start()
        
        # Start data streaming task
        self._streaming_task = asyncio.create_task(self._stream_data())
        logger.info("Data streaming started")
    
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
        """Stream data to child containers using proper data models."""
        try:
            from ..data.streamers import StreamedBar
            from ..data.models import Bar
            
            # Load data using existing data loader (backwards compatible)
            symbols = self._metadata.config.get('symbols', ['SPY'])
            
            for symbol in symbols:
                # Load data - use specific file if provided, otherwise by symbol
                if hasattr(self, '_specific_file') and self._specific_file:
                    # Load the specific file directly
                    import pandas as pd
                    file_path = self.data_loader.data_dir / self._specific_file
                    df = pd.read_csv(file_path)
                    # Normalize columns using the loader's method
                    df = self.data_loader._normalize_columns(df)
                    # Parse dates using the loader's method
                    df = self.data_loader._parse_dates(df)
                    
                    # Apply max_bars limit if specified
                    max_bars = self._metadata.config.get('max_bars')
                    if max_bars and len(df) > max_bars:
                        df = df.head(max_bars)
                        logger.info(f"Limited data to first {max_bars} bars")
                else:
                    # Use normal symbol-based loading
                    df = self.data_loader.load(symbol)
                    
                    # Apply max_bars limit if specified
                    max_bars = self._metadata.config.get('max_bars')
                    if max_bars and df is not None and len(df) > max_bars:
                        df = df.head(max_bars)
                        logger.info(f"Limited data to first {max_bars} bars")
                
                if df is not None and len(df) > 0:
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                    
                    # Stream data bar by bar using proper data models
                    bar_count = 0
                    for timestamp, row in df.iterrows():
                        bar_count += 1
                        if bar_count <= 5 or bar_count % 10 == 0:  # Log first 5 bars and every 10th bar
                            log_bar_event(logger, symbol, timestamp, row['close'], bar_count)
                        
                        # Create StreamedBar object (proper data model)
                        streamed_bar = StreamedBar(
                            timestamp=timestamp,
                            symbol=symbol,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=float(row.get('volume', 0))
                        )
                        
                        # Convert to validated Bar object
                        bar_object = streamed_bar.to_bar()
                        
                        # Create market data dict for backwards compatibility
                        market_data = {
                            symbol: {
                                'open': bar_object.open,
                                'high': bar_object.high,
                                'low': bar_object.low,
                                'close': bar_object.close,
                                'volume': bar_object.volume
                            }
                        }
                        
                        event = Event(
                            event_type=EventType.BAR,
                            payload={
                                'timestamp': timestamp,
                                'market_data': market_data,
                                'bar_objects': {symbol: bar_object}  # Include validated Bar objects
                            },
                            timestamp=timestamp
                        )
                        
                        # Publish to children
                        self.publish_event(event, target_scope="children")
                        
                        # Yield control to allow other tasks to run
                        await asyncio.sleep(0.001)  # Small delay to simulate streaming
                        
                        # Check if container is still running
                        if self.state != ContainerState.RUNNING:
                            break
                            
        except Exception as e:
            logger.error(f"Error streaming data: {e}")
    
    def get_capabilities(self) -> Set[str]:
        """Data container capabilities."""
        capabilities = super().get_capabilities()
        data_source = self._metadata.config.get('source', 'historical')
        capabilities.add(f"data.{data_source}")
        return capabilities


class IndicatorContainer(BaseComposableContainer):
    """Container for shared indicator computation."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.INDICATOR,
            name="IndicatorContainer", 
            config=config,
            container_id=container_id
        )
        self.indicator_hub = None
        self._subscriptions: Dict[str, Set[str]] = {}  # container_id -> required_indicators
        self._indicators_discovered = False  # Flag to track if indicators have been discovered
        
    async def _initialize_self(self) -> None:
        """Initialize indicator hub with indicators inferred from configuration."""
        from ..strategy.components.indicator_hub import IndicatorHub
        from ..strategy.components.indicator_hub import IndicatorConfig, IndicatorType
        
        # Get inferred indicators from configuration  
        required_indicator_names = self._metadata.config.get('required_indicators', [])
        
        if required_indicator_names:
            logger.info(f"Initializing IndicatorContainer with config-inferred indicators: {required_indicator_names}")
            
            # Convert indicator names to IndicatorConfig objects
            indicator_configs = []
            for indicator_name in required_indicator_names:
                config = self._create_indicator_config(indicator_name)
                indicator_configs.append(config)
                logger.info(f"Created indicator config: {indicator_name} ({config.indicator_type})")
            
            # Initialize with inferred indicators
            self.indicator_hub = IndicatorHub(
                indicators=indicator_configs,
                cache_size=self._metadata.config.get('cache_size', 1000)
            )
            
            # For config-driven indicators, we need to set up subscriptions manually
            # since we're not doing runtime discovery. Find all strategy containers
            # that need these indicators.
            await self._setup_config_driven_subscriptions(set(required_indicator_names))
            
            logger.info(f"IndicatorContainer initialized with {len(indicator_configs)} inferred indicators")
        else:
            # Fallback: try runtime discovery for backward compatibility
            self.indicator_hub = IndicatorHub(
                indicators=[],
                cache_size=self._metadata.config.get('cache_size', 1000)
            )
            logger.info(f"IndicatorContainer initialized with no indicators - will try runtime discovery")
    
    def _create_indicator_config(self, indicator_name: str):
        """Create IndicatorConfig from indicator name."""
        from ..strategy.components.indicator_hub import IndicatorConfig, IndicatorType
        
        # Parse indicator name to determine type and parameters
        base_name = indicator_name.split('_')[0]
        
        if base_name in ['SMA', 'EMA']:
            indicator_type = IndicatorType.TREND
            # Extract period from name (e.g., SMA_20 -> period=20)
            try:
                period = int(indicator_name.split('_')[1]) if '_' in indicator_name else 20
            except (IndexError, ValueError):
                period = 20
            params = {'period': period}
        elif base_name == 'RSI':
            indicator_type = IndicatorType.MOMENTUM
            # Extract period if specified (e.g., RSI_14 -> period=14)
            try:
                period = int(indicator_name.split('_')[1]) if '_' in indicator_name else 14
            except (IndexError, ValueError):
                period = 14
            params = {'period': period}
        elif base_name == 'MACD':
            indicator_type = IndicatorType.MOMENTUM
            params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        elif base_name == 'BB':
            indicator_type = IndicatorType.VOLATILITY
            params = {'period': 20, 'std_dev': 2}
        elif base_name == 'ATR':
            indicator_type = IndicatorType.VOLATILITY
            params = {'period': 14}
        else:
            indicator_type = IndicatorType.CUSTOM
            params = {}
        
        return IndicatorConfig(
            name=indicator_name,
            indicator_type=indicator_type,
            parameters=params
        )
    
    async def _setup_config_driven_subscriptions(self, required_indicators: Set[str]) -> None:
        """Set up subscriptions for config-driven indicators by finding strategy containers."""
        # Wait a moment for container hierarchy to be fully built
        await asyncio.sleep(0.1)
        
        # Get root container and scan for StrategyContainers
        root_container = self
        while hasattr(root_container, '_parent_container') and root_container._parent_container:
            root_container = root_container._parent_container
        
        # Find all StrategyContainers that need indicators
        strategy_containers = []
        for container in self._get_all_nested_containers_from_root(root_container):
            if container.metadata.role.value == 'strategy':
                strategy_containers.append(container)
        
        # Subscribe each strategy container to all inferred indicators
        for strategy_container in strategy_containers:
            container_id = strategy_container.metadata.container_id
            self._subscriptions[container_id] = required_indicators
            logger.info(f"Config-driven subscription: {strategy_container.metadata.name} ({container_id}) → {required_indicators}")
        
        if not strategy_containers:
            logger.warning("No StrategyContainers found for config-driven indicator subscriptions")
    
    def _discover_required_indicators(self) -> Set[str]:
        """Discover indicators needed by all containers in the system."""
        required_indicators = set()
        
        # Get root container (DataContainer) and scan ALL nested containers
        root_container = self
        while hasattr(root_container, '_parent_container') and root_container._parent_container:
            root_container = root_container._parent_container
        
        # Scan all nested containers from root for indicator requirements
        for container in self._get_all_nested_containers_from_root(root_container):
            if hasattr(container, 'get_required_indicators'):
                indicators = container.get_required_indicators()
                required_indicators.update(indicators)
                self._subscriptions[container.metadata.container_id] = indicators
                logger.info(f"Found indicators from {container.metadata.name}: {indicators}")
        
        return required_indicators
    
    def _get_all_nested_containers(self) -> List[ComposableContainerProtocol]:
        """Get all nested containers recursively."""
        containers = []
        
        def collect_containers(container):
            containers.append(container)
            for child in container.child_containers:
                collect_containers(child)
        
        for child in self.child_containers:
            collect_containers(child)
        
        return containers
    
    def _get_all_nested_containers_from_root(self, root_container) -> List[ComposableContainerProtocol]:
        """Get all nested containers recursively from root container."""
        containers = []
        
        def collect_containers(container):
            containers.append(container)
            for child in container.child_containers:
                collect_containers(child)
        
        # Start from root and collect all containers
        collect_containers(root_container)
        
        return containers
    
    def _find_container_by_id(self, container_id: str) -> Optional[ComposableContainerProtocol]:
        """Find a container by ID in the entire container tree."""
        # Get root container
        root_container = self
        while hasattr(root_container, '_parent_container') and root_container._parent_container:
            root_container = root_container._parent_container
        
        # Search all containers from root
        for container in self._get_all_nested_containers_from_root(root_container):
            if container.metadata.container_id == container_id:
                return container
        
        return None
    
    async def _discover_and_initialize_indicators(self) -> None:
        """Discover required indicators from child containers and initialize them."""
        logger.info("Discovering indicators from child containers...")
        
        # Discover required indicators from child containers
        required_indicator_names = self._discover_required_indicators()
        
        if not required_indicator_names:
            logger.info("No indicators required by child containers")
            return
        
        logger.info(f"Discovered {len(required_indicator_names)} required indicators: {required_indicator_names}")
        
        # Convert to IndicatorConfig objects
        from ..strategy.components.indicator_hub import IndicatorConfig, IndicatorType
        
        indicator_configs = []
        for indicator_name in required_indicator_names:
            # Parse indicator name to determine type and parameters
            base_name = indicator_name.split('_')[0]
            
            if base_name in ['SMA', 'EMA']:
                indicator_type = IndicatorType.TREND
                # Extract period from name (e.g., SMA_20 -> period=20)
                try:
                    period = int(indicator_name.split('_')[1]) if '_' in indicator_name else 20
                except (IndexError, ValueError):
                    period = 20
                params = {'period': period}
            elif base_name == 'RSI':
                indicator_type = IndicatorType.MOMENTUM
                params = {'period': 14}
            elif base_name == 'MACD':
                indicator_type = IndicatorType.MOMENTUM
                params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            elif base_name == 'BB':
                indicator_type = IndicatorType.VOLATILITY
                params = {'period': 20, 'std_dev': 2}
            elif base_name == 'ATR':
                indicator_type = IndicatorType.VOLATILITY
                params = {'period': 14}
            else:
                indicator_type = IndicatorType.CUSTOM
                params = {}
            
            config = IndicatorConfig(
                name=indicator_name,
                indicator_type=indicator_type,
                parameters=params
            )
            indicator_configs.append(config)
            logger.info(f"Created indicator config: {indicator_name} ({indicator_type})")
        
        # Reinitialize indicator hub with discovered indicators
        from ..strategy.components.indicator_hub import IndicatorHub
        self.indicator_hub = IndicatorHub(
            indicators=indicator_configs,
            cache_size=self._metadata.config.get('cache_size', 1000)
        )
        
        logger.info(f"IndicatorHub reinitialized with {len(indicator_configs)} indicators")
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process market data and compute indicators."""
        await super().process_event(event)
        
        if event.event_type == EventType.BAR:
            logger.debug(f"IndicatorContainer received BAR event")
            
            # Always forward BAR events to children
            self.publish_event(event, target_scope="children")
            
            # Process indicators if we have an indicator hub
            if self.indicator_hub:
                timestamp = event.payload.get('timestamp')
                
                # Convert market data from dict format to object format that IndicatorHub expects
                market_data = event.payload.get('market_data', {})
                converted_market_data = {}
                
                for symbol, data_dict in market_data.items():
                    # Create a simple object with attributes from the dictionary
                    class MarketDataObj:
                        def __init__(self, data):
                            self.open = data.get('open', 0)
                            self.high = data.get('high', 0) 
                            self.low = data.get('low', 0)
                            self.close = data.get('close', 0)
                            self.price = data.get('close', 0)  # Use close as price
                            self.volume = data.get('volume', 0)
                    
                    converted_market_data[symbol] = MarketDataObj(data_dict)
                
                # Create a converted event with the proper market data format
                converted_event = Event(
                    event_type=event.event_type,
                    payload={
                        'timestamp': timestamp,
                        'market_data': converted_market_data
                    },
                    timestamp=timestamp
                )
                
                # Process market data through indicator hub
                self.indicator_hub.process_market_data(converted_event)
                
                # Get computed indicator values
                indicator_values = {}
                for container_id, required_indicators in self._subscriptions.items():
                    # Get latest values for required indicators
                    filtered_indicators = {}
                    for indicator_name in required_indicators:
                        # Try to get values for each symbol in market data
                        for symbol in event.payload.get('market_data', {}).keys():
                            latest_value = self.indicator_hub.get_latest_value(indicator_name, symbol)
                            if latest_value:
                                if symbol not in filtered_indicators:
                                    filtered_indicators[symbol] = {}
                                filtered_indicators[symbol][indicator_name] = latest_value.value
                                log_indicator_event(logger, symbol, indicator_name, latest_value.value)
                    
                    if filtered_indicators:
                        indicator_event = Event(
                            event_type=EventType.INDICATORS,
                            payload={
                                'timestamp': timestamp,
                                'indicators': filtered_indicators,
                                'subscriber': container_id
                            },
                            timestamp=timestamp
                        )
                        
                        # Send event directly to the subscribed container
                        # Find the container by ID and send the event directly to its event bus
                        target_container = self._find_container_by_id(container_id)
                        if target_container:
                            target_container.event_bus.publish(indicator_event)
                            logger.info(f"📤 Sent INDICATOR event to {target_container.metadata.name} ({container_id})")
                        else:
                            logger.warning(f"Could not find subscribed container {container_id} for INDICATOR event")
        
        return None
    
    def get_capabilities(self) -> Set[str]:
        """Indicator container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("indicator.computation")
        capabilities.add("indicator.sharing")
        return capabilities


class StrategyContainer(BaseComposableContainer):
    """Container for strategy execution."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.STRATEGY,
            name="StrategyContainer",
            config=config,
            container_id=container_id
        )
        self.strategy: Optional[Strategy] = None
        self._current_indicators: Dict[str, Any] = {}
        
    async def _initialize_self(self) -> None:
        """Initialize strategy."""
        strategy_type = self._metadata.config.get('type', 'momentum')
        strategy_params = self._metadata.config.get('parameters', {})
        
        # Create strategy instance
        if strategy_type == 'momentum':
            from ..strategy.strategies.momentum import MomentumStrategy
            self.strategy = MomentumStrategy(**strategy_params)
        elif strategy_type == 'mean_reversion':
            from ..strategy.strategies.mean_reversion import MeanReversionStrategy
            self.strategy = MeanReversionStrategy(**strategy_params)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        logger.info(f"StrategyContainer initialized with {strategy_type} strategy")
    
    def get_required_indicators(self) -> Set[str]:
        """Get indicators required by this strategy."""
        if self.strategy and hasattr(self.strategy, 'get_required_indicators'):
            return self.strategy.get_required_indicators()
        return set()
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process events and generate signals."""
        await super().process_event(event)
        
        logger.info(f"StrategyContainer received event: {event.event_type}")
        
        if event.event_type == EventType.INDICATORS:
            # Update current indicators
            subscriber = event.payload.get('subscriber')
            logger.info(f"StrategyContainer received INDICATORS event, subscriber: {subscriber}, my_id: {self.metadata.container_id}")
            if subscriber == self.metadata.container_id:
                indicators = event.payload.get('indicators', {})
                self._current_indicators.update(indicators)
                logger.info(f"StrategyContainer updated indicators: {indicators}")
                
                # Generate signals immediately when we receive updated indicators
                # (since StrategyContainer may not receive BAR events directly)
                if self.strategy and self._current_indicators:
                    timestamp = event.payload.get('timestamp')
                    await self._generate_signals_from_indicators(timestamp)
            else:
                logger.debug(f"INDICATORS event not for me (subscriber: {subscriber})")
        
        elif event.event_type == EventType.BAR and self.strategy:
            # Generate signals using current indicators
            market_data = event.payload.get('market_data', {})
            timestamp = event.payload.get('timestamp')
            
            logger.debug(f"StrategyContainer processing BAR at {timestamp}")
            logger.debug(f"Available indicators: {self._current_indicators}")
            
            # Combine market data with indicators
            strategy_input = {
                'market_data': market_data,
                'indicators': self._current_indicators,
                'timestamp': timestamp
            }
            
            # Generate signals
            signals = self.strategy.generate_signals(strategy_input)
            
            if signals:
                for signal in signals:
                    log_signal_event(logger, signal)
                
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
                
                # Publish to parent (Portfolio, then up to Risk for processing)
                logger.info(f"🚀 StrategyContainer publishing SIGNAL event to parent (Portfolio→Risk)")
                self.publish_event(signal_event, target_scope="parent")
            else:
                logger.debug(f"No signals generated at {timestamp}")
        
        return None
    
    async def _generate_signals_from_indicators(self, timestamp) -> None:
        """Generate signals using current indicators (when no market data available)."""
        logger.info(f"🎯 StrategyContainer generating signals from indicators at {timestamp}")
        logger.info(f"Available indicators: {self._current_indicators}")
        
        # For indicator-driven signal generation, we need to provide market data
        # in a format the strategy expects. We can derive basic market data from indicators.
        market_data = {}
        
        # Try to reconstruct basic market data from indicators
        for symbol, indicators in self._current_indicators.items():
            # Use SMA as a proxy for current price if available
            price = indicators.get('SMA_20') or indicators.get('SMA_10') or indicators.get('close')
            
            # If no price is available from indicators, skip this symbol
            if price is None:
                logger.debug(f"No price data available for {symbol} from indicators: {list(indicators.keys())} - waiting for SMA_20")
                continue
                
            market_data[symbol] = {
                'close': price,
                'price': price,
                # Add other fields that might be needed
                'open': price,
                'high': price,
                'low': price,
                'volume': 0
            }
        
        if not market_data:
            logger.debug("No market data available from indicators, skipping signal generation")
            return
        
        # Combine market data with indicators
        strategy_input = {
            'market_data': market_data,
            'indicators': self._current_indicators,
            'timestamp': timestamp
        }
        
        # Generate signals
        signals = self.strategy.generate_signals(strategy_input)
        
        if signals:
            for signal in signals:
                log_signal_event(logger, signal)
            
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
            
            # Publish to parent (Portfolio, then up to Risk for processing)
            logger.info(f"🚀 StrategyContainer publishing SIGNAL event to parent (Portfolio→Risk)")
            self.publish_event(signal_event, target_scope="parent")
        else:
            logger.debug(f"No signals generated at {timestamp}")
    
    def get_capabilities(self) -> Set[str]:
        """Strategy container capabilities."""
        capabilities = super().get_capabilities()
        if self.strategy:
            strategy_type = self._metadata.config.get('type', 'unknown')
            capabilities.add(f"strategy.{strategy_type}")
        return capabilities


class ExecutionContainer(BaseComposableContainer):
    """Container for order execution and portfolio tracking."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.EXECUTION,
            name="ExecutionContainer",
            config=config,
            container_id=container_id
        )
        self.execution_engine = None
        
    async def _initialize_self(self) -> None:
        """Initialize execution engine."""
        execution_mode = self._metadata.config.get('mode', 'backtest')
        
        if execution_mode == 'backtest':
            from .backtest_engine import UnifiedBacktestEngine
            from .backtest_engine import BacktestConfig
            # Convert config to BacktestConfig format
            try:
                backtest_config = BacktestConfig(
                    start_date=self._metadata.config.get('start_date'),
                    end_date=self._metadata.config.get('end_date'), 
                    initial_capital=self._metadata.config.get('initial_capital', 100000),
                    symbols=self._metadata.config.get('symbols', ['SPY'])
                )
                logger.info(f"Created BacktestConfig: {backtest_config}")
                self.execution_engine = UnifiedBacktestEngine(backtest_config)
                logger.info("UnifiedBacktestEngine created successfully")
            except Exception as e:
                logger.error(f"Failed to create UnifiedBacktestEngine: {e}")
                import traceback
                traceback.print_exc()
                raise
        elif execution_mode == 'live':
            # Live execution not implemented yet
            raise NotImplementedError("Live execution not yet implemented")
        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")
        
        # UnifiedBacktestEngine doesn't have initialize method, remove this line
        logger.info(f"ExecutionContainer initialized with {execution_mode} execution")
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process orders and market data."""
        await super().process_event(event)
        
        logger.debug(f"🔍 ExecutionContainer process_event called with: {event.event_type}")
        
        if self.execution_engine:
            if event.event_type == EventType.ORDER:
                # Process order through UnifiedBacktestEngine
                orders = event.payload.get('orders', [])
                market_data = event.payload.get('market_data', {})
                
                logger.info(f"🎯 ExecutionContainer received ORDER event with {len(orders)} orders")
                for order in orders:
                    logger.info(f"   📝 Order: {order}")
                
                # Process each order through execution engine
                fills = []
                for order in orders:
                    log_order_event(logger, order)
                    
                    try:
                        # Execute order directly through the execution engine
                        logger.info(f"⚡ Executing order: {order.symbol} {order.side.value} {order.quantity}")
                        
                        # Check if execution engine has market data
                        if hasattr(self.execution_engine, 'execution_engine'):
                            market_data_cache = getattr(self.execution_engine.execution_engine, '_market_data', {})
                            logger.info(f"📊 Market data cache: {list(market_data_cache.keys())}")
                        
                        # Execute order (this generates a fill)
                        fill = await self.execution_engine.execution_engine.execute_order(order)
                        
                        if fill:
                            fills.append(fill)
                            logger.info(f"✅ Order executed: Fill ID {fill.fill_id}")
                        else:
                            logger.warning(f"❌ Order execution failed for {order.order_id}")
                            
                    except Exception as e:
                        logger.error(f"💥 Error executing order {order.order_id}: {e}")
                        import traceback
                        traceback.print_exc()
                
                if fills:
                    for fill in fills:
                        log_fill_event(logger, fill)
                    
                    # Create fill event
                    fill_event = Event(
                        event_type=EventType.FILL,
                        payload={
                            'timestamp': event.timestamp,
                            'fills': fills,
                            'source': self.metadata.container_id
                        },
                        timestamp=event.timestamp
                    )
                    
                    # Find RiskContainer sibling and send FILL event directly
                    logger.info(f"📤 ExecutionContainer publishing FILL event with {len(fills)} fills to RiskContainer")
                    # Get parent and find RiskContainer sibling
                    if hasattr(self, '_parent_container') and self._parent_container:
                        risk_containers = self._parent_container.find_containers_by_role(ContainerRole.RISK)
                        if risk_containers:
                            for risk_container in risk_containers:
                                risk_container.event_bus.publish(fill_event)
                                logger.info(f"   📨 Sent FILL event to RiskContainer {risk_container.metadata.container_id}")
                        else:
                            logger.warning("⚠️ No RiskContainer found to send FILL event")
                    else:
                        logger.warning("⚠️ ExecutionContainer has no parent to find RiskContainer")
                    return fill_event
            
            elif event.event_type == EventType.BAR:
                # Forward market data to execution engine for position valuation
                logger.debug(f"ExecutionContainer processing market data for {event.timestamp}")
                
                # Update market data in execution engine for order execution
                market_data = event.payload.get('market_data', {})
                if market_data and hasattr(self.execution_engine, 'execution_engine'):
                    # Update market data cache in execution engine
                    self.execution_engine.execution_engine._market_data.update(market_data)
                    logger.debug(f"Updated execution engine market data: {list(market_data.keys())}")
                
        return None
    
    def get_capabilities(self) -> Set[str]:
        """Execution container capabilities."""
        capabilities = super().get_capabilities()
        execution_mode = self._metadata.config.get('mode', 'backtest')
        capabilities.add(f"execution.{execution_mode}")
        return capabilities


class RiskContainer(BaseComposableContainer):
    """Container for risk management and position sizing."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.RISK,
            name="RiskContainer",
            config=config,
            container_id=container_id
        )
        self.risk_manager = None
        
    async def _initialize_self(self) -> None:
        """Initialize risk management system."""
        from ..risk.improved_risk_portfolio import RiskPortfolioContainer
        from ..core.dependencies.container import DependencyContainer
        from decimal import Decimal
        
        # Create dependency container for risk components
        dependency_container = DependencyContainer()
        
        # Initialize risk manager with proper parameters
        component_id = f"risk_{self.metadata.container_id}"
        initial_capital = Decimal(str(self._metadata.config.get('initial_capital', 100000)))
        
        self.risk_manager = RiskPortfolioContainer(
            component_id=component_id,
            dependency_container=dependency_container,
            initial_capital=initial_capital,
            base_currency="USD"
        )
        
        # Initialize the risk manager with basic context
        context = {
            'event_bus': getattr(self, '_event_bus', None)
        }
        self.risk_manager.initialize(context)
        
        logger.info(f"RiskContainer initialized with capital: {initial_capital}")
        
        # Debug: Check parent container relationship
        if hasattr(self, '_parent_container') and self._parent_container:
            logger.info(f"🔗 RiskContainer parent: {self._parent_container.metadata.name} ({self._parent_container.metadata.container_id})")
        else:
            logger.warning(f"⚠️ RiskContainer has no parent container!")
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process signals and generate orders."""
        await super().process_event(event)
        
        logger.debug(f"🔍 RiskContainer process_event called with: {event.event_type}")
        
        # Forward BAR events to children (especially ExecutionContainer) for market data
        if event.event_type == EventType.BAR:
            logger.debug(f"RiskContainer forwarding BAR event to children")
            self.publish_event(event, target_scope="children")
        
        if event.event_type == EventType.FILL and self.risk_manager:
            # Process fill events to update portfolio
            fills = event.payload.get('fills', [])
            
            logger.info(f"🎯 RiskContainer received FILL event with {len(fills)} fills")
            for fill in fills:
                logger.info(f"   💰 Fill: {fill.symbol} {fill.side.value} {fill.quantity} @ {fill.price}")
            
            # Update portfolio with fills
            for fill in fills:
                self.risk_manager.update_fills([fill])
                logger.info(f"📊 Portfolio updated with fill: {fill.fill_id}")
            
            # Get updated portfolio state for logging
            portfolio_state = self.risk_manager.get_portfolio_state()
            cash_balance = portfolio_state.get_cash_balance()
            positions = portfolio_state.get_all_positions()
            
            logger.info(f"💼 Portfolio Update - Cash: ${cash_balance:.2f}, Positions: {len(positions)}")
            for symbol, position in positions.items():
                if position.quantity != 0:  # Only log non-zero positions
                    logger.info(f"   📈 Position: {symbol} {position.quantity} shares @ ${position.avg_price:.2f}")
        
        elif event.event_type == EventType.SIGNAL and self.risk_manager:
            signals = event.payload.get('signals', [])
            market_data = event.payload.get('market_data', {})
            
            logger.info(f"🔥 RiskContainer received SIGNAL event with {len(signals)} signals")
            for signal in signals:
                logger.info(f"   📊 Signal: {signal.symbol} {signal.side} strength={signal.strength}")
            
            # Debug: Check market data format
            logger.info(f"   💰 Market data keys: {list(market_data.keys())}")
            for symbol, data in market_data.items():
                logger.info(f"   💰 {symbol} data: {data}")
            
            # Transform market data to format expected by position sizer
            # Position sizer expects: market_data["prices"][symbol] = price
            transformed_market_data = {
                "prices": {}
            }
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'close' in data:
                    transformed_market_data["prices"][symbol] = data['close']
                elif isinstance(data, (int, float)):
                    transformed_market_data["prices"][symbol] = data
            
            logger.info(f"   💱 Transformed prices: {transformed_market_data['prices']}")
            
            # Process signals through risk management
            orders = self.risk_manager.process_signals(signals, transformed_market_data)
            
            logger.info(f"📋 RiskContainer processed signals, generated {len(orders)} orders")
            
            if orders:
                for order in orders:
                    log_order_event(logger, order)
                
                order_event = Event(
                    event_type=EventType.ORDER,
                    payload={
                        'timestamp': event.timestamp,
                        'orders': orders,
                        'source': self.metadata.container_id
                    },
                    timestamp=event.timestamp
                )
                
                # Find ExecutionContainer sibling and send ORDER event directly
                logger.info(f"📤 RiskContainer publishing ORDER event to ExecutionContainer")
                # Get parent and find ExecutionContainer sibling
                if hasattr(self, '_parent_container') and self._parent_container:
                    execution_containers = self._parent_container.find_containers_by_role(ContainerRole.EXECUTION)
                    if execution_containers:
                        for exec_container in execution_containers:
                            exec_container.event_bus.publish(order_event)
                            logger.info(f"   📨 Sent ORDER event to ExecutionContainer {exec_container.metadata.container_id}")
                    else:
                        logger.warning("⚠️ No ExecutionContainer found to send ORDER event")
                else:
                    logger.warning("⚠️ RiskContainer has no parent to find ExecutionContainer")
                
                return order_event
            # No orders generated - no logging needed for normal operation
        
        return None
    
    def get_capabilities(self) -> Set[str]:
        """Risk container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("risk.management")
        capabilities.add("risk.position_sizing")
        return capabilities


class PortfolioContainer(BaseComposableContainer):
    """Container for portfolio allocation and management."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.PORTFOLIO,
            name="PortfolioContainer",
            config=config,
            container_id=container_id
        )
        self.allocation_manager = None
        
    async def _initialize_self(self) -> None:
        """Initialize portfolio allocation system."""
        allocation_type = self._metadata.config.get('allocation_type', 'equal_weight')
        allocation_capital = self._metadata.config.get('allocation_capital', 50000)
        
        # Simple allocation manager for now
        self.allocation_manager = {
            'type': allocation_type,
            'capital': allocation_capital,
            'allocations': self._metadata.config.get('allocations', {})
        }
        
        logger.info(f"PortfolioContainer initialized: {allocation_type} with ${allocation_capital}")
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process signals and apply portfolio allocation."""
        await super().process_event(event)
        
        if event.event_type == EventType.SIGNAL and self.allocation_manager:
            signals = event.payload.get('signals', [])
            
            # Apply portfolio allocation to signals
            allocated_signals = []
            for signal in signals:
                # Apply allocation weights (Signal is an object, not a dict)
                symbol = signal.symbol if hasattr(signal, 'symbol') else 'UNKNOWN'
                allocation_weight = self.allocation_manager['allocations'].get(symbol, 1.0)
                
                # For now, just pass through the signal - allocation could be applied in risk management
                allocated_signal = signal
                allocated_signals.append(allocated_signal)
            
            if allocated_signals:
                allocated_event = Event(
                    event_type=EventType.SIGNAL,
                    payload={
                        'timestamp': event.timestamp,
                        'signals': allocated_signals,
                        'market_data': event.payload.get('market_data', {}),  # Preserve market data
                        'source': self.metadata.container_id
                    },
                    timestamp=event.timestamp
                )
                
                # Forward to parent (risk container)
                self.publish_event(allocated_event, target_scope="parent")
                
                return allocated_event
        
        return None
    
    def get_capabilities(self) -> Set[str]:
        """Portfolio container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("portfolio.allocation")
        capabilities.add("portfolio.management")
        return capabilities


class ClassifierContainer(BaseComposableContainer):
    """Container for regime detection and classification."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.CLASSIFIER,
            name="ClassifierContainer",
            config=config,
            container_id=container_id
        )
        self.classifier = None
        self._current_regime = "neutral"
        
    async def _initialize_self(self) -> None:
        """Initialize classifier system."""
        classifier_type = self._metadata.config.get('type', 'hmm')
        
        if classifier_type == 'hmm':
            from ..strategy.classifiers.hmm_classifier import HMMClassifier
            self.classifier = HMMClassifier(**self._metadata.config.get('parameters', {}))
        elif classifier_type == 'pattern':
            from ..strategy.classifiers.pattern_classifier import PatternClassifier
            self.classifier = PatternClassifier(**self._metadata.config.get('parameters', {}))
        else:
            # Simple threshold-based classifier as fallback
            self.classifier = None
            logger.warning(f"Unknown classifier type: {classifier_type}, using simple fallback")
        
        logger.info(f"ClassifierContainer initialized with {classifier_type} classifier")
    
    def get_required_indicators(self) -> Set[str]:
        """Get indicators required by this classifier."""
        if self.classifier and hasattr(self.classifier, 'get_required_indicators'):
            return self.classifier.get_required_indicators()
        # Default indicators for simple classifier
        return {"RSI", "SMA_20", "SMA_50"}
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process indicators and determine regime."""
        await super().process_event(event)
        
        if event.event_type == EventType.INDICATORS:
            indicators = event.payload.get('indicators', {})
            timestamp = event.payload.get('timestamp')
            
            # Determine regime
            if self.classifier:
                regime = self.classifier.classify(indicators)
            else:
                # Simple fallback regime detection
                rsi = indicators.get('RSI', 50)
                if rsi > 70:
                    regime = "overbought"
                elif rsi < 30:
                    regime = "oversold"
                else:
                    regime = "neutral"
            
            if regime != self._current_regime:
                self._current_regime = regime
                
                # Emit regime change event
                regime_event = Event(
                    event_type=EventType.REGIME,
                    payload={
                        'timestamp': timestamp,
                        'regime': regime,
                        'classifier': self.metadata.container_id,
                        'confidence': getattr(self.classifier, 'confidence', 0.8) if self.classifier else 0.5
                    },
                    timestamp=timestamp
                )
                
                # Publish to children
                self.publish_event(regime_event, target_scope="children")
                
                return regime_event
        
        return None
    
    def get_capabilities(self) -> Set[str]:
        """Classifier container capabilities."""
        capabilities = super().get_capabilities()
        classifier_type = self._metadata.config.get('type', 'simple')
        capabilities.add(f"classifier.{classifier_type}")
        capabilities.add("regime.detection")
        return capabilities


class AnalysisContainer(BaseComposableContainer):
    """Container for signal analysis and research."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.ANALYSIS,
            name="AnalysisContainer",
            config=config,
            container_id=container_id
        )
        self.analysis_engine = None
        self._signal_log = []
        
    async def _initialize_self(self) -> None:
        """Initialize analysis engine."""
        analysis_type = self._metadata.config.get('mode', 'signal_generation')
        
        # Simple analysis engine for signal collection
        self.analysis_engine = {
            'mode': analysis_type,
            'signal_count': 0,
            'output_file': self._metadata.config.get('output_file', 'signals.json')
        }
        
        logger.info(f"AnalysisContainer initialized for {analysis_type}")
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Collect and analyze signals."""
        await super().process_event(event)
        
        if event.event_type == EventType.SIGNAL:
            signals = event.payload.get('signals', [])
            timestamp = event.payload.get('timestamp')
            
            # Log signals for analysis
            for signal in signals:
                signal_record = {
                    'timestamp': timestamp.isoformat() if timestamp else None,
                    'signal': signal,
                    'analysis_id': self.metadata.container_id
                }
                self._signal_log.append(signal_record)
            
            self.analysis_engine['signal_count'] += len(signals)
            
            # Periodically save signals
            if len(self._signal_log) % 100 == 0:
                logger.info(f"Collected {len(self._signal_log)} signals for analysis")
        
        return None
    
    def get_capabilities(self) -> Set[str]:
        """Analysis container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("analysis.signals")
        capabilities.add("analysis.research")
        return capabilities


class SignalLogContainer(BaseComposableContainer):
    """Container for signal log streaming and replay."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.SIGNAL_LOG,
            name="SignalLogContainer",
            config=config,
            container_id=container_id
        )
        self.signal_streamer = None
        self._streaming_task = None
        
    async def _initialize_self(self) -> None:
        """Initialize signal log streamer."""
        log_source = self._metadata.config.get('source', 'phase1_output')
        log_file = self._metadata.config.get('log_file', 'signals.json')
        
        # Simple signal streamer
        self.signal_streamer = {
            'source': log_source,
            'file': log_file,
            'current_position': 0
        }
        
        logger.info(f"SignalLogContainer initialized for replay from {log_file}")
    
    async def start(self) -> None:
        """Start signal replay streaming."""
        await super().start()
        
        # Start signal replay task
        self._streaming_task = asyncio.create_task(self._stream_signals())
        logger.info("Signal replay streaming started")
    
    async def _stop_self(self) -> None:
        """Stop signal streaming."""
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
        logger.info("Signal replay streaming stopped")
    
    async def _stream_signals(self) -> None:
        """Stream signals from log file."""
        # Placeholder for signal streaming
        # In real implementation, would read from signal log file
        # and emit signals according to timestamps
        import json
        
        try:
            log_file = self.signal_streamer['file']
            # For now, just log that we would stream signals
            logger.info(f"Would stream signals from {log_file}")
            
            # Simulate signal streaming
            while self.metadata.state == ContainerState.RUNNING:
                # Create mock signal event
                signal_event = Event(
                    event_type=EventType.SIGNAL,
                    payload={
                        'timestamp': datetime.now(),
                        'signals': [],  # Would load from file
                        'source': 'signal_replay'
                    },
                    timestamp=datetime.now()
                )
                
                # Publish to children
                self.publish_event(signal_event, target_scope="children")
                
                # Wait before next signal batch
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Signal streaming cancelled")
    
    def get_capabilities(self) -> Set[str]:
        """Signal log container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("signal_log.replay")
        capabilities.add("signal_log.streaming")
        return capabilities


class EnsembleContainer(BaseComposableContainer):
    """Container for ensemble signal optimization."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.ENSEMBLE,
            name="EnsembleContainer",
            config=config,
            container_id=container_id
        )
        self.ensemble_optimizer = None
        
    async def _initialize_self(self) -> None:
        """Initialize ensemble optimizer."""
        weight_config = self._metadata.config.get('weight_config', {})
        optimization_method = self._metadata.config.get('method', 'equal_weight')
        
        # Simple ensemble optimizer
        self.ensemble_optimizer = {
            'method': optimization_method,
            'weights': weight_config.get('weights', {}),
            'signal_buffer': []
        }
        
        logger.info(f"EnsembleContainer initialized with {optimization_method} method")
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process and combine signals using ensemble weights."""
        await super().process_event(event)
        
        if event.event_type == EventType.SIGNAL:
            signals = event.payload.get('signals', [])
            timestamp = event.payload.get('timestamp')
            
            # Apply ensemble weights to signals
            weighted_signals = []
            for signal in signals:
                strategy_id = signal.get('strategy_id', 'default')
                weight = self.ensemble_optimizer['weights'].get(strategy_id, 1.0)
                
                weighted_signal = signal.copy()
                weighted_signal['ensemble_weight'] = weight
                weighted_signal['original_strength'] = signal.get('strength', 1.0)
                weighted_signal['strength'] = signal.get('strength', 1.0) * weight
                
                weighted_signals.append(weighted_signal)
            
            if weighted_signals:
                ensemble_event = Event(
                    event_type=EventType.SIGNAL,
                    payload={
                        'timestamp': timestamp,
                        'signals': weighted_signals,
                        'source': self.metadata.container_id,
                        'ensemble_method': self.ensemble_optimizer['method']
                    },
                    timestamp=timestamp
                )
                
                # Forward to children (risk/portfolio containers)
                self.publish_event(ensemble_event, target_scope="children")
                
                return ensemble_event
        
        return None
    
    def get_capabilities(self) -> Set[str]:
        """Ensemble container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("ensemble.optimization")
        capabilities.add("ensemble.weighting")
        return capabilities


# Factory functions for container registration
def create_data_container(config: Dict[str, Any], container_id: str = None) -> DataContainer:
    """Factory function for data containers."""
    return DataContainer(config, container_id)


def create_indicator_container(config: Dict[str, Any], container_id: str = None) -> IndicatorContainer:
    """Factory function for indicator containers."""
    return IndicatorContainer(config, container_id)


def create_strategy_container(config: Dict[str, Any], container_id: str = None) -> StrategyContainer:
    """Factory function for strategy containers."""
    return StrategyContainer(config, container_id)


def create_execution_container(config: Dict[str, Any], container_id: str = None) -> ExecutionContainer:
    """Factory function for execution containers."""
    return ExecutionContainer(config, container_id)


def create_risk_container(config: Dict[str, Any], container_id: str = None) -> RiskContainer:
    """Factory function for risk containers."""
    return RiskContainer(config, container_id)


def create_portfolio_container(config: Dict[str, Any], container_id: str = None) -> PortfolioContainer:
    """Factory function for portfolio containers."""
    return PortfolioContainer(config, container_id)


def create_classifier_container(config: Dict[str, Any], container_id: str = None) -> ClassifierContainer:
    """Factory function for classifier containers."""
    return ClassifierContainer(config, container_id)


def create_analysis_container(config: Dict[str, Any], container_id: str = None) -> AnalysisContainer:
    """Factory function for analysis containers."""
    return AnalysisContainer(config, container_id)


def create_signal_log_container(config: Dict[str, Any], container_id: str = None) -> SignalLogContainer:
    """Factory function for signal log containers."""
    return SignalLogContainer(config, container_id)


def create_ensemble_container(config: Dict[str, Any], container_id: str = None) -> EnsembleContainer:
    """Factory function for ensemble containers."""
    return EnsembleContainer(config, container_id)


# Register container types with global registry
def register_execution_containers():
    """Register all execution container types."""
    from ..core.containers.composition_engine import register_container_type
    
    register_container_type(
        ContainerRole.DATA,
        create_data_container,
        {"data.historical", "data.live"}
    )
    
    register_container_type(
        ContainerRole.INDICATOR,
        create_indicator_container,
        {"indicator.computation", "indicator.sharing"}
    )
    
    register_container_type(
        ContainerRole.STRATEGY,
        create_strategy_container,
        {"strategy.momentum", "strategy.mean_reversion", "strategy.custom"}
    )
    
    register_container_type(
        ContainerRole.EXECUTION,
        create_execution_container,
        {"execution.backtest", "execution.live"}
    )
    
    register_container_type(
        ContainerRole.RISK,
        create_risk_container,
        {"risk.management", "risk.position_sizing"}
    )
    
    register_container_type(
        ContainerRole.PORTFOLIO,
        create_portfolio_container,
        {"portfolio.allocation", "portfolio.management"}
    )
    
    register_container_type(
        ContainerRole.CLASSIFIER,
        create_classifier_container,
        {"classifier.hmm", "classifier.pattern", "regime.detection"}
    )
    
    register_container_type(
        ContainerRole.ANALYSIS,
        create_analysis_container,
        {"analysis.signals", "analysis.research"}
    )
    
    register_container_type(
        ContainerRole.SIGNAL_LOG,
        create_signal_log_container,
        {"signal_log.replay", "signal_log.streaming"}
    )
    
    register_container_type(
        ContainerRole.ENSEMBLE,
        create_ensemble_container,
        {"ensemble.optimization", "ensemble.weighting"}
    )


# Auto-register on import
register_execution_containers()