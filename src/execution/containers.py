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

# Import signal aggregation for ensemble container
try:
    from ..strategy.signal_aggregation import (
        WeightedVotingAggregator, Direction, TradingSignal, 
        AggregatedSignal, ConsensusSignal
    )
except ImportError:
    # Fallback if signal aggregation not available
    WeightedVotingAggregator = None


class DataContainer(BaseComposableContainer):
    """Container for data streaming and management."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Configure external communication for broadcasting data
        self._configure_data_broadcasting(config)
        
        super().__init__(
            role=ContainerRole.DATA,
            name="DataContainer",
            config=config,
            container_id=container_id
        )
        self.data_loader: Optional[DataLoader] = None
        self._streaming_task: Optional[asyncio.Task] = None
    
    def _configure_data_broadcasting(self, config: Dict[str, Any]) -> None:
        """Configure external Event Router communication for data broadcasting"""
        if 'external_events' not in config:
            config['external_events'] = {}
        
        ext_config = config['external_events']
        
        # Configure publications for broadcasting data events
        if 'publishes' not in ext_config:
            ext_config['publishes'] = []
        
        ext_config['publishes'].extend([
            {
                'events': ['BAR'],
                'scope': 'GLOBAL',  # Broadcast to all subscribers
                'tier': 'fast'     # Use Fast Tier for high-frequency data
            },
            {
                'events': ['SYSTEM'],
                'scope': 'GLOBAL',  # System events to all containers
                'tier': 'reliable' # Use Reliable Tier for critical events
            }
        ])
        
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
        
        # Broadcast end-of-backtest event for position closure
        await self._broadcast_end_of_backtest()
        
        logger.info("Data streaming stopped")
    
    async def _broadcast_end_of_backtest(self) -> None:
        """Broadcast end-of-backtest event to trigger position closure."""
        try:
            from ..core.events.types import Event, EventType
            
            # Get the last bar price for position closure
            last_bar_price = getattr(self, '_last_bar_price', None)
            
            end_event = Event(
                event_type=EventType.SYSTEM,  # Use SYSTEM event type
                payload={
                    'action': 'END_OF_BACKTEST',
                    'last_prices': last_bar_price or {},
                    'timestamp': datetime.now().isoformat()
                },
                timestamp=datetime.now()
            )
            
            # Broadcast to ALL containers in hierarchy (not just direct children)
            await self._broadcast_to_all_containers(end_event)
            logger.info("📢 Broadcasted END_OF_BACKTEST event")
            
        except Exception as e:
            logger.error(f"Failed to broadcast end-of-backtest event: {e}")
    
    async def _broadcast_to_all_containers(self, event: Event) -> None:
        """Recursively broadcast event to all containers in the hierarchy."""
        # Send to self first
        if hasattr(self, 'process_event'):
            try:
                await self.process_event(event)
            except Exception as e:
                logger.error(f"Error processing END_OF_BACKTEST in {self.metadata.name}: {e}")
        
        # Recursively send to all child containers
        for child in self.child_containers:
            await self._broadcast_to_child_and_descendants(child, event)
    
    async def _broadcast_to_child_and_descendants(self, container, event: Event) -> None:
        """Recursively broadcast event to a container and all its descendants."""
        try:
            # Send to this container
            if hasattr(container, 'process_event'):
                await container.process_event(event)
                logger.debug(f"Sent END_OF_BACKTEST to {container.metadata.name}")
            
            # Recursively send to its children
            if hasattr(container, 'child_containers'):
                for child in container.child_containers:
                    await self._broadcast_to_child_and_descendants(child, event)
        except Exception as e:
            logger.error(f"Error broadcasting END_OF_BACKTEST to {container.metadata.name}: {e}")
    
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
                    
                    # Initialize last price tracking
                    if not hasattr(self, '_last_bar_price'):
                        self._last_bar_price = {}
                    
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
                        
                        # Track last price for end-of-backtest position closure
                        self._last_bar_price[symbol] = float(row['close'])
                        
                        # Broadcast BAR events via Event Router Fast Tier for selective subscriptions
                        logger.debug(f"📡 DataContainer broadcasting BAR event for {symbol} via Fast Tier")
                        from ..core.events.hybrid_interface import CommunicationTier
                        self.publish_external(event, tier=CommunicationTier.FAST)
                        
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
        # Add external event routing configuration for hybrid interface
        if 'external_events' not in config:
            config['external_events'] = {}
        
        # Declare what events this container publishes and subscribes to via Event Router
        config['external_events']['publishes'] = config['external_events'].get('publishes', []) + [
            {
                'events': ['INDICATORS'],
                'scope': 'GLOBAL',  # Send INDICATORS globally
                'tier': 'standard'
            }
        ]
        
        config['external_events']['subscribes'] = config['external_events'].get('subscribes', []) + [
            {
                'source': '*',  # Subscribe to BAR events from any source (DataContainer will be the only publisher)
                'events': ['BAR'],
                'tier': 'fast'
            }
        ]
        
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
            logger.info(f"📊 IndicatorContainer received BAR event")
            
            # Always forward BAR events to children (using old pattern for broad distribution)
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
                        
                        # Route INDICATOR event to subscribing container
                        # This is a workaround for cross-hierarchy event delivery since each
                        # container has its own isolated event bus
                        target_container = self._find_container_by_id(container_id)
                        if target_container:
                            # Deliver event directly to target container's process_event method
                            # This maintains the event-driven pattern while working around isolation
                            await target_container.process_event(indicator_event)
                            logger.info(f"📤 Routed INDICATOR event to {target_container.metadata.name} ({container_id})")
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
    """Container for strategy execution that automatically creates sub-containers for multiple strategies."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Configure external communication via Event Router
        self._configure_external_events(config)
        
        super().__init__(
            role=ContainerRole.STRATEGY,
            name="StrategyContainer",
            config=config,
            container_id=container_id
        )
        # Strategy execution components
        self.strategy: Optional[Strategy] = None
        self.signal_aggregator = None
        
        # State management
        self._current_indicators: Dict[str, Any] = {}
        self._current_market_data: Dict[str, Any] = {}  # Store latest market data
        
        # Determine if this is multi-strategy
        self.strategies_config = self._get_strategies_config()
        self.is_multi_strategy = len(self.strategies_config) > 1
    
    def _configure_external_events(self, config: Dict[str, Any]) -> None:
        """Configure external Event Router communication"""
        if 'external_events' not in config:
            config['external_events'] = {}
        
        ext_config = config['external_events']
        
        # Configure publications (what this container publishes externally)
        if 'publishes' not in ext_config:
            ext_config['publishes'] = []
        
        ext_config['publishes'].extend([
            {
                'events': ['SIGNAL'],
                'scope': 'PARENT',  # Send to PortfolioContainer
                'tier': 'standard'
            }
        ])
        
        # Configure subscriptions (what this container subscribes to externally)
        if 'subscribes' not in ext_config:
            ext_config['subscribes'] = []
        
        ext_config['subscribes'].extend([
            {
                'source': '*',  # Subscribe to BAR events from any source
                'events': ['BAR'],
                'tier': 'fast',
                'filters': {}  # Will be configured per instance
            },
            {
                'source': '*',  # Subscribe to INDICATORS events from any source  
                'events': ['INDICATORS'],
                'tier': 'standard',
                'filters': {'subscriber': config.get('container_id', 'strategy_container')}
            }
        ])
    
    def _get_strategies_config(self) -> List[Dict[str, Any]]:
        """Extract strategies configuration from various config formats."""
        # Check for direct strategies list
        if 'strategies' in self._metadata.config:
            return self._metadata.config['strategies']
        
        # Check for single strategy config (convert to list)
        if 'type' in self._metadata.config:
            return [{
                'name': 'primary_strategy',
                'type': self._metadata.config['type'],
                'parameters': self._metadata.config.get('parameters', {}),
                'weight': 1.0
            }]
        
        # Fallback to empty list
        return []
        
    async def _initialize_self(self) -> None:
        """Initialize strategy system - single strategy or sub-containers for multiple."""
        if self.is_multi_strategy:
            await self._initialize_multi_strategy()
        else:
            await self._initialize_single_strategy()
        
        logger.info(f"StrategyContainer initialized ({'multi-strategy' if self.is_multi_strategy else 'single-strategy'})")
    
    async def _initialize_single_strategy(self) -> None:
        """Initialize single strategy."""
        if not self.strategies_config:
            raise ValueError("No strategy configuration found")
        
        strategy_config = self.strategies_config[0]  # Use first strategy for single mode
        strategy_type = strategy_config.get('type', 'momentum')
        strategy_params = strategy_config.get('parameters', {})
        
        # Create strategy instance
        self.strategy = self._create_strategy_instance(strategy_type, strategy_params)
        if not self.strategy:
            raise ValueError(f"Failed to create strategy of type: {strategy_type}")
        
        logger.info(f"Single strategy initialized: {strategy_type}")
    
    async def _initialize_multi_strategy(self) -> None:
        """Initialize multiple strategies as sub-containers."""
        aggregation_config = self._metadata.config.get('aggregation', {})
        
        # Initialize signal aggregator for consensus
        aggregation_method = aggregation_config.get('method', 'weighted_voting')
        min_confidence = aggregation_config.get('min_confidence', 0.6)
        
        if WeightedVotingAggregator and aggregation_method == 'weighted_voting':
            self.signal_aggregator = WeightedVotingAggregator(
                min_confidence=min_confidence,
                container_id=self.metadata.container_id
            )
        
        # Create sub-containers for each strategy
        for i, strategy_config in enumerate(self.strategies_config):
            strategy_name = strategy_config.get('name', f"strategy_{i+1}")
            
            # Create sub-container for this strategy
            sub_container_config = {
                'type': strategy_config.get('type', 'momentum'),
                'parameters': strategy_config.get('parameters', {}),
                'weight': strategy_config.get('weight', 1.0),
                'allocation': strategy_config.get('allocation', 1.0)
            }
            
            sub_container = StrategyContainer(
                config=sub_container_config,
                container_id=f"{self.metadata.container_id}_{strategy_name}"
            )
            sub_container._metadata.name = f"Strategy_{strategy_name}"
            
            # Add as child container
            self.add_child_container(sub_container)
            
            logger.info(f"Created sub-container for strategy: {strategy_name} ({strategy_config.get('type')})")
        
        logger.info(f"Multi-strategy initialized with {len(self.strategies_config)} sub-containers")
    
    def _create_strategy_instance(self, strategy_type: str, strategy_params: Dict[str, Any]) -> Optional[Strategy]:
        """Factory method to create strategy instances."""
        try:
            if strategy_type == 'momentum':
                from ..strategy.strategies.momentum import MomentumStrategy
                return MomentumStrategy(**strategy_params)
            elif strategy_type == 'mean_reversion':
                from ..strategy.strategies.mean_reversion import MeanReversionStrategy
                return MeanReversionStrategy(**strategy_params)
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create strategy {strategy_type}: {e}")
            return None
    
    def get_required_indicators(self) -> Set[str]:
        """Get indicators required by this strategy or sub-containers."""
        required_indicators = set()
        
        if self.is_multi_strategy:
            # Collect indicators from all sub-containers
            for child in self.child_containers:
                if hasattr(child, 'get_required_indicators'):
                    required_indicators.update(child.get_required_indicators())
        else:
            # Single strategy - get indicators from our strategy
            if self.strategy and hasattr(self.strategy, 'get_required_indicators'):
                required_indicators.update(self.strategy.get_required_indicators())
        
        return required_indicators
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process events and generate signals."""
        await super().process_event(event)
        
        logger.info(f"StrategyContainer received event: {event.event_type}")
        
        if self.is_multi_strategy:
            # Multi-strategy: forward events to sub-containers and collect signals
            if event.event_type == EventType.SIGNAL:
                # Collect signals from sub-containers for aggregation
                await self._handle_sub_container_signal(event)
            else:
                # Forward other events to all sub-containers
                for child in self.child_containers:
                    await child.process_event(event)
        else:
            # Single strategy: handle events directly
            if event.event_type == EventType.INDICATORS:
                await self._handle_indicators_event(event)
            elif event.event_type == EventType.BAR:
                await self._handle_bar_event(event)
        
        return None
    
    async def _handle_indicators_event(self, event: Event) -> None:
        """Handle INDICATORS event for single strategy."""
        subscriber = event.payload.get('subscriber')
        
        # Accept INDICATORS events if:
        # 1. We're the direct subscriber, OR
        # 2. We're a sub-container and subscriber is our parent
        should_process = (
            subscriber == self.metadata.container_id or
            (self.parent_container and subscriber == self.parent_container.metadata.container_id)
        )
        
        if should_process:
            indicators = event.payload.get('indicators', {})
            self._current_indicators.update(indicators)
            logger.info(f"StrategyContainer ({self.metadata.name}) updated indicators: {indicators}")
            
            # Generate signals using stored market data
            timestamp = event.payload.get('timestamp')
            if self._current_market_data:
                await self._generate_single_signals(timestamp, self._current_market_data)
            else:
                logger.warning("Received indicators but no market data available yet")
        else:
            logger.debug(f"INDICATORS event not for me (subscriber: {subscriber}, my_id: {self.metadata.container_id})")
    
    async def _handle_bar_event(self, event: Event) -> None:
        """Handle BAR event for single strategy."""
        market_data = event.payload.get('market_data', {})
        timestamp = event.payload.get('timestamp')
        
        # Update current market data
        self._current_market_data = market_data
        
        # Generate signals if we have both market data and indicators
        if self._current_indicators:
            await self._generate_single_signals(timestamp, market_data)
        else:
            logger.debug("Waiting for indicators before generating signals")
    
    async def _handle_sub_container_signal(self, event: Event) -> None:
        """Collect and aggregate signals from sub-containers."""
        # For now, just forward the signal up - aggregation can be added later
        # This maintains the composable design where parent just coordinates
        signals = event.payload.get('signals', [])
        if signals:
            logger.info(f"Multi-strategy container received {len(signals)} signals from sub-container")
            # Forward to parent (Portfolio -> Risk) via Event Router
            from ..core.events.routing.protocols import EventScope
            self.publish_external(event)
    
    async def _generate_single_signals(self, timestamp, market_data: Dict[str, Any]) -> None:
        """Generate signals from single strategy."""
        if not self.strategy:
            return
        
        # Prepare strategy input
        strategy_input = self._prepare_strategy_input(timestamp, market_data)
        if not strategy_input:
            return
        
        # Generate signals
        signals = self.strategy.generate_signals(strategy_input)
        
        if signals:
            # Use the prepared market data, not the original empty market_data
            await self._emit_signals(signals, timestamp, strategy_input['market_data'])
    
    def _prepare_strategy_input(self, timestamp, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare strategy input, deriving market data from indicators if needed."""
        # If no market data provided, try to derive from indicators
        if not market_data:
            market_data = {}
            for symbol, indicators in self._current_indicators.items():
                # Use SMA as proxy for current price if available
                price = indicators.get('SMA_20') or indicators.get('SMA_10') or indicators.get('close')
                
                if price is None:
                    logger.debug(f"No price data available for {symbol} from indicators")
                    continue
                    
                market_data[symbol] = {
                    'close': price, 'price': price,
                    'open': price, 'high': price, 'low': price, 'volume': 0
                }
        
        if not market_data:
            logger.debug("No market data available, skipping signal generation")
            return None
        
        return {
            'market_data': market_data,
            'indicators': self._current_indicators,
            'timestamp': timestamp
        }
    
    async def _emit_signals(self, signals: List, timestamp, market_data: Dict[str, Any]) -> None:
        """Emit signals to parent container."""
        if signals:
            for signal in signals:
                log_signal_event(logger, signal)
            
            signal_event = Event(
                event_type=EventType.SIGNAL,
                payload={
                    'timestamp': timestamp,
                    'signals': signals,
                    'market_data': market_data,
                    'source': self.metadata.container_id,
                    'container_type': 'strategy'
                },
                timestamp=timestamp
            )
            
            # Use hybrid communication: internal for parent, external for cross-container
            if self.parent_container:
                # Internal: Direct to parent Portfolio via internal event bus
                logger.info(f"🚀 StrategyContainer publishing SIGNAL to parent via internal bus")
                self.publish_internal(signal_event, scope="parent")
            else:
                # External: Direct to Risk/Portfolio via Event Router
                logger.info(f"🚀 StrategyContainer publishing SIGNAL via Event Router")
                from ..core.events.hybrid_interface import CommunicationTier
                self.publish_external(signal_event, tier=CommunicationTier.STANDARD)
        else:
            logger.debug(f"No signals to emit at {timestamp}")
    
    def get_capabilities(self) -> Set[str]:
        """Strategy container capabilities."""
        capabilities = super().get_capabilities()
        
        if self.is_multi_strategy:
            capabilities.add("strategy.multi")
            capabilities.add("strategy.sub_containers")
        else:
            capabilities.add("strategy.single")
        
        return capabilities


class ExecutionContainer(BaseComposableContainer):
    """Container for order execution and portfolio tracking."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Add event routing configuration
        if 'events' not in config:
            config['events'] = {}
        
        # Declare what events this container subscribes to and publishes
        config['events']['subscribes'] = config['events'].get('subscribes', []) + [
            {
                'events': ['BAR'],
                'scope': 'UPWARD'  # Receive BAR events from DataContainer ancestor
            },
            {
                'events': ['ORDER'],
                'scope': 'SIBLINGS'  # Receive from RiskContainer sibling
            }
        ]
        
        config['events']['publishes'] = config['events'].get('publishes', []) + [
            {
                'events': ['FILL'],
                'scope': 'SIBLINGS'  # Send to RiskContainer sibling
            }
        ]
        
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
                
                # CRITICAL: Replace the broker's portfolio state with the one from RiskContainer
                # This ensures position tracking is synchronized
                if hasattr(self, '_parent_container') and self._parent_container:
                    risk_container = self._parent_container
                    if hasattr(risk_container, 'risk_manager') and risk_container.risk_manager:
                        shared_portfolio_state = risk_container.risk_manager.get_portfolio_state()
                        if shared_portfolio_state and hasattr(self.execution_engine, 'broker'):
                            logger.info("Replacing broker's portfolio state with RiskContainer's portfolio state")
                            self.execution_engine.broker.portfolio_state = shared_portfolio_state
                            # Also update the risk_portfolio reference
                            self.execution_engine.risk_portfolio = risk_container.risk_manager
                
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
                        
                        # Execute order (this generates a fill) - run in thread to avoid blocking
                        import asyncio
                        loop = asyncio.get_event_loop()
                        fill = await loop.run_in_executor(None, self.execution_engine.execution_engine.execute_order, order)
                        
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
                    
                    # Send FILL event to RiskContainer via Event Router
                    logger.info(f"📤 ExecutionContainer publishing FILL event with {len(fills)} fills to RiskContainer")
                    
                    from ..core.events.routing.protocols import EventScope
                    self.publish_external(fill_event)
                    logger.info(f"   📨 Sent FILL event to RiskContainer via Event Router")
                    
                    return fill_event
            
            elif event.event_type == EventType.BAR:
                # Forward market data to execution engine for position valuation
                logger.debug(f"ExecutionContainer processing market data for {event.timestamp}")
                
                # Update market data in execution engine for order execution
                market_data = event.payload.get('market_data', {})
                if market_data and hasattr(self.execution_engine, 'execution_engine'):
                    # Update market data cache in execution engine
                    self.execution_engine.execution_engine._market_data.update(market_data)
            
            elif event.event_type == EventType.SYSTEM:
                # Handle system events like END_OF_BACKTEST
                action = event.payload.get('action')
                if action == 'END_OF_BACKTEST':
                    logger.info("🏁 Received END_OF_BACKTEST event - closing all positions")
                    await self._close_all_positions(event.payload.get('last_prices', {}))
                
        return None
    
    async def _close_all_positions(self, last_prices: Dict[str, float]) -> None:
        """Force close all open positions at end of backtest."""
        try:
            if not self.execution_engine:
                logger.warning("No execution engine available for position closure")
                return
            
            # Get positions from RiskContainer (parent) instead of broker
            # since that's where the actual portfolio state is maintained
            if not hasattr(self, '_parent_container') or not self._parent_container:
                logger.warning("No parent container to get positions from")
                return
            
            # Find RiskContainer parent and get positions from its portfolio state
            risk_container = self._parent_container
            if not hasattr(risk_container, 'risk_manager') or not risk_container.risk_manager:
                logger.warning("No risk manager available to get positions")
                return
            
            portfolio_state = risk_container.risk_manager.get_portfolio_state()
            if not portfolio_state:
                logger.warning("No portfolio state available")
                return
            
            # Get all current positions from portfolio state
            positions = portfolio_state.get_all_positions()
            if not positions:
                logger.info("No positions to close")
                return
            
            logger.info(f"Closing {len(positions)} positions at end of backtest")
            
            # Generate market orders to close each position
            from .protocols import Order, OrderSide, OrderType
            import uuid
            
            close_orders = []
            for symbol, position in positions.items():
                # Position is a Position object from portfolio state
                quantity = abs(float(position.quantity)) if hasattr(position, 'quantity') else 0
                if quantity > 0:
                    # Determine side to close position
                    current_quantity = float(position.quantity) if hasattr(position, 'quantity') else 0
                    close_side = OrderSide.SELL if current_quantity > 0 else OrderSide.BUY
                    
                    # Use last price if available, otherwise use current price from position
                    close_price = last_prices.get(symbol)
                    if not close_price and hasattr(position, 'current_price'):
                        close_price = float(position.current_price)
                    if not close_price and hasattr(position, 'average_price'):
                        close_price = float(position.average_price)
                    
                    if close_price and close_price > 0:
                        close_order = Order(
                            order_id=str(uuid.uuid4()),
                            symbol=symbol,
                            side=close_side,
                            order_type=OrderType.MARKET,
                            quantity=quantity,
                            price=close_price,
                            created_at=datetime.now()
                        )
                        close_orders.append(close_order)
                        logger.info(f"Generated close order: {symbol} {close_side.name} {quantity} @ {close_price}")
                    else:
                        logger.warning(f"No valid price found for closing position {symbol}")
                else:
                    logger.debug(f"Position {symbol} has zero quantity, skipping")
            
            # Execute close orders immediately
            if close_orders:
                for order in close_orders:
                    try:
                        fill = await asyncio.get_event_loop().run_in_executor(
                            None, self.execution_engine.execution_engine.execute_order, order
                        )
                        if fill:
                            logger.info(f"✅ Position closed: {order.symbol} - Fill: {fill.fill_id}")
                        else:
                            logger.warning(f"❌ Failed to close position: {order.symbol}")
                    except Exception as e:
                        logger.error(f"Error closing position {order.symbol}: {e}")
                
                logger.info(f"🏁 End-of-backtest position closure completed: {len(close_orders)} orders processed")
            
        except Exception as e:
            logger.error(f"Error in position closure: {e}")
            import traceback
            traceback.print_exc()
    
    def get_capabilities(self) -> Set[str]:
        """Execution container capabilities."""
        capabilities = super().get_capabilities()
        execution_mode = self._metadata.config.get('mode', 'backtest')
        capabilities.add(f"execution.{execution_mode}")
        return capabilities


class RiskContainer(BaseComposableContainer):
    """Container for risk management and position sizing."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Add external event routing configuration for hybrid communication
        if 'external_events' not in config:
            config['external_events'] = {}
        
        # Declare what events this container publishes and subscribes to externally
        config['external_events']['publishes'] = config['external_events'].get('publishes', []) + [
            {
                'events': ['ORDER'],
                'scope': 'GLOBAL',  # Send to ExecutionContainer via Event Router
                'tier': 'standard'
            }
        ]
        
        config['external_events']['subscribes'] = config['external_events'].get('subscribes', []) + [
            {
                'source': '*',  # Accept from any source
                'events': ['SIGNAL'],
                'tier': 'standard'
            },
            {
                'source': '*',  # Accept from any source  
                'events': ['FILL'],
                'tier': 'standard'
            }
        ]
        
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
        
        # Try to get initial_capital from multiple locations in config
        initial_capital = None
        
        # Try direct config first
        initial_capital = self._metadata.config.get('initial_capital')
        
        # Try parent container config if available
        if not initial_capital and hasattr(self, '_parent_container') and self._parent_container:
            parent_config = getattr(self._parent_container, '_metadata', {}).config or {}
            initial_capital = parent_config.get('initial_capital')
        
        # Try backtest section if available (common location)
        if not initial_capital:
            initial_capital = self._metadata.config.get('backtest', {}).get('initial_capital')
        
        # Try portfolio section if available
        if not initial_capital:
            initial_capital = self._metadata.config.get('portfolio', {}).get('initial_capital')
            
        # Default fallback
        if not initial_capital:
            initial_capital = 100000
            
        initial_capital = Decimal(str(initial_capital))
        logger.info(f"RiskContainer using initial_capital: ${initial_capital}")
        
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
                    logger.info(f"   📈 Position: {symbol} {position.quantity} shares @ ${position.average_price:.2f}")
        
        elif event.event_type == EventType.SIGNAL and self.risk_manager:
            signals = event.payload.get('signals', [])
            market_data = event.payload.get('market_data', {})
            
            logger.info(f"🔥 RiskContainer received SIGNAL event with {len(signals)} signals")
            for signal in signals:
                # Handle both dict and object signal formats
                if hasattr(signal, 'symbol'):
                    logger.info(f"   📊 Signal: {signal.symbol} {signal.side} strength={signal.strength}")
                else:
                    # Signal is a dict
                    symbol = signal.get('symbol', 'UNKNOWN')
                    side = signal.get('direction', signal.get('side', 'UNKNOWN'))
                    strength = signal.get('strength', 0)
                    logger.info(f"   📊 Signal: {symbol} {side} strength={strength}")
            
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
            
            # Convert dict signals to Signal objects for risk manager
            signal_objects = []
            for signal in signals:
                if hasattr(signal, 'symbol'):
                    # Already a Signal object
                    signal_objects.append(signal)
                else:
                    # Convert dict to Signal object
                    from ..risk.protocols import Signal as RiskSignal, SignalType
                    from ..execution.protocols import OrderSide
                    
                    # Map direction string to OrderSide enum
                    direction = signal.get('direction', signal.get('side', 'BUY'))
                    if isinstance(direction, str):
                        side = OrderSide.BUY if direction.upper() in ['BUY', 'LONG'] else OrderSide.SELL
                    else:
                        side = direction
                    
                    # Map signal type string to enum
                    sig_type_str = signal.get('signal_type', 'entry')
                    signal_type = SignalType(sig_type_str) if isinstance(sig_type_str, str) else sig_type_str
                    
                    signal_obj = RiskSignal(
                        signal_id=signal.get('signal_id', f"consensus_{signal.get('symbol', 'UNKNOWN')}_{event.timestamp}"),
                        strategy_id=signal.get('strategy_id', 'consensus'),
                        symbol=signal.get('symbol', 'UNKNOWN'),
                        signal_type=signal_type,
                        side=side,
                        strength=signal.get('strength', 1.0),
                        timestamp=event.timestamp,
                        metadata=signal.get('metadata', {})
                    )
                    signal_objects.append(signal_obj)
            
            # Process signals through risk management
            orders = self.risk_manager.process_signals(signal_objects, transformed_market_data)
            
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
                
                # Publish ORDER event to ExecutionContainer via Event Router
                logger.info(f"📤 RiskContainer publishing ORDER event to ExecutionContainer via Event Router")
                from ..core.events.routing.protocols import EventScope
                self.publish_external(order_event)
                logger.info(f"   📨 Sent ORDER event to ExecutionContainer via Event Router")
                
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
        # Add external event routing configuration for hybrid communication
        if 'external_events' not in config:
            config['external_events'] = {}
        
        # Declare what events this container publishes and subscribes to externally
        config['external_events']['publishes'] = config['external_events'].get('publishes', []) + [
            {
                'events': ['SIGNAL'],
                'scope': 'GLOBAL',  # Send to RiskContainer via Event Router
                'tier': 'standard'
            }
        ]
        
        config['external_events']['subscribes'] = config['external_events'].get('subscribes', []) + [
            {
                'source': '*',  # Accept from any source
                'events': ['BAR', 'INDICATORS'],
                'tier': 'fast'
            }
        ]
        
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
        
        # Forward BAR and INDICATOR events to children (StrategyContainer)
        if event.event_type in [EventType.BAR, EventType.INDICATORS]:
            logger.debug(f"PortfolioContainer forwarding {event.event_type} to children")
            self.publish_event(event, target_scope="children")
        
        if event.event_type == EventType.SIGNAL and self.allocation_manager:
            signals = event.payload.get('signals', [])
            logger.info(f"📊 PortfolioContainer received SIGNAL event with {len(signals)} signals")
            
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
                
                # Forward to parent (risk container) via Event Router
                logger.info(f"📤 PortfolioContainer forwarding {len(allocated_signals)} signals to RiskContainer")
                from ..core.events.routing.protocols import EventScope
                self.publish_external(allocated_event)
                
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


# EnsembleContainer removed - functionality moved to enhanced StrategyContainer


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


# create_ensemble_container removed - functionality moved to enhanced StrategyContainer


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
    
    # Ensemble functionality moved to enhanced StrategyContainer


# Auto-register on import
register_execution_containers()