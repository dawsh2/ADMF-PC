"""
Simple Container Factory - Component-Based

This is a simplified container factory that creates containers based on 
component composition rather than predefined roles.
"""

from typing import Dict, List, Any, Optional, Set
import logging
from dataclasses import dataclass, field

from .container import Container, ContainerConfig

logger = logging.getLogger(__name__)


@dataclass
class ComponentSpec:
    """Specification for a component to be injected into a container."""
    name: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)


class ContainerFactory:
    """
    Simplified container factory focused on component composition.
    
    Creates containers by injecting specified components rather than 
    following predefined role patterns.
    """
    
    def __init__(self):
        # Registry of available component factories
        self._component_factories: Dict[str, Any] = {}
        self._default_components = self._load_default_components()
        
        # Store system configuration for access during component creation
        self._system_config = {}
    
    def set_system_config(self, config: Dict[str, Any]) -> None:
        """Set the system configuration for access during component creation."""
        self._system_config = config
    
    def _load_default_components(self) -> Dict[str, ComponentSpec]:
        """Load default component specifications."""
        return {
            # Data components
            'data_streamer': ComponentSpec('data_streamer', 'data.streamers.BarStreamer'),
            'signal_streamer': ComponentSpec('signal_streamer', 'data.streamers.SignalStreamer'),
            
            # Strategy components
            'strategy_state': ComponentSpec('strategy_state', 'strategy.state.StrategyState'),
            
            # Feature Hub component
            'feature_hub': ComponentSpec('feature_hub', 'core.containers.components.FeatureHubComponent'),
            
            # Portfolio components
            'portfolio_state': ComponentSpec('portfolio_state', 'portfolio.PortfolioState'),
            
            # Risk components
            'risk_manager': ComponentSpec('risk_manager', 'risk.RiskManager'),
            'position_sizer': ComponentSpec('position_sizer', 'risk.PositionSizer'),
            
            # Execution components
            'execution_engine': ComponentSpec('execution_engine', 'execution.ExecutionEngine'),
            'order_manager': ComponentSpec('order_manager', 'execution.OrderManager'),
            
            # Analytics components
            'metrics_collector': ComponentSpec('metrics_collector', 'analytics.MetricsCollector'),
            'performance_analyzer': ComponentSpec('performance_analyzer', 'analytics.PerformanceAnalyzer'),
        }
    
    def create_container(
        self, 
        name: str, 
        components: List[str],
        config: Optional[Dict[str, Any]] = None,
        container_type: Optional[str] = None,
        parent_event_bus: Optional[Any] = None
    ) -> Container:
        """
        Create a container with specified components.
        
        Args:
            name: Container name
            components: List of component names to inject
            config: Optional container configuration
            container_type: Optional explicit container type (will be inferred if not provided)
            parent_event_bus: Optional parent event bus (None for root container)
            
        Returns:
            Configured container instance
        """
        container_config = ContainerConfig(
            name=name,
            components=components,
            config=config or {},
            container_type=container_type
        )
        
        container = Container(container_config, parent_event_bus=parent_event_bus)
        
        # Inject specified components
        for component_name in components:
            if component_name in self._default_components:
                component = self._create_component(component_name, config)
                if component:
                    container.add_component(component_name, component)
                    logger.info(f"✅ Injected {component_name} into container {name} (component type: {type(component).__name__})")
            else:
                logger.warning(f"Unknown component: {component_name}")
        
        # Wire components together if both feature_hub and strategy_state exist
        if 'feature_hub' in components and 'strategy_state' in components:
            feature_hub_component = container.get_component('feature_hub')
            strategy_state = container.get_component('strategy_state')
            
            if feature_hub_component and strategy_state:
                # Connect ComponentState directly to FeatureHub in same container
                if hasattr(feature_hub_component, 'get_feature_hub'):
                    feature_hub = feature_hub_component.get_feature_hub()
                    strategy_state._feature_hub = feature_hub
                    logger.info(f"Wired strategy_state to feature_hub in container {name}")
        
        # Wire portfolio and risk manager together
        if 'portfolio_state' in components and 'risk_manager' in components:
            portfolio_state = container.get_component('portfolio_state')
            risk_manager = container.get_component('risk_manager')
            
            if portfolio_state and risk_manager:
                # Connect portfolio to risk manager
                if hasattr(portfolio_state, 'set_risk_manager'):
                    portfolio_state.set_risk_manager(risk_manager)
                    logger.info(f"Wired portfolio_state to risk_manager in container {name}")
                else:
                    logger.warning(f"Portfolio state has no set_risk_manager method")
        
        return container
    
    def _create_data_handler(self, data_source: str, config: Dict[str, Any]) -> Any:
        """
        Dynamically create data handler based on data source configuration.
        
        Args:
            data_source: Type of data source (file, alpaca_websocket, etc.)
            config: Configuration dictionary
            
        Returns:
            Data handler instance
        """
        # Data source to handler mapping
        data_handler_map = {
            'file': self._create_file_data_handler,
            'local': self._create_file_data_handler,  # Alias for file
            'alpaca_websocket': self._create_alpaca_data_handler,
            'live': self._create_alpaca_data_handler,  # Alias for alpaca_websocket
            'signal_replay': self._create_signal_replay_handler,  # New signal replay mode
        }
        
        handler_creator = data_handler_map.get(data_source)
        if not handler_creator:
            logger.warning(f"Unknown data source '{data_source}', falling back to file handler")
            handler_creator = self._create_file_data_handler
        
        return handler_creator(config)
    
    def _create_file_data_handler(self, config: Dict[str, Any]) -> Any:
        """Create file-based historical data handler."""
        from ...data.handlers import SimpleHistoricalDataHandler
        
        # Pass config to handler including dataset and split_ratio
        handler = SimpleHistoricalDataHandler(
            handler_id=f"data_{config.get('symbol', 'unknown')}",
            data_dir=config.get('data_dir', './data'),
            dataset=config.get('dataset'),
            split_ratio=config.get('split_ratio', 0.8)
        )
        
        # Configure symbols - handle both 'symbol' and 'symbols' fields
        symbols = config.get('symbol') or config.get('symbols', [])
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # If we have a data file specification, use it directly
        data_file = config.get('data_file') or config.get('file')
        if data_file:
            # Direct file specification takes precedence
            handler.symbols = [data_file]
            logger.info(f"📄 Using direct file specification: {data_file}")
        else:
            handler.symbols = symbols or []
        
        # Set max bars if configured
        if 'max_bars' in config:
            handler.max_bars = config['max_bars']
        
        # Configure WFV window if specified
        if all(k in config for k in ['wfv_window', 'wfv_windows', 'wfv_phase']):
            wfv_dataset = config.get('wfv_dataset', 'train')
            handler.setup_wfv_window(
                window_num=config['wfv_window'],
                total_windows=config['wfv_windows'],
                phase=config['wfv_phase'],
                dataset_split=wfv_dataset
            )
        
        logger.info(f"📁 Created file data handler for symbols: {handler.symbols}")
        return handler
    
    def _create_alpaca_data_handler(self, config: Dict[str, Any]) -> Any:
        """Create Alpaca WebSocket live data handler."""
        try:
            from ...data.live_data_handler import LiveDataHandler
            
            # Debug logging
            logger.debug(f"Alpaca handler config: {config}")
            logger.debug(f"System config: {self._system_config}")
            
            # Get symbols - check multiple places
            symbols = config.get('symbol', config.get('symbols', []))
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Get live trading config from system config
            live_trading_config = self._system_config.get('live_trading', {})
            
            # Create live data handler with Alpaca configuration
            handler = LiveDataHandler(
                handler_id=f"live_data_{'+'.join(symbols)}",
                symbols=symbols or ['SPY'],
                live_config=live_trading_config,
                config=config
            )
            
            logger.info(f"🔴 Created Alpaca live data handler for symbols: {handler.symbols}")
            return handler
            
        except ImportError as e:
            logger.error(f"Failed to import LiveDataHandler: {e}")
            logger.info("Falling back to file data handler")
            return self._create_file_data_handler(config)
    
    def _create_signal_replay_handler(self, config: Dict[str, Any]) -> Any:
        """Create signal replay handler that reads pre-computed traces."""
        from ...data.handlers import SignalReplayHandler
        
        # Create handler
        handler = SignalReplayHandler(
            handler_id=f"signal_replay_{config.get('symbol', 'unknown')}",
            traces_dir=config.get('traces_dir', './traces')
        )
        
        # Configure symbols
        symbols = config.get('symbol') or config.get('symbols', [])
        if isinstance(symbols, str):
            symbols = [symbols]
        handler.symbols = symbols or []
        
        # Get strategy configurations from config
        strategy_configs = []
        if 'strategies' in config:
            # Strategies already parsed by clean syntax parser
            strategy_configs = config['strategies']
        elif 'parameter_space' in config and 'strategies' in config['parameter_space']:
            # Strategies in parameter space format
            strategy_configs = config['parameter_space']['strategies']
        
        logger.info(f"📈 Created signal replay handler for symbols: {handler.symbols}")
        logger.info(f"📈 Will replay signals from {len(strategy_configs)} strategies")
        
        # Store strategy configs for later loading
        handler.strategy_configs = strategy_configs
        
        return handler

    def _create_component(self, component_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Create a component instance.
        
        Creates real components when available, falls back to mock for missing ones.
        """
        config = config or {}
        
        try:
            # Map component names to actual implementations
            if component_name == 'data_streamer':
                # Dynamic data handler creation based on config
                data_source = config.get('data_source', 'file')
                logger.info(f"🔍 Creating data_streamer for data_source: {data_source}")
                logger.debug(f"🔍 Full component config: {config}")
                
                # Create data handler based on configuration
                handler = self._create_data_handler(data_source, config)
                return handler
            elif component_name == 'signal_streamer':
                from ...data.streamers import SignalStreamerComponent
                return SignalStreamerComponent()
            elif component_name == 'portfolio_state':
                from ...portfolio import PortfolioState
                portfolio = PortfolioState()
                self._last_portfolio = portfolio  # Store for execution engine
                logger.info(f"✅ Created PortfolioState instance: {id(portfolio)}")
                return portfolio
            elif component_name == 'risk_manager':
                from ...risk.strategy_risk_manager import StrategyRiskManager
                return StrategyRiskManager()
            elif component_name == 'position_sizer':
                from ...risk import FixedPositionSizer
                return FixedPositionSizer()
            elif component_name == 'execution_engine':
                # Determine broker type from config
                broker_type = config.get('broker', 'simulated')
                live_config = self._system_config.get('live_trading', {})
                execution_mode = config.get('execution_mode', 'sync')
                
                
                if broker_type == 'alpaca' and execution_mode == 'async':
                    # Use async execution engine with Alpaca
                    from ...execution.asynchronous.brokers.alpaca_clean import create_alpaca_broker
                    from ...execution.asynchronous.clean_engine import create_async_execution_engine
                    
                    # Create async Alpaca broker
                    broker = create_alpaca_broker(
                        api_key=live_config.get('api_key'),
                        secret_key=live_config.get('secret_key'),
                        paper_trading=live_config.get('paper_trading', True)
                    )
                    
                    # Get portfolio reference if available
                    portfolio = None
                    if hasattr(self, '_last_portfolio'):
                        portfolio = self._last_portfolio
                    
                    # Create async execution engine with adapter
                    adapter = create_async_execution_engine(
                        component_id=f"exec_{component_name}",
                        broker=broker,
                        portfolio=portfolio
                    )
                    
                    logger.info("Created async Alpaca execution engine")
                    return adapter
                    
                elif broker_type == 'alpaca' and live_config:
                    # For sync mode with Alpaca, use simulated broker for now
                    logger.warning("Alpaca broker requested but sync mode selected. Using simulated broker.")
                    from ...execution.synchronous.broker import SimulatedBroker
                    from ...execution import ExecutionEngine
                    
                    broker = SimulatedBroker(
                        broker_id="alpaca_simulated",
                        slippage_model=None,
                        commission_model=None,
                        liquidity_model=None
                    )
                    logger.info("Created simulated broker (sync Alpaca broker pending implementation)")
                    return ExecutionEngine(f"exec_{component_name}", broker)
                    
                else:
                    # Default to simulated broker
                    from ...execution.synchronous.broker import SimulatedBroker
                    from ...execution import ExecutionEngine
                    
                    execution_config = config.get('execution', {})
                    broker = SimulatedBroker(
                        broker_id="simulated",
                        slippage_model=None,
                        commission_model=None,
                        liquidity_model=None
                    )
                    logger.info("Created simulated broker")
                    return ExecutionEngine(f"exec_{component_name}", broker)
            elif component_name == 'order_manager':
                from ...execution import SyncOrderManager
                return SyncOrderManager(component_id="order_manager")
            elif component_name == 'strategy':
                from ...strategy.strategies import NullStrategy
                return NullStrategy()
            elif component_name == 'strategy_state' or component_name == 'component_state':
                from ...strategy.state import ComponentState
                return ComponentState(
                    symbols=config.get('symbols', []),
                    feature_configs=config.get('features', {}),
                    verbose_signals=True  # Enable signal logging to console
                )
            elif component_name == 'feature_hub':
                from .components.feature_hub_component import create_feature_hub_component
                return create_feature_hub_component(config)
            else:
                logger.error(f"Unknown component: {component_name}")
                raise ValueError(f"Unknown component: {component_name}")
                
        except ImportError as e:
            logger.error(f"Failed to import {component_name}: {e}")
            raise
    
    
    
    def create_portfolio_container(
        self, 
        name: str,
        strategies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Container:
        """
        Convenience method to create a portfolio container.
        
        Args:
            name: Container name
            strategies: List of strategy IDs this portfolio manages
            config: Optional configuration
            
        Returns:
            Portfolio container
        """
        components = ['portfolio_state', 'risk_manager']
        
        portfolio_config = config or {}
        if strategies:
            portfolio_config['managed_strategies'] = strategies
        
        return self.create_container(
            name=name,
            components=components,
            config=portfolio_config,
            container_type='portfolio'
        )
    
    def create_strategy_container(
        self, 
        name: str,
        strategy_type: str = 'momentum',
        config: Optional[Dict[str, Any]] = None
    ) -> Container:
        """
        Convenience method to create a strategy container.
        
        Args:
            name: Container name
            strategy_type: Type of strategy
            config: Optional configuration
            
        Returns:
            Strategy container
        """
        components = ['strategy']
        
        # Add classifier if needed
        if config and config.get('use_classifier', False):
            components.append('classifier')
        
        strategy_config = config or {}
        strategy_config['strategy_type'] = strategy_type
        
        return self.create_container(
            name=name,
            components=components,
            config=strategy_config,
            container_type='strategy'
        )
    
    def create_data_container(
        self, 
        name: str,
        symbols: List[str],
        timeframes: List[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Container:
        """
        Convenience method to create a data container.
        
        Args:
            name: Container name
            symbols: List of symbols to stream
            timeframes: List of timeframes (default: ['1m'])
            config: Optional configuration
            
        Returns:
            Data container
        """
        components = ['data_streamer']
        
        data_config = config or {}
        data_config.update({
            'symbols': symbols,
            'timeframes': timeframes or ['1m']
        })
        
        return self.create_container(
            name=name,
            components=components,
            config=data_config,
            container_type='data'
        )


# Global factory instance
_factory = ContainerFactory()


def create_container(name: str, components: List[str], **kwargs) -> Container:
    """Global function to create containers."""
    return _factory.create_container(name, components, **kwargs)


def create_portfolio_container(name: str, strategies: List[str] = None, **kwargs) -> Container:
    """Global function to create portfolio containers."""
    return _factory.create_portfolio_container(name, strategies, **kwargs)


def create_strategy_container(name: str, strategy_type: str = 'momentum', **kwargs) -> Container:
    """Global function to create strategy containers."""
    return _factory.create_strategy_container(name, strategy_type, **kwargs)


def create_data_container(name: str, symbols: List[str], **kwargs) -> Container:
    """Global function to create data containers.""" 
    return _factory.create_data_container(name, symbols, **kwargs)


# Utility functions for container setup
def setup_simple_container(container: Container, components: List[Any]) -> None:
    """Setup a container with basic components."""
    for component in components:
        if hasattr(component, 'name'):
            container.add_component(component.name, component)
        else:
            # Use class name as component name
            name = component.__class__.__name__.lower()
            container.add_component(name, component)


def create_symbol_group_requirement(symbols: List[str], timeframe: str = '1m'):
    """Create requirement for a group of symbols."""
    # Import here to avoid circular dependency
    from ..events.barriers import DataRequirement, AlignmentMode
    return DataRequirement(
        symbols=symbols,
        timeframes=[timeframe],
        alignment_mode=AlignmentMode.ALL
    )


def create_multi_timeframe_requirement(symbol: str, timeframes: List[str]):
    """Create requirement for multiple timeframes of same symbol."""
    from ..events.barriers import DataRequirement, AlignmentMode
    return DataRequirement(
        symbols=[symbol],
        timeframes=timeframes,
        alignment_mode=AlignmentMode.ALL
    )


def create_pairs_requirement(symbol1: str, symbol2: str, timeframe: str = '1m'):
    """Create requirement for a pair of symbols."""
    from ..events.barriers import DataRequirement, AlignmentMode
    return DataRequirement(
        symbols=[symbol1, symbol2],
        timeframes=[timeframe],
        alignment_mode=AlignmentMode.ALL
    )
