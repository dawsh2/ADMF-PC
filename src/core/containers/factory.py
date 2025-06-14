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
    
    def _load_default_components(self) -> Dict[str, ComponentSpec]:
        """Load default component specifications."""
        return {
            # Data components
            'data_streamer': ComponentSpec('data_streamer', 'data.streamers.BarStreamer'),
            'signal_streamer': ComponentSpec('signal_streamer', 'data.streamers.SignalStreamer'),
            
            # Strategy components
            'strategy_state': ComponentSpec('strategy_state', 'strategy.state.StrategyState'),
            
            # Portfolio components
            'portfolio_manager': ComponentSpec('portfolio_manager', 'portfolio.PortfolioManager'),
            'position_manager': ComponentSpec('position_manager', 'portfolio.PositionManager'),
            
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
                    logger.debug(f"Injected {component_name} into container {name}")
            else:
                logger.warning(f"Unknown component: {component_name}")
        
        return container
    
    def _create_component(self, component_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Create a component instance.
        
        Creates real components when available, falls back to mock for missing ones.
        """
        config = config or {}
        
        try:
            # Map component names to actual implementations
            if component_name == 'data_streamer':
                from ...data.handlers import SimpleHistoricalDataHandler
                # Pass config to handler
                handler = SimpleHistoricalDataHandler(
                    handler_id=f"data_{config.get('symbol', 'unknown')}",
                    data_dir=config.get('data_dir', './data')
                )
                # Load data if symbols are configured
                symbols = config.get('symbol', [])
                if isinstance(symbols, str):
                    symbols = [symbols]
                if symbols:
                    handler.load_data(symbols)
                # Set max bars if configured
                if 'max_bars' in config:
                    handler.max_bars = config['max_bars']
                
                # Configure train/test split if specified
                if config.get('split_ratio'):
                    handler.setup_split(train_ratio=config['split_ratio'])
                    if config.get('dataset'):
                        handler.set_active_split(config['dataset'])
                
                # Configure WFV window if specified
                if all(k in config for k in ['wfv_window', 'wfv_windows', 'wfv_phase']):
                    wfv_dataset = config.get('wfv_dataset', 'train')
                    handler.setup_wfv_window(
                        window_num=config['wfv_window'],
                        total_windows=config['wfv_windows'],
                        phase=config['wfv_phase'],
                        dataset_split=wfv_dataset
                    )
                
                return handler
            elif component_name == 'signal_streamer':
                from ...data.streamers import SignalStreamerComponent
                return SignalStreamerComponent()
            elif component_name == 'portfolio_manager':
                from ...portfolio import PortfolioState
                return PortfolioState()
            elif component_name == 'position_manager':
                # Position manager is part of PortfolioState
                from ...portfolio import PortfolioState
                return PortfolioState()
            elif component_name == 'risk_manager':
                from ...risk import RiskLimits
                return RiskLimits()
            elif component_name == 'position_sizer':
                from ...risk import FixedPositionSizer
                return FixedPositionSizer()
            elif component_name == 'execution_engine':
                from ...execution import ExecutionEngine
                # ExecutionEngine needs component_id and broker
                from ...execution.brokers import SimulatedBroker
                broker = SimulatedBroker(config.get('execution', {}))
                return ExecutionEngine(f"exec_{component_name}", broker)
            elif component_name == 'order_manager':
                from ...execution import SyncOrderManager
                return SyncOrderManager()
            elif component_name == 'strategy':
                from ...strategy.strategies import NullStrategy
                return NullStrategy()
            elif component_name == 'strategy_state' or component_name == 'component_state':
                from ...strategy.state import ComponentState
                return ComponentState(
                    symbols=config.get('symbols', []),
                    feature_configs=config.get('features', {})
                )
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
        components = ['portfolio_manager', 'position_manager', 'risk_manager']
        
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
