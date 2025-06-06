"""
Helper for wiring containers together with adapters.

This module handles creating communication adapters between containers
based on the topology structure.
"""

from typing import Dict, Any, List
import logging

from ....communication.factory import AdapterFactory
from ....events import EventType

logger = logging.getLogger(__name__)


def wire_backtest_topology(containers: Dict[str, Any], config: Dict[str, Any]) -> List[Any]:
    """
    Wire containers for backtest topology.
    
    Pipeline: data → features → strategies → portfolios → risk → execution
    """
    adapter_factory = AdapterFactory()
    adapters = []
    root_event_bus = config.get('root_event_bus')
    
    if not root_event_bus:
        raise ValueError("Root event bus required for wiring")
    
    # 1. Wire Data → Features
    data_containers = {k: v for k, v in containers.items() if '_data' in k}
    feature_containers = {k: v for k, v in containers.items() if '_features' in k}
    
    for data_id, data_container in data_containers.items():
        # Find matching feature container
        symbol_timeframe = data_id.replace('_data', '')
        feature_id = f"{symbol_timeframe}_features"
        
        if feature_id in feature_containers:
            # Already wired in topology creation
            logger.debug(f"Data → Features already wired for {symbol_timeframe}")
    
    # 2. Create Feature Dispatcher for strategies
    portfolio_containers = {k: v for k, v in containers.items() if 'portfolio_' in k}
    
    if feature_containers and portfolio_containers:
        from ....components.feature_dispatcher import FeatureDispatcher
        feature_dispatcher = FeatureDispatcher(root_event_bus=root_event_bus)
        
        # Register strategies with dispatcher based on config
        stateless_components = config.get('stateless_components', {})
        strategies = stateless_components.get('strategies', {})
        
        for combo in config.get('parameter_combinations', []):
            strategy_config = combo['strategy_params']
            strategy_type = strategy_config.get('type')
            combo_id = combo['combo_id']
            
            if strategy_type and strategy_type in strategies:
                # Get feature requirements from helper
                from .component_builder import get_strategy_feature_requirements
                required_features = get_strategy_feature_requirements(strategy_type, strategy_config)
                
                feature_dispatcher.register_strategy(
                    strategy_id=f"{combo_id}_{strategy_type}",
                    strategy_type=strategy_type,
                    required_features=required_features
                )
        
        # Wire feature containers to dispatcher
        for feature_container_name, feature_container in feature_containers.items():
            feature_container.event_bus.subscribe(EventType.FEATURES, feature_dispatcher.handle_features)
            logger.info(f"Wired {feature_container_name} → Feature Dispatcher")
        
        # Wire dispatcher to strategies (handled internally by dispatcher)
        # Wire portfolios to receive signals from root bus
        for portfolio_name, portfolio_container in portfolio_containers.items():
            # Get signal processor component
            signal_processor = portfolio_container.get_component('signal_processor')
            if signal_processor and hasattr(signal_processor, 'on_signal'):
                root_event_bus.subscribe(EventType.SIGNAL, signal_processor.on_signal)
                logger.info(f"Wired Signal events → {portfolio_name}")
    
    # 3. Wire Portfolios → Risk → Execution
    if 'risk_manager' in containers:
        # Create risk service adapter
        risk_validators = stateless_components.get('risk_validators', {})
        risk_adapter = adapter_factory.create_adapter(
            name='risk_service',
            config={
                'type': 'risk_service',
                'risk_validators': risk_validators,
                'root_event_bus': root_event_bus
            }
        )
        adapters.append(risk_adapter)
        
        # Subscribe execution to ORDER events
        if 'execution' in containers:
            execution = containers['execution']
            execution_engine = execution.get_component('execution_engine')
            if execution_engine and hasattr(execution_engine, 'on_order'):
                root_event_bus.subscribe(EventType.ORDER, execution_engine.on_order)
                logger.info("Wired ORDER events → Execution")
    else:
        # Direct portfolio → execution
        if 'execution' in containers and portfolio_containers:
            execution = containers['execution']
            execution_engine = execution.get_component('execution_engine')
            if execution_engine and hasattr(execution_engine, 'on_order'):
                root_event_bus.subscribe(EventType.ORDER, execution_engine.on_order)
                logger.info("Wired ORDER events → Execution (direct)")
    
    # 4. Wire Execution → Portfolios (fill broadcast)
    if 'execution' in containers and portfolio_containers:
        fill_broadcast = adapter_factory.create_adapter(
            name='fill_broadcast',
            config={
                'type': 'broadcast',
                'source': 'execution',
                'targets': list(portfolio_containers.keys()),
                'allowed_types': [EventType.FILL]
            }
        )
        adapters.append(fill_broadcast)
        logger.info(f"Created fill broadcast to {len(portfolio_containers)} portfolios")
    
    # Wire and start adapters
    all_containers = containers
    for adapter in adapters:
        try:
            adapter.setup(all_containers)
            adapter.start()
        except Exception as e:
            logger.error(f"Failed to setup adapter {adapter.name}: {e}")
            raise
    
    return adapters


def wire_signal_generation_topology(containers: Dict[str, Any], config: Dict[str, Any]) -> List[Any]:
    """
    Wire containers for signal generation topology.
    
    Pipeline: data → features → strategies (signals saved to disk)
    """
    adapter_factory = AdapterFactory()
    adapters = []
    root_event_bus = config.get('root_event_bus')
    
    if not root_event_bus:
        raise ValueError("Root event bus required for wiring")
    
    # Get containers
    data_containers = {k: v for k, v in containers.items() if '_data' in k}
    feature_containers = {k: v for k, v in containers.items() if '_features' in k}
    
    # Data → Features wiring already done in topology creation
    
    # Create Feature Dispatcher for strategies
    if feature_containers:
        from ....components.feature_dispatcher import FeatureDispatcher
        feature_dispatcher = FeatureDispatcher(root_event_bus=root_event_bus)
        
        # Register strategies with dispatcher based on config
        stateless_components = config.get('stateless_components', {})
        strategies = stateless_components.get('strategies', {})
        
        for combo in config.get('parameter_combinations', []):
            strategy_config = combo['strategy_params']
            strategy_type = strategy_config.get('type')
            combo_id = combo['combo_id']
            
            if strategy_type and strategy_type in strategies:
                # Get feature requirements from helper
                from .component_builder import get_strategy_feature_requirements
                required_features = get_strategy_feature_requirements(strategy_type, strategy_config)
                
                feature_dispatcher.register_strategy(
                    strategy_id=f"{combo_id}_{strategy_type}",
                    strategy_type=strategy_type,
                    required_features=required_features
                )
        
        # Wire feature containers to dispatcher
        for feature_container_name, feature_container in feature_containers.items():
            feature_container.event_bus.subscribe(EventType.FEATURES, feature_dispatcher.handle_features)
            logger.info(f"Wired {feature_container_name} → Feature Dispatcher")
        
        # Create signal saver adapter to capture signals
        signal_saver = adapter_factory.create_adapter(
            name='signal_saver',
            config={
                'type': 'signal_saver',
                'save_directory': config.get('signal_save_directory', './results/signals/'),
                'root_event_bus': root_event_bus
            }
        )
        adapters.append(signal_saver)
        logger.info("Created signal saver to capture strategy signals")
    
    # Wire and start adapters
    for adapter in adapters:
        try:
            adapter.setup(containers)
            adapter.start()
        except Exception as e:
            logger.error(f"Failed to setup adapter {adapter.name}: {e}")
            raise
    
    logger.info(f"Signal generation topology wired: {len(adapters)} adapters created")
    return adapters


def wire_signal_replay_topology(containers: Dict[str, Any], config: Dict[str, Any]) -> List[Any]:
    """
    Wire containers for signal replay topology.
    
    Pipeline: saved signals → portfolios → risk → execution
    """
    adapter_factory = AdapterFactory()
    adapters = []
    root_event_bus = config.get('root_event_bus')
    
    if not root_event_bus:
        raise ValueError("Root event bus required for wiring")
    
    # Get containers
    portfolio_containers = {k: v for k, v in containers.items() if 'portfolio_' in k}
    replay_container = containers.get('signal_replay')
    
    # 1. Wire Signal Replay → Portfolios
    if replay_container and portfolio_containers:
        # Signal replay publishes to root bus, portfolios subscribe
        for portfolio_name, portfolio_container in portfolio_containers.items():
            signal_processor = portfolio_container.get_component('signal_processor')
            if signal_processor and hasattr(signal_processor, 'on_signal'):
                root_event_bus.subscribe(EventType.SIGNAL, signal_processor.on_signal)
                logger.info(f"Wired Signal events → {portfolio_name}")
    
    # 2. Wire Portfolios → Risk (if risk validators exist)
    stateless_components = config.get('stateless_components', {})
    risk_validators = stateless_components.get('risk_validators', {})
    
    if risk_validators:
        # Create risk service adapter
        risk_adapter = adapter_factory.create_adapter(
            name='risk_service',
            config={
                'type': 'risk_service',
                'risk_validators': risk_validators,
                'root_event_bus': root_event_bus
            }
        )
        adapters.append(risk_adapter)
        logger.info(f"Created risk service with {len(risk_validators)} validators")
    
    # 3. Wire Risk/Portfolios → Execution
    if 'execution' in containers:
        execution = containers['execution']
        execution_engine = execution.get_component('execution_engine')
        if execution_engine and hasattr(execution_engine, 'on_order'):
            root_event_bus.subscribe(EventType.ORDER, execution_engine.on_order)
            logger.info("Wired ORDER events → Execution")
    
    # 4. Wire Execution → Portfolios (fill broadcast)
    if 'execution' in containers and portfolio_containers:
        fill_broadcast = adapter_factory.create_adapter(
            name='fill_broadcast',
            config={
                'type': 'broadcast', 
                'source': 'execution',
                'targets': list(portfolio_containers.keys()),
                'allowed_types': [EventType.FILL]
            }
        )
        adapters.append(fill_broadcast)
        logger.info(f"Created fill broadcast to {len(portfolio_containers)} portfolios")
    
    # Wire and start adapters
    for adapter in adapters:
        try:
            adapter.setup(containers)
            adapter.start()
        except Exception as e:
            logger.error(f"Failed to setup adapter {adapter.name}: {e}")
            raise
    
    logger.info(f"Signal replay topology wired: {len(adapters)} adapters created")
    return adapters