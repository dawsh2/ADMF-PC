"""
Helper for routing containers together.

This module handles creating communication routes between containers
based on the topology structure.
"""

from typing import Dict, Any, List
import logging

from ....routing.factory import RoutingFactory
from ....events import EventType

logger = logging.getLogger(__name__)


def route_backtest_topology(containers: Dict[str, Any], config: Dict[str, Any]) -> List[Any]:
    """
    Route containers for backtest topology.
    
    Pipeline: data → features → strategies → portfolios → risk → execution
    """
    routing_factory = RoutingFactory()
    routes = []
    root_event_bus = config.get('root_event_bus')
    
    if not root_event_bus:
        raise ValueError("Root event bus required for routing")
    
    # 1. Route Data → Features
    data_containers = {k: v for k, v in containers.items() if '_data' in k}
    feature_containers = {k: v for k, v in containers.items() if '_features' in k}
    
    for data_id, data_container in data_containers.items():
        # Find matching feature container
        symbol_timeframe = data_id.replace('_data', '')
        feature_id = f"{symbol_timeframe}_features"
        
        if feature_id in feature_containers:
            # Already routed in topology creation
            logger.debug(f"Data → Features already routed for {symbol_timeframe}")
    
    # 2. Create Feature Filter for strategies
    portfolio_containers = {k: v for k, v in containers.items() if 'portfolio_' in k}
    
    if feature_containers and portfolio_containers:
        # Create feature filter route
        feature_filter = routing_factory.create_route(
            name='feature_filter',
            config={
                'type': 'filter',
                'filter_field': 'payload.features',
                'event_types': [EventType.FEATURES]
            }
        )
        
        # Register strategies with filter based on config
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
                
                # Create strategy transform function
                def create_strategy_transform(sid, stype, sconfig):
                    def transform(event):
                        # Create strategy instance
                        from ....components.factory import create_component
                        strategy = create_component(
                            stype,
                            context={
                                'event_bus': root_event_bus,
                                'config': sconfig
                            },
                            capabilities=['events']
                        )
                        
                        # Process features
                        if hasattr(strategy, 'handle_features'):
                            result = strategy.handle_features(event)
                            if result:
                                # Create signal event
                                return Event(
                                    event_type=EventType.SIGNAL,
                                    timestamp=event.timestamp,
                                    payload=result,
                                    metadata={
                                        'strategy_id': sid,
                                        'strategy_type': stype,
                                        'source_event': event.event_id
                                    }
                                )
                        return None
                    return transform
                
                # Register strategy requirements
                strategy_id = f"{combo_id}_{strategy_type}"
                feature_filter.register_requirements(
                    target_id=strategy_id,
                    required_keys=required_features,
                    transform=create_strategy_transform(strategy_id, strategy_type, strategy_config)
                )
        
        # Set up the filter with containers
        feature_filter.setup(containers)
        
        # Subscribe filter to feature events from each feature container
        for feature_container_name, feature_container in feature_containers.items():
            feature_container.event_bus.subscribe(EventType.FEATURES, feature_filter.handle_event)
            logger.info(f"Routed {feature_container_name} → Feature Filter")
        
        routes.append(feature_filter)
        
        # Route dispatcher to strategies (handled internally by dispatcher)
        # Route portfolios to receive signals from root bus
        for portfolio_name, portfolio_container in portfolio_containers.items():
            # Get signal processor component
            signal_processor = portfolio_container.get_component('signal_processor')
            if signal_processor and hasattr(signal_processor, 'on_signal'):
                root_event_bus.subscribe(EventType.SIGNAL, signal_processor.on_signal)
                logger.info(f"Routed Signal events → {portfolio_name}")
    
    # 3. Route Portfolios → Risk → Execution
    if 'risk_manager' in containers:
        # Create risk service route
        risk_validators = stateless_components.get('risk_validators', {})
        risk_route = routing_factory.create_route(
            name='risk_service',
            config={
                'type': 'risk_service',
                'risk_validators': risk_validators,
                'root_event_bus': root_event_bus
            }
        )
        routes.append(risk_route)
        
        # Subscribe execution to ORDER events
        if 'execution' in containers:
            execution = containers['execution']
            execution_engine = execution.get_component('execution_engine')
            if execution_engine and hasattr(execution_engine, 'on_order'):
                root_event_bus.subscribe(EventType.ORDER, execution_engine.on_order)
                logger.info("Routed ORDER events → Execution")
    else:
        # Direct portfolio → execution
        if 'execution' in containers and portfolio_containers:
            execution = containers['execution']
            execution_engine = execution.get_component('execution_engine')
            if execution_engine and hasattr(execution_engine, 'on_order'):
                root_event_bus.subscribe(EventType.ORDER, execution_engine.on_order)
                logger.info("Routed ORDER events → Execution (direct)")
    
    # 4. Route Execution → Portfolios (fill broadcast)
    if 'execution' in containers and portfolio_containers:
        fill_broadcast = routing_factory.create_route(
            name='fill_broadcast',
            config={
                'type': 'broadcast',
                'source': 'execution',
                'targets': list(portfolio_containers.keys()),
                'allowed_types': [EventType.FILL]
            }
        )
        routes.append(fill_broadcast)
        logger.info(f"Created fill broadcast to {len(portfolio_containers)} portfolios")
    
    # Route and start routes
    all_containers = containers
    for route in routes:
        try:
            route.setup(all_containers)
            route.start()
        except Exception as e:
            logger.error(f"Failed to setup route {route.name}: {e}")
            raise
    
    return routes


def route_signal_generation_topology(containers: Dict[str, Any], config: Dict[str, Any]) -> List[Any]:
    """
    Route containers for signal generation topology.
    
    Pipeline: data → features → strategies (signals saved to disk)
    """
    routing_factory = RoutingFactory()
    routes = []
    root_event_bus = config.get('root_event_bus')
    
    if not root_event_bus:
        raise ValueError("Root event bus required for routing")
    
    # Get containers
    data_containers = {k: v for k, v in containers.items() if '_data' in k}
    feature_containers = {k: v for k, v in containers.items() if '_features' in k}
    
    # Data → Features routing already done in topology creation
    
    # Create Feature Filter for strategies
    if feature_containers:
        # Create feature filter route
        feature_filter = routing_factory.create_route(
            name='feature_filter',
            config={
                'type': 'filter',
                'filter_field': 'payload.features',
                'event_types': [EventType.FEATURES]
            }
        )
        
        # Register strategies with filter based on config
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
                
                # Create strategy transform function
                def create_strategy_transform(sid, stype, sconfig):
                    def transform(event):
                        # Create strategy instance
                        from ....components.factory import create_component
                        strategy = create_component(
                            stype,
                            context={
                                'event_bus': root_event_bus,
                                'config': sconfig
                            },
                            capabilities=['events']
                        )
                        
                        # Process features
                        if hasattr(strategy, 'handle_features'):
                            result = strategy.handle_features(event)
                            if result:
                                # Create signal event
                                return Event(
                                    event_type=EventType.SIGNAL,
                                    timestamp=event.timestamp,
                                    payload=result,
                                    metadata={
                                        'strategy_id': sid,
                                        'strategy_type': stype,
                                        'source_event': event.event_id
                                    }
                                )
                        return None
                    return transform
                
                # Register strategy requirements
                strategy_id = f"{combo_id}_{strategy_type}"
                feature_filter.register_requirements(
                    target_id=strategy_id,
                    required_keys=required_features,
                    transform=create_strategy_transform(strategy_id, strategy_type, strategy_config)
                )
        
        # Set up the filter with containers
        feature_filter.setup(containers)
        
        # Subscribe filter to feature events from each feature container
        for feature_container_name, feature_container in feature_containers.items():
            feature_container.event_bus.subscribe(EventType.FEATURES, feature_filter.handle_event)
            logger.info(f"Routed {feature_container_name} → Feature Filter")
        
        routes.append(feature_filter)
        
        # Create signal saver route to capture signals
        signal_saver = routing_factory.create_route(
            name='signal_saver',
            config={
                'type': 'signal_saver',
                'save_directory': config.get('signal_save_directory', './results/signals/'),
                'root_event_bus': root_event_bus
            }
        )
        routes.append(signal_saver)
        logger.info("Created signal saver to capture strategy signals")
    
    # Route and start routes
    for route in routes:
        try:
            route.setup(containers)
            route.start()
        except Exception as e:
            logger.error(f"Failed to setup route {route.name}: {e}")
            raise
    
    logger.info(f"Signal generation topology routed: {len(routes)} routes created")
    return routes


def route_signal_replay_topology(containers: Dict[str, Any], config: Dict[str, Any]) -> List[Any]:
    """
    Route containers for signal replay topology.
    
    Pipeline: saved signals → portfolios → risk → execution
    """
    routing_factory = RoutingFactory()
    routes = []
    root_event_bus = config.get('root_event_bus')
    
    if not root_event_bus:
        raise ValueError("Root event bus required for routing")
    
    # Get containers
    portfolio_containers = {k: v for k, v in containers.items() if 'portfolio_' in k}
    replay_container = containers.get('signal_replay')
    
    # 1. Route Signal Replay → Portfolios
    if replay_container and portfolio_containers:
        # Signal replay publishes to root bus, portfolios subscribe
        for portfolio_name, portfolio_container in portfolio_containers.items():
            signal_processor = portfolio_container.get_component('signal_processor')
            if signal_processor and hasattr(signal_processor, 'on_signal'):
                root_event_bus.subscribe(EventType.SIGNAL, signal_processor.on_signal)
                logger.info(f"Routed Signal events → {portfolio_name}")
    
    # 2. Route Portfolios → Risk (if risk validators exist)
    stateless_components = config.get('stateless_components', {})
    risk_validators = stateless_components.get('risk_validators', {})
    
    if risk_validators:
        # Create risk service route
        risk_route = routing_factory.create_route(
            name='risk_service',
            config={
                'type': 'risk_service',
                'risk_validators': risk_validators,
                'root_event_bus': root_event_bus
            }
        )
        routes.append(risk_route)
        logger.info(f"Created risk service with {len(risk_validators)} validators")
    
    # 3. Route Risk/Portfolios → Execution
    if 'execution' in containers:
        execution = containers['execution']
        execution_engine = execution.get_component('execution_engine')
        if execution_engine and hasattr(execution_engine, 'on_order'):
            root_event_bus.subscribe(EventType.ORDER, execution_engine.on_order)
            logger.info("Routed ORDER events → Execution")
    
    # 4. Route Execution → Portfolios (fill broadcast)
    if 'execution' in containers and portfolio_containers:
        fill_broadcast = routing_factory.create_route(
            name='fill_broadcast',
            config={
                'type': 'broadcast', 
                'source': 'execution',
                'targets': list(portfolio_containers.keys()),
                'allowed_types': [EventType.FILL]
            }
        )
        routes.append(fill_broadcast)
        logger.info(f"Created fill broadcast to {len(portfolio_containers)} portfolios")
    
    # Route and start routes
    for route in routes:
        try:
            route.setup(containers)
            route.start()
        except Exception as e:
            logger.error(f"Failed to setup route {route.name}: {e}")
            raise
    
    logger.info(f"Signal replay topology routed: {len(routes)} routes created")
    return routes