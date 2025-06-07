"""
Backtest topology creation.

Creates the full pipeline: data → features → strategies → portfolios → risk → execution
"""

from typing import Dict, Any, List, Optional
import logging

from ...container_factory import ContainerFactory
from .helpers.component_builder import create_stateless_components
from .helpers.routing import route_backtest_topology

logger = logging.getLogger(__name__)


def create_backtest_topology(config: Dict[str, Any], event_tracer: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create a complete backtest topology.
    
    Args:
        config: Configuration containing symbols, strategies, risk profiles, etc.
        event_tracer: Optional event tracer for debugging
        
    Returns:
        Dictionary with:
            - containers: All created containers
            - routes: Communication routes
            - parameter_combinations: Strategy/risk combinations
            - stateless_components: Strategy/risk/execution components
    """
    logger.info("Creating backtest topology")
    
    # Initialize topology structure
    topology = {
        'containers': {},
        'routes': [],
        'parameter_combinations': [],
        'stateless_components': {}
    }
    
    # Create stateless components first
    logger.info("Creating stateless components")
    topology['stateless_components'] = create_stateless_components(config)
    
    # Pass stateless components in config for container creation
    config['stateless_components'] = topology['stateless_components']
    
    # Create containers
    container_factory = ContainerFactory()
    
    # Extract execution config to pass to all containers
    execution_config = config.get('execution', {})
    
    # 1. Create data containers (one per symbol/timeframe)
    symbols = config.get('symbols', ['SPY'])
    if isinstance(symbols, str):
        symbols = [symbols]
    
    timeframes = config.get('timeframes', ['1T'])
    if isinstance(timeframes, str):
        timeframes = [timeframes]
    
    for symbol in symbols:
        for timeframe in timeframes:
            # Data container
            data_container_name = f"{symbol}_{timeframe}_data"
            data_config = {
                'type': 'data',
                'symbol': symbol,
                'timeframe': timeframe,
                'data_source': config.get('data_source', 'file'),
                'data_path': config.get('data_path'),
                'start_date': config.get('start_date'),
                'end_date': config.get('end_date'),
                'execution': execution_config  # Pass execution config for tracing
            }
            data_container = container_factory.create_container(data_container_name, data_config)
            topology['containers'][data_container_name] = data_container
            
            # Feature container (wired to data container)
            feature_container_name = f"{symbol}_{timeframe}_features"
            feature_config = {
                'type': 'features',
                'symbol': symbol,
                'timeframe': timeframe,
                'features': config.get('features', {}),
                'data_container': data_container,  # Direct reference for wiring
                'execution': execution_config  # Pass execution config for tracing
            }
            feature_container = container_factory.create_container(feature_container_name, feature_config)
            topology['containers'][feature_container_name] = feature_container
            
            logger.info(f"Created data and feature containers for {symbol}/{timeframe}")
    
    # 2. Create parameter combinations
    strategies = config.get('strategies', [{'type': 'momentum'}])
    risk_profiles = config.get('risk_profiles', [{'type': 'conservative'}])
    
    combo_id = 0
    for strategy_config in strategies:
        for risk_config in risk_profiles:
            combo = {
                'combo_id': f"c{combo_id:04d}",
                'strategy_params': strategy_config,
                'risk_params': risk_config
            }
            topology['parameter_combinations'].append(combo)
            
            # Create portfolio container for this combination
            portfolio_name = f"portfolio_c{combo_id:04d}"
            portfolio_config = {
                'type': 'portfolio',
                'combo_id': f"c{combo_id:04d}",
                'strategy_type': strategy_config.get('type'),
                'strategy_params': strategy_config,
                'risk_type': risk_config.get('type'),
                'risk_params': risk_config,
                'initial_capital': config.get('initial_capital', 100000),
                'stateless_components': topology['stateless_components'],  # Pass components
                'execution': execution_config,  # Pass execution config for tracing
                'objective_function': config.get('objective_function', {'name': 'sharpe_ratio'}),  # Pass objective function
                'results': config.get('results', {}),  # Pass results config for metrics
                'metrics': config.get('metrics', {})  # Pass metrics config
            }
            portfolio_container = container_factory.create_container(portfolio_name, portfolio_config)
            topology['containers'][portfolio_name] = portfolio_container
            
            combo_id += 1
    
    logger.info(f"Created {len(topology['parameter_combinations'])} parameter combinations")
    
    # 3. Create execution container
    exec_container_config = {
        'type': 'execution',
        'mode': 'backtest',
        'execution_models': config.get('execution_models', []),
        'stateless_components': topology['stateless_components'],
        'execution': execution_config  # Pass execution config for tracing
    }
    # Merge any additional execution settings from config
    exec_container_config.update(config.get('execution_container', {}))
    
    execution_container = container_factory.create_container('execution', exec_container_config)
    topology['containers']['execution'] = execution_container
    
    # 4. Route containers together
    logger.info("Routing containers together")
    
    # Add root event bus to config for routing
    if hasattr(container_factory, 'root_event_bus'):
        config['root_event_bus'] = container_factory.root_event_bus
    else:
        # Create a root event bus if needed
        from ...events import EventBus
        config['root_event_bus'] = EventBus()
    
    # Pass parameter combinations for routing
    config['parameter_combinations'] = topology['parameter_combinations']
    
    topology['routes'] = route_backtest_topology(topology['containers'], config)
    
    # 5. Add event tracer if provided
    if event_tracer:
        topology['event_tracer'] = event_tracer
        # Register tracer with all containers
        for container in topology['containers'].values():
            if hasattr(container, 'event_bus'):
                event_tracer.register_bus(container.event_bus, container.name)
    
    logger.info(f"Created backtest topology with {len(topology['containers'])} containers and {len(topology['routes'])} routes")
    
    return topology


def create_parameter_combinations(strategies: List[Dict], risk_profiles: List[Dict]) -> List[Dict]:
    """
    Create all combinations of strategies and risk profiles.
    
    Args:
        strategies: List of strategy configurations
        risk_profiles: List of risk profile configurations
        
    Returns:
        List of parameter combinations with unique IDs
    """
    combinations = []
    combo_id = 0
    
    for strategy in strategies:
        for risk in risk_profiles:
            combo = {
                'combo_id': f"c{combo_id:04d}",
                'strategy_params': strategy.copy(),
                'risk_params': risk.copy()
            }
            combinations.append(combo)
            combo_id += 1
    
    return combinations
