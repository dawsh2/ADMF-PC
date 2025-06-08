"""
Backtest topology creation.

Creates the full pipeline: data → features → strategies → portfolios → risk → execution
"""

from typing import Dict, Any, List, Optional
import logging

from ...container_factory import ContainerFactory
from ..topology import create_stateless_components

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
            - parameter_combinations: Strategy/risk combinations
            - stateless_components: Strategy/risk/execution components
    """
    logger.info("Creating backtest topology")
    
    # Initialize topology structure
    topology = {
        'containers': {},
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
    
    # 4. Wire up event subscriptions
    logger.info("Setting up event subscriptions")
    
    # Get or create root event bus
    if hasattr(container_factory, 'root_event_bus'):
        root_bus = container_factory.root_event_bus
    else:
        # Create a root event bus if needed
        from ...events import EventBus
        root_bus = EventBus()
    
    # Setup subscriptions for the backtest flow
    setup_backtest_subscriptions(topology['containers'], root_bus, topology['parameter_combinations'])
    
    # 5. Add event tracer if provided
    if event_tracer:
        topology['event_tracer'] = event_tracer
        # Register tracer with all containers
        for container in topology['containers'].values():
            if hasattr(container, 'event_bus'):
                event_tracer.register_bus(container.event_bus, container.name)
    
    logger.info(f"Created backtest topology with {len(topology['containers'])} containers")
    
    return topology


def setup_backtest_subscriptions(containers: Dict[str, Any], root_bus: Any, parameter_combinations: List[Dict]) -> None:
    """
    Setup event subscriptions for backtest topology.
    
    Since we no longer have FEATURES events and strategies are functional,
    the flow is simpler:
    1. Feature containers process BAR events and call strategies directly
    2. Strategies return signals which are published to root bus
    3. Portfolios subscribe to SIGNAL events with filters
    4. Risk validation happens in portfolio containers
    5. Execution subscribes to ORDER events
    6. Portfolios subscribe to FILL events
    """
    from ...events import EventType
    
    # Get container groups
    portfolio_containers = {k: v for k, v in containers.items() if 'portfolio_' in k}
    execution_container = containers.get('execution')
    
    # 1. Portfolios subscribe to SIGNAL events with strategy filters
    for portfolio_name, portfolio in portfolio_containers.items():
        # Get the strategy assignments for this portfolio
        combo_id = portfolio.config.config.get('combo_id')
        strategy_type = portfolio.config.config.get('strategy_type')
        
        # Find matching parameter combination
        matching_combo = next((c for c in parameter_combinations if c['combo_id'] == combo_id), None)
        if matching_combo:
            # Create strategy ID for filtering
            strategy_id = f"{combo_id}_{strategy_type}"
            
            # Get signal processor component
            signal_processor = portfolio.get_component('signal_processor')
            if signal_processor and hasattr(signal_processor, 'on_signal'):
                # Subscribe with filter for this portfolio's strategy
                # Note: This would use the enhanced EventBus with filtering
                # For now, using standard subscription
                root_bus.subscribe(EventType.SIGNAL, signal_processor.on_signal)
                logger.info(f"Portfolio {portfolio_name} subscribed to signals from strategy {strategy_id}")
    
    # 2. Execution subscribes to ORDER events
    if execution_container:
        execution_engine = execution_container.get_component('execution_engine')
        if execution_engine and hasattr(execution_engine, 'on_order'):
            root_bus.subscribe(EventType.ORDER, execution_engine.on_order)
            logger.info("Execution engine subscribed to ORDER events")
    
    # 3. Portfolios subscribe to FILL events
    for portfolio_name, portfolio in portfolio_containers.items():
        portfolio_state = portfolio.get_component('portfolio_state')
        if portfolio_state and hasattr(portfolio_state, 'on_fill'):
            root_bus.subscribe(EventType.FILL, portfolio_state.on_fill)
            logger.info(f"Portfolio {portfolio_name} subscribed to FILL events")


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
