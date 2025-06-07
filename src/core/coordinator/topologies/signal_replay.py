"""
Signal replay topology creation.

Creates pipeline starting from saved signals: disk → portfolios → risk → execution
"""

from typing import Dict, Any, Optional
import logging

from ...container_factory import ContainerFactory
from .helpers.component_builder import create_stateless_components
from .helpers.routing import route_signal_replay_topology

logger = logging.getLogger(__name__)


def create_signal_replay_topology(config: Dict[str, Any], event_tracer: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create signal replay topology.
    
    This topology loads previously generated signals and executes them.
    
    Args:
        config: Configuration containing signal paths, risk profiles, execution settings
        event_tracer: Optional event tracer for debugging
        
    Returns:
        Dictionary with containers, routes, and configuration
    """
    logger.info("Creating signal replay topology")
    
    # Initialize topology structure
    topology = {
        'containers': {},
        'routes': [],
        'parameter_combinations': [],
        'stateless_components': {}
    }
    
    # Create stateless components (risk validators, execution models)
    logger.info("Creating stateless components")
    topology['stateless_components'] = create_stateless_components(config)
    config['stateless_components'] = topology['stateless_components']
    
    # Create containers
    container_factory = ContainerFactory()
    
    # Extract execution config to pass to all containers
    execution_config = config.get('execution', {})
    
    # 1. Create signal replay container
    signal_replay_config = {
        'type': 'signal_replay',
        'signal_directory': config.get('signal_directory', './results/signals/'),
        'signal_files': config.get('signal_files', []),  # Optional specific files
        'replay_speed': config.get('replay_speed', 'max'),  # 'max' or time multiplier
        'start_date': config.get('start_date'),
        'end_date': config.get('end_date'),
        'execution': execution_config  # Pass execution config for tracing
    }
    
    replay_container = container_factory.create_container('signal_replay', signal_replay_config)
    topology['containers']['signal_replay'] = replay_container
    
    # 2. Create portfolio containers based on risk profiles
    # Since strategies are already in the signals, we only need risk profiles
    risk_profiles = config.get('risk_profiles', [{'type': 'conservative'}])
    
    combo_id = 0
    for risk_config in risk_profiles:
        combo = {
            'combo_id': f"c{combo_id:04d}",
            'strategy_params': None,  # Strategy already determined in signals
            'risk_params': risk_config
        }
        topology['parameter_combinations'].append(combo)
        
        # Create portfolio container for this risk profile
        portfolio_name = f"portfolio_c{combo_id:04d}"
        portfolio_config = {
            'type': 'portfolio',
            'combo_id': f"c{combo_id:04d}",
            'risk_type': risk_config.get('type'),
            'risk_params': risk_config,
            'initial_capital': config.get('initial_capital', 100000),
            'stateless_components': topology['stateless_components'],
            'signal_replay_mode': True,  # Flag to indicate replay mode
            'execution': execution_config,  # Pass execution config for tracing
            'objective_function': config.get('objective_function', {'name': 'sharpe_ratio'}),  # Pass objective function
            'results': config.get('results', {}),  # Pass results config for metrics
            'metrics': config.get('metrics', {})  # Pass metrics config
        }
        portfolio_container = container_factory.create_container(portfolio_name, portfolio_config)
        topology['containers'][portfolio_name] = portfolio_container
        
        combo_id += 1
    
    logger.info(f"Created {len(topology['parameter_combinations'])} portfolio configurations")
    
    # 3. Create execution container
    exec_container_config = {
        'type': 'execution',
        'mode': 'replay',  # Special mode for replay
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
    
    topology['routes'] = route_signal_replay_topology(topology['containers'], config)
    
    # 5. Add event tracer if provided
    if event_tracer:
        topology['event_tracer'] = event_tracer
        # Register tracer with all containers
        for container in topology['containers'].values():
            if hasattr(container, 'event_bus'):
                event_tracer.register_bus(container.event_bus, container.name)
    
    logger.info(f"Created signal replay topology with {len(topology['containers'])} containers")
    
    return topology
