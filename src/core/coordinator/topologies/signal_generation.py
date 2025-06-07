"""
Signal generation topology creation.

Creates pipeline that stops at signals: data → features → strategies → disk
"""

from typing import Dict, Any, Optional
import logging

from ...container_factory import ContainerFactory
from .helpers.component_builder import create_stateless_components
from .helpers.routing import route_signal_generation_topology

logger = logging.getLogger(__name__)


def create_signal_generation_topology(config: Dict[str, Any], event_tracer: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create signal generation topology.
    
    This topology generates and saves signals without portfolio management or execution.
    
    Args:
        config: Configuration containing symbols, strategies, output settings
        event_tracer: Optional event tracer for debugging
        
    Returns:
        Dictionary with containers, routes, and configuration
    """
    logger.info("Creating signal generation topology")
    
    # Initialize topology structure
    topology = {
        'containers': {},
        'routes': [],
        'parameter_combinations': [],
        'stateless_components': {}
    }
    
    # Create stateless components (strategies)
    logger.info("Creating stateless components")
    topology['stateless_components'] = create_stateless_components(config)
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
            
            # Feature container (routed to data container)
            feature_container_name = f"{symbol}_{timeframe}_features"
            feature_config = {
                'type': 'features',
                'symbol': symbol,
                'timeframe': timeframe,
                'features': config.get('features', {}),
                'data_container': data_container,  # Direct reference for routing
                'execution': execution_config  # Pass execution config for tracing
            }
            feature_container = container_factory.create_container(feature_container_name, feature_config)
            topology['containers'][feature_container_name] = feature_container
            
            logger.info(f"Created data and feature containers for {symbol}/{timeframe}")
    
    # 2. Create parameter combinations (for strategies only)
    strategies = config.get('strategies', [{'type': 'momentum'}])
    
    combo_id = 0
    for strategy_config in strategies:
        combo = {
            'combo_id': f"c{combo_id:04d}",
            'strategy_params': strategy_config,
            'risk_params': None  # No risk profiles for signal generation
        }
        topology['parameter_combinations'].append(combo)
        combo_id += 1
    
    logger.info(f"Created {len(topology['parameter_combinations'])} strategy configurations")
    
    # 3. Configure signal saving
    config['signal_save_directory'] = config.get('signal_output_dir', './results/signals/')
    config['save_signals'] = True
    
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
    
    topology['routes'] = route_signal_generation_topology(topology['containers'], config)
    
    # 5. Add event tracer if provided
    if event_tracer:
        topology['event_tracer'] = event_tracer
        # Register tracer with all containers
        for container in topology['containers'].values():
            if hasattr(container, 'event_bus'):
                event_tracer.register_bus(container.event_bus, container.name)
    
    logger.info(f"Created signal generation topology with {len(topology['containers'])} containers")
    
    return topology
