"""
Backtest topology creation with container-based tracing.

Shows how tracing configuration flows from user config to containers,
not from orchestration.
"""

from typing import Dict, Any, List
import logging

from ...container_factory import ContainerFactory
from .helpers.component_builder import create_stateless_components
from .helpers.routing import route_backtest_topology

logger = logging.getLogger(__name__)


def create_backtest_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a backtest topology with optional container tracing.
    
    Tracing is configured through the config, not passed from orchestration.
    
    Args:
        config: Configuration containing:
            - execution.enable_event_tracing: Whether to enable tracing
            - execution.trace_settings: Tracing configuration
            - metadata: Context metadata from orchestration (optional)
            
    Returns:
        Topology with containers that may have tracing enabled
    """
    logger.info("Creating backtest topology")
    
    # Extract tracing configuration from execution settings
    execution_config = config.get('execution', {})
    enable_tracing = execution_config.get('enable_event_tracing', False)
    trace_settings = execution_config.get('trace_settings', {})
    
    # Extract metadata (passed from orchestration for context only)
    metadata = config.get('metadata', {})
    
    if enable_tracing:
        logger.info(f"Tracing enabled for containers with settings: {trace_settings}")
    
    # Initialize topology structure
    topology = {
        'containers': {},
        'routes': [],
        'parameter_combinations': [],
        'stateless_components': {}
    }
    
    # Create stateless components first
    topology['stateless_components'] = create_stateless_components(config)
    
    # Pass stateless components in config for container creation
    config['stateless_components'] = topology['stateless_components']
    
    # Create containers
    container_factory = ContainerFactory()
    
    # 1. Create data containers
    symbols = config.get('symbols', ['SPY'])
    if isinstance(symbols, str):
        symbols = [symbols]
    
    for symbol in symbols:
        # Data container config with tracing
        data_container_name = f"{symbol}_data"
        data_config = {
            'type': 'data',
            'symbol': symbol,
            'data_source': config.get('data_source', 'file'),
            'data_path': config.get('data_path'),
            'start_date': config.get('start_date'),
            'end_date': config.get('end_date'),
            # Container tracing configuration
            'enable_tracing': enable_tracing,
            'trace_settings': trace_settings,
            # Metadata for context
            'metadata': metadata
        }
        
        data_container = container_factory.create_container(data_container_name, data_config)
        topology['containers'][data_container_name] = data_container
        
        # Feature container config with tracing
        feature_container_name = f"{symbol}_features"
        feature_config = {
            'type': 'features',
            'symbol': symbol,
            'features': config.get('features', {}),
            'data_container': data_container,
            # Container tracing configuration
            'enable_tracing': enable_tracing,
            'trace_settings': trace_settings,
            # Metadata for context
            'metadata': metadata
        }
        
        feature_container = container_factory.create_container(feature_container_name, feature_config)
        topology['containers'][feature_container_name] = feature_container
    
    # 2. Create portfolio containers with parameter combinations
    strategies = config.get('strategies', [{'type': 'momentum'}])
    risk_profiles = config.get('risk_profiles', [{'type': 'conservative'}])
    
    # Check if this is an optimization (parameter space provided)
    if 'parameter_space' in config:
        # Use optimizer to expand parameters
        from ....optimization import ParameterOptimizer
        optimizer = ParameterOptimizer(method=config.get('expansion_method', 'grid'))
        param_combinations = optimizer.expand_parameters(config['parameter_space'])
        
        # Create container for each combination
        for i, params in enumerate(param_combinations):
            combo_id = f"c{i:04d}"
            
            topology['parameter_combinations'].append({
                'container_id': f"portfolio_{combo_id}",
                'parameters': params
            })
            
            portfolio_config = create_portfolio_config(
                combo_id=combo_id,
                parameters=params,
                base_config=config,
                enable_tracing=enable_tracing,
                trace_settings=trace_settings,
                metadata=metadata
            )
            
            portfolio_container = container_factory.create_container(
                f"portfolio_{combo_id}", 
                portfolio_config
            )
            topology['containers'][f"portfolio_{combo_id}"] = portfolio_container
            
        # Add optimization info to topology
        topology['optimization'] = {
            'parameter_combinations': topology['parameter_combinations'],
            'objective': config.get('optimization', {}).get('objective', 'sharpe_ratio'),
            'constraints': config.get('optimization', {}).get('constraints', {})
        }
    else:
        # Normal execution - create combinations of strategies and risk profiles
        combo_id = 0
        for strategy_config in strategies:
            for risk_config in risk_profiles:
                combo = {
                    'combo_id': f"c{combo_id:04d}",
                    'strategy_params': strategy_config,
                    'risk_params': risk_config
                }
                topology['parameter_combinations'].append(combo)
                
                portfolio_config = {
                    'type': 'portfolio',
                    'combo_id': f"c{combo_id:04d}",
                    'strategy_type': strategy_config.get('type'),
                    'strategy_params': strategy_config,
                    'risk_type': risk_config.get('type'),
                    'risk_params': risk_config,
                    'initial_capital': config.get('initial_capital', 100000),
                    'stateless_components': topology['stateless_components'],
                    # Container tracing configuration
                    'enable_tracing': enable_tracing,
                    'trace_settings': trace_settings,
                    # Metadata for context
                    'metadata': metadata
                }
                
                portfolio_container = container_factory.create_container(
                    f"portfolio_c{combo_id:04d}", 
                    portfolio_config
                )
                topology['containers'][f"portfolio_c{combo_id:04d}"] = portfolio_container
                
                combo_id += 1
    
    # 3. Create execution container with tracing
    execution_config = {
        'type': 'execution',
        'mode': 'backtest',
        'execution_models': config.get('execution_models', []),
        'stateless_components': topology['stateless_components'],
        # Container tracing configuration
        'enable_tracing': enable_tracing,
        'trace_settings': trace_settings,
        # Metadata for context
        'metadata': metadata
    }
    
    execution_container = container_factory.create_container('execution', execution_config)
    topology['containers']['execution'] = execution_container
    
    # 4. Route containers together
    topology['routes'] = route_backtest_topology(topology['containers'], config)
    
    logger.info(
        f"Created backtest topology with {len(topology['containers'])} containers "
        f"(tracing {'enabled' if enable_tracing else 'disabled'})"
    )
    
    return topology


def create_portfolio_config(
    combo_id: str,
    parameters: Dict[str, Any],
    base_config: Dict[str, Any],
    enable_tracing: bool,
    trace_settings: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create portfolio container configuration with parameters and tracing.
    
    This shows how optimization parameters are merged with tracing config.
    """
    # Extract strategy and risk parameters
    strategy_params = {}
    risk_params = {}
    
    for param_path, value in parameters.items():
        if param_path.startswith('strategy.'):
            key = param_path.replace('strategy.', '')
            strategy_params[key] = value
        elif param_path.startswith('risk.'):
            key = param_path.replace('risk.', '')
            risk_params[key] = value
    
    return {
        'type': 'portfolio',
        'combo_id': combo_id,
        'strategy_params': strategy_params,
        'risk_params': risk_params,
        'initial_capital': base_config.get('initial_capital', 100000),
        'stateless_components': base_config.get('stateless_components', {}),
        # Container tracing configuration
        'enable_tracing': enable_tracing,
        'trace_settings': trace_settings,
        # Metadata for context
        'metadata': metadata
    }