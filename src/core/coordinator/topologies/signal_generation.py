"""
Signal generation topology - just data → features → signals, then stop.

This topology is used for generating and storing signals without running
portfolio or execution logic. Useful for:
- Pre-computing signals for parameter optimization  
- Analyzing signal quality before backtesting
- Building signal datasets for machine learning
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import itertools
import functools
import operator

from ...containers.factory import ContainerFactory
from ...containers.container import Container, ContainerConfig, ContainerRole
from ..topology import create_stateless_components

logger = logging.getLogger(__name__)


def build_signal_generation_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build topology for signal generation only.
    
    This creates a truncated backtest topology that:
    1. Loads market data from symbol/timeframe containers
    2. Processes through feature container with TimeAlignmentBuffer
    3. Generates signals using stateless classifiers/strategies
    4. Captures signals via EventBus subscription and stores to disk
    5. Stops (no portfolio or execution)
    
    Args:
        config: Configuration dict with:
            - data_sources: List of (symbol, timeframe) tuples
            - data_files: Dict of symbol -> file path
            - strategies: List of strategy configurations
            - classifiers: List of classifier configurations  
            - signal_output_dir: Where to store signals
            - workflow_id: Unique identifier for this run
            
    Returns:
        Topology configuration dict
    """
    # Extract workflow metadata
    workflow_id = config.get('workflow_id', 'signal_gen')
    signal_output_dir = config.get('signal_output_dir', './signals')
    
    # Build topology structure
    topology = {
        'root_container': {
            'role': ContainerRole.BACKTEST,
            'config': {
                'workflow_id': workflow_id,
                'enable_event_tracing': config.get('enable_event_tracing', True),
                'trace_settings': {
                    'events_to_trace': ['SIGNAL', 'CLASSIFICATION_CHANGE'],
                    'trace_dir': signal_output_dir
                }
            }
        },
        'containers': {}
    }
    
    # 1. Create symbol/timeframe containers
    for symbol, timeframe in config.get('data_sources', []):
        container_id = f"{symbol}_{timeframe}"
        
        container_config = ContainerConfig(
            role=ContainerRole.DATA,
            name=container_id,
            config={
                'symbol': symbol,
                'timeframe': timeframe,
                'data_file': config['data_files'].get(symbol),
                'components': {
                    'data_loader': {
                        'type': 'CSVDataLoader',
                        'file_path': config['data_files'].get(symbol)
                    },
                    'data_streamer': {
                        'type': 'DataStreamer',
                        'batch_size': 1
                    }
                }
            }
        )
        topology['containers'][container_id] = container_config
    
    # 2. Build strategy configurations (with grid search)
    strategy_configs = _build_strategy_configs(config)
    
    # 3. Create feature container with signal generation
    feature_config = ContainerConfig(
        role=ContainerRole.FEATURE,
        name='feature_processor',
        config={
            'components': {
                'synchronizer': {
                    'type': 'TimeAlignmentBuffer',
                    'strategy_requirements': _build_strategy_requirements(config, strategy_configs)
                },
                'feature_calculator': {
                    'type': 'FeatureCalculator',
                    'indicators': config.get('indicators', ['sma', 'rsi', 'volatility'])
                },
                'signal_generator': {
                    'type': 'SignalGeneratorComponent',
                    'storage_enabled': True,
                    'storage_path': signal_output_dir,
                    'workflow_id': workflow_id
                }
            }
        }
    )
    topology['containers']['feature_processor'] = feature_config
    
    # 4. Configure stateless functions
    topology['stateless_functions'] = {
        'classifiers': _build_classifier_configs(config),
        'strategies': strategy_configs
    }
    
    # 5. Add metadata
    topology['metadata'] = {
        'workflow_id': workflow_id,
        'workflow_type': 'signal_generation',
        'output_dir': signal_output_dir,
        'data_sources': config.get('data_sources', []),
        'total_strategies': len(strategy_configs),
        'grid_search_size': calculate_grid_search_size(config)
    }
    
    logger.info(f"Built signal generation topology with {len(config.get('data_sources', []))} data sources "
               f"and {len(strategy_configs)} strategy variants")
    
    return topology


def _build_strategy_requirements(config: Dict[str, Any], strategy_configs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build strategy requirements for TimeAlignmentBuffer."""
    requirements = []
    
    # Create requirement for each strategy variant
    for strategy_id, strategy_info in strategy_configs.items():
        requirement = {
            'strategy_id': strategy_id,
            'strategy_function': strategy_info['function'],
            'required_data': strategy_info.get('required_data', []),
            'classifier_id': strategy_info.get('classifier_id'),
            'alignment_mode': 'wait_for_all'
        }
        requirements.append(requirement)
    
    return requirements


def _build_classifier_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build classifier configurations."""
    classifiers = {}
    
    for clf_config in config.get('classifiers', []):
        classifiers[clf_config['name']] = {
            'function': clf_config['function'],
            'parameters': clf_config.get('parameters', {})
        }
    
    return classifiers


def _build_strategy_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build strategy configurations for grid search."""
    strategies = {}
    
    for strategy_config in config.get('strategies', []):
        base_name = strategy_config['name']
        base_params = strategy_config.get('base_parameters', {})
        
        # Check if this is a grid search
        param_grid = strategy_config.get('parameter_grid', {})
        
        if param_grid:
            # Generate all parameter combinations
            param_names = list(param_grid.keys())
            param_values = [param_grid[name] for name in param_names]
            
            # Generate all combinations
            for i, values in enumerate(itertools.product(*param_values)):
                # Create unique strategy ID
                param_str = '_'.join(f"{name}_{value}" for name, value in zip(param_names, values))
                strategy_id = f"{base_name}_{param_str}"
                
                # Merge parameters
                parameters = base_params.copy()
                for name, value in zip(param_names, values):
                    parameters[name] = value
                
                strategies[strategy_id] = {
                    'name': base_name,
                    'function': strategy_config['function'],
                    'parameters': parameters,
                    'required_data': strategy_config.get('required_data', []),
                    'classifier_id': strategy_config.get('classifier_id')
                }
        else:
            # Single strategy instance
            strategy_id = f"{base_name}_default"
            strategies[strategy_id] = {
                'name': base_name,
                'function': strategy_config['function'],
                'parameters': base_params,
                'required_data': strategy_config.get('required_data', []),
                'classifier_id': strategy_config.get('classifier_id')
            }
    
    return strategies


def calculate_grid_search_size(config: Dict[str, Any]) -> int:
    """Calculate total number of strategy variants in grid search."""
    total = 0
    
    for strategy_config in config.get('strategies', []):
        param_grid = strategy_config.get('parameter_grid', {})
        
        if param_grid:
            # Calculate combinations
            combinations = functools.reduce(
                operator.mul,
                [len(values) for values in param_grid.values()],
                1
            )
            total += combinations
        else:
            total += 1
    
    return total
