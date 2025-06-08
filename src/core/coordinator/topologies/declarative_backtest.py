"""
Declarative backtest topology following the agreed structure.

Shows the parent-child relationships clearly:
Root Container
├── Symbol_Timeframe Containers (publish BARs)
├── Feature Container (processes all signals)
├── Portfolio Containers (subscribe by strategy_id)
└── Execution Container
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def build_declarative_backtest_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build declarative backtest topology with proper parent-child structure.
    
    Returns a topology that can be directly used by the coordinator.
    """
    # Build the complete topology structure
    topology = {
        'name': 'backtest_topology',
        'root': {
            'name': 'backtest_root',
            'type': 'root',
            'config': {
                'workflow_id': config.get('workflow_id', 'backtest'),
                'enable_event_tracing': config.get('enable_event_tracing', False)
            },
            'children': []
        }
    }
    
    root_children = topology['root']['children']
    
    # 1. Add Symbol_Timeframe containers as children of root
    for symbol, timeframe in config.get('data_sources', []):
        symbol_tf_container = {
            'name': f'{symbol}_{timeframe}',
            'type': 'symbol_timeframe',
            'config': {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_file': config['data_files'].get(symbol),
                'components': [
                    {
                        'type': 'CSVDataLoader',
                        'config': {
                            'file_path': config['data_files'].get(symbol)
                        }
                    },
                    {
                        'type': 'DataStreamer',
                        'config': {
                            'batch_size': 1,
                            'publish_to': 'parent'  # Publish BARs to root
                        }
                    }
                ]
            }
        }
        root_children.append(symbol_tf_container)
    
    # 2. Add Feature Container as child of root
    feature_container = {
        'name': 'feature_processor',
        'type': 'feature',
        'config': {
            'components': [
                {
                    'type': 'TimeAlignmentBuffer',
                    'config': {
                        'strategy_requirements': _build_strategy_requirements(config),
                        'max_buffer_size': 1000
                    }
                },
                {
                    'type': 'FeatureCalculator',
                    'config': {
                        'indicators': config.get('indicators', ['sma', 'rsi', 'volatility'])
                    }
                },
                {
                    'type': 'SignalGeneratorComponent',
                    'config': {
                        'storage_enabled': config.get('store_signals', False),
                        'publish_to': 'parent'  # Publish signals to root
                    }
                }
            ],
            'subscriptions': [
                {
                    'event_type': 'BAR',
                    'source': 'parent'  # Subscribe to BARs from root
                }
            ],
            'stateless_functions': {
                'classifiers': config.get('classifiers', {}),
                'strategies': config.get('strategies', {})
            }
        }
    }
    root_children.append(feature_container)
    
    # 3. Add Portfolio Containers as children of root
    for portfolio_config in config.get('portfolios', []):
        portfolio_container = {
            'name': portfolio_config['id'],
            'type': 'portfolio',
            'config': {
                'initial_capital': portfolio_config.get('initial_capital', 100000),
                'risk_params': portfolio_config.get('risk_params', {}),
                'strategy_assignments': portfolio_config.get('strategy_assignments', []),
                'components': [
                    {
                        'type': 'PortfolioState',
                        'config': {
                            'track_pending_orders': True
                        }
                    },
                    {
                        'type': 'SignalProcessor',
                        'config': {
                            'position_sizer': portfolio_config.get('position_sizer', 'fixed')
                        }
                    },
                    {
                        'type': 'OrderGenerator',
                        'config': {
                            'publish_to': 'parent'  # Publish ORDER_REQUEST to root
                        }
                    }
                ],
                'subscriptions': [
                    {
                        'event_type': 'SIGNAL',
                        'source': 'parent',
                        'filter': {
                            'strategy_id': portfolio_config.get('strategy_assignments', [])
                        }
                    },
                    {
                        'event_type': 'FILL',
                        'source': 'parent'
                    }
                ],
                # Event tracing configuration
                'event_tracing': portfolio_config.get('event_tracing', [
                    'POSITION_OPEN', 'POSITION_CLOSE', 'FILL'
                ]),
                'retention_policy': portfolio_config.get('retention_policy', 'trade_complete')
            }
        }
        root_children.append(portfolio_container)
    
    # 4. Add Risk Validator as child of root (stateless)
    risk_container = {
        'name': 'risk_validator',
        'type': 'risk',
        'config': {
            'risk_limits': config.get('risk_limits', {}),
            'validation_mode': 'stateless',
            'components': [
                {
                    'type': 'RiskValidator',
                    'config': {
                        'max_position_size': 0.1,
                        'max_portfolio_risk': 0.02
                    }
                }
            ],
            'subscriptions': [
                {
                    'event_type': 'ORDER_REQUEST',
                    'source': 'parent'
                }
            ],
            'publications': [
                {
                    'event_type': 'ORDER',
                    'target': 'parent'
                },
                {
                    'event_type': 'ORDER_REJECTED',
                    'target': 'parent'
                }
            ]
        }
    }
    root_children.append(risk_container)
    
    # 5. Add Execution Container as child of root
    execution_container = {
        'name': 'execution_engine',
        'type': 'execution',
        'config': {
            'execution_mode': config.get('execution_mode', 'simulated'),
            'slippage_model': config.get('slippage', {}),
            'commission_model': config.get('commission', {}),
            'components': [
                {
                    'type': 'ExecutionEngine',
                    'config': {
                        'fill_ratio': 1.0  # Perfect fills in backtest
                    }
                }
            ],
            'subscriptions': [
                {
                    'event_type': 'ORDER',
                    'source': 'parent'
                }
            ],
            'publications': [
                {
                    'event_type': 'FILL',
                    'target': 'parent'
                }
            ]
        }
    }
    root_children.append(execution_container)
    
    # Add metadata
    topology['metadata'] = {
        'container_count': len(root_children) + 1,  # +1 for root
        'data_sources': config.get('data_sources', []),
        'portfolio_count': len(config.get('portfolios', [])),
        'event_flow': 'BAR -> Feature -> SIGNAL -> Portfolio -> ORDER_REQUEST -> Risk -> ORDER -> Execution -> FILL'
    }
    
    return topology


def build_declarative_signal_generation_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build declarative topology for signal generation only.
    
    Structure:
    Root Container
    ├── Symbol_Timeframe Containers (publish BARs)
    └── Feature Container (generates and stores signals)
    """
    topology = {
        'name': 'signal_generation_topology',
        'root': {
            'name': 'signal_gen_root',
            'type': 'root',
            'config': {
                'workflow_id': config.get('workflow_id', 'signal_gen'),
                'enable_event_tracing': True,
                'trace_settings': {
                    'events_to_trace': ['SIGNAL', 'CLASSIFICATION_CHANGE'],
                    'trace_dir': config.get('signal_output_dir', './signals')
                }
            },
            'children': []
        }
    }
    
    root_children = topology['root']['children']
    
    # 1. Add Symbol_Timeframe containers
    for symbol, timeframe in config.get('data_sources', []):
        symbol_tf_container = {
            'name': f'{symbol}_{timeframe}',
            'type': 'symbol_timeframe',
            'config': {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_file': config['data_files'].get(symbol),
                'components': [
                    {
                        'type': 'CSVDataLoader',
                        'config': {
                            'file_path': config['data_files'].get(symbol)
                        }
                    },
                    {
                        'type': 'DataStreamer',
                        'config': {
                            'batch_size': 1,
                            'publish_to': 'parent'
                        }
                    }
                ]
            }
        }
        root_children.append(symbol_tf_container)
    
    # 2. Add Feature Container with signal storage
    # Build strategy configs for grid search
    strategy_configs = _build_strategy_configs_for_grid_search(config)
    
    feature_container = {
        'name': 'feature_processor',
        'type': 'feature',
        'config': {
            'components': [
                {
                    'type': 'TimeAlignmentBuffer',
                    'config': {
                        'strategy_requirements': _build_strategy_requirements_with_grid(config, strategy_configs),
                        'max_buffer_size': 1000
                    }
                },
                {
                    'type': 'FeatureCalculator',
                    'config': {
                        'indicators': config.get('indicators', ['sma', 'rsi', 'volatility'])
                    }
                },
                {
                    'type': 'SignalGeneratorComponent',
                    'config': {
                        'storage_enabled': True,
                        'storage_path': config.get('signal_output_dir', './signals'),
                        'workflow_id': config.get('workflow_id', 'signal_gen'),
                        'strategy_configs': strategy_configs
                    }
                }
            ],
            'subscriptions': [
                {
                    'event_type': 'BAR',
                    'source': 'parent'
                }
            ],
            'stateless_functions': {
                'classifiers': config.get('classifiers', {}),
                'strategies': strategy_configs
            }
        }
    }
    root_children.append(feature_container)
    
    # Add metadata
    topology['metadata'] = {
        'container_count': len(root_children) + 1,
        'data_sources': config.get('data_sources', []),
        'total_strategies': len(strategy_configs),
        'signal_output_dir': config.get('signal_output_dir', './signals')
    }
    
    return topology


def build_declarative_signal_replay_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build declarative topology for signal replay.
    
    Structure:
    Root Container
    ├── Signal Replay Container (streams stored signals)
    ├── Portfolio Containers (process signals)
    ├── Risk Validator
    └── Execution Container
    """
    topology = {
        'name': 'signal_replay_topology',
        'root': {
            'name': 'replay_root',
            'type': 'root',
            'config': {
                'workflow_id': f"{config['workflow_id']}_replay",
                'replay_mode': True
            },
            'children': []
        }
    }
    
    root_children = topology['root']['children']
    
    # 1. Add Signal Replay Container
    replay_container = {
        'name': 'signal_streamer',
        'type': 'data',  # Reuse data type
        'config': {
            'components': [
                {
                    'type': 'SignalStreamerComponent',
                    'config': {
                        'signal_storage_path': config['signal_storage_path'],
                        'workflow_id': config['workflow_id'],
                        'strategy_filter': config.get('strategy_filter'),
                        'regime_filter': config.get('regime_filter'),
                        'sparse_replay': config.get('sparse_replay', True),
                        'publish_to': 'parent'
                    }
                },
                {
                    'type': 'ReplayCoordinator',
                    'config': {
                        'replay_speed': config.get('replay_speed', 1.0)
                    }
                }
            ]
        }
    }
    root_children.append(replay_container)
    
    # 2. Add Portfolio Containers
    for portfolio_config in config.get('portfolios', []):
        portfolio_container = {
            'name': portfolio_config['id'],
            'type': 'portfolio',
            'config': {
                'initial_capital': portfolio_config.get('initial_capital', 100000),
                'risk_params': portfolio_config.get('risk_params', {}),
                'strategy_assignments': portfolio_config.get('strategy_assignments', []),
                'components': [
                    {
                        'type': 'PortfolioState',
                        'config': {
                            'track_pending_orders': True
                        }
                    },
                    {
                        'type': 'SignalProcessor',
                        'config': {
                            'replay_mode': True  # Signals already have context
                        }
                    },
                    {
                        'type': 'OrderGenerator',
                        'config': {
                            'publish_to': 'parent'
                        }
                    }
                ],
                'subscriptions': [
                    {
                        'event_type': 'SIGNAL',
                        'source': 'parent',
                        'filter': {
                            'strategy_id': portfolio_config.get('strategy_assignments', [])
                        }
                    },
                    {
                        'event_type': 'FILL',
                        'source': 'parent'
                    }
                ],
                'event_tracing': portfolio_config.get('event_tracing', [
                    'POSITION_OPEN', 'POSITION_CLOSE', 'FILL'
                ]),
                'retention_policy': portfolio_config.get('retention_policy', 'trade_complete')
            }
        }
        root_children.append(portfolio_container)
    
    # 3. Add Risk and Execution (same as backtest)
    risk_container = {
        'name': 'risk_validator',
        'type': 'risk',
        'config': {
            'risk_limits': config.get('risk_limits', {}),
            'validation_mode': 'stateless',
            'components': [
                {
                    'type': 'RiskValidator',
                    'config': config.get('risk_validator', {})
                }
            ],
            'subscriptions': [
                {'event_type': 'ORDER_REQUEST', 'source': 'parent'}
            ],
            'publications': [
                {'event_type': 'ORDER', 'target': 'parent'},
                {'event_type': 'ORDER_REJECTED', 'target': 'parent'}
            ]
        }
    }
    root_children.append(risk_container)
    
    execution_container = {
        'name': 'execution_engine',
        'type': 'execution',
        'config': {
            'execution_mode': config.get('execution_mode', 'simulated'),
            'replay_mode': True,
            'components': [
                {
                    'type': 'ExecutionEngine',
                    'config': config.get('execution', {})
                }
            ],
            'subscriptions': [
                {'event_type': 'ORDER', 'source': 'parent'}
            ],
            'publications': [
                {'event_type': 'FILL', 'target': 'parent'}
            ]
        }
    }
    root_children.append(execution_container)
    
    # Add metadata
    topology['metadata'] = {
        'container_count': len(root_children) + 1,
        'source_workflow': config['workflow_id'],
        'portfolio_count': len(config.get('portfolios', [])),
        'filters': {
            'strategies': config.get('strategy_filter'),
            'regime': config.get('regime_filter')
        }
    }
    
    return topology


# Helper functions

def _build_strategy_requirements(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build strategy requirements for TimeAlignmentBuffer."""
    requirements = []
    
    for strategy in config.get('strategies', []):
        requirements.append({
            'strategy_id': strategy['id'],
            'required_data': strategy.get('required_data', []),
            'classifier_id': strategy.get('classifier_id'),
            'alignment_mode': 'wait_for_all'
        })
    
    return requirements


def _build_strategy_configs_for_grid_search(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build all strategy configurations including grid search expansion."""
    import itertools
    
    all_strategies = {}
    
    for strategy_config in config.get('strategies', []):
        base_name = strategy_config['name']
        base_params = strategy_config.get('base_parameters', {})
        param_grid = strategy_config.get('parameter_grid', {})
        
        if param_grid:
            # Generate all combinations
            param_names = list(param_grid.keys())
            param_values = [param_grid[name] for name in param_names]
            
            for values in itertools.product(*param_values):
                # Create unique strategy ID
                param_str = '_'.join(f"{name}_{value}" for name, value in zip(param_names, values))
                strategy_id = f"{base_name}_{param_str}"
                
                # Merge parameters
                parameters = base_params.copy()
                for name, value in zip(param_names, values):
                    parameters[name] = value
                
                all_strategies[strategy_id] = {
                    'name': base_name,
                    'function': strategy_config['function'],
                    'parameters': parameters,
                    'required_data': strategy_config.get('required_data', []),
                    'classifier_id': strategy_config.get('classifier_id')
                }
        else:
            # Single strategy
            strategy_id = f"{base_name}_default"
            all_strategies[strategy_id] = {
                'name': base_name,
                'function': strategy_config['function'],
                'parameters': base_params,
                'required_data': strategy_config.get('required_data', []),
                'classifier_id': strategy_config.get('classifier_id')
            }
    
    return all_strategies


def _build_strategy_requirements_with_grid(config: Dict[str, Any], 
                                         strategy_configs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build requirements for all strategy variants."""
    requirements = []
    
    for strategy_id, strategy_info in strategy_configs.items():
        requirements.append({
            'strategy_id': strategy_id,
            'strategy_function': strategy_info['function'],
            'required_data': strategy_info.get('required_data', []),
            'classifier_id': strategy_info.get('classifier_id'),
            'alignment_mode': 'wait_for_all'
        })
    
    return requirements