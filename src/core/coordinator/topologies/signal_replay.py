"""
Signal replay topology - streams stored signals to portfolios.

This topology replays previously generated signals, optionally filtering
by regime or strategy, to test different portfolio configurations without
regenerating signals.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from ...containers.container import ContainerConfig, ContainerRole

logger = logging.getLogger(__name__)


def build_signal_replay_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build topology for signal replay.
    
    Skips data and feature processing entirely:
    - Signal replay container(s) stream stored signals
    - Portfolio containers receive filtered signals
    - Execution container processes orders
    
    Args:
        config: Configuration dict with:
            - signal_storage_path: Base path to stored signals
            - workflow_id: ID of signal generation run to replay
            - portfolios: List of portfolio configurations
            - strategy_filter: Optional list of strategy_ids to replay
            - regime_filter: Optional regime to filter by
            - sparse_replay: Whether to skip bars without signals
            
    Returns:
        Topology configuration dict
    """
    # Extract configuration
    signal_path = Path(config['signal_storage_path'])
    workflow_id = config['workflow_id']
    
    # Build topology structure
    topology = {
        'root_container': {
            'role': ContainerRole.BACKTEST,
            'config': {
                'workflow_id': f"{workflow_id}_replay",
                'replay_mode': True
            }
        },
        'containers': {}
    }
    
    # 1. Create signal replay container
    replay_config = ContainerConfig(
        role=ContainerRole.DATA,  # Reuse DATA role
        name='signal_replay',
        config={
            'components': {
                'signal_streamer': {
                    'type': 'SignalStreamerComponent',
                    'signal_storage_path': signal_path,
                    'workflow_id': workflow_id,
                    'strategy_filter': config.get('strategy_filter'),
                    'regime_filter': config.get('regime_filter'),
                    'sparse_replay': config.get('sparse_replay', True)
                },
                'coordinator': {
                    'type': 'CoordinatorComponent',
                    'mode': 'replay',
                    'replay_speed': config.get('replay_speed', 1.0)
                }
            }
        }
    )
    topology['containers']['signal_replay'] = replay_config
    
    # 2. Create portfolio containers
    for i, portfolio_config in enumerate(config.get('portfolios', [])):
        portfolio_id = portfolio_config.get('id', f'portfolio_{i}')
        
        # Determine which strategies this portfolio handles
        strategy_assignments = portfolio_config.get('strategy_assignments', [])
        
        container_config = ContainerConfig(
            role=ContainerRole.PORTFOLIO,
            name=portfolio_id,
            config={
                'initial_capital': portfolio_config.get('initial_capital', 100000),
                'risk_params': portfolio_config.get('risk_params', {}),
                'strategy_assignments': strategy_assignments,
                # Event tracing configuration
                'event_tracing': portfolio_config.get('event_tracing', [
                    'POSITION_OPEN', 'POSITION_CLOSE', 'FILL'
                ]),
                'retention_policy': portfolio_config.get('retention_policy', 'trade_complete'),
                # Signal filtering
                'signal_filter': {
                    'strategy_ids': strategy_assignments
                }
            }
        )
        topology['containers'][portfolio_id] = container_config
    
    # 3. Create risk validation container (stateless)
    risk_config = ContainerConfig(
        role=ContainerRole.RISK,
        name='risk_validator',
        config={
            'risk_limits': config.get('risk_limits', {}),
            'validation_mode': 'stateless'
        }
    )
    topology['containers']['risk_validator'] = risk_config
    
    # 4. Create execution container
    execution_config = ContainerConfig(
        role=ContainerRole.EXECUTION,
        name='execution_engine',
        config={
            'execution_mode': config.get('execution_mode', 'simulated'),
            'slippage_model': config.get('slippage', {}),
            'commission_model': config.get('commission', {}),
            'replay_mode': True  # Signals already have bar data
        }
    )
    topology['containers']['execution_engine'] = execution_config
    
    # 5. Add metadata
    topology['metadata'] = {
        'workflow_id': f"{workflow_id}_replay",
        'workflow_type': 'signal_replay',
        'source_workflow': workflow_id,
        'signal_path': str(signal_path),
        'portfolio_count': len(config.get('portfolios', [])),
        'filters': {
            'strategies': config.get('strategy_filter'),
            'regime': config.get('regime_filter')
        }
    }
    
    logger.info(f"Built signal replay topology for workflow {workflow_id} "
               f"with {len(config.get('portfolios', []))} portfolios")
    
    return topology


def build_regime_filtered_replay(config: Dict[str, Any], regime: str) -> Dict[str, Any]:
    """
    Build replay topology filtered for a specific regime.
    
    This is a convenience function that sets up regime filtering
    and boundary-aware replay.
    
    Args:
        config: Base replay configuration
        regime: Regime to filter for (e.g., 'TRENDING', 'VOLATILE')
        
    Returns:
        Topology with regime filtering configured
    """
    # Copy config and add regime filter
    filtered_config = config.copy()
    filtered_config['regime_filter'] = regime
    filtered_config['sparse_replay'] = True  # Only replay relevant bars
    
    # Update workflow ID to indicate filtering
    base_workflow = config['workflow_id']
    filtered_config['workflow_id'] = f"{base_workflow}_{regime.lower()}"
    
    return build_signal_replay_topology(filtered_config)


def build_multi_regime_replay(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build multiple replay topologies, one per regime.
    
    Useful for analyzing performance across different market conditions.
    
    Args:
        config: Base replay configuration with 'regimes' list
        
    Returns:
        Dict mapping regime -> topology
    """
    topologies = {}
    
    for regime in config.get('regimes', ['TRENDING', 'CHOPPY', 'VOLATILE']):
        topologies[regime] = build_regime_filtered_replay(config, regime)
    
    return topologies
