"""
Communication patterns for workflow containers.

Defines how containers should communicate for different workflow patterns.
"""

import logging
from typing import Dict, Any, List

from ....containers.protocols import ComposableContainer

logger = logging.getLogger(__name__)


def get_communication_config(pattern_name: str, containers: List[ComposableContainer]) -> Dict[str, Any]:
    """Get communication configuration for a specific pattern."""
    
    # Basic pipeline configuration for standard patterns
    if pattern_name in ['simple_backtest', 'full_backtest']:
        return _get_backtest_communication_config(pattern_name, containers)
    
    elif pattern_name == 'signal_generation':
        return _get_signal_generation_communication_config(containers)
    
    elif pattern_name == 'signal_replay':
        return _get_signal_replay_communication_config(containers)
    
    elif pattern_name in ['multi_parameter_backtest', 'optimization_grid']:
        return _get_multi_parameter_communication_config(pattern_name, containers)
    
    # Default: no communication setup
    return {'adapters': []}


def _get_backtest_communication_config(pattern_name: str, containers: List[ComposableContainer]) -> Dict[str, Any]:
    """Get communication config for backtest patterns."""
    
    # Determine pipeline order based on container roles
    pipeline_order = _determine_pipeline_order(pattern_name, containers)
    container_names = [c.metadata.name for c in pipeline_order]
    
    if len(container_names) < 2:
        logger.warning(f"Insufficient containers for pipeline: {container_names}")
        return {'adapters': []}
    
    return {
        'adapters': [{
            'name': f'{pattern_name}_pipeline',
            'type': 'pipeline',
            'containers': container_names,
            'log_level': 'INFO'
        }]
    }


def _get_signal_generation_communication_config(containers: List[ComposableContainer]) -> Dict[str, Any]:
    """Get communication config for signal generation."""
    
    # Signal generation doesn't need execution pipeline
    signal_containers = [c.metadata.name for c in containers 
                       if c.metadata.role.value in ['data', 'indicator', 'strategy']]
    
    if len(signal_containers) < 2:
        return {'adapters': []}
    
    return {
        'adapters': [{
            'name': 'signal_pipeline', 
            'type': 'pipeline',
            'containers': signal_containers,
            'log_level': 'INFO'
        }]
    }


def _get_signal_replay_communication_config(containers: List[ComposableContainer]) -> Dict[str, Any]:
    """Get communication config for signal replay."""
    
    # Signal replay uses ensemble logic
    ensemble_containers = [c.metadata.name for c in containers
                         if c.metadata.role.value in ['signal_log', 'ensemble', 'risk', 'execution']]
    
    if len(ensemble_containers) < 2:
        return {'adapters': []}
    
    return {
        'adapters': [{
            'name': 'replay_pipeline',
            'type': 'pipeline', 
            'containers': ensemble_containers,
            'log_level': 'INFO'
        }]
    }


def _get_multi_parameter_communication_config(pattern_name: str, containers: List[ComposableContainer]) -> Dict[str, Any]:
    """Get communication config for multi-parameter patterns."""
    
    # Multi-parameter uses hub-based communication
    # Find hub container
    hub_containers = [c for c in containers if 'hub' in c.metadata.name.lower()]
    if not hub_containers:
        logger.warning("No hub container found for multi-parameter communication")
        return {'adapters': []}
    
    hub_container = hub_containers[0]
    
    # Find portfolio containers (one per parameter combination)
    portfolio_containers = [c for c in containers 
                          if c.metadata.role.value == 'portfolio' and c != hub_container]
    
    if not portfolio_containers:
        logger.warning("No portfolio containers found for multi-parameter communication")
        return {'adapters': []}
    
    # Create hub-to-portfolio broadcast adapter
    adapters = [{
        'name': f'{pattern_name}_hub_broadcast',
        'type': 'broadcast',
        'source': hub_container.metadata.name,
        'targets': [c.metadata.name for c in portfolio_containers],
        'log_level': 'INFO'
    }]
    
    # Create pipeline adapters within each portfolio
    for portfolio in portfolio_containers:
        # Find containers belonging to this portfolio
        portfolio_children = _collect_portfolio_containers(portfolio)
        if len(portfolio_children) >= 2:
            adapters.append({
                'name': f'{portfolio.metadata.name}_pipeline',
                'type': 'pipeline',
                'containers': [c.metadata.name for c in portfolio_children],
                'log_level': 'DEBUG'
            })
    
    return {'adapters': adapters}


def _determine_pipeline_order(pattern_name: str, containers: List[ComposableContainer]) -> List[ComposableContainer]:
    """Determine the order of containers in the pipeline based on pattern."""
    
    # Create a map by role
    role_map = {}
    for container in containers:
        role = container.metadata.role.value
        if role not in role_map:
            role_map[role] = []
        role_map[role].append(container)
    
    # Define pipeline order for different patterns
    if pattern_name == 'simple_backtest':
        # Portfolio is NOT in the main pipeline - it receives FILL events via reverse routing
        pipeline_roles = ['data', 'indicator', 'strategy', 'risk', 'execution']
    elif pattern_name == 'full_backtest':
        # Full backtest includes classifier
        pipeline_roles = ['data', 'indicator', 'classifier', 'strategy', 'risk', 'execution']
    else:
        # Default order
        pipeline_roles = ['data', 'indicator', 'strategy', 'risk', 'execution']
    
    # Build ordered list
    pipeline_order = []
    for role in pipeline_roles:
        if role in role_map:
            pipeline_order.extend(role_map[role])
    
    logger.info(f"Pipeline order for {pattern_name}: {[c.metadata.name for c in pipeline_order]}")
    return pipeline_order


def _collect_portfolio_containers(portfolio_container: ComposableContainer) -> List[ComposableContainer]:
    """Collect all containers that belong to a portfolio."""
    
    containers = []
    
    # Add the portfolio itself
    containers.append(portfolio_container)
    
    # Add child containers (strategy, risk, execution within this portfolio)
    def collect_children(container):
        for child in container.child_containers:
            containers.append(child)
            collect_children(child)  # Recursive for nested structures
    
    collect_children(portfolio_container)
    
    # Sort by typical execution order
    role_order = {'portfolio': 0, 'strategy': 1, 'risk': 2, 'execution': 3}
    containers.sort(key=lambda c: role_order.get(c.metadata.role.value, 999))
    
    return containers


def get_available_communication_patterns() -> Dict[str, str]:
    """Get available communication patterns and their descriptions."""
    
    return {
        'pipeline': 'Linear pipeline communication (A → B → C)',
        'broadcast': 'One-to-many broadcast communication (Hub → [A, B, C])',
        'hierarchical': 'Parent-child hierarchical communication',
        'selective': 'Rule-based selective routing',
        'hub': 'Hub-based multi-parameter communication'
    }