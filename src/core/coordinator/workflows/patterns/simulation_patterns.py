"""
Simulation-specific workflow patterns.
Defines container patterns and communication configs for simulation workflows
including market simulation, Monte Carlo analysis, and stress testing.
"""
from typing import Dict, Any, List

def get_simulation_patterns() -> Dict[str, Dict[str, Any]]:
    """
    Get all simulation workflow patterns.
    
    Returns:
        Dictionary mapping pattern names to their configurations
    """
    return {
        'market_simulation': _market_simulation_pattern(),
        'monte_carlo': _monte_carlo_pattern(),
        'stress_test': _stress_test_pattern(),
        'scenario_analysis': _scenario_analysis_pattern(),
        'liquidity_simulation': _liquidity_simulation_pattern()
    }

def _market_simulation_pattern() -> Dict[str, Any]:
    """Market simulation with realistic order book dynamics."""
    return {
        'container_pattern': 'market_simulation',
        'communication_config': [
            {
                'type': 'pipeline',
                'source': 'market_data_container',
                'target': 'simulation_engine_container',
                'event_types': ['market_data_update', 'order_book_update']
            },
            {
                'type': 'pipeline', 
                'source': 'simulation_engine_container',
                'target': 'execution_container',
                'event_types': ['fill_event', 'market_impact_event']
            },
            {
                'type': 'broadcast',
                'source': 'execution_container',
                'targets': ['risk_container', 'analytics_container'],
                'event_types': ['trade_execution', 'portfolio_update']
            }
        ],
        'execution_strategy': 'standard',
        'isolation_level': 'full',
        'resource_sharing': {
            'market_data': True,
            'analytics': True,
            'execution': False  # Keep execution isolated
        }
    }

def _monte_carlo_pattern() -> Dict[str, Any]:
    """Monte Carlo simulation for portfolio analysis."""
    return {
        'container_pattern': 'monte_carlo',
        'communication_config': [
            {
                'type': 'pipeline',
                'source': 'scenario_generator_container', 
                'target': 'simulation_container',
                'event_types': ['scenario_data', 'parameter_update']
            },
            {
                'type': 'pipeline',
                'source': 'simulation_container',
                'target': 'analytics_container', 
                'event_types': ['simulation_result', 'portfolio_state']
            },
            {
                'type': 'selective',
                'source': 'analytics_container',
                'target': 'reporting_container',
                'event_types': ['aggregated_stats', 'risk_metrics'],
                'conditions': {'confidence_level': '>=0.95'}
            }
        ],
        'execution_strategy': 'multi_pattern',
        'isolation_level': 'moderate',
        'parallel_simulations': True,
        'batch_size': 1000
    }

def _stress_test_pattern() -> Dict[str, Any]:
    """Stress testing with extreme market scenarios."""
    return {
        'container_pattern': 'stress_test',
        'communication_config': [
            {
                'type': 'broadcast',
                'source': 'stress_scenario_container',
                'targets': ['market_data_container', 'volatility_container'],
                'event_types': ['stress_scenario', 'shock_event']
            },
            {
                'type': 'pipeline',
                'source': 'market_data_container',
                'target': 'portfolio_container',
                'event_types': ['shocked_market_data', 'liquidity_crisis']
            },
            {
                'type': 'hierarchical',
                'parent': 'risk_aggregator_container',
                'children': ['portfolio_container', 'exposure_container'],
                'event_types': ['position_update', 'risk_breach']
            }
        ],
        'execution_strategy': 'nested',
        'isolation_level': 'high',
        'scenario_types': ['market_crash', 'liquidity_crisis', 'correlation_breakdown'],
        'risk_monitoring': 'real_time'
    }

def _scenario_analysis_pattern() -> Dict[str, Any]:
    """Scenario analysis for strategic planning."""
    return {
        'container_pattern': 'scenario_analysis',
        'communication_config': [
            {
                'type': 'selective',
                'source': 'scenario_container',
                'target': 'strategy_container',
                'event_types': ['scenario_parameters', 'market_regime'],
                'conditions': {'scenario_probability': '>0.1'}
            },
            {
                'type': 'pipeline',
                'source': 'strategy_container', 
                'target': 'simulation_container',
                'event_types': ['strategy_signals', 'allocation_changes']
            },
            {
                'type': 'broadcast',
                'source': 'simulation_container',
                'targets': ['analytics_container', 'comparison_container'],
                'event_types': ['scenario_result', 'performance_metrics']
            }
        ],
        'execution_strategy': 'pipeline',
        'isolation_level': 'moderate',
        'scenario_weighting': 'probability_based',
        'comparison_baseline': 'current_strategy'
    }

def _liquidity_simulation_pattern() -> Dict[str, Any]:
    """Liquidity simulation for execution analysis."""
    return {
        'container_pattern': 'liquidity_simulation',
        'communication_config': [
            {
                'type': 'pipeline',
                'source': 'order_flow_container',
                'target': 'liquidity_engine_container',
                'event_types': ['order_submission', 'market_order_flow']
            },
            {
                'type': 'pipeline',
                'source': 'liquidity_engine_container',
                'target': 'execution_container',
                'event_types': ['liquidity_impact', 'slippage_estimate']
            },
            {
                'type': 'hierarchical',
                'parent': 'execution_analytics_container',
                'children': ['execution_container', 'cost_analysis_container'],
                'event_types': ['execution_cost', 'market_impact']
            }
        ],
        'execution_strategy': 'standard',
        'isolation_level': 'full',
        'market_impact_model': 'sqrt_law',
        'temporary_impact': True,
        'permanent_impact': True
    }

# Utility functions for simulation patterns

def get_simulation_container_requirements(pattern_name: str) -> List[str]:
    """
    Get required container types for a simulation pattern.
    
    Args:
        pattern_name: Name of simulation pattern
        
    Returns:
        List of required container types
    """
    requirements = {
        'market_simulation': [
            'market_data_container',
            'simulation_engine_container', 
            'execution_container',
            'risk_container',
            'analytics_container'
        ],
        'monte_carlo': [
            'scenario_generator_container',
            'simulation_container',
            'analytics_container',
            'reporting_container'
        ],
        'stress_test': [
            'stress_scenario_container',
            'market_data_container',
            'volatility_container',
            'portfolio_container',
            'risk_aggregator_container',
            'exposure_container'
        ],
        'scenario_analysis': [
            'scenario_container',
            'strategy_container',
            'simulation_container', 
            'analytics_container',
            'comparison_container'
        ],
        'liquidity_simulation': [
            'order_flow_container',
            'liquidity_engine_container',
            'execution_container',
            'execution_analytics_container',
            'cost_analysis_container'
        ]
    }
    
    return requirements.get(pattern_name, [])

def validate_simulation_pattern(pattern_config: Dict[str, Any]) -> List[str]:
    """
    Validate a simulation pattern configuration.
    
    Args:
        pattern_config: Pattern configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields
    required_fields = ['container_pattern', 'communication_config', 'execution_strategy']
    for field in required_fields:
        if field not in pattern_config:
            errors.append(f"Missing required field: {field}")
    
    # Validate execution strategy
    valid_strategies = ['standard', 'pipeline', 'nested', 'multi_pattern']
    if 'execution_strategy' in pattern_config:
        strategy = pattern_config['execution_strategy']
        if strategy not in valid_strategies:
            errors.append(f"Invalid execution strategy: {strategy}")
    
    # Validate isolation level
    valid_isolation = ['minimal', 'moderate', 'high', 'full']
    if 'isolation_level' in pattern_config:
        isolation = pattern_config['isolation_level']
        if isolation not in valid_isolation:
            errors.append(f"Invalid isolation level: {isolation}")
    
    # Validate communication config
    if 'communication_config' in pattern_config:
        comm_config = pattern_config['communication_config']
        if not isinstance(comm_config, list):
            errors.append("communication_config must be a list")
        else:
            for i, config in enumerate(comm_config):
                if 'type' not in config:
                    errors.append(f"Communication config {i} missing 'type'")
                elif config['type'] not in ['pipeline', 'broadcast', 'selective', 'hierarchical']:
                    errors.append(f"Invalid communication type: {config['type']}")
    
    return errors