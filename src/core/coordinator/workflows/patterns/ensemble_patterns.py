"""
Ensemble-specific workflow patterns.
Defines container patterns and communication configs for ensemble workflows
including strategy ensembles, model ensembles, and risk ensembles.
"""
from typing import Dict, Any, List

def get_ensemble_patterns() -> Dict[str, Dict[str, Any]]:
    """
    Get all ensemble workflow patterns.
    
    Returns:
        Dictionary mapping pattern names to their configurations
    """
    return {
        'strategy_ensemble': _strategy_ensemble_pattern(),
        'model_ensemble': _model_ensemble_pattern(),
        'risk_ensemble': _risk_ensemble_pattern(),
        'signal_ensemble': _signal_ensemble_pattern(),
        'portfolio_ensemble': _portfolio_ensemble_pattern()
    }

def _strategy_ensemble_pattern() -> Dict[str, Any]:
    """Strategy ensemble pattern for combining multiple strategies."""
    return {
        'container_pattern': 'strategy_ensemble',
        'communication_config': [
            {
                'type': 'broadcast',
                'source': 'data_container',
                'targets': ['momentum_strategy_container', 'mean_reversion_strategy_container', 'breakout_strategy_container'],
                'event_types': ['market_data', 'price_update']
            },
            {
                'type': 'pipeline',
                'source': 'momentum_strategy_container',
                'target': 'signal_aggregator_container',
                'event_types': ['momentum_signal', 'strategy_confidence']
            },
            {
                'type': 'pipeline',
                'source': 'mean_reversion_strategy_container',
                'target': 'signal_aggregator_container',
                'event_types': ['mean_reversion_signal', 'strategy_confidence']
            },
            {
                'type': 'pipeline',
                'source': 'breakout_strategy_container',
                'target': 'signal_aggregator_container',
                'event_types': ['breakout_signal', 'strategy_confidence']
            },
            {
                'type': 'hierarchical',
                'parent': 'ensemble_coordinator_container',
                'children': ['signal_aggregator_container', 'weight_optimizer_container'],
                'event_types': ['aggregated_signal', 'optimal_weights']
            },
            {
                'type': 'pipeline',
                'source': 'ensemble_coordinator_container',
                'target': 'execution_container',
                'event_types': ['ensemble_signal', 'position_target']
            }
        ],
        'execution_strategy': 'nested',
        'isolation_level': 'moderate',
        'aggregation_methods': ['weighted_average', 'voting', 'stacking'],
        'weight_optimization': 'dynamic',
        'rebalance_frequency': 'daily'
    }

def _model_ensemble_pattern() -> Dict[str, Any]:
    """Model ensemble pattern for combining multiple predictive models."""
    return {
        'container_pattern': 'model_ensemble',
        'communication_config': [
            {
                'type': 'broadcast',
                'source': 'feature_container',
                'targets': ['linear_model_container', 'tree_model_container', 'neural_net_container'],
                'event_types': ['feature_vector', 'training_data']
            },
            {
                'type': 'selective',
                'source': 'linear_model_container',
                'target': 'prediction_aggregator_container',
                'event_types': ['linear_prediction', 'model_confidence'],
                'conditions': {'model_status': 'trained', 'confidence': '>0.5'}
            },
            {
                'type': 'selective',
                'source': 'tree_model_container',
                'target': 'prediction_aggregator_container',
                'event_types': ['tree_prediction', 'model_confidence'],
                'conditions': {'model_status': 'trained', 'confidence': '>0.5'}
            },
            {
                'type': 'selective',
                'source': 'neural_net_container',
                'target': 'prediction_aggregator_container',
                'event_types': ['neural_prediction', 'model_confidence'],
                'conditions': {'model_status': 'trained', 'confidence': '>0.5'}
            },
            {
                'type': 'pipeline',
                'source': 'prediction_aggregator_container',
                'target': 'ensemble_validator_container',
                'event_types': ['ensemble_prediction', 'prediction_uncertainty']
            }
        ],
        'execution_strategy': 'multi_pattern',
        'isolation_level': 'high',
        'ensemble_methods': ['bagging', 'boosting', 'stacking', 'blending'],
        'cross_validation': 'time_series_aware',
        'uncertainty_quantification': True
    }

def _risk_ensemble_pattern() -> Dict[str, Any]:
    """Risk ensemble pattern for comprehensive risk assessment."""
    return {
        'container_pattern': 'risk_ensemble',
        'communication_config': [
            {
                'type': 'broadcast',
                'source': 'portfolio_container',
                'targets': ['var_risk_container', 'stress_risk_container', 'liquidity_risk_container'],
                'event_types': ['position_update', 'portfolio_state']
            },
            {
                'type': 'pipeline',
                'source': 'var_risk_container',
                'target': 'risk_aggregator_container',
                'event_types': ['var_estimate', 'expected_shortfall']
            },
            {
                'type': 'pipeline',
                'source': 'stress_risk_container',
                'target': 'risk_aggregator_container',
                'event_types': ['stress_loss', 'scenario_impact']
            },
            {
                'type': 'pipeline',
                'source': 'liquidity_risk_container',
                'target': 'risk_aggregator_container',
                'event_types': ['liquidity_cost', 'funding_risk']
            },
            {
                'type': 'hierarchical',
                'parent': 'risk_coordinator_container',
                'children': ['risk_aggregator_container', 'limit_monitor_container'],
                'event_types': ['aggregated_risk', 'limit_check']
            },
            {
                'type': 'selective',
                'source': 'risk_coordinator_container',
                'target': 'alert_container',
                'event_types': ['risk_breach', 'risk_warning'],
                'conditions': {'severity': 'high', 'immediate_action': True}
            }
        ],
        'execution_strategy': 'nested',
        'isolation_level': 'high',
        'risk_measures': ['var', 'expected_shortfall', 'maximum_drawdown'],
        'confidence_levels': [0.95, 0.99, 0.999],
        'stress_scenarios': ['historical', 'monte_carlo', 'hypothetical']
    }

def _signal_ensemble_pattern() -> Dict[str, Any]:
    """Signal ensemble pattern for combining multiple signal sources."""
    return {
        'container_pattern': 'signal_ensemble',
        'communication_config': [
            {
                'type': 'broadcast',
                'source': 'market_data_container',
                'targets': ['technical_signal_container', 'fundamental_signal_container', 'sentiment_signal_container'],
                'event_types': ['price_data', 'volume_data', 'fundamental_data']
            },
            {
                'type': 'pipeline',
                'source': 'technical_signal_container',
                'target': 'signal_scorer_container',
                'event_types': ['technical_signal', 'signal_strength']
            },
            {
                'type': 'pipeline',
                'source': 'fundamental_signal_container',
                'target': 'signal_scorer_container',
                'event_types': ['fundamental_signal', 'signal_strength']
            },
            {
                'type': 'pipeline',
                'source': 'sentiment_signal_container',
                'target': 'signal_scorer_container',
                'event_types': ['sentiment_signal', 'signal_strength']
            },
            {
                'type': 'pipeline',
                'source': 'signal_scorer_container',
                'target': 'signal_combiner_container',
                'event_types': ['scored_signal', 'signal_weight']
            },
            {
                'type': 'pipeline',
                'source': 'signal_combiner_container',
                'target': 'signal_filter_container',
                'event_types': ['combined_signal', 'ensemble_confidence']
            }
        ],
        'execution_strategy': 'pipeline',
        'isolation_level': 'moderate',
        'signal_types': ['technical', 'fundamental', 'sentiment', 'macro'],
        'combination_methods': ['linear', 'rank_based', 'ml_meta_model'],
        'signal_decay': 'exponential'
    }

def _portfolio_ensemble_pattern() -> Dict[str, Any]:
    """Portfolio ensemble pattern for managing multiple portfolios."""
    return {
        'container_pattern': 'portfolio_ensemble',
        'communication_config': [
            {
                'type': 'broadcast',
                'source': 'allocation_container',
                'targets': ['conservative_portfolio_container', 'aggressive_portfolio_container', 'balanced_portfolio_container'],
                'event_types': ['allocation_signal', 'rebalance_trigger']
            },
            {
                'type': 'pipeline',
                'source': 'conservative_portfolio_container',
                'target': 'portfolio_aggregator_container',
                'event_types': ['portfolio_performance', 'risk_metrics']
            },
            {
                'type': 'pipeline',
                'source': 'aggressive_portfolio_container',
                'target': 'portfolio_aggregator_container',
                'event_types': ['portfolio_performance', 'risk_metrics']
            },
            {
                'type': 'pipeline',
                'source': 'balanced_portfolio_container',
                'target': 'portfolio_aggregator_container',
                'event_types': ['portfolio_performance', 'risk_metrics']
            },
            {
                'type': 'hierarchical',
                'parent': 'meta_portfolio_container',
                'children': ['portfolio_aggregator_container', 'allocation_optimizer_container'],
                'event_types': ['aggregated_performance', 'optimal_allocation']
            },
            {
                'type': 'pipeline',
                'source': 'meta_portfolio_container',
                'target': 'execution_coordinator_container',
                'event_types': ['meta_allocation', 'execution_instruction']
            }
        ],
        'execution_strategy': 'nested',
        'isolation_level': 'moderate',
        'portfolio_types': ['conservative', 'aggressive', 'balanced'],
        'allocation_method': 'risk_parity',
        'rebalance_threshold': 0.05
    }

# Utility functions for ensemble patterns

def get_ensemble_container_requirements(pattern_name: str) -> List[str]:
    """
    Get required container types for an ensemble pattern.
    
    Args:
        pattern_name: Name of ensemble pattern
        
    Returns:
        List of required container types
    """
    requirements = {
        'strategy_ensemble': [
            'data_container',
            'momentum_strategy_container',
            'mean_reversion_strategy_container',
            'breakout_strategy_container',
            'signal_aggregator_container',
            'ensemble_coordinator_container',
            'weight_optimizer_container',
            'execution_container'
        ],
        'model_ensemble': [
            'feature_container',
            'linear_model_container',
            'tree_model_container',
            'neural_net_container',
            'prediction_aggregator_container',
            'ensemble_validator_container'
        ],
        'risk_ensemble': [
            'portfolio_container',
            'var_risk_container',
            'stress_risk_container',
            'liquidity_risk_container',
            'risk_aggregator_container',
            'risk_coordinator_container',
            'limit_monitor_container',
            'alert_container'
        ],
        'signal_ensemble': [
            'market_data_container',
            'technical_signal_container',
            'fundamental_signal_container',
            'sentiment_signal_container',
            'signal_scorer_container',
            'signal_combiner_container',
            'signal_filter_container'
        ],
        'portfolio_ensemble': [
            'allocation_container',
            'conservative_portfolio_container',
            'aggressive_portfolio_container',
            'balanced_portfolio_container',
            'portfolio_aggregator_container',
            'meta_portfolio_container',
            'allocation_optimizer_container',
            'execution_coordinator_container'
        ]
    }
    
    return requirements.get(pattern_name, [])

def validate_ensemble_pattern(pattern_config: Dict[str, Any]) -> List[str]:
    """
    Validate an ensemble pattern configuration.
    
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
    
    # Ensemble-specific validations
    pattern_type = pattern_config.get('container_pattern', '')
    
    if 'ensemble' in pattern_type:
        # Check for minimum number of ensemble members
        comm_config = pattern_config.get('communication_config', [])
        broadcast_configs = [c for c in comm_config if c.get('type') == 'broadcast']
        
        if broadcast_configs:
            for config in broadcast_configs:
                targets = config.get('targets', [])
                if len(targets) < 2:
                    errors.append("Ensemble patterns require at least 2 ensemble members")
    
    if pattern_type == 'risk_ensemble':
        confidence_levels = pattern_config.get('confidence_levels', [])
        for level in confidence_levels:
            if not (0 < level < 1):
                errors.append(f"Invalid confidence level: {level}")
    
    if pattern_type == 'portfolio_ensemble':
        rebalance_threshold = pattern_config.get('rebalance_threshold')
        if rebalance_threshold is not None and not (0 < rebalance_threshold < 1):
            errors.append("rebalance_threshold must be between 0 and 1")
    
    return errors

def get_ensemble_aggregation_methods(pattern_name: str) -> List[str]:
    """
    Get available aggregation methods for an ensemble pattern.
    
    Args:
        pattern_name: Name of ensemble pattern
        
    Returns:
        List of available aggregation methods
    """
    methods = {
        'strategy_ensemble': ['weighted_average', 'voting', 'stacking', 'rank_based'],
        'model_ensemble': ['bagging', 'boosting', 'stacking', 'blending', 'bayesian_model_averaging'],
        'risk_ensemble': ['worst_case', 'average', 'weighted_average', 'convex_combination'],
        'signal_ensemble': ['linear', 'rank_based', 'ml_meta_model', 'bayesian_fusion'],
        'portfolio_ensemble': ['equal_weight', 'risk_parity', 'mean_variance', 'black_litterman']
    }
    
    return methods.get(pattern_name, ['weighted_average'])