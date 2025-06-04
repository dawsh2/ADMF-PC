"""
Research-specific workflow patterns.
Defines container patterns and communication configs for research workflows
including data mining, feature discovery, and model development.
"""
from typing import Dict, Any, List

def get_research_patterns() -> Dict[str, Dict[str, Any]]:
    """
    Get all research workflow patterns.
    
    Returns:
        Dictionary mapping pattern names to their configurations
    """
    return {
        'data_mining': _data_mining_pattern(),
        'feature_discovery': _feature_discovery_pattern(),
        'model_development': _model_development_pattern(),
        'hypothesis_testing': _hypothesis_testing_pattern(),
        'correlation_analysis': _correlation_analysis_pattern()
    }

def _data_mining_pattern() -> Dict[str, Any]:
    """Data mining pattern for discovering market patterns."""
    return {
        'container_pattern': 'data_mining',
        'communication_config': [
            {
                'type': 'pipeline',
                'source': 'data_loader_container',
                'target': 'preprocessing_container',
                'event_types': ['raw_data', 'data_batch']
            },
            {
                'type': 'pipeline',
                'source': 'preprocessing_container',
                'target': 'pattern_miner_container',
                'event_types': ['cleaned_data', 'feature_vector']
            },
            {
                'type': 'broadcast',
                'source': 'pattern_miner_container',
                'targets': ['analysis_container', 'storage_container'],
                'event_types': ['pattern_discovered', 'mining_result']
            }
        ],
        'execution_strategy': 'pipeline',
        'isolation_level': 'moderate',
        'resource_sharing': {
            'data_loader': True,
            'storage': True,
            'preprocessing': False  # Keep preprocessing isolated
        }
    }

def _feature_discovery_pattern() -> Dict[str, Any]:
    """Feature discovery pattern for identifying predictive signals."""
    return {
        'container_pattern': 'feature_discovery',
        'communication_config': [
            {
                'type': 'broadcast',
                'source': 'data_container',
                'targets': ['technical_feature_container', 'fundamental_feature_container', 'alternative_feature_container'],
                'event_types': ['market_data', 'fundamental_data', 'alternative_data']
            },
            {
                'type': 'selective',
                'source': 'technical_feature_container',
                'target': 'feature_evaluator_container',
                'event_types': ['technical_signal', 'indicator_value'],
                'conditions': {'signal_strength': '>0.7'}
            },
            {
                'type': 'selective',
                'source': 'fundamental_feature_container',
                'target': 'feature_evaluator_container',
                'event_types': ['fundamental_signal', 'ratio_value'],
                'conditions': {'significance': '>0.05'}
            },
            {
                'type': 'pipeline',
                'source': 'feature_evaluator_container',
                'target': 'ranking_container',
                'event_types': ['feature_score', 'evaluation_result']
            }
        ],
        'execution_strategy': 'multi_pattern',
        'isolation_level': 'moderate',
        'feature_types': ['technical', 'fundamental', 'alternative'],
        'evaluation_metrics': ['information_ratio', 'sharpe_ratio', 'correlation']
    }

def _model_development_pattern() -> Dict[str, Any]:
    """Model development pattern for strategy creation."""
    return {
        'container_pattern': 'model_development',
        'communication_config': [
            {
                'type': 'pipeline',
                'source': 'feature_container',
                'target': 'model_trainer_container',
                'event_types': ['training_data', 'feature_set']
            },
            {
                'type': 'pipeline',
                'source': 'model_trainer_container',
                'target': 'validation_container',
                'event_types': ['trained_model', 'model_parameters']
            },
            {
                'type': 'hierarchical',
                'parent': 'experiment_tracker_container',
                'children': ['model_trainer_container', 'validation_container'],
                'event_types': ['experiment_start', 'experiment_result']
            },
            {
                'type': 'selective',
                'source': 'validation_container',
                'target': 'deployment_container',
                'event_types': ['validated_model', 'performance_metrics'],
                'conditions': {'validation_score': '>0.6', 'stability_check': 'passed'}
            }
        ],
        'execution_strategy': 'nested',
        'isolation_level': 'high',
        'model_types': ['linear', 'tree_based', 'neural_network'],
        'validation_methods': ['walk_forward', 'time_series_split', 'purged_cv']
    }

def _hypothesis_testing_pattern() -> Dict[str, Any]:
    """Hypothesis testing pattern for research validation."""
    return {
        'container_pattern': 'hypothesis_testing',
        'communication_config': [
            {
                'type': 'pipeline',
                'source': 'hypothesis_container',
                'target': 'test_design_container',
                'event_types': ['hypothesis_definition', 'test_parameters']
            },
            {
                'type': 'broadcast',
                'source': 'test_design_container',
                'targets': ['control_group_container', 'treatment_group_container'],
                'event_types': ['test_design', 'group_assignment']
            },
            {
                'type': 'pipeline',
                'source': 'control_group_container',
                'target': 'statistical_test_container',
                'event_types': ['control_results', 'baseline_metrics']
            },
            {
                'type': 'pipeline',
                'source': 'treatment_group_container',
                'target': 'statistical_test_container',
                'event_types': ['treatment_results', 'experimental_metrics']
            },
            {
                'type': 'pipeline',
                'source': 'statistical_test_container',
                'target': 'conclusion_container',
                'event_types': ['test_statistic', 'p_value', 'confidence_interval']
            }
        ],
        'execution_strategy': 'standard',
        'isolation_level': 'full',
        'statistical_tests': ['t_test', 'mann_whitney', 'kolmogorov_smirnov'],
        'significance_level': 0.05,
        'multiple_testing_correction': 'bonferroni'
    }

def _correlation_analysis_pattern() -> Dict[str, Any]:
    """Correlation analysis pattern for relationship discovery."""
    return {
        'container_pattern': 'correlation_analysis',
        'communication_config': [
            {
                'type': 'broadcast',
                'source': 'data_synchronizer_container',
                'targets': ['asset_data_container', 'factor_data_container', 'macro_data_container'],
                'event_types': ['synchronized_data', 'timestamp_aligned']
            },
            {
                'type': 'pipeline',
                'source': 'asset_data_container',
                'target': 'correlation_engine_container',
                'event_types': ['asset_returns', 'price_series']
            },
            {
                'type': 'pipeline',
                'source': 'factor_data_container',
                'target': 'correlation_engine_container',
                'event_types': ['factor_values', 'factor_returns']
            },
            {
                'type': 'pipeline',
                'source': 'correlation_engine_container',
                'target': 'visualization_container',
                'event_types': ['correlation_matrix', 'rolling_correlation']
            },
            {
                'type': 'selective',
                'source': 'correlation_engine_container',
                'target': 'alert_container',
                'event_types': ['correlation_break', 'regime_change'],
                'conditions': {'correlation_change': '>0.3', 'statistical_significance': '>0.95'}
            }
        ],
        'execution_strategy': 'pipeline',
        'isolation_level': 'moderate',
        'correlation_types': ['pearson', 'spearman', 'kendall'],
        'rolling_window': 252,  # 1 year
        'regime_detection': True
    }

# Utility functions for research patterns

def get_research_container_requirements(pattern_name: str) -> List[str]:
    """
    Get required container types for a research pattern.
    
    Args:
        pattern_name: Name of research pattern
        
    Returns:
        List of required container types
    """
    requirements = {
        'data_mining': [
            'data_loader_container',
            'preprocessing_container',
            'pattern_miner_container',
            'analysis_container',
            'storage_container'
        ],
        'feature_discovery': [
            'data_container',
            'technical_feature_container',
            'fundamental_feature_container',
            'alternative_feature_container',
            'feature_evaluator_container',
            'ranking_container'
        ],
        'model_development': [
            'feature_container',
            'model_trainer_container',
            'validation_container',
            'experiment_tracker_container',
            'deployment_container'
        ],
        'hypothesis_testing': [
            'hypothesis_container',
            'test_design_container',
            'control_group_container',
            'treatment_group_container',
            'statistical_test_container',
            'conclusion_container'
        ],
        'correlation_analysis': [
            'data_synchronizer_container',
            'asset_data_container',
            'factor_data_container',
            'macro_data_container',
            'correlation_engine_container',
            'visualization_container',
            'alert_container'
        ]
    }
    
    return requirements.get(pattern_name, [])

def validate_research_pattern(pattern_config: Dict[str, Any]) -> List[str]:
    """
    Validate a research pattern configuration.
    
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
    
    # Research-specific validations
    if pattern_config.get('container_pattern') == 'hypothesis_testing':
        if 'significance_level' in pattern_config:
            sig_level = pattern_config['significance_level']
            if not (0 < sig_level < 1):
                errors.append("significance_level must be between 0 and 1")
    
    if pattern_config.get('container_pattern') == 'correlation_analysis':
        if 'rolling_window' in pattern_config:
            window = pattern_config['rolling_window']
            if not isinstance(window, int) or window <= 0:
                errors.append("rolling_window must be a positive integer")
    
    return errors