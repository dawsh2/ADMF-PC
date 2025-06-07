"""
Helper for creating stateless components.

This module handles creation of strategy, classifier, risk, and execution components
using the discovery system.
"""

from typing import Dict, Any, List
import logging

from ....components.discovery import get_component_registry

logger = logging.getLogger(__name__)


def create_stateless_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create all stateless components from configuration."""
    components = {
        'strategies': {},
        'classifiers': {},
        'risk_validators': {},
        'execution_models': {}
    }
    
    # Create strategy instances from config
    for strat_config in config.get('strategies', []):
        if 'type' not in strat_config:
            logger.error(f"Strategy config missing 'type': {strat_config}")
            continue
        strat_type = strat_config['type']
        strategy = create_strategy(strat_type, strat_config)
        if strategy:
            components['strategies'][strat_type] = strategy
    
    # Create classifier instances from config
    for class_config in config.get('classifiers', []):
        if 'type' not in class_config:
            logger.error(f"Classifier config missing 'type': {class_config}")
            continue
        class_type = class_config['type']
        classifier = create_classifier(class_type, class_config)
        if classifier:
            components['classifiers'][class_type] = classifier
    
    # Create risk validator instances from config
    for risk_config in config.get('risk_profiles', []):
        if 'type' not in risk_config:
            logger.error(f"Risk profile config missing 'type': {risk_config}")
            continue
        risk_type = risk_config['type']
        validator = create_risk_validator(risk_type, risk_config)
        if validator:
            components['risk_validators'][risk_type] = validator
    
    # Create execution model instances from config
    for exec_config in config.get('execution_models', []):
        if 'type' not in exec_config:
            logger.error(f"Execution model config missing 'type': {exec_config}")
            continue
        exec_type = exec_config['type']
        exec_models = create_execution_models(exec_type, exec_config)
        if exec_models:
            components['execution_models'][exec_type] = exec_models
    
    return components


def create_strategy(strategy_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless strategy using discovery system."""
    registry = get_component_registry()
    
    # Get strategy from registry
    strategy_info = registry.get_component(strategy_type)
    
    if strategy_info:
        logger.info(f"Found strategy '{strategy_type}' in registry")
        return strategy_info.factory
    
    # If not in registry, log error
    logger.error(f"Strategy type '{strategy_type}' not found in registry. Available strategies: {list(registry.list_by_type('strategy'))}")
    return None


def create_classifier(classifier_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless classifier using discovery system."""
    registry = get_component_registry()
    
    # Try exact match first
    classifier_info = registry.get_component(classifier_type)
    
    # If not found, try with _classifier suffix
    if not classifier_info:
        classifier_info = registry.get_component(f"{classifier_type}_classifier")
    
    if classifier_info:
        logger.info(f"Found classifier '{classifier_type}' in registry")
        return classifier_info.factory
    
    # If not in registry, log error
    logger.error(f"Classifier type '{classifier_type}' not found in registry. Available classifiers: {list(registry.list_by_type('classifier'))}")
    return None


def create_risk_validator(risk_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless risk validator."""
    # First check registry
    registry = get_component_registry()
    
    # Try exact match
    validator_info = registry.get_component(risk_type)
    
    # Try with _validator suffix
    if not validator_info:
        validator_info = registry.get_component(f"{risk_type}_validator")
    
    # Try with validate_ prefix
    if not validator_info:
        validator_info = registry.get_component(f"validate_{risk_type}")
    
    if validator_info:
        logger.info(f"Found risk validator '{risk_type}' in registry")
        return validator_info.factory
    
    # If not in registry, try direct import as fallback
    try:
        from ....risk import validators
        validator_func = getattr(validators, f"validate_{risk_type}", None)
        if validator_func:
            logger.warning(f"Risk validator '{risk_type}' not in registry but found via import")
            return validator_func
    except (ImportError, AttributeError):
        pass
    
    logger.error(f"Risk validator type '{risk_type}' not found")
    return None


def create_execution_models(exec_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create stateless execution models from config."""
    models = {}
    registry = get_component_registry()
    
    # Slippage model
    slippage_config = config.get('slippage', {})
    if 'type' in slippage_config:
        slippage_type = slippage_config['type']
        slippage_info = registry.get_component(f'{slippage_type}_slippage')
        
        if slippage_info:
            params = slippage_config.get('params', {})
            models['slippage'] = slippage_info.factory(**params)
            logger.info(f"Created slippage model '{slippage_type}' with params: {params}")
        else:
            logger.error(f"Slippage model '{slippage_type}' not found in registry")
    
    # Commission model  
    commission_config = config.get('commission', {})
    if 'type' in commission_config:
        commission_type = commission_config['type']
        commission_info = registry.get_component(f'{commission_type}_commission')
        
        if commission_info:
            params = commission_config.get('params', {})
            models['commission'] = commission_info.factory(**params)
            logger.info(f"Created commission model '{commission_type}' with params: {params}")
        else:
            logger.error(f"Commission model '{commission_type}' not found in registry")
    
    # Other execution models from config
    for model_key in ['liquidity', 'market_impact', 'latency']:
        if model_key in config:
            model_config = config[model_key]
            if 'type' in model_config:
                model_type = model_config['type']
                model_info = registry.get_component(f'{model_type}_{model_key}')
                
                if model_info:
                    params = model_config.get('params', {})
                    models[model_key] = model_info.factory(**params)
                    logger.info(f"Created {model_key} model '{model_type}' with params: {params}")
    
    return models


def get_strategy_feature_requirements(strategy_type: str, config: Dict[str, Any]) -> List[str]:
    """Get required features for a strategy from registry metadata."""
    registry = get_component_registry()
    strategy_info = registry.get_component(strategy_type)
    
    if strategy_info:
        # Get features from metadata
        base_features = strategy_info.metadata.get('features', [])
        feature_config = strategy_info.metadata.get('feature_config', {})
        if feature_config:
            base_features.extend(feature_config.keys())
        
        # Add any custom features from config
        custom_features = config.get('required_features', [])
        all_features = list(set(base_features + custom_features))
        
        logger.info(f"Strategy '{strategy_type}' requires features: {all_features}")
        return all_features
    else:
        logger.error(f"Strategy '{strategy_type}' not found in registry, cannot determine feature requirements")
        # Return any explicitly configured features
        return config.get('required_features', [])
