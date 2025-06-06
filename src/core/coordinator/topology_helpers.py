"""
Topology Helper Functions - Container and Component Creation

This module contains helper functions for creating containers, stateless components,
and other topology building blocks used by the TopologyBuilder.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..events import EventBus
from ..containers.discovery import get_component_registry
from ..containers.symbol_timeframe_container import SymbolTimeframeContainer
from ..containers.portfolio_container import PortfolioContainer
from ..containers.execution_container import ExecutionContainer

logger = logging.getLogger(__name__)


def create_event_bus(tracing_enabled: bool, bus_name: str, event_tracer: Optional[Any] = None) -> EventBus:
    """
    Create an event bus with optional tracing support.
    
    Args:
        tracing_enabled: Whether to enable event tracing
        bus_name: Name for the event bus
        event_tracer: Optional event tracer instance
        
    Returns:
        EventBus or TracedEventBus instance
    """
    if tracing_enabled:
        from ..events.tracing import TracedEventBus
        traced_bus = TracedEventBus(bus_name)
        if event_tracer:
            traced_bus.set_tracer(event_tracer)
        return traced_bus
    else:
        return EventBus(bus_name)


def extract_symbol_timeframe_configs(config: Any) -> List[Dict[str, Any]]:
    """
    Extract symbol-timeframe configurations from workflow config.
    
    Supports multiple formats:
    1. Simple: symbols: ['SPY', 'QQQ'] (defaults to 1d timeframe)
    2. Detailed: symbol_configs with explicit timeframes
    3. Multi-timeframe: symbols with multiple timeframes each
    """
    symbol_configs = []
    
    # Check for simple symbols list (backward compatibility)
    if 'symbols' in config.parameters:
        symbols = config.parameters['symbols']
        if isinstance(symbols, list) and all(isinstance(s, str) for s in symbols):
            # Simple symbol list - default to daily timeframe
            for symbol in symbols:
                symbol_configs.append({
                    'symbol': symbol,
                    'timeframe': '1d',
                    'data_config': config.parameters.get('backtest', {}).get('data', config.data_config),
                    'features': config.parameters.get('backtest', {}).get('features', {})
                })
            return symbol_configs
    
    # Check for detailed symbol_configs
    if 'symbol_configs' in config.parameters:
        for sc in config.parameters['symbol_configs']:
            symbol = sc['symbol']
            timeframes = sc.get('timeframes', ['1d'])
            if not isinstance(timeframes, list):
                timeframes = [timeframes]
            
            # Create config for each symbol-timeframe combination
            for timeframe in timeframes:
                symbol_configs.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_config': sc.get('data_config', config.data_config),
                    'features': sc.get('features', config.parameters.get('features', {}))
                })
    
    # Default case - at least one symbol container
    if not symbol_configs:
        logger.warning("No symbol configurations found, creating default SPY_1d container")
        symbol_configs.append({
            'symbol': 'SPY',
            'timeframe': '1d',
            'data_config': config.data_config,
            'features': config.parameters.get('backtest', {}).get('features', {})
        })
    
    return symbol_configs


def expand_parameter_combinations(config: Any) -> List[Dict[str, Any]]:
    """
    Expand parameter grid into individual combinations.
    
    Example:
    - 20 strategies × 3 risk profiles × 2 execution models = 120 combinations
    - Each gets its own portfolio container with unique combo_id
    """
    combinations = []
    
    # Get parameter grids from config
    # The YAML config is stored under parameters, so extract from backtest section
    backtest_config = config.parameters.get('backtest', {})
    strategy_params = backtest_config.get('strategies', [{}])
    risk_params = backtest_config.get('risk_profiles', [{}])
    classifier_params = backtest_config.get('classifiers', [{}])
    execution_params = backtest_config.get('execution_models', [{}])
    
    # Ensure they're lists
    if not isinstance(strategy_params, list):
        strategy_params = [strategy_params]
    if not isinstance(risk_params, list):
        risk_params = [risk_params]
    if not isinstance(classifier_params, list):
        classifier_params = [classifier_params]
    if not isinstance(execution_params, list):
        execution_params = [execution_params]
    
    # Generate all combinations
    combo_id = 0
    for strat in strategy_params:
        for risk in risk_params:
            for classifier in classifier_params:
                for execution in execution_params:
                    combinations.append({
                        'combo_id': f'c{combo_id:04d}',
                        'strategy_params': strat,
                        'risk_params': risk,
                        'classifier_params': classifier,
                        'execution_params': execution
                    })
                    combo_id += 1
    
    return combinations


def create_stateless_strategy(strategy_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless strategy instance using discovery system."""
    registry = get_component_registry()
    
    # Get strategy from registry
    strategy_info = registry.get_component(strategy_type)
    
    if strategy_info:
        # Return the factory (which is the strategy function itself)
        return strategy_info.factory
    else:
        # Fallback - try importing directly for backward compatibility
        logger.warning(f"Strategy '{strategy_type}' not found in registry, trying direct import")
        
        try:
            if strategy_type == 'momentum':
                from ...strategy.strategies.momentum import momentum_strategy
                return momentum_strategy
            elif strategy_type == 'mean_reversion':
                from ...strategy.strategies.mean_reversion_simple import mean_reversion_strategy
                return mean_reversion_strategy
            elif strategy_type == 'trend_following':
                from ...strategy.strategies.trend_following import trend_following_strategy
                return trend_following_strategy
            elif strategy_type == 'simple_trend':
                from ...strategy.strategies.simple_trend import simple_trend_strategy
                return simple_trend_strategy
        except ImportError as e:
            logger.error(f"Failed to import strategy '{strategy_type}': {e}")
        
        # Final fallback
        logger.error(f"Strategy type '{strategy_type}' not found")
        return lambda features, bar, params: None


def create_stateless_classifier(classifier_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless classifier instance using discovery system."""
    registry = get_component_registry()
    
    # Handle common name mappings
    name_mapping = {
        'trend': 'trend_classifier',
        'volatility': 'volatility_classifier',
        'momentum_regime': 'momentum_regime_classifier',
        'simple': 'trend_classifier',  # Default simple to trend
        'composite': 'trend_classifier'  # For now, use trend as default
    }
    
    # Map the type to registry name
    registry_name = name_mapping.get(classifier_type, classifier_type)
    
    # Get classifier from registry
    classifier_info = registry.get_component(registry_name)
    
    if classifier_info:
        # Return the factory (which is the classifier function itself)
        return classifier_info.factory
    else:
        # Fallback - try importing directly for backward compatibility
        logger.warning(f"Classifier '{registry_name}' not found in registry, trying direct import")
        
        try:
            from ...strategy.classifiers.classifiers import (
                trend_classifier,
                volatility_classifier,
                momentum_regime_classifier
            )
            
            if classifier_type in ['trend', 'simple', 'composite']:
                return trend_classifier
            elif classifier_type == 'volatility':
                return volatility_classifier
            elif classifier_type == 'momentum_regime':
                return momentum_regime_classifier
        except ImportError as e:
            logger.error(f"Failed to import classifier '{classifier_type}': {e}")
        
        # Final fallback
        logger.error(f"Classifier type '{classifier_type}' not found")
        return lambda features, params: {'regime': 'unknown', 'confidence': 0.0}


def create_stateless_risk_validator(risk_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless risk validator instance."""
    # Import risk validator modules
    from ...risk.validators import (
        validate_max_position,
        validate_drawdown,
        validate_composite
    )
    
    if risk_type == 'position':
        return validate_max_position
    elif risk_type == 'drawdown':
        return validate_drawdown
    elif risk_type in ['conservative', 'moderate', 'aggressive', 'composite']:
        # All risk profiles use composite validator with different params
        return validate_composite
    else:
        # Fallback placeholder for other validators
        logger.warning(f"Risk validator type '{risk_type}' not yet converted to stateless")
        return {'type': risk_type, 'config': config, 'stateless': True}


def create_stateless_execution_models(exec_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create stateless execution models (slippage, commission) using discovery system."""
    registry = get_component_registry()
    
    models = {}
    
    # Create slippage model
    slippage_config = config.get('slippage', {})
    slippage_type = slippage_config.get('type', 'percentage')
    
    # Try to get from registry first
    slippage_info = registry.get_component(slippage_type)
    if not slippage_info:
        # Try with model suffix
        slippage_info = registry.get_component(f'{slippage_type}_slippage')
    
    if slippage_info:
        # Use factory to create instance with config params
        params = slippage_config.get('params', {})
        models['slippage'] = slippage_info.factory(**params)
    else:
        # Fallback - import directly
        logger.warning(f"Slippage model '{slippage_type}' not found in registry, using fallback")
        from ...execution.brokers.slippage import PercentageSlippageModel, ZeroSlippageModel
        
        if slippage_type == 'zero':
            models['slippage'] = ZeroSlippageModel()
        else:
            models['slippage'] = PercentageSlippageModel()
    
    # Create commission model
    commission_config = config.get('commission', {})
    commission_type = commission_config.get('type', 'zero')
    
    # Try to get from registry first
    commission_info = registry.get_component(commission_type)
    if not commission_info:
        # Try with model suffix
        commission_info = registry.get_component(f'{commission_type}_commission')
    
    if commission_info:
        # Use factory to create instance with config params
        params = commission_config.get('params', {})
        models['commission'] = commission_info.factory(**params)
    else:
        # Fallback - import directly
        logger.warning(f"Commission model '{commission_type}' not found in registry, using fallback")
        from ...execution.brokers.commission import ZeroCommissionModel, PerShareCommissionModel
        
        if commission_type == 'per_share':
            models['commission'] = PerShareCommissionModel()
        else:
            models['commission'] = ZeroCommissionModel()
    
    # Add liquidity model if specified
    liquidity_config = config.get('liquidity', {})
    if liquidity_config:
        liquidity_type = liquidity_config.get('type', 'perfect')
        # Similar pattern for liquidity models when they have decorators
        models['liquidity'] = {'type': liquidity_type, 'config': liquidity_config}
    
    return models


def create_stateless_components(config: Any) -> Dict[str, Any]:
    """Create all stateless strategy, classifier, risk, and execution components."""
    components = {
        'strategies': {},
        'classifiers': {},
        'risk_validators': {},
        'execution_models': {}
    }
    
    # Create strategy instances (stateless)
    backtest_config = config.parameters.get('backtest', {})
    for strat_config in backtest_config.get('strategies', []):
        strat_type = strat_config.get('type', 'momentum')
        strategy = create_stateless_strategy(strat_type, strat_config)
        components['strategies'][strat_type] = strategy
    
    # Create classifier instances (stateless)
    for class_config in backtest_config.get('classifiers', []):
        class_type = class_config.get('type', 'simple')
        classifier = create_stateless_classifier(class_type, class_config)
        components['classifiers'][class_type] = classifier
    
    # Create risk validator instances (stateless)
    for risk_config in backtest_config.get('risk_profiles', []):
        risk_type = risk_config.get('type', 'conservative')
        validator = create_stateless_risk_validator(risk_type, risk_config)
        components['risk_validators'][risk_type] = validator
    
    # Create execution model instances (stateless)
    for exec_config in backtest_config.get('execution_models', []):
        exec_type = exec_config.get('type', 'standard')
        exec_models = create_stateless_execution_models(exec_type, exec_config)
        components['execution_models'][exec_type] = exec_models
    
    return components


def get_strategy_feature_requirements(strategy_type: str, config: Dict[str, Any]) -> List[str]:
    """
    Determine which features a strategy needs using discovery system.
    
    This avoids sending all features to every strategy.
    """
    registry = get_component_registry()
    
    # Get strategy info from registry
    strategy_info = registry.get_component(strategy_type)
    
    if strategy_info:
        # Get features from metadata
        base_features = strategy_info.metadata.get('features', [])
        
        # If feature_config is present, extract feature names
        feature_config = strategy_info.metadata.get('feature_config', {})
        if feature_config:
            base_features.extend(feature_config.keys())
    else:
        # Fallback to hardcoded requirements
        logger.warning(f"Strategy '{strategy_type}' not in registry, using fallback feature requirements")
        feature_requirements = {
            'momentum': ['sma', 'rsi'],  # Momentum uses moving averages and RSI
            'mean_reversion': ['rsi', 'bollinger_bands'],  # Mean reversion uses RSI and BB
            'breakout': ['atr', 'high', 'low'],  # Breakout uses ATR and price extremes
            'trend_following': ['sma', 'ema', 'adx'],  # Trend following uses various MAs and ADX
        }
        base_features = feature_requirements.get(strategy_type, [])
    
    # Add any custom features from config
    custom_features = config.get('required_features', [])
    
    # Combine and deduplicate
    all_features = list(set(base_features + custom_features))
    
    return all_features


def infer_features_from_strategies(strategies: List[Dict[str, Any]]) -> List[str]:
    """
    Infer required features from strategy configurations.
    
    This ensures feature containers compute the right indicators.
    """
    from ...strategy.components.feature_inference import infer_features_from_strategies as infer_func
    return infer_func(strategies)