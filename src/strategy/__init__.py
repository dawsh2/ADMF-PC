"""
Strategy module for ADMF-PC.

This module provides trading strategies using protocol-based design
with no inheritance. Components can be enhanced with capabilities
through composition.

Example Usage:
    ```python
    from src.strategy import MomentumStrategy
    from src.core.components import ComponentFactory
    
    # Create strategy with capabilities
    strategy = ComponentFactory().create_component({
        'class': 'MomentumStrategy',
        'params': {
            'lookback_period': 20,
            'momentum_threshold': 0.02
        },
        'capabilities': ['strategy', 'optimization', 'events']
    })
    
    # Use strategy
    signal = strategy.generate_signal(market_data)
    ```
"""

# Core protocols
from .protocols import (
    Strategy,
    Indicator,
    Feature,
    Rule,
    SignalAggregator,
    Classifier,
    RegimeAdaptive,
    Optimizable,
    PerformanceTracker,
    SignalDirection
)

# Capabilities
from .capabilities import (
    StrategyCapability,
    IndicatorCapability,
    RuleManagementCapability,
    RegimeAdaptiveCapability
)

# Strategies
from .strategies.momentum import MomentumStrategy, create_momentum_strategy

# Optimization
from .optimization import (
    # Protocols
    Optimizer,
    Objective,
    Constraint,
    ParameterSpace,
    OptimizationWorkflow,
    RegimeAnalyzer,
    
    # Capabilities
    OptimizationCapability,
    
    # Containers
    OptimizationContainer,
    RegimeAwareOptimizationContainer
)


__all__ = [
    # Protocols
    "Strategy",
    "Indicator", 
    "Feature",
    "Rule",
    "SignalAggregator",
    "Classifier",
    "RegimeAdaptive",
    "Optimizable",
    "PerformanceTracker",
    "SignalDirection",
    
    # Capabilities
    "StrategyCapability",
    "IndicatorCapability",
    "RuleManagementCapability",
    "RegimeAdaptiveCapability",
    
    # Strategies
    "MomentumStrategy",
    "create_momentum_strategy",
    
    # Optimization
    "Optimizer",
    "Objective",
    "Constraint",
    "ParameterSpace",
    "OptimizationWorkflow",
    "RegimeAnalyzer",
    "OptimizationCapability",
    "OptimizationContainer",
    "RegimeAwareOptimizationContainer"
]