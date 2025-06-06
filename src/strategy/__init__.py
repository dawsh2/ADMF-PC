"""
Strategy module for ADMF-PC.

Architecture Reference: docs/SYSTEM_ARCHITECTURE_V5.MD#strategy-module
Style Guide: STYLE.md - Canonical strategy implementations

This module provides trading strategies using Protocol + Composition architecture
with zero inheritance. All components implement protocols directly.

Example Usage:
    ```python
    from src.strategy import MomentumStrategy
    
    # Create strategy directly - no framework overhead
    strategy = MomentumStrategy(
        lookback_period=20,
        momentum_threshold=0.02
    )
    
    # Use strategy - implements Strategy protocol
    signal = strategy.generate_signal(market_data)
    ```
"""

# Core protocols
from .protocols import (
    Strategy,
    SignalDirection
)

# Note: No capabilities.py - components ARE capabilities through protocol implementation

# Strategies - now using pure functions
# from .strategies.momentum import MomentumStrategy, create_momentum_strategy

# # Optimization - commented out for unified architecture testing
# from .optimization import (
#     # Protocols
#     Optimizer,
#     Objective,
#     Constraint,
#     ParameterSpace,
#     OptimizationWorkflow,
#     RegimeAnalyzer,
#     
#     # Capabilities
#     OptimizationCapability,
#     
#     # Containers
#     OptimizationContainer,
#     RegimeAwareOptimizationContainer
# )


__all__ = [
    # Protocols
    "Strategy",
    "SignalDirection",
    
    # No capabilities exported - components implement protocols directly
    
    # Strategies - now using pure functions
    # "MomentumStrategy",
    # "create_momentum_strategy",
    
    # # Optimization - commented out for unified architecture testing
    # "Optimizer",
    # "Objective",
    # "Constraint",
    # "ParameterSpace",
    # "OptimizationWorkflow",
    # "RegimeAnalyzer",
    # "OptimizationCapability",
    # "OptimizationContainer",
    # "RegimeAwareOptimizationContainer"
]