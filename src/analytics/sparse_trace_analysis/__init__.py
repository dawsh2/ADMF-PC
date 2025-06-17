"""
Sparse Trace Analysis Module

Provides analytics tools for analyzing sparse trace data from ADMF-PC backtests.
Handles classifier state analysis, strategy performance by regime, and execution cost modeling.
"""

from .classifier_analysis import ClassifierAnalyzer, calculate_state_durations, analyze_classifier_balance
from .strategy_analysis import StrategyAnalyzer, analyze_strategy_performance_by_regime
from .performance_calculation import (
    calculate_log_returns_with_costs,
    apply_execution_costs,
    ExecutionCostConfig,
    ZERO_COST,
    TYPICAL_RETAIL,
    INSTITUTIONAL,
    HIGH_FREQUENCY
)
from .regime_attribution import (
    RegimeAttributor,
    get_regime_at_bar,
    analyze_regime_transitions
)

__all__ = [
    'ClassifierAnalyzer',
    'StrategyAnalyzer', 
    'RegimeAttributor',
    'calculate_state_durations',
    'analyze_classifier_balance',
    'analyze_strategy_performance_by_regime',
    'calculate_log_returns_with_costs',
    'apply_execution_costs',
    'ExecutionCostConfig',
    'ZERO_COST',
    'TYPICAL_RETAIL', 
    'INSTITUTIONAL',
    'HIGH_FREQUENCY',
    'get_regime_at_bar',
    'analyze_regime_transitions'
]