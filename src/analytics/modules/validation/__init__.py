"""
Validation framework for comparing expected vs actual trade execution.
"""

from .signal_execution import validate_execution, create_validation_report
from .exit_analysis import analyze_exit_reasons, plot_exit_breakdown
from .performance_gap import calculate_performance_gap, analyze_slippage_impact

__all__ = [
    'validate_execution',
    'create_validation_report',
    'analyze_exit_reasons',
    'plot_exit_breakdown',
    'calculate_performance_gap',
    'analyze_slippage_impact'
]