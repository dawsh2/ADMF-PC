"""
Container patterns for workflow execution.

This module defines different workflow patterns that combine:
- Container structures (what containers to create)
- Communication patterns (how containers communicate)
- Execution strategies (how to execute the workflow)

Note: No live_patterns.py - the same patterns work for both backtest and live
trading. Only data sources and execution targets change via --live flag.
"""

from .backtest_patterns import get_backtest_patterns
from .optimization_patterns import get_optimization_patterns
from .analysis_patterns import get_analysis_patterns
from .communication_patterns import get_communication_config
from .simulation_patterns import get_simulation_patterns
from .research_patterns import get_research_patterns
from .ensemble_patterns import get_ensemble_patterns

__all__ = [
    'get_backtest_patterns',
    'get_optimization_patterns', 
    'get_analysis_patterns',
    'get_communication_config',
    'get_simulation_patterns',
    'get_research_patterns',
    'get_ensemble_patterns'
]