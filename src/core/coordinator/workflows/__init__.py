"""
Workflow definitions for the coordinator.

Workflows define multi-phase processes that combine sequences and topologies.
Each workflow specifies:
- What phases to execute
- Which topology to use for each phase
- How phases depend on each other
- Configuration overrides per phase
"""

from .adaptive_ensemble import adaptive_ensemble_workflow
from .walk_forward_optimization import walk_forward_optimization_workflow
from .regime_adaptive_trading import regime_adaptive_trading_workflow
from .simple_backtest import simple_backtest_workflow
from .signal_generation import signal_generation_workflow

# Registry of all available workflows
WORKFLOW_REGISTRY = {
    'adaptive_ensemble': adaptive_ensemble_workflow,
    'walk_forward_optimization': walk_forward_optimization_workflow,
    'regime_adaptive_trading': regime_adaptive_trading_workflow,
    'simple_backtest': simple_backtest_workflow,
    'signal_generation': signal_generation_workflow
}

__all__ = [
    'WORKFLOW_REGISTRY',
    'adaptive_ensemble_workflow',
    'walk_forward_optimization_workflow',
    'regime_adaptive_trading_workflow',
    'simple_backtest_workflow',
    'signal_generation_workflow'
]
