"""
Unified workflow management system.

This module provides simplified workflow management for the unified architecture:
- Three execution modes (backtest, signal_generation, signal_replay)
- Universal topology for all workflows
- No pattern detection needed!

The main WorkflowManager is in topology.py
"""

# Import the result type
from ...types.workflow import WorkflowResult

# Configuration helpers still useful
from .config import ParameterAnalyzer, ConfigBuilder

__all__ = [
    # Types
    'WorkflowResult',
    
    # Configuration helpers (still useful)
    'ParameterAnalyzer', 
    'ConfigBuilder',
]

# Usage in unified architecture:
#
# ```python
# from src.core.coordinator.topology import WorkflowManager
# 
# manager = WorkflowManager()
# result = await manager.execute(config, context)
# ```
#
# Configuration is now simple:
# ```yaml
# parameters:
#   mode: backtest  # or signal_generation, or signal_replay
#   symbols: ['SPY']
#   strategies:
#     - type: momentum
#       threshold: [0.01, 0.02, 0.03]  # Arrays create parameter grid
# ```