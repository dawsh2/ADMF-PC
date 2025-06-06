"""
SIMPLIFIED FOR UNIFIED ARCHITECTURE - NO LONGER NEEDED

This module previously provided different execution strategies for container patterns.
With the unified architecture, execution is handled directly in topology.py using
three simple modes: backtest, signal_generation, and signal_replay.

KEPT FOR BACKWARD COMPATIBILITY ONLY - WILL BE REMOVED
"""

# Legacy imports kept for backward compatibility
from typing import Dict, Any
from ....types.workflow import WorkflowResult

# Stub implementations for backward compatibility
class ExecutionStrategy:
    """DEPRECATED - Use WorkflowManager directly with unified modes."""
    pass

def get_executor(mode: str, workflow_manager: Any) -> Any:
    """DEPRECATED - Execution is now handled in WorkflowManager._execute_* methods."""
    raise NotImplementedError(
        "Pattern-based executors have been removed. "
        "Use WorkflowManager with modes: backtest, signal_generation, signal_replay"
    )


__all__ = [
    'ExecutionStrategy',
    'get_executor'
]