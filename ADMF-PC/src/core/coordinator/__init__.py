"""
Core Coordinator Module.

The Coordinator serves as the primary entry point for all high-level operations.
It interprets workflow configurations and delegates execution to specialized managers.
"""

from .coordinator import Coordinator
from .types import (
    WorkflowConfig,
    WorkflowType,
    WorkflowPhase,
    WorkflowResult,
    ExecutionContext
)

__all__ = [
    'Coordinator',
    'WorkflowConfig',
    'WorkflowType',
    'WorkflowPhase',
    'WorkflowResult',
    'ExecutionContext'
]