"""
Core Coordinator Module.

The Coordinator serves as the primary entry point for all high-level operations.
It interprets workflow configurations and delegates execution to specialized managers.

Now includes enhanced phase management with:
- Event flow between phases
- Container naming & tracking strategy
- Result streaming & aggregation
- Cross-regime strategy identity
- State management with checkpointing
- Walk-forward validation support
"""

from .coordinator import Coordinator
from .types import (
    WorkflowConfig,
    WorkflowType,
    WorkflowPhase,
    WorkflowResult,
    ExecutionContext
)
from .phase_management import (
    PhaseTransition,
    ContainerNamingStrategy,
    StreamingResultWriter,
    ResultAggregator,
    StrategyIdentity,
    WorkflowState,
    CheckpointManager,
    WorkflowCoordinator,
    SharedServiceRegistry,
    WalkForwardValidator,
    integrate_phase_management
)
from .multi_symbol_architecture import (
    SymbolAllocation,
    RiskContainer,
    MultiSymbolCoordinator,
    integrate_multi_symbol_support
)

__all__ = [
    'Coordinator',
    'WorkflowConfig',
    'WorkflowType',
    'WorkflowPhase',
    'WorkflowResult',
    'ExecutionContext',
    # Phase management
    'PhaseTransition',
    'ContainerNamingStrategy',
    'StreamingResultWriter',
    'ResultAggregator',
    'StrategyIdentity',
    'WorkflowState',
    'CheckpointManager',
    'WorkflowCoordinator',
    'SharedServiceRegistry',
    'WalkForwardValidator',
    'integrate_phase_management',
    # Multi-symbol architecture
    'SymbolAllocation',
    'RiskContainer',
    'MultiSymbolCoordinator',
    'integrate_multi_symbol_support'
]