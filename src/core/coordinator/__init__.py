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

# Delayed imports to avoid circular dependencies
def get_coordinator():
    from .coordinator import Coordinator
    return Coordinator

def get_workflow_manager():
    from .topology import WorkflowManager
    return WorkflowManager

# Import simple types that don't have dependencies
from ..types.workflow import (
    WorkflowConfig,
    WorkflowType,
    WorkflowPhase,
    ExecutionContext
)

# Import types only if needed
try:
    from ..types.workflow import WorkflowResult
except ImportError:
    # Use a simple dataclass if pydantic not available
    from dataclasses import dataclass, field
    from typing import Dict, Any, List, Optional
    from datetime import datetime
    
    @dataclass
    class WorkflowResult:
        workflow_id: str
        workflow_type: WorkflowType
        success: bool = True
        results: Dict[str, Any] = field(default_factory=dict)
        errors: List[str] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
        duration_seconds: Optional[float] = None
        _start_time: Optional[datetime] = field(default=None, init=False)
        
        def finalize(self):
            if self._start_time:
                self.duration_seconds = (datetime.now() - self._start_time).total_seconds()
# Phase management imports (using actual sequencer module)
try:
    from .sequencer import (
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
except ImportError as e:
    # Phase management not available - use stubs
    class PhaseTransition: pass
    class ContainerNamingStrategy: pass
    class StreamingResultWriter: pass
    class ResultAggregator: pass
    class StrategyIdentity: pass
    class WorkflowState: pass
    class CheckpointManager: pass
    class WorkflowCoordinator: pass
    class SharedServiceRegistry: pass
    class WalkForwardValidator: pass
    def integrate_phase_management(*args, **kwargs): pass
# from .multi_symbol_architecture import (
#     SymbolAllocation,
#     RiskContainer,
#     MultiSymbolCoordinator,
#     integrate_multi_symbol_support
# )

__all__ = [
    'get_coordinator',  # Changed from Coordinator to avoid circular import
    'get_workflow_manager',  # Modular workflow manager
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
    'integrate_phase_management'
    # Multi-symbol architecture (commented out)
    # 'SymbolAllocation',
    # 'RiskContainer', 
    # 'MultiSymbolCoordinator',
    # 'integrate_multi_symbol_support'
]