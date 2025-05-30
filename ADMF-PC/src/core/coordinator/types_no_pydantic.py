"""
Coordinator type definitions without pydantic dependency.
"""
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field


class WorkflowType(str, Enum):
    """Supported workflow types."""
    OPTIMIZATION = "optimization"
    BACKTEST = "backtest"
    LIVE_TRADING = "live_trading"
    ANALYSIS = "analysis"
    VALIDATION = "validation"


class WorkflowPhase(str, Enum):
    """Workflow execution phases."""
    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    COMPUTATION = "computation"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"
    FINALIZATION = "finalization"


@dataclass
class ExecutionContext:
    """Context for workflow execution."""
    workflow_id: str
    workflow_type: WorkflowType
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    shared_resources: Dict[str, Any] = field(default_factory=dict)
    current_phase: Optional[WorkflowPhase] = None
    completed_phases: Set[WorkflowPhase] = field(default_factory=set)
    
    def update_phase(self, phase: WorkflowPhase) -> None:
        """Update current phase and track completion."""
        if self.current_phase:
            self.completed_phases.add(self.current_phase)
        self.current_phase = phase


@dataclass
class WorkflowConfig:
    """Configuration for a workflow."""
    workflow_type: WorkflowType
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    backtest_config: Dict[str, Any] = field(default_factory=dict)
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    
    def dict(self):
        """Convert to dictionary (pydantic compatibility)."""
        return {
            'workflow_type': self.workflow_type.value,
            'parameters': self.parameters,
            'data_config': self.data_config,
            'backtest_config': self.backtest_config,
            'optimization_config': self.optimization_config
        }


@dataclass  
class PhaseResult:
    """Result from a phase execution."""
    phase: WorkflowPhase
    success: bool = True
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    workflow_id: str
    workflow_type: WorkflowType
    success: bool = True
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list) 
    warnings: List[str] = field(default_factory=list)
    phase_results: Dict[WorkflowPhase, PhaseResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: Optional[float] = None
    _start_time: Optional[datetime] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize start time."""
        self._start_time = datetime.now()
        
    def finalize(self) -> None:
        """Finalize result and calculate duration."""
        if self._start_time:
            self.duration_seconds = (datetime.now() - self._start_time).total_seconds()
            
    def dict(self):
        """Convert to dictionary (pydantic compatibility)."""
        return {
            'workflow_id': self.workflow_id,
            'workflow_type': self.workflow_type.value,
            'success': self.success,
            'results': self.results,
            'errors': self.errors,
            'warnings': self.warnings,
            'phase_results': {k.value: v.__dict__ for k, v in self.phase_results.items()},
            'metadata': self.metadata,
            'duration_seconds': self.duration_seconds
        }