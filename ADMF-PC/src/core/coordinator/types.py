"""
Coordinator type definitions.
"""
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


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


class WorkflowConfig(BaseModel):
    """Configuration for a workflow."""
    workflow_type: WorkflowType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Common configuration
    data_config: Dict[str, Any] = Field(default_factory=dict)
    infrastructure_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Type-specific configuration
    optimization_config: Optional[Dict[str, Any]] = None
    backtest_config: Optional[Dict[str, Any]] = None
    live_config: Optional[Dict[str, Any]] = None
    analysis_config: Optional[Dict[str, Any]] = None
    validation_config: Optional[Dict[str, Any]] = None
    
    # Execution settings
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_seconds: Optional[int] = None
    
    # Resource requirements
    memory_limit_mb: Optional[int] = None
    cpu_cores: Optional[int] = None
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


@dataclass
class PhaseResult:
    """Result from a workflow phase."""
    phase: WorkflowPhase
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: Optional[float] = None


@dataclass
class WorkflowResult:
    """Aggregated result from workflow execution."""
    workflow_id: str
    workflow_type: WorkflowType
    success: bool
    
    # Phase results
    phase_results: Dict[WorkflowPhase, PhaseResult] = field(default_factory=dict)
    
    # Aggregated data
    final_results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_phase_result(self, result: PhaseResult) -> None:
        """Add a phase result."""
        self.phase_results[result.phase] = result
        
        # Update overall success status
        if not result.success:
            self.success = False
            
        # Collect errors and warnings
        self.errors.extend(result.errors)
        self.warnings.extend(result.warnings)
        
    def finalize(self) -> None:
        """Finalize the workflow result."""
        self.end_time = datetime.now()
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()