"""
Coordinator type definitions with automatic pydantic fallback.
"""

# Try to use pydantic if available, otherwise use dataclasses
try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

if not HAS_PYDANTIC:
    # If no pydantic, import everything from the no-pydantic version
    from .types_no_pydantic import *
else:
    # Define types using pydantic
    from enum import Enum
    from typing import Dict, Any, List, Optional, Set
    from datetime import datetime
    
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


    class ExecutionContext(BaseModel):
        """Context for workflow execution."""
        workflow_id: str
        workflow_type: WorkflowType
        start_time: datetime = Field(default_factory=datetime.now)
        metadata: Dict[str, Any] = Field(default_factory=dict)
        shared_resources: Dict[str, Any] = Field(default_factory=dict)
        current_phase: Optional[WorkflowPhase] = None
        completed_phases: Set[WorkflowPhase] = Field(default_factory=set)
        
        def update_phase(self, phase: WorkflowPhase) -> None:
            """Update current phase and track completion."""
            if self.current_phase:
                self.completed_phases.add(self.current_phase)
            self.current_phase = phase


    class WorkflowConfig(BaseModel):
        """Configuration for a workflow."""
        workflow_type: WorkflowType
        parameters: Dict[str, Any] = Field(default_factory=dict)
        data_config: Dict[str, Any] = Field(default_factory=dict)
        backtest_config: Dict[str, Any] = Field(default_factory=dict)
        optimization_config: Dict[str, Any] = Field(default_factory=dict)


    class PhaseResult(BaseModel):
        """Result from a phase execution."""
        phase: WorkflowPhase
        success: bool = True
        results: Dict[str, Any] = Field(default_factory=dict)
        errors: List[str] = Field(default_factory=list)
        warnings: List[str] = Field(default_factory=list)
        duration_seconds: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)


    class WorkflowResult(BaseModel):
        """Result from workflow execution."""
        workflow_id: str
        workflow_type: WorkflowType
        success: bool = True
        results: Dict[str, Any] = Field(default_factory=dict)
        errors: List[str] = Field(default_factory=list)
        warnings: List[str] = Field(default_factory=list)
        phase_results: Dict[WorkflowPhase, PhaseResult] = Field(default_factory=dict)
        metadata: Dict[str, Any] = Field(default_factory=dict)
        duration_seconds: Optional[float] = None
        _start_time: Optional[datetime] = None
        
        def __init__(self, **data):
            super().__init__(**data)
            self._start_time = datetime.now()
            
        def finalize(self) -> None:
            """Finalize result and calculate duration."""
            if self._start_time:
                self.duration_seconds = (datetime.now() - self._start_time).total_seconds()