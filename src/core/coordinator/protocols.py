"""
Coordinator protocols for manager interfaces.
"""
from typing import Protocol, Dict, Any, Optional, List
from abc import abstractmethod
from pathlib import Path

# Use simple types to avoid circular imports
try:
    from ..types.workflow import WorkflowConfig, WorkflowResult, ExecutionContext
    from .types import PhaseResult
except ImportError:
    from ..types.workflow import WorkflowConfig, ExecutionContext
    WorkflowResult = Any
    PhaseResult = Any


class WorkflowManager(Protocol):
    """Protocol for workflow-specific managers."""
    
    @abstractmethod
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute the workflow with given configuration."""
        ...
        
    @abstractmethod
    async def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Validate workflow-specific configuration."""
        ...
        
    @abstractmethod
    def get_required_capabilities(self) -> Dict[str, Any]:
        """Get required infrastructure capabilities."""
        ...


class PhaseExecutor(Protocol):
    """Protocol for phase-specific executors."""
    
    @abstractmethod
    async def execute(
        self,
        config: Dict[str, Any],
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute a specific phase."""
        ...
        
    @abstractmethod
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate phase inputs."""
        ...


class ResultStreamer(Protocol):
    """Protocol for streaming results during execution."""
    
    @abstractmethod
    def write_result(self, result: Dict[str, Any]) -> None:
        """Write a single result."""
        ...
        
    @abstractmethod
    def write_batch(self, results: List[Dict[str, Any]]) -> None:
        """Write a batch of results."""
        ...
        
    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered results."""
        ...
        
    @abstractmethod
    def close(self) -> None:
        """Close the streamer."""
        ...


class CheckpointManager(Protocol):
    """Protocol for managing workflow checkpoints."""
    
    @abstractmethod
    def save_checkpoint(
        self,
        workflow_id: str,
        phase: str,
        state: Dict[str, Any]
    ) -> Path:
        """Save a checkpoint."""
        ...
        
    @abstractmethod
    def load_checkpoint(
        self,
        workflow_id: str,
        phase: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load a checkpoint."""
        ...
        
    @abstractmethod
    def list_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        ...
        
    @abstractmethod
    def delete_checkpoint(self, workflow_id: str, phase: Optional[str] = None) -> bool:
        """Delete a checkpoint."""
        ...


class ResourceManager(Protocol):
    """Protocol for resource management."""
    
    @abstractmethod
    async def allocate_resources(
        self,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Allocate required resources."""
        ...
        
    @abstractmethod
    async def release_resources(self, resources: Dict[str, Any]) -> None:
        """Release allocated resources."""
        ...
        
    @abstractmethod
    def check_availability(self, requirements: Dict[str, Any]) -> bool:
        """Check if required resources are available."""
        ...


class ResultAggregator(Protocol):
    """Protocol for result aggregation."""
    
    @abstractmethod
    def aggregate(
        self,
        phase_results: Dict[str, PhaseResult]
    ) -> Dict[str, Any]:
        """Aggregate phase results into final output."""
        ...
        
    @abstractmethod
    def validate_completeness(
        self,
        results: Dict[str, Any]
    ) -> bool:
        """Validate that all required results are present."""
        ...