"""
Coordinator protocols for manager interfaces.
"""
from typing import Protocol, Dict, Any, Optional
from abc import abstractmethod

from .types import WorkflowConfig, WorkflowResult, ExecutionContext, PhaseResult


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