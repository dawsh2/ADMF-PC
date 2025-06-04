"""
Execution strategies for workflow patterns.

This module provides different execution strategies for containers:
- Standard: Basic sequential container execution
- Nested: Hierarchical container structures (Risk > Portfolio > Strategy)
- Pipeline: Pipeline communication with adapters
- Multi-pattern: Multi-phase workflow execution
"""

from typing import Dict, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ...topology import WorkflowManager

from ....types.workflow import WorkflowConfig, ExecutionContext, WorkflowResult


class ExecutionStrategy(ABC):
    """Base class for all execution strategies."""
    
    def __init__(self, workflow_manager: 'WorkflowManager'):
        """Initialize execution strategy with workflow manager reference."""
        self.workflow_manager = workflow_manager
        self.factory = workflow_manager.factory
        self.adapter_factory = workflow_manager.adapter_factory
        
    @abstractmethod
    async def execute_single_pattern(
        self,
        pattern_info: Dict[str, Any],
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute a single container pattern."""
        pass
    
    async def execute_multi_pattern(
        self,
        patterns: list[Dict[str, Any]],
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Default multi-pattern execution (can be overridden)."""
        # Default: execute patterns sequentially
        results = []
        for i, pattern_info in enumerate(patterns):
            pattern_result = await self.execute_single_pattern(pattern_info, config, context)
            results.append(pattern_result)
            
        # Aggregate results
        overall_success = all(r.success for r in results)
        return WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=config.workflow_type,
            success=overall_success,
            final_results={
                'pattern_results': {f'pattern_{i}': r.final_results for i, r in enumerate(results)},
                'execution_order': [p['name'] for p in patterns]
            },
            metadata={'execution_mode': 'multi_pattern', 'patterns_executed': len(patterns)}
        )


def get_executor(mode: str, workflow_manager: 'WorkflowManager') -> ExecutionStrategy:
    """Factory function to get appropriate executor."""
    if mode == 'standard':
        from .standard_executor import StandardExecutor
        return StandardExecutor(workflow_manager)
    elif mode == 'nested':
        from .nested_executor import NestedExecutor
        return NestedExecutor(workflow_manager)
    elif mode == 'pipeline':
        from .pipeline_executor import PipelineExecutor
        return PipelineExecutor(workflow_manager)
    elif mode == 'multi_pattern':
        from .multi_pattern_executor import MultiPatternExecutor
        # Multi-pattern executor has different constructor signature
        from ....containers.factory import ContainerFactory
        from ....communication.factory import CommunicationFactory
        return MultiPatternExecutor(
            container_factory=ContainerFactory(),
            communication_factory=CommunicationFactory()
        )
    else:
        raise ValueError(f"Unknown execution mode: {mode}")


__all__ = [
    'ExecutionStrategy',
    'get_executor'
]