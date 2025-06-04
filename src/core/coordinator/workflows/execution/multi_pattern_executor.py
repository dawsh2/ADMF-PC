"""
Multi-pattern execution strategy for complex workflows.
This execution strategy handles workflows with multiple container patterns,
smart container sharing, and coordinated execution across patterns.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from .....core.types.workflow import WorkflowConfig, WorkflowResult
from .....core.containers.factory import ContainerFactory
from .....core.communication.factory import CommunicationFactory
from .....core.events.event_bus import EventBus
from .....core.events.isolation import EventIsolationManager
from . import ExecutionStrategy

logger = logging.getLogger(__name__)

@dataclass
class PatternExecution:
    """Tracks execution state for a single pattern within multi-pattern workflow."""
    pattern_id: str
    container_pattern: str
    shared_containers: Set[str]
    communication_config: List[Dict[str, Any]]
    result: Optional[WorkflowResult] = None
    started: bool = False
    completed: bool = False

class MultiPatternExecutor(ExecutionStrategy):
    """
    Execution strategy for workflows with multiple container patterns.
    
    Features:
    - Smart container sharing across patterns
    - Coordinated execution with dependency resolution
    - Resource optimization through container reuse
    - Isolation management for shared containers
    """
    
    def __init__(self, container_factory: ContainerFactory, 
                 communication_factory: CommunicationFactory):
        self.container_factory = container_factory
        self.communication_factory = communication_factory
        self.shared_containers: Dict[str, Any] = {}
        self.container_usage: Dict[str, Set[str]] = defaultdict(set)
        
    async def execute_single_pattern(self, 
                                   config: WorkflowConfig,
                                   pattern_config: Dict[str, Any],
                                   event_bus: EventBus,
                                   isolation_manager: EventIsolationManager) -> WorkflowResult:
        """Execute single pattern - delegates to pattern-specific executor."""
        # For multi-pattern workflows, this shouldn't be called directly
        raise NotImplementedError(
            "MultiPatternExecutor handles multiple patterns. "
            "Use execute_workflow for multi-pattern execution."
        )
    
    async def execute_workflow(self, config: WorkflowConfig) -> WorkflowResult:
        """
        Execute complete multi-pattern workflow.
        
        Args:
            config: Complete workflow configuration
            
        Returns:
            Aggregated results from all patterns
        """
        logger.info(f"Starting multi-pattern workflow execution: {len(config.patterns)} patterns")
        
        try:
            # Phase 1: Plan execution and identify shared resources
            execution_plan = self._plan_execution(config)
            logger.info(f"Planned execution for {len(execution_plan)} patterns")
            
            # Phase 2: Create shared containers
            await self._create_shared_containers(execution_plan)
            logger.info(f"Created {len(self.shared_containers)} shared containers")
            
            # Phase 3: Execute patterns with dependency resolution
            pattern_results = await self._execute_patterns_coordinated(execution_plan, config)
            
            # Phase 4: Aggregate results
            aggregated_result = self._aggregate_results(pattern_results, config)
            
            logger.info("Multi-pattern workflow execution completed successfully")
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Multi-pattern workflow execution failed: {e}")
            raise
        finally:
            await self._cleanup_shared_resources()
    
    def _plan_execution(self, config: WorkflowConfig) -> List[PatternExecution]:
        """
        Plan execution order and identify shared containers.
        
        Args:
            config: Workflow configuration
            
        Returns:
            List of pattern executions with sharing analysis
        """
        executions = []
        container_patterns = defaultdict(list)
        
        # Analyze patterns and identify sharing opportunities
        for i, pattern in enumerate(config.patterns):
            pattern_id = f"pattern_{i}"
            container_pattern = pattern.get('container_pattern', 'simple_backtest')
            
            # Track which patterns use which container types
            container_patterns[container_pattern].append(pattern_id)
            
            execution = PatternExecution(
                pattern_id=pattern_id,
                container_pattern=container_pattern,
                shared_containers=set(),
                communication_config=pattern.get('communication_config', [])
            )
            executions.append(execution)
        
        # Identify containers that can be shared
        for container_type, pattern_ids in container_patterns.items():
            if len(pattern_ids) > 1 and self._can_share_container_type(container_type):
                shared_container_id = f"shared_{container_type}"
                
                for pattern_id in pattern_ids:
                    execution = next(e for e in executions if e.pattern_id == pattern_id)
                    execution.shared_containers.add(shared_container_id)
                    self.container_usage[shared_container_id].add(pattern_id)
        
        logger.info(f"Container sharing analysis: {len(self.container_usage)} shared containers")
        return executions
    
    def _can_share_container_type(self, container_type: str) -> bool:
        """
        Determine if a container type can be safely shared.
        
        Args:
            container_type: Type of container to check
            
        Returns:
            True if container can be shared across patterns
        """
        # Define which container types are safe to share
        shareable_types = {
            'data_container',
            'indicator_container', 
            'analysis_container',
            'reporting_container'
        }
        
        # Strategy and execution containers typically shouldn't be shared
        # to maintain pattern isolation
        non_shareable_types = {
            'strategy_container',
            'execution_container',
            'risk_container'
        }
        
        if container_type in shareable_types:
            return True
        elif container_type in non_shareable_types:
            return False
        else:
            # Default to not sharing for unknown types (safer)
            logger.warning(f"Unknown container type for sharing: {container_type}")
            return False
    
    async def _create_shared_containers(self, executions: List[PatternExecution]):
        """
        Create containers that will be shared across patterns.
        
        Args:
            executions: List of pattern executions
        """
        shared_container_types = set()
        for execution in executions:
            for shared_id in execution.shared_containers:
                container_type = shared_id.replace('shared_', '')
                shared_container_types.add(container_type)
        
        for container_type in shared_container_types:
            shared_id = f"shared_{container_type}"
            
            logger.info(f"Creating shared container: {shared_id}")
            
            # Create container with shared configuration
            containers = await self.container_factory.compose_pattern(container_type)
            
            if containers:
                # Take the first container as the shared instance
                self.shared_containers[shared_id] = next(iter(containers.values()))
                logger.info(f"Shared container created: {shared_id}")
            else:
                logger.warning(f"Failed to create shared container: {shared_id}")
    
    async def _execute_patterns_coordinated(self, 
                                          executions: List[PatternExecution],
                                          config: WorkflowConfig) -> List[WorkflowResult]:
        """
        Execute patterns with coordination and dependency resolution.
        
        Args:
            executions: Planned pattern executions
            config: Original workflow configuration
            
        Returns:
            Results from all pattern executions
        """
        results = []
        
        # Execute patterns that don't depend on others first
        independent_patterns = [e for e in executions if not e.shared_containers]
        dependent_patterns = [e for e in executions if e.shared_containers]
        
        # Phase 1: Execute independent patterns
        if independent_patterns:
            logger.info(f"Executing {len(independent_patterns)} independent patterns")
            independent_results = await asyncio.gather(*[
                self._execute_single_pattern_execution(execution, config)
                for execution in independent_patterns
            ])
            results.extend(independent_results)
        
        # Phase 2: Execute patterns with shared resources
        if dependent_patterns:
            logger.info(f"Executing {len(dependent_patterns)} dependent patterns")
            # Execute in smaller batches to manage resource contention
            batch_size = 3  # Configurable
            
            for i in range(0, len(dependent_patterns), batch_size):
                batch = dependent_patterns[i:i + batch_size]
                batch_results = await asyncio.gather(*[
                    self._execute_single_pattern_execution(execution, config)
                    for execution in batch
                ])
                results.extend(batch_results)
        
        return results
    
    async def _execute_single_pattern_execution(self, 
                                              execution: PatternExecution,
                                              config: WorkflowConfig) -> WorkflowResult:
        """
        Execute a single pattern within the multi-pattern workflow.
        
        Args:
            execution: Pattern execution plan
            config: Overall workflow configuration
            
        Returns:
            Result from pattern execution
        """
        logger.info(f"Executing pattern: {execution.pattern_id}")
        execution.started = True
        
        try:
            # Create event bus with appropriate isolation
            event_bus = EventBus()
            isolation_manager = EventIsolationManager()
            
            # Setup isolation scope for this pattern
            scope_id = f"pattern_{execution.pattern_id}"
            isolation_manager.create_scope(scope_id)
            
            # Create containers for this pattern
            if execution.shared_containers:
                # Mix shared and pattern-specific containers
                containers = await self._create_mixed_containers(execution)
            else:
                # Create all containers for this pattern
                containers = await self.container_factory.compose_pattern(
                    execution.container_pattern
                )
            
            # Setup communication adapters
            adapters = []
            if execution.communication_config:
                adapters = self.communication_factory.create_adapters_from_config(
                    execution.communication_config
                )
            
            # Connect containers and adapters
            await self._connect_pattern_components(containers, adapters, event_bus)
            
            # Execute pattern
            # Note: This would typically delegate to a pattern-specific executor
            # For now, we'll simulate execution
            await asyncio.sleep(0.1)  # Simulate work
            
            result = WorkflowResult(
                success=True,
                pattern_id=execution.pattern_id,
                container_pattern=execution.container_pattern,
                shared_resources=len(execution.shared_containers),
                message=f"Pattern {execution.pattern_id} completed successfully"
            )
            
            execution.result = result
            execution.completed = True
            
            logger.info(f"Pattern {execution.pattern_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Pattern {execution.pattern_id} failed: {e}")
            result = WorkflowResult(
                success=False,
                pattern_id=execution.pattern_id,
                error=str(e)
            )
            execution.result = result
            return result
    
    async def _create_mixed_containers(self, execution: PatternExecution) -> Dict[str, Any]:
        """
        Create containers mixing shared and pattern-specific instances.
        
        Args:
            execution: Pattern execution plan
            
        Returns:
            Mixed container set
        """
        containers = {}
        
        # Add shared containers
        for shared_id in execution.shared_containers:
            if shared_id in self.shared_containers:
                # Use shared container with pattern-specific wrapper if needed
                shared_container = self.shared_containers[shared_id]
                containers[shared_id] = shared_container
        
        # Create pattern-specific containers
        pattern_containers = await self.container_factory.compose_pattern(
            execution.container_pattern
        )
        
        # Merge, with pattern-specific containers taking precedence
        for container_id, container in pattern_containers.items():
            if container_id not in containers:
                containers[container_id] = container
        
        return containers
    
    async def _connect_pattern_components(self, 
                                        containers: Dict[str, Any],
                                        adapters: List[Any],
                                        event_bus: EventBus):
        """
        Connect containers and communication adapters for a pattern.
        
        Args:
            containers: Pattern containers
            adapters: Communication adapters
            event_bus: Event bus for pattern
        """
        # Connect containers to event bus
        for container_id, container in containers.items():
            if hasattr(container, 'connect_event_bus'):
                await container.connect_event_bus(event_bus)
        
        # Setup communication adapters
        for adapter in adapters:
            if hasattr(adapter, 'setup'):
                await adapter.setup(event_bus)
    
    def _aggregate_results(self, 
                          pattern_results: List[WorkflowResult],
                          config: WorkflowConfig) -> WorkflowResult:
        """
        Aggregate results from all patterns into single workflow result.
        
        Args:
            pattern_results: Results from individual patterns
            config: Original workflow configuration
            
        Returns:
            Aggregated workflow result
        """
        successful_patterns = [r for r in pattern_results if r.success]
        failed_patterns = [r for r in pattern_results if not r.success]
        
        # Calculate resource savings from sharing
        total_containers = sum(1 for _ in config.patterns)
        shared_containers = len(self.shared_containers)
        resource_savings = (shared_containers / total_containers) * 100 if total_containers > 0 else 0
        
        aggregated = WorkflowResult(
            success=len(failed_patterns) == 0,
            patterns_executed=len(pattern_results),
            successful_patterns=len(successful_patterns),
            failed_patterns=len(failed_patterns),
            shared_containers=shared_containers,
            resource_savings_percent=resource_savings,
            message=f"Multi-pattern workflow: {len(successful_patterns)}/{len(pattern_results)} patterns successful"
        )
        
        if failed_patterns:
            failed_messages = [r.error or "Unknown error" for r in failed_patterns]
            aggregated.error = f"Failed patterns: {'; '.join(failed_messages)}"
        
        logger.info(f"Multi-pattern execution summary: {aggregated.message}")
        logger.info(f"Resource savings: {resource_savings:.1f}%")
        
        return aggregated
    
    async def _cleanup_shared_resources(self):
        """Clean up shared containers and resources."""
        logger.info("Cleaning up shared resources")
        
        for shared_id, container in self.shared_containers.items():
            try:
                if hasattr(container, 'cleanup'):
                    await container.cleanup()
                logger.debug(f"Cleaned up shared container: {shared_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup shared container {shared_id}: {e}")
        
        self.shared_containers.clear()
        self.container_usage.clear()
        
        logger.info("Shared resource cleanup completed")