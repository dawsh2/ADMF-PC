"""
Clean Coordinator implementation following BACKTEST.MD architecture.

The Coordinator orchestrates workflows but delegates all execution to containers.
No circular dependencies, no knowledge of execution details.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..containers import UniversalScopedContainer, ContainerLifecycleManager
from ..containers.backtest import (
    BacktestContainerFactory,
    BacktestPattern,
    BacktestConfig
)
from ..events import EventBus, EventType, Event
from ..logging import StructuredLogger

from .protocols import WorkflowManager
from .types import (
    WorkflowConfig,
    WorkflowType,
    WorkflowPhase,
    WorkflowResult,
    ExecutionContext
)

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Main Coordinator for the ADMF-PC system.
    
    The Coordinator:
    - Reads workflow configurations
    - Delegates to specialized containers
    - Manages workflow phases
    - Aggregates results
    - NO execution logic
    """
    
    def __init__(
        self,
        shared_services: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """Initialize the Coordinator."""
        self.shared_services = shared_services or {}
        self.logger = StructuredLogger("coordinator")
        
        # Container lifecycle management
        self.container_manager = ContainerLifecycleManager(
            max_containers=100,
            shared_services=self.shared_services
        )
        
        # Event bus for coordinator-level events
        self.event_bus = EventBus()
        
        # Workflow managers by type
        self._workflow_managers: Dict[WorkflowType, WorkflowManager] = {}
        self._setup_workflow_managers()
        
    def _setup_workflow_managers(self):
        """Setup workflow managers for each workflow type."""
        # Import here to avoid circular dependencies
        from .managers import (
            BacktestWorkflowManager,
            OptimizationWorkflowManager,
            LiveWorkflowManager
        )
        
        self._workflow_managers[WorkflowType.BACKTEST] = BacktestWorkflowManager(self)
        self._workflow_managers[WorkflowType.OPTIMIZATION] = OptimizationWorkflowManager(self)
        self._workflow_managers[WorkflowType.LIVE] = LiveWorkflowManager(self)
    
    async def execute_workflow(
        self,
        config: WorkflowConfig,
        workflow_id: Optional[str] = None
    ) -> WorkflowResult:
        """
        Execute a workflow based on configuration.
        
        The coordinator orchestrates but does not execute.
        All execution happens in specialized containers.
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        
        self.logger.info(
            "Starting workflow execution",
            workflow_id=workflow_id,
            workflow_type=config.workflow_type
        )
        
        # Create execution context
        context = ExecutionContext(
            workflow_id=workflow_id,
            start_time=datetime.now(),
            config=config,
            shared_resources={}
        )
        
        # Get appropriate workflow manager
        manager = self._workflow_managers.get(config.workflow_type)
        if not manager:
            raise ValueError(f"Unknown workflow type: {config.workflow_type}")
        
        # Execute through workflow manager
        try:
            result = await manager.execute(config, context)
            
            self.logger.info(
                "Workflow completed",
                workflow_id=workflow_id,
                success=result.success
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Workflow failed",
                workflow_id=workflow_id,
                error=str(e)
            )
            
            return WorkflowResult(
                workflow_id=workflow_id,
                success=False,
                errors=[str(e)],
                results={}
            )
        
        finally:
            # Cleanup containers
            await self.container_manager.cleanup_completed()
    
    async def create_backtest_container(
        self,
        workflow_id: str,
        config: WorkflowConfig,
        pattern: BacktestPattern = BacktestPattern.FULL
    ) -> str:
        """
        Create a backtest container using standardized factory.
        
        This is the ONLY way to create backtest containers,
        ensuring consistency per BACKTEST.MD.
        """
        # Build backtest configuration
        backtest_config = BacktestConfig(
            container_id=f"backtest_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pattern=pattern,
            data_config=config.data_config,
            indicator_config=self._extract_indicators(config),
            classifiers=self._extract_classifiers(config),
            execution_config=config.backtest_config or {}
        )
        
        # Use factory to create container
        container = BacktestContainerFactory.create_instance(backtest_config)
        
        # Register with lifecycle manager
        await self.container_manager.register_container(container)
        
        self.logger.info(
            "Created backtest container",
            container_id=container.container_id,
            pattern=pattern.value
        )
        
        return container.container_id
    
    async def execute_container(
        self,
        container_id: str,
        method: str = "execute",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a method on a container.
        
        The coordinator doesn't know what the container does,
        just that it can execute.
        """
        container = self.container_manager.get_container(container_id)
        if not container:
            raise ValueError(f"Container not found: {container_id}")
        
        # Get the execution method
        executor = getattr(container, method, None)
        if not executor:
            raise ValueError(f"Container {container_id} has no method: {method}")
        
        # Execute and return results
        return await executor(**kwargs)
    
    def _extract_indicators(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract indicator configuration from workflow config."""
        # Look in various places indicators might be defined
        indicators = config.parameters.get('indicators', {})
        
        # Also check in backtest config
        if config.backtest_config:
            indicators.update(config.backtest_config.get('indicators', {}))
        
        # And in strategy configs
        for strategy in config.parameters.get('strategies', []):
            indicators.update(strategy.get('indicators', {}))
        
        return indicators
    
    def _extract_classifiers(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """Extract classifier configurations from workflow config."""
        classifiers = []
        
        # Direct classifiers
        if 'classifiers' in config.parameters:
            classifiers.extend(config.parameters['classifiers'])
        
        # Classifiers in backtest config
        if config.backtest_config and 'classifiers' in config.backtest_config:
            classifiers.extend(config.backtest_config['classifiers'])
        
        # Default if none specified
        if not classifiers:
            classifiers = [{
                'type': 'simple',
                'parameters': {},
                'risk_profiles': [{
                    'name': 'default',
                    'capital_allocation': 100000,
                    'strategies': config.parameters.get('strategies', [])
                }]
            }]
        
        return classifiers
    
    async def shutdown(self):
        """Shutdown coordinator and cleanup resources."""
        self.logger.info("Shutting down coordinator")
        
        # Stop all containers
        await self.container_manager.stop_all()
        
        # Cleanup
        await self.container_manager.cleanup_all()
        
        self.logger.info("Coordinator shutdown complete")