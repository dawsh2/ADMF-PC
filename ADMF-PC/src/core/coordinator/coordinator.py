"""
Main Coordinator implementation.

The Coordinator is the primary entry point for all high-level operations.
It interprets workflow configurations and delegates execution to specialized managers.
"""
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from ..containers import UniversalScopedContainer, ContainerLifecycleManager, ContainerFactory
from ..events import EventBus, EventType, Event
from ..infrastructure import InfrastructureCapability
from ..logging import StructuredLogger
from ..components import ComponentFactory

from .types import (
    WorkflowConfig, WorkflowResult, ExecutionContext,
    WorkflowType, WorkflowPhase
)
from .infrastructure import InfrastructureSetup
from .managers import WorkflowManagerFactory
from .protocols import WorkflowManager


logger = logging.getLogger(__name__)


class Coordinator:
    """
    Main Coordinator for the ADMF-PC system.
    
    The Coordinator:
    - Reads workflow configurations
    - Sets up required infrastructure
    - Delegates to specialized managers
    - Aggregates results in standardized format
    - Ensures clean resource management
    """
    
    def __init__(
        self,
        shared_services: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        logger: Optional[StructuredLogger] = None
    ):
        """Initialize the Coordinator."""
        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Shared services across all containers
        self.shared_services = shared_services or {}
        
        # Core infrastructure
        self.logger = logger or StructuredLogger("coordinator")
        self.container_manager = ContainerLifecycleManager(self.shared_services)
        self.container_factory = ContainerFactory(self.shared_services)
        self.component_factory = ComponentFactory()
        
        # Create coordinator's own container
        self.coordinator_container = UniversalScopedContainer(
            container_id="coordinator",
            container_type="coordinator",
            shared_services=self.shared_services
        )
        
        # Initialize components
        self.infrastructure = InfrastructureSetup(
            self.coordinator_container, 
            self.coordinator_container.event_bus
        )
        self.manager_factory = WorkflowManagerFactory(
            self.container_manager,
            self.shared_services
        )
        
        # Track active workflows
        self._active_workflows: Dict[str, ExecutionContext] = {}
        self._workflow_containers: Dict[str, str] = {}  # workflow_id -> container_id
        self._workflow_lock = asyncio.Lock()
        
        # Register event handlers
        self._register_event_handlers()
        
        self.logger.info(
            "Coordinator initialized",
            shared_services=list(self.shared_services.keys())
        )
        
    async def execute_workflow(
        self,
        config: Union[WorkflowConfig, Dict[str, Any]]
    ) -> WorkflowResult:
        """
        Execute a workflow with the given configuration.
        
        This is the main entry point for all workflow execution.
        """
        # Convert dict to WorkflowConfig if needed
        if isinstance(config, dict):
            config = WorkflowConfig(**config)
            
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Create execution context
        context = ExecutionContext(
            workflow_id=workflow_id,
            workflow_type=config.workflow_type,
            metadata={
                'config': config.dict(),
                'start_time': datetime.now().isoformat()
            }
        )
        
        # Track active workflow
        async with self._workflow_lock:
            self._active_workflows[workflow_id] = context
            
        try:
            # Log workflow start
            self.logger.info(
                "Starting workflow execution",
                workflow_id=workflow_id,
                workflow_type=config.workflow_type.value
            )
            
            # Create container for this workflow
            container_id = await self._create_workflow_container(
                workflow_id, config, context
            )
            context.metadata['container_id'] = container_id
            
            # Get the container's event bus
            container = self.container_manager.active_containers[container_id]
            event_bus = container.event_bus
            
            # Emit workflow start event
            event = Event(
                event_type=EventType.INFO,
                payload={
                    'type': 'workflow.start',
                    'workflow_id': workflow_id,
                    'workflow_type': config.workflow_type.value,
                    'config': config.dict()
                },
                source_id="coordinator",
                container_id=container_id
            )
            event_bus.publish(event)
            
            # Validate configuration
            validation_result = await self._validate_workflow_config(config)
            if not validation_result['valid']:
                return self._create_validation_error_result(
                    workflow_id,
                    config.workflow_type,
                    validation_result['errors']
                )
                
            # Set up shared infrastructure
            await self._setup_shared_infrastructure(config, context)
            
            # Get appropriate manager
            manager = self.manager_factory.create_manager(
                config.workflow_type,
                container_id
            )
            
            # Execute workflow in its container
            result = await manager.execute(config, context)
            
            # Log completion
            self.logger.info(
                "Workflow execution completed",
                workflow_id=workflow_id,
                success=result.success,
                duration_seconds=result.duration_seconds
            )
            
            # Emit workflow complete event
            complete_event = Event(
                event_type=EventType.INFO,
                payload={
                    'type': 'workflow.complete',
                    'workflow_id': workflow_id,
                    'success': result.success,
                    'duration_seconds': result.duration_seconds
                },
                source_id="coordinator",
                container_id=container_id
            )
            event_bus.publish(complete_event)
            
            return result
            
        except Exception as e:
            # Log error
            self.logger.error(
                "Workflow execution failed",
                workflow_id=workflow_id,
                error=str(e)
            )
            
            # Create error result
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)]
            )
            result.finalize()
            
            # Emit error event
            if 'container_id' in context.metadata:
                container_id = context.metadata['container_id']
                container = self.container_manager.active_containers.get(container_id)
                if container:
                    error_event = Event(
                        event_type=EventType.ERROR,
                        payload={
                            'type': 'workflow.error',
                            'workflow_id': workflow_id,
                            'error': str(e)
                        },
                        source_id="coordinator",
                        container_id=container_id
                    )
                    container.event_bus.publish(error_event)
            
            return result
            
        finally:
            # Clean up container
            if workflow_id in self._workflow_containers:
                container_id = self._workflow_containers[workflow_id]
                await self._cleanup_workflow_container(workflow_id, container_id)
            
            # Remove from active workflows
            async with self._workflow_lock:
                self._active_workflows.pop(workflow_id, None)
                self._workflow_containers.pop(workflow_id, None)
                
    async def get_workflow_status(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Get the status of a workflow."""
        async with self._workflow_lock:
            context = self._active_workflows.get(workflow_id)
            
        if not context:
            return {
                'workflow_id': workflow_id,
                'status': 'not_found',
                'active': False
            }
            
        return {
            'workflow_id': workflow_id,
            'status': 'active',
            'active': True,
            'workflow_type': context.workflow_type.value,
            'current_phase': context.current_phase.value if context.current_phase else None,
            'completed_phases': [p.value for p in context.completed_phases],
            'start_time': context.start_time.isoformat()
        }
        
    async def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows."""
        async with self._workflow_lock:
            workflows = []
            
            for workflow_id, context in self._active_workflows.items():
                workflows.append({
                    'workflow_id': workflow_id,
                    'workflow_type': context.workflow_type.value,
                    'current_phase': context.current_phase.value if context.current_phase else None,
                    'start_time': context.start_time.isoformat()
                })
                
        return workflows
        
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow."""
        async with self._workflow_lock:
            context = self._active_workflows.get(workflow_id)
            
        if not context:
            return False
            
        # Emit cancellation event
        await self.event_bus.emit({
            'type': 'workflow.cancel',
            'workflow_id': workflow_id
        })
        
        # The actual cancellation would be handled by the manager
        # through the event system
        
        return True
        
    async def _validate_workflow_config(
        self,
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Validate workflow configuration."""
        # Get manager for validation
        try:
            manager = self.manager_factory.create_manager(config.workflow_type)
            return await manager.validate_config(config)
        except ValueError as e:
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': []
            }
            
    def _create_validation_error_result(
        self,
        workflow_id: str,
        workflow_type: WorkflowType,
        errors: List[str]
    ) -> WorkflowResult:
        """Create a result for validation errors."""
        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            success=False,
            errors=errors
        )
        result.finalize()
        return result
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            self.logger.warning("PyYAML not installed, using empty config")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    async def _create_workflow_container(
        self,
        workflow_id: str,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> str:
        """Create a container for workflow execution."""
        # Determine container type based on workflow
        if config.workflow_type == WorkflowType.OPTIMIZATION:
            container_id = self.container_manager.create_and_start_container(
                "optimization",
                {
                    'workflow_id': workflow_id,
                    'optimization_config': config.optimization_config,
                    'shared_services': self.shared_services
                }
            )
        elif config.workflow_type == WorkflowType.BACKTEST:
            container_id = self.container_manager.create_and_start_container(
                "backtest",
                {
                    'workflow_id': workflow_id,
                    'backtest_config': config.backtest_config,
                    'shared_services': self.shared_services
                }
            )
        else:
            # Generic container for other types
            container_id = self.container_manager.create_and_start_container(
                config.workflow_type.value,
                {
                    'workflow_id': workflow_id,
                    'config': config.dict(),
                    'shared_services': self.shared_services
                }
            )
        
        async with self._workflow_lock:
            self._workflow_containers[workflow_id] = container_id
        
        return container_id
    
    async def _setup_shared_infrastructure(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> None:
        """Set up shared infrastructure for the workflow."""
        # Set up shared indicators if specified
        if 'shared_indicators' in config.parameters:
            await self._setup_shared_indicators(
                config.parameters['shared_indicators'],
                context
            )
        
        # Set up data feeds
        if config.data_config:
            await self._setup_data_feeds(config.data_config, context)
    
    async def _setup_shared_indicators(
        self,
        indicators_config: Dict[str, Any],
        context: ExecutionContext
    ) -> None:
        """Set up shared indicator containers."""
        # Create indicator hub container
        indicator_container_id = self.container_factory.create_indicator_container(
            indicators=indicators_config.get('indicators', [])
        )
        
        # Initialize the container
        self.container_manager.initialize_container(indicator_container_id)
        
        # Store reference in context
        context.shared_resources['indicator_hub'] = indicator_container_id
        
        self.logger.info(
            "Created shared indicator hub",
            workflow_id=context.workflow_id,
            container_id=indicator_container_id
        )
    
    async def _setup_data_feeds(
        self,
        data_config: Dict[str, Any],
        context: ExecutionContext
    ) -> None:
        """Set up data sources for the workflow."""
        # This would integrate with your data manager
        # For now, store config in context
        context.shared_resources['data_config'] = data_config
        
        self.logger.info(
            "Configured data feeds",
            workflow_id=context.workflow_id,
            sources=list(data_config.get('sources', {}).keys())
        )
    
    async def _cleanup_workflow_container(
        self,
        workflow_id: str,
        container_id: str
    ) -> None:
        """Clean up workflow container."""
        try:
            self.container_manager.stop_and_destroy_container(container_id)
            self.logger.info(
                "Cleaned up workflow container",
                workflow_id=workflow_id,
                container_id=container_id
            )
        except Exception as e:
            self.logger.error(
                f"Failed to cleanup container {container_id}: {e}"
            )
    
    def _register_event_handlers(self) -> None:
        """Register internal event handlers."""
        # Use coordinator container's event bus
        self.coordinator_container.event_bus.subscribe(
            EventType.SYSTEM,
            self._handle_workflow_event
        )
        
    def _handle_workflow_event(self, event: Event) -> None:
        """Handle workflow lifecycle events."""
        if isinstance(event.payload, dict):
            event_type = event.payload.get('type', '')
            workflow_id = event.payload.get('workflow_id')
            
            # Log significant events
            if event_type in ['workflow.start', 'workflow.complete', 'workflow.error']:
                self.logger.info(
                    f"Workflow event: {event_type}",
                    workflow_id=workflow_id,
                    details=event.payload
                )
            
    async def shutdown(self) -> None:
        """Shutdown the coordinator and clean up resources."""
        self.logger.info("Shutting down Coordinator")
        
        # Cancel any active workflows
        async with self._workflow_lock:
            active_ids = list(self._active_workflows.keys())
            
        for workflow_id in active_ids:
            await self.cancel_workflow(workflow_id)
            
        # Wait for workflows to complete
        wait_time = 0
        while self._active_workflows and wait_time < 30:
            await asyncio.sleep(1)
            wait_time += 1
            
        if self._active_workflows:
            self.logger.warning(
                "Some workflows still active after shutdown",
                count=len(self._active_workflows)
            )