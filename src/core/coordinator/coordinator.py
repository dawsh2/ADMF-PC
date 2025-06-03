"""
Canonical Coordinator Implementation

This is the single canonical coordinator for the ADMF-PC system that:
1. Uses clean imports (no deep dependencies)
2. Supports both traditional and composable container patterns
3. Uses lazy loading and dependency injection
4. Provides flexible workflow execution

Key principles:
- Minimal imports at module level
- Lazy loading of complex dependencies
- Plugin architecture for extensibility
- Clean separation of concerns
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Only import basic types and protocols
from .simple_types import WorkflowConfig, ExecutionContext, WorkflowType, WorkflowPhase
from .protocols import WorkflowManager

# Communication imports (lazy loaded when needed)
# from ..communication import EventCommunicationFactory, CommunicationLayer


logger = logging.getLogger(__name__)


@dataclass
class CoordinatorResult:
    """Result from coordinator workflow execution."""
    workflow_id: str
    workflow_type: WorkflowType
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def finalize(self) -> None:
        """Finalize result with completion time."""
        self.end_time = datetime.now()
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.metadata['duration_seconds'] = duration
    
    def add_error(self, error: str) -> None:
        """Add error and mark as failed."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str) -> None:
        """Add warning."""
        self.warnings.append(warning)


class ExecutionMode(str, Enum):
    """Supported execution modes."""
    AUTO = "auto"           # Coordinator chooses best mode
    TRADITIONAL = "traditional"  # Traditional workflow managers
    COMPOSABLE = "composable"    # Composable container patterns
    HYBRID = "hybrid"           # Mix of traditional and composable


class Coordinator:
    """
    Canonical Coordinator for the ADMF-PC system.
    
    This coordinator:
    - Orchestrates workflows without deep dependencies
    - Supports traditional and composable execution modes
    - Uses lazy loading to avoid import issues
    - Provides clean plugin architecture
    """
    
    def __init__(
        self,
        enable_composable_containers: bool = True,
        shared_services: Optional[Dict[str, Any]] = None,
        enable_communication: bool = True
    ):
        """Initialize coordinator."""
        self.enable_composable_containers = enable_composable_containers
        self.shared_services = shared_services or {}
        self.enable_communication = enable_communication
        
        # Lazy-loaded components
        self._composition_engine = None
        self._container_registry = None
        self._workflow_manager_factory = None
        self._container_manager = None
        self._communication_factory = None
        self._communication_layer = None
        self._log_manager = None
        
        # Active workflows tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Coordinator ID for communication
        self.coordinator_id = f"coordinator_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Coordinator initialized (composable: {enable_composable_containers}, communication: {enable_communication})")
    
    @property
    def communication_layer(self):
        """Get the communication layer."""
        return self._communication_layer
    
    async def execute_workflow(
        self,
        config: WorkflowConfig,
        workflow_id: Optional[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.AUTO
    ) -> CoordinatorResult:
        """
        Execute workflow using specified execution mode.
        
        Args:
            config: Workflow configuration
            workflow_id: Optional workflow ID
            execution_mode: How to execute the workflow
            
        Returns:
            CoordinatorResult with execution details
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        
        # Create result
        result = CoordinatorResult(
            workflow_id=workflow_id,
            workflow_type=config.workflow_type,
            success=True
        )
        
        # Create execution context
        context = ExecutionContext(
            workflow_id=workflow_id,
            workflow_type=config.workflow_type,
            metadata={'execution_mode': execution_mode.value}
        )
        
        try:
            # Determine execution mode
            if execution_mode == ExecutionMode.AUTO:
                execution_mode = self._determine_execution_mode(config)
                result.metadata['auto_selected_mode'] = execution_mode.value
            
            logger.info(f"Executing workflow {workflow_id} in {execution_mode.value} mode")
            
            # Execute based on mode
            if execution_mode == ExecutionMode.TRADITIONAL:
                await self._execute_traditional_workflow(config, context, result)
            elif execution_mode == ExecutionMode.COMPOSABLE:
                await self._execute_composable_workflow(config, context, result)
            elif execution_mode == ExecutionMode.HYBRID:
                await self._execute_hybrid_workflow(config, context, result)
            else:
                result.add_error(f"Unknown execution mode: {execution_mode}")
                
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            result.add_error(str(e))
        finally:
            # Clean up and finalize
            await self._cleanup_workflow(workflow_id)
            result.finalize()
        
        return result
    
    def _determine_execution_mode(self, config: WorkflowConfig) -> ExecutionMode:
        """Automatically determine best execution mode."""
        
        # Check if config explicitly requests container pattern
        container_pattern = config.parameters.get('container_pattern')
        if container_pattern and self.enable_composable_containers:
            return ExecutionMode.COMPOSABLE
        
        # Check for complex multi-pattern workflows
        if config.parameters.get('use_multiple_patterns', False):
            return ExecutionMode.COMPOSABLE
        
        # Check for parallel optimization with different patterns
        if (config.workflow_type == WorkflowType.OPTIMIZATION and 
            config.parameters.get('parallel_patterns', False)):
            return ExecutionMode.COMPOSABLE
        
        # Check if composable containers would provide benefit
        if self._would_benefit_from_composable(config):
            return ExecutionMode.COMPOSABLE
        
        # Default to traditional for backward compatibility
        return ExecutionMode.TRADITIONAL
    
    def _would_benefit_from_composable(self, config: WorkflowConfig) -> bool:
        """Determine if config would benefit from composable containers."""
        
        if not self.enable_composable_containers:
            return False
        
        # Check for multi-classifier scenarios
        optimization_config = config.optimization_config or {}
        classifiers = optimization_config.get('classifiers', [])
        strategies = optimization_config.get('strategies', [])
        risk_profiles = optimization_config.get('risk_profiles', [])
        
        # Benefit from composable if:
        # - Multiple classifiers/strategies/risk profiles
        # - Signal generation/replay workflows
        # - Complex indicator sharing scenarios
        
        has_multiple_components = (
            len(classifiers) > 1 or 
            len(strategies) > 1 or 
            len(risk_profiles) > 1
        )
        
        is_signal_workflow = (
            config.workflow_type == WorkflowType.ANALYSIS or
            config.analysis_config.get('mode') == 'signal_generation'
        )
        
        return has_multiple_components or is_signal_workflow
    
    async def _execute_traditional_workflow(
        self,
        config: WorkflowConfig,
        context: ExecutionContext,
        result: CoordinatorResult
    ) -> None:
        """Execute workflow using traditional workflow managers."""
        
        try:
            # Lazy import and create workflow manager factory
            workflow_manager_factory = await self._get_workflow_manager_factory()
            
            # Create workflow manager
            manager = workflow_manager_factory.create_manager(
                workflow_type=config.workflow_type,
                container_id=f"workflow_{context.workflow_id}",
                use_composable=False
            )
            
            # Store workflow info
            self.active_workflows[context.workflow_id] = {
                'config': config,
                'context': context,
                'manager': manager,
                'mode': 'traditional'
            }
            
            # Execute workflow
            workflow_result = await manager.execute(config, context)
            
            # Convert to coordinator result
            result.success = workflow_result.success
            result.data = workflow_result.final_results
            result.errors.extend(workflow_result.errors)
            result.metadata.update(workflow_result.metadata)
            result.metadata['execution_mode'] = 'traditional'
            
            # Generate reports if configured
            if result.success:
                await self._handle_reporting(config, context, result)
            
        except ImportError as e:
            result.add_error(f"Traditional workflow manager not available: {e}")
        except Exception as e:
            result.add_error(f"Traditional workflow execution failed: {e}")
    
    async def _execute_composable_workflow(
        self,
        config: WorkflowConfig,
        context: ExecutionContext,
        result: CoordinatorResult
    ) -> None:
        """Execute workflow using composable container patterns."""
        
        if not self.enable_composable_containers:
            result.add_error("Composable containers not enabled")
            return
        
        try:
            # Ensure pipeline-enabled containers are registered
            await self._ensure_containers_registered()
            
            # Lazy import composable workflow manager
            ComposableWorkflowManager = await self._get_composable_workflow_manager()
            
            # Create composable workflow manager
            # Check if this is the pipeline manager (has different constructor)
            if ComposableWorkflowManager.__name__ == 'ComposableWorkflowManagerPipeline':
                manager = ComposableWorkflowManager(coordinator=self)
            else:
                manager = ComposableWorkflowManager(
                    container_id=f"workflow_{context.workflow_id}",
                    shared_services=self.shared_services
                )
            
            # Store workflow info
            self.active_workflows[context.workflow_id] = {
                'config': config,
                'context': context,
                'manager': manager,
                'mode': 'composable'
            }
            
            # Execute workflow using appropriate method
            if ComposableWorkflowManager.__name__ == 'ComposableWorkflowManagerPipeline':
                # Set up communication for pipeline manager
                await self.setup_communication()
                workflow_result = await manager.execute_workflow(config)
            else:
                workflow_result = await manager.execute(config, context)
            
            # Convert to coordinator result - handle different return formats
            if isinstance(workflow_result, dict):
                # Pipeline manager returns dict
                result.success = workflow_result.get('success', True)
                result.data = workflow_result
                result.metadata['execution_mode'] = 'composable_pipeline'
            else:
                # Regular manager returns WorkflowResult object
                result.success = workflow_result.success
                result.data = workflow_result.final_results
                result.errors.extend(workflow_result.errors)
                result.metadata.update(workflow_result.metadata)
                result.metadata['execution_mode'] = 'composable'
            
            # Generate reports if configured
            if result.success:
                await self._handle_reporting(config, context, result)
            
        except ImportError as e:
            result.add_error(f"Composable workflow manager not available: {e}")
            # Fall back to traditional
            result.add_warning("Falling back to traditional execution")
            await self._execute_traditional_workflow(config, context, result)
        except Exception as e:
            result.add_error(f"Composable workflow execution failed: {e}")
    
    async def _execute_hybrid_workflow(
        self,
        config: WorkflowConfig,
        context: ExecutionContext,
        result: CoordinatorResult
    ) -> None:
        """Execute workflow using hybrid approach."""
        
        try:
            # Use traditional for orchestration, composable for specific phases
            phase_configs = config.parameters.get('phase_patterns', {})
            
            if not phase_configs:
                # No phase-specific config, default to composable
                await self._execute_composable_workflow(config, context, result)
                return
            
            phase_results = {}
            
            for phase_name, phase_config in phase_configs.items():
                if 'container_pattern' in phase_config:
                    # Use composable for this phase
                    phase_context = ExecutionContext(
                        workflow_id=f"{context.workflow_id}_{phase_name}",
                        workflow_type=config.workflow_type,
                        metadata={'phase': phase_name}
                    )
                    
                    phase_result = CoordinatorResult(
                        workflow_id=phase_context.workflow_id,
                        workflow_type=config.workflow_type,
                        success=True
                    )
                    
                    await self._execute_composable_workflow(config, phase_context, phase_result)
                    phase_results[phase_name] = phase_result
                else:
                    # Use traditional for this phase
                    phase_context = ExecutionContext(
                        workflow_id=f"{context.workflow_id}_{phase_name}",
                        workflow_type=config.workflow_type,
                        metadata={'phase': phase_name}
                    )
                    
                    phase_result = CoordinatorResult(
                        workflow_id=phase_context.workflow_id,
                        workflow_type=config.workflow_type,
                        success=True
                    )
                    
                    await self._execute_traditional_workflow(config, phase_context, phase_result)
                    phase_results[phase_name] = phase_result
            
            # Aggregate results
            result.success = all(r.success for r in phase_results.values())
            result.data = {
                'phase_results': {name: r.data for name, r in phase_results.items()}
            }
            result.metadata.update({
                'execution_mode': 'hybrid',
                'phases': list(phase_results.keys()),
                'successful_phases': sum(1 for r in phase_results.values() if r.success)
            })
            
            # Collect all errors and warnings
            for phase_result in phase_results.values():
                result.errors.extend(phase_result.errors)
                result.warnings.extend(phase_result.warnings)
            
        except Exception as e:
            result.add_error(f"Hybrid workflow execution failed: {e}")
    
    async def _cleanup_workflow(self, workflow_id: str) -> None:
        """Clean up resources for completed workflow."""
        
        if workflow_id not in self.active_workflows:
            return
        
        workflow_info = self.active_workflows[workflow_id]
        mode = workflow_info.get('mode')
        
        try:
            # Mode-specific cleanup
            if mode == 'traditional':
                # Traditional workflows clean up automatically
                pass
            elif mode == 'composable':
                # Composable workflows might need container disposal
                manager = workflow_info.get('manager')
                if manager and hasattr(manager, 'cleanup'):
                    await manager.cleanup()
            
            # Remove from active workflows
            del self.active_workflows[workflow_id]
            
            logger.debug(f"Cleaned up workflow {workflow_id} ({mode} mode)")
            
        except Exception as e:
            logger.error(f"Error cleaning up workflow {workflow_id}: {e}")
    
    # Lazy loading methods to avoid deep imports
    
    async def _get_communication_factory(self):
        """Lazy load communication factory."""
        if self._communication_factory is None and self.enable_communication:
            try:
                from ..communication import EventCommunicationFactory
                
                # Get or create log manager
                if self._log_manager is None:
                    from ..logging.log_manager import LogManager
                    self._log_manager = LogManager(coordinator_id=self.coordinator_id)
                
                self._communication_factory = EventCommunicationFactory(
                    coordinator_id=self.coordinator_id,
                    log_manager=self._log_manager
                )
            except ImportError as e:
                logger.warning(f"Communication module not available: {e}")
                self.enable_communication = False
        
        return self._communication_factory
    
    async def setup_communication(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Setup communication layer with given configuration.
        
        Args:
            config: Communication configuration with adapter definitions
            
        Returns:
            True if setup successful, False otherwise
        """
        if not self.enable_communication:
            logger.info("Communication disabled, skipping setup")
            return False
        
        try:
            # Get communication factory
            factory = await self._get_communication_factory()
            if not factory:
                return False
            
            # Default configuration if none provided
            if config is None:
                config = {
                    'adapters': [{
                        'name': 'default_pipeline',
                        'type': 'pipeline',
                        'containers': [],  # Will be populated with active containers
                        'log_level': 'INFO'
                    }]
                }
            
            # Collect active containers from workflows
            active_containers = {}
            for workflow_info in self.active_workflows.values():
                manager = workflow_info.get('manager')
                if manager and hasattr(manager, 'containers'):
                    active_containers.update(manager.containers)
            
            # Create communication layer
            self._communication_layer = factory.create_communication_layer(
                config=config,
                containers=active_containers
            )
            
            # Setup all adapters
            await self._communication_layer.setup_all_adapters()
            
            logger.info(f"Communication layer setup complete with {len(self._communication_layer.adapters)} adapters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup communication: {e}")
            return False
    
    async def _get_workflow_manager_factory(self):
        """Lazy load workflow manager factory."""
        if self._workflow_manager_factory is None:
            try:
                from .managers import WorkflowManagerFactory
                
                # Create container manager if needed
                if self._container_manager is None:
                    from ..containers import ContainerLifecycleManager
                    self._container_manager = ContainerLifecycleManager()
                
                self._workflow_manager_factory = WorkflowManagerFactory(
                    container_manager=self._container_manager,
                    shared_services=self.shared_services
                )
            except ImportError as e:
                raise ImportError(f"Cannot load workflow manager factory: {e}")
        
        return self._workflow_manager_factory
    
    async def _get_composable_workflow_manager(self):
        """Lazy load composable workflow manager."""
        try:
            # Use pipeline-enabled workflow manager for better event flow
            from .composable_workflow_manager_pipeline import ComposableWorkflowManagerPipeline
            return ComposableWorkflowManagerPipeline
        except ImportError as e:
            # Fall back to regular composable workflow manager
            logger.warning("Pipeline workflow manager not available, falling back to regular manager")
            from .composable_workflow_manager import ComposableWorkflowManager
            return ComposableWorkflowManager
    
    async def _get_composition_engine(self):
        """Lazy load composition engine."""
        if self._composition_engine is None:
            try:
                from ..containers.composition_engine import get_global_composition_engine
                self._composition_engine = get_global_composition_engine()
            except ImportError as e:
                raise ImportError(f"Cannot load composition engine: {e}")
        
        return self._composition_engine
    
    async def _get_container_registry(self):
        """Lazy load container registry."""
        if self._container_registry is None:
            try:
                from ..containers.composition_engine import get_global_registry
                self._container_registry = get_global_registry()
            except ImportError as e:
                raise ImportError(f"Cannot load container registry: {e}")
        
        return self._container_registry
    
    async def _ensure_containers_registered(self) -> None:
        """Ensure pipeline-enabled containers are registered with the composition engine."""
        try:
            # Import and register pipeline-enabled containers
            from ...execution import containers_pipeline
            containers_pipeline.register_execution_containers()
            logger.info("Pipeline-enabled containers registered with composition engine")
        except ImportError as e:
            logger.error(f"Failed to register pipeline containers: {e}")
            raise ImportError(f"Pipeline containers not available: {e}")
        except Exception as e:
            logger.error(f"Error registering pipeline containers: {e}")
            raise
    
    # Public API methods
    
    async def get_available_patterns(self) -> Dict[str, Any]:
        """Get all available container patterns."""
        if not self.enable_composable_containers:
            return {}
        
        try:
            registry = await self._get_container_registry()
            patterns = registry.list_available_patterns()
            
            pattern_info = {}
            for pattern_name in patterns:
                pattern = registry.get_pattern(pattern_name)
                if pattern:
                    pattern_info[pattern_name] = {
                        'description': pattern.description,
                        'required_capabilities': list(pattern.required_capabilities),
                        'default_config': pattern.default_config
                    }
            
            return pattern_info
        except ImportError:
            return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including communication metrics.
        
        Returns:
            Dictionary containing system status information
        """
        status = {
            'coordinator_id': self.coordinator_id,
            'composable_containers_enabled': self.enable_composable_containers,
            'communication_enabled': self.enable_communication,
            'active_workflows': len(self.active_workflows),
            'workflows': {}
        }
        
        # Add workflow details
        for workflow_id, workflow_info in self.active_workflows.items():
            status['workflows'][workflow_id] = {
                'type': workflow_info['config'].workflow_type.value,
                'mode': workflow_info.get('mode', 'unknown'),
                'start_time': workflow_info.get('context', {}).metadata.get('start_time')
            }
        
        # Add communication metrics if enabled
        if self.enable_communication and self._communication_layer:
            try:
                comm_metrics = self._communication_layer.get_system_metrics()
                status['communication'] = {
                    'total_adapters': comm_metrics.get('total_adapters', 0),
                    'active_adapters': comm_metrics.get('active_adapters', 0),
                    'connected_adapters': comm_metrics.get('connected_adapters', 0),
                    'total_events_sent': comm_metrics.get('total_events_sent', 0),
                    'total_events_received': comm_metrics.get('total_events_received', 0),
                    'overall_error_rate': comm_metrics.get('overall_error_rate', 0),
                    'events_per_second': comm_metrics.get('events_per_second', 0),
                    'overall_health': comm_metrics.get('overall_health', 'unknown'),
                    'adapter_status': self._communication_layer.get_adapter_status_summary()
                }
                
                # Add latency percentiles if available
                if 'latency_percentiles' in comm_metrics:
                    status['communication']['latency_percentiles'] = comm_metrics['latency_percentiles']
                    
            except Exception as e:
                logger.error(f"Failed to get communication metrics: {e}")
                status['communication'] = {'error': str(e)}
        else:
            status['communication'] = {'status': 'disabled'}
        
        # Add container pattern info if available
        if self.enable_composable_containers:
            try:
                patterns = await self.get_available_patterns()
                status['available_patterns'] = list(patterns.keys())
            except Exception as e:
                logger.error(f"Failed to get pattern info: {e}")
                status['available_patterns'] = []
        
        return status
    
    async def validate_workflow_config(
        self,
        config: WorkflowConfig,
        execution_mode: ExecutionMode = ExecutionMode.AUTO
    ) -> Dict[str, Any]:
        """Validate workflow configuration."""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggested_mode': None
        }
        
        # Basic validation
        if not config.data_config:
            validation_result['errors'].append("Missing data configuration")
        
        if config.workflow_type == WorkflowType.BACKTEST and not config.backtest_config:
            validation_result['warnings'].append("Missing backtest configuration, using defaults")
        
        if config.workflow_type == WorkflowType.OPTIMIZATION and not config.optimization_config:
            validation_result['errors'].append("Missing optimization configuration")
        
        # Execution mode validation
        if execution_mode == ExecutionMode.AUTO:
            suggested_mode = self._determine_execution_mode(config)
            validation_result['suggested_mode'] = suggested_mode.value
        
        # Container pattern validation
        container_pattern = config.parameters.get('container_pattern')
        if container_pattern and self.enable_composable_containers:
            try:
                registry = await self._get_container_registry()
                pattern = registry.get_pattern(container_pattern)
                if not pattern:
                    validation_result['errors'].append(f"Unknown container pattern: {container_pattern}")
            except ImportError:
                validation_result['warnings'].append("Cannot validate container pattern - composable containers not available")
        
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    async def shutdown(self) -> None:
        """Shutdown coordinator and clean up all resources."""
        logger.info("Shutting down Coordinator...")
        
        # Clean up all active workflows
        workflow_ids = list(self.active_workflows.keys())
        for workflow_id in workflow_ids:
            await self._cleanup_workflow(workflow_id)
        
        # Clean up communication layer if created
        if self._communication_layer:
            try:
                await self._communication_layer.cleanup()
                logger.info("Communication layer cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up communication layer: {e}")
        
        # Clean up communication factory if created
        if self._communication_factory:
            try:
                await self._communication_factory.cleanup_all_adapters()
                logger.info("Communication factory cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up communication factory: {e}")
        
        # Clean up container manager if created
        if self._container_manager and hasattr(self._container_manager, 'shutdown'):
            await self._container_manager.shutdown()
        
        logger.info("Coordinator shutdown complete")
    
    async def _handle_reporting(self, config: WorkflowConfig, context: ExecutionContext, result: CoordinatorResult) -> None:
        """Handle report generation after successful workflow completion"""
        try:
            # Check if reporting is enabled in configuration
            raw_config = config.parameters
            reporting_config = raw_config.get('reporting', {})
            
            if not reporting_config.get('enabled', False):
                logger.debug("Reporting disabled, skipping report generation")
                return
            
            # Import reporting integration
            from ...reporting.coordinator_integration import add_reporting_to_coordinator_workflow
            
            # Determine workspace path from reporting config or default
            output_dir = reporting_config.get('output_dir', 'reports')
            workspace_path = f'./{output_dir}'
            
            # Prepare workflow results for reporting
            workflow_results = {
                'container_status': result.data.get('container_status', {}),
                'container_structure': result.data.get('container_structure', {}),
                'metrics': result.data.get('metrics', {}),
                'final_state': result.data.get('final_state', 'unknown'),
                'execution_time': result.metadata.get('execution_time', 0),
                'workflow_id': context.workflow_id,
                'backtest_data': result.data.get('backtest_data', {})  # Include actual backtest data!
            }
            
            # Save backtest data to workspace for reporting
            self._save_backtest_data_to_workspace(workspace_path, workflow_results)
            
            # Generate reports
            logger.info("🔄 Generating reports...")
            updated_results = add_reporting_to_coordinator_workflow(
                config=raw_config,
                workspace_path=workspace_path,
                workflow_results=workflow_results
            )
            
            # Update result with reporting information
            if 'reporting' in updated_results:
                result.data['reporting'] = updated_results['reporting']
                result.metadata['reporting_enabled'] = True
                
                # Log report generation results
                reporting_info = updated_results['reporting']
                if 'error' in reporting_info:
                    logger.error(f"❌ Report generation failed: {reporting_info['error']}")
                else:
                    report_count = reporting_info.get('reports_generated', 0)
                    logger.info(f"✅ Report generation completed - {report_count} reports created")
                    
                    # Log report paths for user
                    for report_path in reporting_info.get('report_paths', []):
                        logger.info(f"📄 Report available: {report_path}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            # Don't fail the workflow for reporting errors
            result.metadata['reporting_error'] = str(e)
    
    def _save_backtest_data_to_workspace(self, workspace_path: str, workflow_results: Dict[str, Any]) -> None:
        """Save backtest data to workspace for reporting"""
        try:
            import json
            from pathlib import Path
            
            workspace = Path(workspace_path)
            workspace.mkdir(parents=True, exist_ok=True)
            
            # Save the raw workflow results
            backtest_file = workspace / 'backtest_data.json'
            with open(backtest_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_results = self._make_json_serializable(workflow_results)
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"💾 Saved backtest data to {backtest_file}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest data: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


# Convenience functions for backward compatibility

async def execute_backtest(
    config: WorkflowConfig,
    coordinator: Optional[Coordinator] = None,
    container_pattern: Optional[str] = None
) -> CoordinatorResult:
    """Execute backtest workflow."""
    
    if coordinator is None:
        coordinator = Coordinator(enable_composable_containers=True)
    
    # Add container pattern if specified
    if container_pattern:
        config.parameters['container_pattern'] = container_pattern
    
    try:
        result = await coordinator.execute_workflow(
            config=config,
            execution_mode=ExecutionMode.COMPOSABLE if container_pattern else ExecutionMode.AUTO
        )
        return result
    finally:
        await coordinator.shutdown()


async def execute_optimization(
    config: WorkflowConfig,
    coordinator: Optional[Coordinator] = None,
    use_composable: bool = True
) -> CoordinatorResult:
    """Execute optimization workflow."""
    
    if coordinator is None:
        coordinator = Coordinator(enable_composable_containers=use_composable)
    
    try:
        result = await coordinator.execute_workflow(
            config=config,
            execution_mode=ExecutionMode.COMPOSABLE if use_composable else ExecutionMode.TRADITIONAL
        )
        return result
    finally:
        await coordinator.shutdown()