"""
Canonical Coordinator Implementation

This is the single canonical coordinator for the ADMF-PC system that:
1. Uses clean imports (no deep dependencies)
2. Supports container-based execution patterns
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
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Only import basic types and protocols
from ..types.workflow import WorkflowConfig, ExecutionContext, WorkflowType, WorkflowPhase
from .topology import TopologyBuilder

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
    """Supported execution modes - simplified to remove confusion."""
    AUTO = "auto"           # Default mode - uses containers automatically


class Coordinator:
    """
    Canonical Coordinator for the ADMF-PC system.
    
    This coordinator:
    - Orchestrates workflows without deep dependencies
    - Supports container-based execution patterns
    - Uses lazy loading to avoid import issues
    - Provides clean plugin architecture
    """
    
    def __init__(
        self,
        shared_services: Optional[Dict[str, Any]] = None,
        enable_communication: bool = True,
        enable_yaml: bool = True,
        enable_phase_management: bool = True
    ):
        """Initialize coordinator with configurable features.
        
        The Coordinator is the main orchestration component that manages:
        - Workflow execution
        - Container creation and lifecycle
        - Communication between components
        - System-level services
        
        Note: Containers are now the default and only execution model.
        """
        self.shared_services = shared_services or {}
        self.enable_communication = enable_communication
        self.enable_yaml = enable_yaml
        self.enable_phase_management = enable_phase_management
        
        # Deprecated attributes - kept for backward compatibility
        self.enable_nesting = False  # Deprecated
        self.enable_pipeline_communication = False  # Deprecated
        
        # Lazy-loaded components
        self._container_factory = None
        self._container_registry = None
        self._container_manager = None
        self._communication_factory = None
        self._communication_layer = None  # Deprecated - use _communication_adapters
        self._communication_adapters = []
        
        # Active workflows tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Coordinator ID for communication
        self.coordinator_id = f"coordinator_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Coordinator initialized (communication: {enable_communication})")
    
    @property
    def communication_layer(self):
        """Get the communication layer."""
        return self._communication_layer
    
    async def execute_workflow(
        self,
        config: WorkflowConfig,
        workflow_id: Optional[str] = None
    ) -> CoordinatorResult:
        """
        Execute workflow using delegation to WorkflowManager and Sequencer.
        
        This method now follows the pattern-based architecture:
        - Coordinator orchestrates but doesn't execute
        - WorkflowManager handles pattern execution
        - Sequencer handles multi-phase workflows
        - Analytics storage integrated with correlation IDs
        
        Args:
            config: Workflow configuration
            workflow_id: Optional workflow ID
            
        Returns:
            CoordinatorResult with execution details
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        correlation_id = self._generate_correlation_id()
        
        # Create result
        result = CoordinatorResult(
            workflow_id=workflow_id,
            workflow_type=config.workflow_type,
            success=True
        )
        
        # Add correlation ID to metadata
        result.metadata['correlation_id'] = correlation_id
        
        # Create execution context with correlation ID
        context = ExecutionContext(
            workflow_id=workflow_id,
            workflow_type=config.workflow_type,
            metadata={
                'correlation_id': correlation_id
            }
        )
        
        try:
            # Initialize analytics storage if available
            analytics_connection = await self._get_analytics_connection()
            
            # Determine if multi-phase workflow
            if self._is_multi_phase(config):
                # Delegate to sequencer for multi-phase workflows
                logger.info(f"Delegating multi-phase workflow {workflow_id} to sequencer")
                workflow_result = await self._execute_via_sequencer(config, context, analytics_connection)
            else:
                # Delegate to topology builder for single-phase workflows
                logger.info(f"Delegating single-phase workflow {workflow_id} to topology builder")
                workflow_result = await self._execute_via_topology_builder(config, context, analytics_connection)
            
            # Convert WorkflowResult to CoordinatorResult
            result.success = workflow_result.success
            result.data = workflow_result.final_results
            result.errors.extend(workflow_result.errors)
            result.metadata.update(workflow_result.metadata)
            
            # Store results in analytics if available
            if analytics_connection and result.success:
                await self._store_workflow_results(analytics_connection, workflow_result, correlation_id)
                
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            result.add_error(str(e))
        finally:
            # Clean up and finalize
            await self._cleanup_workflow(workflow_id)
            result.finalize()
        
        return result
    
    def _determine_execution_mode(self, config: WorkflowConfig) -> ExecutionMode:
        """Return default execution mode - containers are now the only mode."""
        return ExecutionMode.AUTO
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for workflow tracking."""
        return f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _is_multi_phase(self, config: WorkflowConfig) -> bool:
        """Check if workflow has multiple phases."""
        # Check for explicit phases in config
        if hasattr(config, 'phases') and len(getattr(config, 'phases', [])) > 1:
            return True
        
        # Check parameters for phases
        if 'phases' in config.parameters and len(config.parameters['phases']) > 1:
            return True
        
        # Check for multi-phase workflow types
        if hasattr(WorkflowType, 'WALK_FORWARD') and config.workflow_type == WorkflowType.WALK_FORWARD:
            return True
        
        return False
    
    async def _execute_via_topology_builder(
        self, 
        config: WorkflowConfig, 
        context: ExecutionContext,
        analytics_connection: Optional[Any] = None
    ) -> 'WorkflowResult':
        """Execute single-phase workflow via TopologyBuilder."""
        # Get or create topology builder
        topology_builder = await self._get_topology_builder()
        
        # Store workflow info
        self.active_workflows[context.workflow_id] = {
            'config': config,
            'context': context,
            'manager': topology_builder,
            'mode': 'topology_builder'
        }
        
        # Execute workflow using topology builder
        result = await topology_builder.execute(config, context)
        
        # Generate reports if configured
        if result.success:
            await self._handle_reporting(config, context, result)
        
        return result
    
    async def _execute_via_sequencer(
        self, 
        config: WorkflowConfig, 
        context: ExecutionContext,
        analytics_connection: Optional[Any] = None
    ) -> 'WorkflowResult':
        """Execute multi-phase workflow via Sequencer."""
        # Get or create sequencer
        sequencer = await self._get_sequencer()
        
        # Store workflow info
        self.active_workflows[context.workflow_id] = {
            'config': config,
            'context': context,
            'sequencer': sequencer,
            'mode': 'sequencer'
        }
        
        # Execute phases using sequencer
        result = await sequencer.execute_phases(config, context)
        
        # Generate reports if configured
        if result.success:
            await self._handle_reporting(config, context, result)
        
        return result
    
    async def _get_analytics_connection(self) -> Optional[Any]:
        """Get analytics database connection if available."""
        try:
            from ...analytics.mining.storage.connections import DatabaseManager
            
            # Use SQLite for development
            db_config = {
                'type': 'sqlite',
                'path': './analytics/workflow_analytics.db'
            }
            
            manager = DatabaseManager(db_config)
            connection = manager.connect()
            
            # Ensure schema exists
            try:
                manager.create_schema()
            except Exception as e:
                logger.debug(f"Schema creation skipped (may already exist): {e}")
            
            return connection
            
        except ImportError:
            logger.debug("Analytics module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize analytics: {e}")
            return None
    
    async def _store_workflow_results(
        self, 
        connection: Any, 
        workflow_result: 'WorkflowResult',
        correlation_id: str
    ) -> None:
        """Store workflow results in analytics database."""
        try:
            # Prepare workflow record
            workflow_record = {
                'workflow_id': workflow_result.workflow_id,
                'correlation_id': correlation_id,
                'workflow_type': str(workflow_result.workflow_type.value),
                'success': workflow_result.success,
                'start_time': workflow_result.start_time,
                'end_time': workflow_result.end_time,
                'metadata': json.dumps(workflow_result.metadata),
                'errors': json.dumps(workflow_result.errors) if workflow_result.errors else None
            }
            
            # Store in database
            connection.insert_many('workflow_executions', [workflow_record])
            
            # Store individual results if available
            if hasattr(workflow_result, 'trial_results'):
                trial_records = []
                for trial_id, trial_data in workflow_result.trial_results.items():
                    trial_records.append({
                        'workflow_id': workflow_result.workflow_id,
                        'correlation_id': correlation_id,
                        'trial_id': trial_id,
                        'parameters': json.dumps(trial_data.get('parameters', {})),
                        'metrics': json.dumps(trial_data.get('metrics', {})),
                        'timestamp': datetime.now()
                    })
                
                if trial_records:
                    connection.insert_many('optimization_trials', trial_records)
            
            logger.info(f"Stored workflow results with correlation_id: {correlation_id}")
            
        except Exception as e:
            logger.error(f"Failed to store workflow results: {e}")
            # Don't fail the workflow for analytics errors
    
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
            elif mode == 'container':
                # Container workflows might need container disposal
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
                from ..communication import AdapterFactory
                
                # Use the new AdapterFactory instead of legacy EventCommunicationFactory
                self._communication_factory = AdapterFactory()
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
            
            # Create adapters using the new factory method
            adapters = factory.create_adapters_from_config(
                config.get('adapters', []),
                active_containers
            )
            
            # Store adapters for management
            self._communication_adapters = adapters
            
            # Start all adapters
            factory.start_all()
            
            logger.info(f"Communication setup complete with {len(adapters)} adapters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup communication: {e}")
            return False
    
    async def _get_workflow_manager_factory(self):
        """Deprecated - workflow manager factory no longer used."""
        raise DeprecationWarning("Workflow manager factory is deprecated. Use container-based patterns instead.")
    
    async def _get_workflow_manager_class(self):
        """Lazy load the canonical workflow manager class."""
        try:
            # Use the workflow topology manager at coordinator level
            # which properly leverages core.containers.factory
            from .topology import TopologyBuilder
            return TopologyBuilder
        except ImportError as e:
            raise ImportError(f"Cannot load workflow manager: {e}")
    
    async def _get_workflow_manager(self) -> 'WorkflowManager':
        """Get or create workflow manager instance."""
        # Create workflow manager (TopologyBuilder) with coordinator reference
        WorkflowManagerClass = await self._get_workflow_manager_class()
        
        # Create instance with minimal parameters
        # The deprecated execution_mode, enable_nesting, and enable_pipeline_communication
        # are handled internally by TopologyBuilder for backward compatibility
        workflow_manager = WorkflowManagerClass(
            container_id=f"workflow_{uuid.uuid4().hex[:8]}",
            shared_services=self.shared_services,
            coordinator=self
        )
        
        return workflow_manager
    
    async def _get_sequencer(self) -> 'WorkflowSequencer':
        """Get or create sequencer instance."""
        try:
            from .sequencer import WorkflowCoordinator
            
            # Create sequencer with checkpointing
            sequencer = WorkflowCoordinator(
                checkpoint_dir="./checkpoints"
            )
            
            # Set up result aggregator
            output_dir = f"./results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            from .sequencer import ResultAggregator
            sequencer.result_aggregator = ResultAggregator(output_dir)
            
            return sequencer
            
        except ImportError as e:
            raise ImportError(f"Cannot load sequencer: {e}")
    
    async def _get_container_factory(self):
        """Lazy load container factory."""
        if self._container_factory is None:
            try:
                from ..containers.factory import get_global_factory
                self._container_factory = get_global_factory()
            except ImportError as e:
                raise ImportError(f"Cannot load container factory: {e}")
        
        return self._container_factory
    
    async def _get_container_registry(self):
        """Lazy load container registry."""
        if self._container_registry is None:
            try:
                from ..containers.factory import get_global_registry
                self._container_registry = get_global_registry()
            except ImportError as e:
                raise ImportError(f"Cannot load container registry: {e}")
        
        return self._container_registry
    
    async def _ensure_containers_registered(self) -> None:
        """Ensure pipeline-enabled containers are registered with the container factory."""
        try:
            # Try to import execution containers module
            try:
                from ...execution import containers_pipeline
                containers_pipeline.register_execution_containers()
                logger.info("Pipeline-enabled containers registered with container factory")
            except ImportError:
                # Execution containers module not available - register basic factories
                from ..containers import register_container_type, Container, ContainerConfig
                from ..containers.protocols import ContainerRole
                
                # Create a generic factory function for containers
                def create_container_factory(role: ContainerRole):
                    def factory(config: Dict[str, Any], container_id: Optional[str] = None):
                        return Container(ContainerConfig(
                            role=role,
                            name=config.get('name', f'{role.value}_container'),
                            container_id=container_id,
                            config=config,
                            capabilities=set()
                        ))
                    return factory
                
                # Register container factories for all needed roles
                register_container_type(ContainerRole.DATA, create_container_factory(ContainerRole.DATA))
                register_container_type(ContainerRole.PORTFOLIO, create_container_factory(ContainerRole.PORTFOLIO))
                register_container_type(ContainerRole.EXECUTION, create_container_factory(ContainerRole.EXECUTION))
                register_container_type(ContainerRole.STRATEGY, create_container_factory(ContainerRole.STRATEGY))
                register_container_type(ContainerRole.RISK, create_container_factory(ContainerRole.RISK))
                register_container_type(ContainerRole.FEATURE, create_container_factory(ContainerRole.FEATURE))
                logger.info("Container factories registered with proper Container creation")
                
        except Exception as e:
            logger.error(f"Error registering containers: {e}")
            # Don't raise - allow system to continue without pipeline containers
    
    # Public API methods
    
    async def get_available_patterns(self) -> Dict[str, Any]:
        """Get all available container patterns."""
        
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
            'containers_enabled': True,
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
        if self.enable_communication and self._communication_adapters:
            try:
                # Aggregate metrics from all adapters
                total_events_sent = 0
                total_events_received = 0
                active_adapters = len(self._communication_adapters)
                
                for adapter in self._communication_adapters:
                    if hasattr(adapter, 'metrics'):
                        metrics = adapter.metrics
                        total_events_sent += getattr(metrics, 'events_sent', 0)
                        total_events_received += getattr(metrics, 'events_received', 0)
                
                status['communication'] = {
                    'total_adapters': active_adapters,
                    'active_adapters': active_adapters,
                    'total_events_sent': total_events_sent,
                    'total_events_received': total_events_received
                }
                
                    
            except Exception as e:
                logger.error(f"Failed to get communication metrics: {e}")
                status['communication'] = {'error': str(e)}
        else:
            status['communication'] = {'status': 'disabled'}
        
        # Add container pattern info if available
        try:
            patterns = await self.get_available_patterns()
            status['available_patterns'] = list(patterns.keys())
        except Exception as e:
            logger.error(f"Failed to get pattern info: {e}")
            status['available_patterns'] = []
        
        return status
    
    async def execute_yaml_workflow(self, yaml_path: str) -> CoordinatorResult:
        """Execute workflow from YAML configuration file."""
        
        if not self.enable_yaml:
            raise ValueError("YAML support not enabled in coordinator")
        
        try:
            # Lazy import YAML interpreter
            from .yaml_interpreter import YAMLInterpreter, YAMLWorkflowBuilder
            import yaml
            
            # Load and parse YAML
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Convert to WorkflowConfig
            workflow_type_str = yaml_config.get('type', 'backtest')
            workflow_type = WorkflowType(workflow_type_str)
            
            workflow_config = WorkflowConfig(
                workflow_type=workflow_type,
                parameters=yaml_config,
                data_config=yaml_config.get('data', {}),
                backtest_config=yaml_config.get('backtest', {}),
                optimization_config=yaml_config.get('optimization', {}),
                analysis_config=yaml_config.get('analysis', {})
            )
            
            # Execute using standard workflow execution
            result = await self.execute_workflow(workflow_config)
            
            result.metadata['yaml_source'] = yaml_path
            result.metadata['execution_type'] = 'yaml_driven'
            
            return result
            
        except Exception as e:
            logger.error(f"YAML workflow execution failed: {e}")
            return CoordinatorResult(
                workflow_id=str(uuid.uuid4()),
                workflow_type=WorkflowType.BACKTEST,
                success=False,
                errors=[f"YAML workflow execution failed: {e}"]
            )
    
    async def validate_workflow_config(
        self,
        config: WorkflowConfig
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
        
        # Suggest execution mode
        suggested_mode = self._determine_execution_mode(config)
        validation_result['suggested_mode'] = suggested_mode.value
        
        # Container pattern validation
        container_pattern = config.parameters.get('container_pattern')
        if container_pattern:
            try:
                registry = await self._get_container_registry()
                pattern = registry.get_pattern(container_pattern)
                if not pattern:
                    validation_result['errors'].append(f"Unknown container pattern: {container_pattern}")
            except ImportError:
                validation_result['warnings'].append("Cannot validate container pattern - container registry not available")
        
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    async def shutdown(self) -> None:
        """Shutdown coordinator and clean up all resources."""
        logger.info("Shutting down Coordinator...")
        
        # Clean up all active workflows
        workflow_ids = list(self.active_workflows.keys())
        for workflow_id in workflow_ids:
            await self._cleanup_workflow(workflow_id)
        
        # Clean up communication adapters
        if self._communication_factory and hasattr(self._communication_factory, 'stop_all'):
            try:
                self._communication_factory.stop_all()
                logger.info("Communication adapters stopped")
            except Exception as e:
                logger.error(f"Error stopping communication adapters: {e}")
        
        # Clean up container manager if created
        if self._container_manager and hasattr(self._container_manager, 'shutdown'):
            await self._container_manager.shutdown()
        
        logger.info("Coordinator shutdown complete")
    
    async def _handle_reporting(self, config: WorkflowConfig, context: ExecutionContext, result: 'WorkflowResult') -> None:
        """Handle report generation after successful workflow completion"""
        try:
            # Check if reporting is enabled in configuration
            raw_config = config.parameters
            reporting_config = raw_config.get('reporting', {})
            
            if not reporting_config.get('enabled', False):
                logger.debug("Reporting disabled, skipping report generation")
                return
            
            # Import reporting integration
            from ...analytics.coordinator_integration import add_reporting_to_coordinator_workflow
            
            # Determine workspace path from reporting config or default
            output_dir = reporting_config.get('output_dir', 'reports')
            workspace_path = f'./{output_dir}'
            
            # Prepare workflow results for reporting
            workflow_results = {
                'container_status': result.final_results.get('container_status', {}),
                'container_structure': result.final_results.get('container_structure', {}),
                'metrics': result.final_results.get('metrics', {}),
                'final_state': result.final_results.get('final_state', 'unknown'),
                'execution_time': result.metadata.get('execution_time', 0),
                'workflow_id': context.workflow_id,
                'backtest_data': result.final_results.get('backtest_data', {})  # Include actual backtest data!
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
                result.final_results['reporting'] = updated_results['reporting']
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
        coordinator = Coordinator()
    
    # Add container pattern if specified
    if container_pattern:
        config.parameters['container_pattern'] = container_pattern
    
    try:
        result = await coordinator.execute_workflow(
            config=config
        )
        return result
    finally:
        await coordinator.shutdown()


async def execute_optimization(
    config: WorkflowConfig,
    coordinator: Optional[Coordinator] = None
) -> CoordinatorResult:
    """Execute optimization workflow."""
    
    if coordinator is None:
        coordinator = Coordinator()
    
    try:
        result = await coordinator.execute_workflow(
            config=config
        )
        return result
    finally:
        await coordinator.shutdown()