"""
Workflow Topology Manager - Creates and Manages Workflow Topologies

This module creates workflow topologies by arranging containers, patterns, and
execution strategies into connected graphs. It handles:
- Pattern detection and topology construction
- Communication configuration between components  
- Execution strategy delegation
- Multi-parameter workflow coordination
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from ..containers.factory import get_global_factory, get_global_registry
from ..communication import AdapterFactory
from ..types.workflow import WorkflowConfig, WorkflowType, ExecutionContext, WorkflowResult
from ..containers.protocols import ComposableContainer

from .workflows.config import PatternDetector, ParameterAnalyzer, ConfigBuilder
from .workflows.execution import get_executor, ExecutionStrategy

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Modular workflow manager using execution strategies and pattern detection.
    
    This manager orchestrates workflows by:
    1. Detecting appropriate patterns based on configuration
    2. Delegating execution to specialized execution strategies
    3. Coordinating multi-parameter workflows
    4. Managing container lifecycle and communication
    """
    
    def __init__(
        self,
        container_id: Optional[str] = None,
        shared_services: Optional[Dict[str, Any]] = None,
        coordinator: Optional[Any] = None,
        execution_mode: str = 'standard',
        enable_nesting: bool = False,
        enable_pipeline_communication: bool = False
    ):
        """Initialize modular workflow manager."""
        self.container_id = container_id
        self.shared_services = shared_services or {}
        self.coordinator = coordinator
        self.execution_mode = execution_mode
        self.enable_nesting = enable_nesting
        self.enable_pipeline_communication = enable_pipeline_communication
        
        # Core factories
        self.factory = get_global_factory()
        self.registry = get_global_registry()
        self.adapter_factory = AdapterFactory()
        
        # Modular components
        self.pattern_detector = PatternDetector()
        self.parameter_analyzer = ParameterAnalyzer()
        self.config_builder = ConfigBuilder()
        
        # Execution strategy cache
        self._executors: Dict[str, ExecutionStrategy] = {}
        
        # Active resources
        self.active_containers: Dict[str, ComposableContainer] = {}
        self.active_adapters = []
        
        logger.info(f"WorkflowManager initialized (mode: {execution_mode}, nesting: {enable_nesting}, pipeline: {enable_pipeline_communication})")
    
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute workflow using modular pattern detection and execution strategies."""
        
        logger.info(f"Executing {config.workflow_type.value} workflow")
        
        try:
            # 1. Detect patterns using pattern detector
            patterns = self.pattern_detector.determine_patterns(config)
            logger.info(f"Detected {len(patterns)} patterns: {[p['name'] for p in patterns]}")
            
            # 2. Choose execution strategy
            execution_mode = self._determine_execution_mode(config, patterns)
            logger.info(f"Using execution mode: {execution_mode}")
            
            # 3. Execute using appropriate strategy
            if len(patterns) == 1:
                executor = self._get_executor(execution_mode)
                result = await executor.execute_single_pattern(patterns[0], config, context)
            else:
                executor = self._get_executor('multi_pattern')
                result = await executor.execute_multi_pattern(patterns, config, context)
            
            logger.info(f"Workflow execution completed: success={result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)],
                metadata={'execution_mode': 'failed'}
            )
    
    async def execute_pattern(
        self,
        pattern_name: str,
        config: Dict[str, Any],
        correlation_id: str
    ) -> WorkflowResult:
        """
        Execute a specific workflow pattern.
        
        This method is called by the Coordinator for single-phase workflows.
        It delegates to the appropriate execution strategy based on the pattern.
        
        Args:
            pattern_name: Name of the pattern to execute
            config: Configuration for the pattern
            correlation_id: Correlation ID for tracking
            
        Returns:
            WorkflowResult with execution details
        """
        logger.info(f"Executing pattern '{pattern_name}' with correlation_id: {correlation_id}")
        
        try:
            # Get pattern from registry
            pattern = self.registry.get_pattern(pattern_name)
            if not pattern:
                raise ValueError(f"Unknown pattern: {pattern_name}")
            
            # Create pattern info structure
            pattern_info = {
                'name': pattern_name,
                'config': config,
                'pattern': pattern
            }
            
            # Create execution context with correlation ID
            context = ExecutionContext(
                workflow_id=correlation_id.split('_')[1],  # Extract workflow ID from correlation ID
                workflow_type=WorkflowType.BACKTEST,  # Default, will be overridden if needed
                metadata={'correlation_id': correlation_id}
            )
            
            # Get executor and execute pattern
            executor = self._get_executor(self.execution_mode)
            result = await executor.execute_single_pattern(pattern_info, config, context)
            
            # Add correlation ID to result metadata
            result.metadata['correlation_id'] = correlation_id
            
            logger.info(f"Pattern execution completed: {pattern_name}, success={result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Pattern execution failed: {e}")
            return WorkflowResult(
                workflow_id=correlation_id.split('_')[1],
                workflow_type=WorkflowType.BACKTEST,
                success=False,
                errors=[str(e)],
                metadata={
                    'pattern_name': pattern_name,
                    'correlation_id': correlation_id,
                    'error': str(e)
                }
            )
    
    def _determine_execution_mode(self, config: WorkflowConfig, patterns: List[Dict[str, Any]]) -> str:
        """Determine execution mode based on configuration and detected patterns."""
        
        # Check for explicit execution mode configuration
        if hasattr(config, 'execution_mode') and config.execution_mode:
            return config.execution_mode
        
        # Use instance configuration
        if self.enable_nesting:
            return 'nested'
        elif self.enable_pipeline_communication:
            return 'pipeline'
        
        # Auto-detect based on patterns
        pattern_names = [p['name'] for p in patterns]
        
        if any('multi_parameter' in name or 'optimization_grid' in name for name in pattern_names):
            return 'multi_parameter'
        elif len(patterns) > 1:
            return 'multi_pattern'
        else:
            return self.execution_mode
    
    def _get_executor(self, mode: str) -> ExecutionStrategy:
        """Get executor for specified mode (with caching)."""
        if mode not in self._executors:
            self._executors[mode] = get_executor(mode, self)
        return self._executors[mode]
    
    async def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Validate workflow configuration using modular components."""
        
        errors = []
        warnings = []
        
        # Basic configuration validation
        if not config.data_config:
            errors.append("Missing data configuration")
        
        # Workflow-specific validation
        if config.workflow_type == WorkflowType.BACKTEST:
            if not config.backtest_config:
                warnings.append("Missing backtest configuration, using defaults")
        elif config.workflow_type == WorkflowType.OPTIMIZATION:
            if not config.optimization_config:
                errors.append("Missing optimization configuration")
        
        # Pattern validation
        try:
            patterns = self.pattern_detector.determine_patterns(config)
            pattern_info = []
            
            for pattern_data in patterns:
                pattern_name = pattern_data['name']
                pattern = self.registry.get_pattern(pattern_name)
                
                if not pattern:
                    errors.append(f"Unknown container pattern: {pattern_name}")
                elif not self.factory.validate_pattern(pattern):
                    errors.append(f"Invalid container pattern: {pattern_name}")
                else:
                    pattern_info.append({
                        'name': pattern_name,
                        'description': pattern.description,
                        'required_capabilities': list(pattern.required_capabilities)
                    })
            
        except Exception as e:
            errors.append(f"Pattern detection failed: {e}")
            pattern_info = []
        
        # Multi-parameter analysis
        complexity_analysis = None
        if self.parameter_analyzer.requires_multi_parameter(config):
            try:
                complexity_analysis = self.parameter_analyzer.estimate_execution_complexity(config)
            except Exception as e:
                warnings.append(f"Could not analyze multi-parameter complexity: {e}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'patterns': pattern_info,
            'complexity_analysis': complexity_analysis
        }
    
    async def get_execution_preview(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Get a preview of how the workflow would be executed."""
        
        try:
            # Detect patterns
            patterns = self.pattern_detector.determine_patterns(config)
            execution_mode = self._determine_execution_mode(config, patterns)
            
            # Analyze complexity
            complexity_analysis = None
            if self.parameter_analyzer.requires_multi_parameter(config):
                complexity_analysis = self.parameter_analyzer.estimate_execution_complexity(config)
            
            # Get available patterns info
            available_patterns = self.registry.list_available_patterns()
            
            return {
                'detected_patterns': [p['name'] for p in patterns],
                'execution_mode': execution_mode,
                'pattern_details': patterns,
                'complexity_analysis': complexity_analysis,
                'available_patterns': available_patterns,
                'estimated_resources': {
                    'containers': complexity_analysis['estimated_containers'] if complexity_analysis else 5,
                    'duration_minutes': complexity_analysis['estimated_duration_minutes'] if complexity_analysis else 1
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'detected_patterns': [],
                'execution_mode': 'unknown'
            }
    
    def get_supported_patterns(self) -> Dict[str, Any]:
        """Get information about supported workflow patterns."""
        
        available_patterns = self.registry.list_available_patterns()
        pattern_info = {}
        
        for pattern_name in available_patterns:
            pattern = self.registry.get_pattern(pattern_name)
            if pattern:
                pattern_info[pattern_name] = {
                    'description': pattern.description,
                    'required_capabilities': list(pattern.required_capabilities),
                    'default_config': pattern.default_config
                }
        
        return {
            'available_patterns': pattern_info,
            'execution_modes': ['standard', 'nested', 'pipeline', 'multi_pattern'],
            'communication_patterns': ['pipeline', 'broadcast', 'hierarchical', 'selective']
        }
    
    async def cleanup(self) -> None:
        """Clean up all active resources."""
        
        logger.info("Cleaning up workflow manager resources...")
        
        # Clean up active containers
        for container_id, container in list(self.active_containers.items()):
            try:
                await container.dispose()
                del self.active_containers[container_id]
            except Exception as e:
                logger.error(f"Error disposing container {container_id}: {e}")
        
        # Clean up communication adapters
        try:
            self.adapter_factory.stop_all()
            self.active_adapters.clear()
        except Exception as e:
            logger.error(f"Error stopping adapters: {e}")
        
        # Clear executor cache
        self._executors.clear()
        
        logger.info("Workflow manager cleanup complete")


# No backward compatibility aliases - use WorkflowManager directly
# Following STYLE.md: ONE canonical implementation per concept