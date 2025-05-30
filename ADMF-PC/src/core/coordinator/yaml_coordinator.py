"""YAML-aware Coordinator implementation.

This extends the base Coordinator to support YAML-driven workflow execution,
making ADMF-PC truly a zero-code trading system.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

from .coordinator import Coordinator
from .types import WorkflowConfig, WorkflowResult, WorkflowType
from .yaml_interpreter import YAMLInterpreter, YAMLWorkflowBuilder
from ..config import ConfigSchemaValidator, ValidationResult
from ..containers import ContainerCapability
from ..components import ComponentRegistry


logger = logging.getLogger(__name__)


class YAMLCoordinator(Coordinator):
    """YAML-aware Coordinator for zero-code workflow execution."""
    
    def __init__(
        self,
        shared_services: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """Initialize the YAML Coordinator."""
        super().__init__(shared_services, config_path)
        
        # YAML interpretation
        self.yaml_interpreter = YAMLInterpreter()
        self.workflow_builder = YAMLWorkflowBuilder(self.yaml_interpreter)
        
        # Component registry for dynamic component creation
        self.component_registry = ComponentRegistry()
        
        # Track YAML configs for reload capability
        self._yaml_configs: Dict[str, Path] = {}
        
        logger.info("YAML Coordinator initialized")
    
    async def execute_yaml_workflow(
        self,
        yaml_path: Union[str, Path]
    ) -> WorkflowResult:
        """Execute a workflow from a YAML configuration file.
        
        This is the main entry point for YAML-driven execution.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            WorkflowResult with execution details
        """
        yaml_path = Path(yaml_path)
        
        # Log execution start
        logger.info(f"Executing YAML workflow: {yaml_path}")
        
        try:
            # Load and interpret YAML
            workflow_config, validation_result = self.yaml_interpreter.load_and_interpret(yaml_path)
            
            # Check validation
            if not validation_result.is_valid:
                return self._create_validation_error_result(
                    workflow_id="validation_failed",
                    workflow_type=WorkflowType.VALIDATION,
                    errors=validation_result.errors,
                    warnings=validation_result.warnings
                )
            
            # Log any warnings
            for warning in validation_result.warnings:
                logger.warning(f"Configuration warning: {warning}")
            
            # Store YAML path for potential reload
            workflow_name = workflow_config.parameters.get("name", "unnamed")
            self._yaml_configs[workflow_name] = yaml_path
            
            # Build container hierarchy
            container_spec = self.workflow_builder.build_container_hierarchy(workflow_config)
            
            # Enhance workflow config with container spec
            workflow_config.infrastructure_config["container_hierarchy"] = container_spec
            
            # Execute using base coordinator
            result = await self.execute_workflow(workflow_config)
            
            # Enhance result with YAML-specific info
            result.metadata["yaml_path"] = str(yaml_path)
            result.metadata["validation_warnings"] = validation_result.warnings
            
            return result
            
        except Exception as e:
            logger.error(f"YAML workflow execution failed: {str(e)}", exc_info=True)
            return self._create_error_result(
                workflow_id="execution_failed",
                workflow_type=WorkflowType.VALIDATION,
                error=str(e)
            )
    
    async def execute_yaml_string(
        self,
        yaml_content: str,
        workflow_name: Optional[str] = None
    ) -> WorkflowResult:
        """Execute a workflow from a YAML string.
        
        Args:
            yaml_content: YAML configuration as string
            workflow_name: Optional name for the workflow
            
        Returns:
            WorkflowResult with execution details
        """
        try:
            # Parse YAML string
            import yaml
            yaml_dict = yaml.safe_load(yaml_content)
            
            # Add name if provided
            if workflow_name:
                yaml_dict["name"] = workflow_name
            
            # Interpret configuration
            workflow_config = self.yaml_interpreter.interpret_dict(yaml_dict)
            
            # Build container hierarchy
            container_spec = self.workflow_builder.build_container_hierarchy(workflow_config)
            workflow_config.infrastructure_config["container_hierarchy"] = container_spec
            
            # Execute
            result = await self.execute_workflow(workflow_config)
            
            result.metadata["source"] = "yaml_string"
            return result
            
        except Exception as e:
            logger.error(f"YAML string execution failed: {str(e)}", exc_info=True)
            return self._create_error_result(
                workflow_id="execution_failed",
                workflow_type=WorkflowType.VALIDATION,
                error=str(e)
            )
    
    async def validate_yaml(
        self,
        yaml_path: Union[str, Path]
    ) -> ValidationResult:
        """Validate a YAML configuration file.
        
        Args:
            yaml_path: Path to YAML configuration
            
        Returns:
            ValidationResult with details
        """
        validator = ConfigSchemaValidator()
        return validator.validate_file(yaml_path)
    
    async def reload_workflow(
        self,
        workflow_name: str
    ) -> WorkflowResult:
        """Reload and execute a previously loaded workflow.
        
        Args:
            workflow_name: Name of workflow to reload
            
        Returns:
            WorkflowResult from re-execution
        """
        if workflow_name not in self._yaml_configs:
            return self._create_error_result(
                workflow_id="reload_failed",
                workflow_type=WorkflowType.VALIDATION,
                error=f"No YAML configuration found for workflow: {workflow_name}"
            )
        
        yaml_path = self._yaml_configs[workflow_name]
        logger.info(f"Reloading workflow '{workflow_name}' from {yaml_path}")
        
        return await self.execute_yaml_workflow(yaml_path)
    
    def list_available_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows that have been loaded.
        
        Returns:
            List of workflow information
        """
        workflows = []
        for name, path in self._yaml_configs.items():
            workflows.append({
                "name": name,
                "path": str(path),
                "exists": path.exists()
            })
        return workflows
    
    async def _create_workflow_container(
        self,
        workflow_id: str,
        config: WorkflowConfig,
        context
    ) -> str:
        """Create workflow container with YAML-specified hierarchy."""
        # Check if we have a container hierarchy specification
        if "container_hierarchy" in config.infrastructure_config:
            container_spec = config.infrastructure_config["container_hierarchy"]
            return await self._create_hierarchical_containers(
                workflow_id, container_spec, config
            )
        else:
            # Fall back to base implementation
            return await super()._create_workflow_container(workflow_id, config, context)
    
    async def _create_hierarchical_containers(
        self,
        workflow_id: str,
        container_spec: Dict[str, Any],
        config: WorkflowConfig
    ) -> str:
        """Create hierarchical container structure from specification."""
        # Create root container
        root_spec = container_spec["root"]
        root_container_id = self.container_manager.create_and_start_container(
            root_spec["type"],
            {
                'workflow_id': workflow_id,
                'container_spec': root_spec,
                'config': config.dict()
            }
        )
        
        # Get root container
        root_container = self.container_manager.active_containers[root_container_id]
        
        # Apply capabilities to root
        for capability_name in root_spec.get("capabilities", []):
            self._apply_capability(root_container, capability_name, root_spec.get("config", {}))
        
        # Create child containers
        for child_spec in root_spec.get("children", []):
            await self._create_child_container(
                root_container,
                child_spec,
                workflow_id
            )
        
        return root_container_id
    
    async def _create_child_container(
        self,
        parent_container,
        child_spec: Dict[str, Any],
        workflow_id: str
    ) -> None:
        """Create a child container within parent."""
        # Create child container
        child_container = parent_container.create_child_container(
            child_spec["id"],
            child_spec["type"]
        )
        
        # Apply capabilities
        for capability_name in child_spec.get("capabilities", []):
            self._apply_capability(
                child_container,
                capability_name,
                child_spec.get("config", {})
            )
        
        # Configure components based on type
        if child_spec["type"] == "strategy":
            await self._configure_strategy_container(
                child_container,
                child_spec["config"]
            )
        elif child_spec["type"] == "risk_portfolio":
            await self._configure_risk_container(
                child_container,
                child_spec["config"]
            )
        elif child_spec["type"] == "data":
            await self._configure_data_container(
                child_container,
                child_spec["config"]
            )
        
        # Recursively create children
        for grandchild_spec in child_spec.get("children", []):
            await self._create_child_container(
                child_container,
                grandchild_spec,
                workflow_id
            )
    
    def _apply_capability(
        self,
        container,
        capability_name: str,
        config: Dict[str, Any]
    ) -> None:
        """Apply a capability to a container."""
        # Map capability names to actual capability classes
        capability_map = {
            "monitoring": "MonitoringCapability",
            "error_handling": "ErrorHandlingCapability",
            "event_bus": "EventBusCapability",
            "risk_portfolio": "RiskPortfolioCapability",
            "backtesting": "BacktestingCapability",
            "optimization": "OptimizationCapability",
            "live_trading": "LiveTradingCapability",
            "data_loading": "DataLoadingCapability",
            "technical_indicators": "TechnicalIndicatorCapability",
            "performance_analytics": "PerformanceAnalyticsCapability"
        }
        
        if capability_name in capability_map:
            capability_class_name = capability_map[capability_name]
            # In real implementation, would dynamically load and apply capability
            logger.debug(f"Applied {capability_name} to {container.container_id}")
    
    async def _configure_strategy_container(
        self,
        container,
        strategy_config: Dict[str, Any]
    ) -> None:
        """Configure a strategy container."""
        # Create strategy component based on type
        strategy_type = strategy_config["type"]
        
        # In real implementation, would create actual strategy component
        logger.debug(f"Configured strategy container: {strategy_type}")
    
    async def _configure_risk_container(
        self,
        container,
        risk_config: Dict[str, Any]
    ) -> None:
        """Configure a risk & portfolio container."""
        # Configure position sizers
        for sizer_config in risk_config.get("position_sizers", []):
            logger.debug(f"Added position sizer: {sizer_config['type']}")
        
        # Configure risk limits
        for limit_config in risk_config.get("risk_limits", []):
            logger.debug(f"Added risk limit: {limit_config['type']}")
    
    async def _configure_data_container(
        self,
        container,
        data_config: Dict[str, Any]
    ) -> None:
        """Configure a data container."""
        # Configure data source
        logger.debug(f"Configured data source: {data_config.get('source', 'default')}")
    
    def _create_validation_error_result(
        self,
        workflow_id: str,
        workflow_type: WorkflowType,
        errors: List[str],
        warnings: Optional[List[str]] = None
    ) -> WorkflowResult:
        """Create a workflow result for validation errors."""
        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            success=False,
            errors=errors,
            warnings=warnings or []
        )
        result.finalize()
        return result
    
    def _create_error_result(
        self,
        workflow_id: str,
        workflow_type: WorkflowType,
        error: str
    ) -> WorkflowResult:
        """Create a workflow result for execution errors."""
        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            success=False,
            errors=[error]
        )
        result.finalize()
        return result


# Convenience functions for YAML-driven execution
async def run_backtest(yaml_path: Union[str, Path]) -> WorkflowResult:
    """Run a backtest from YAML configuration.
    
    Args:
        yaml_path: Path to backtest YAML configuration
        
    Returns:
        WorkflowResult with backtest results
    """
    coordinator = YAMLCoordinator()
    return await coordinator.execute_yaml_workflow(yaml_path)


async def run_optimization(yaml_path: Union[str, Path]) -> WorkflowResult:
    """Run an optimization from YAML configuration.
    
    Args:
        yaml_path: Path to optimization YAML configuration
        
    Returns:
        WorkflowResult with optimization results
    """
    coordinator = YAMLCoordinator()
    return await coordinator.execute_yaml_workflow(yaml_path)


async def run_live_trading(yaml_path: Union[str, Path]) -> WorkflowResult:
    """Run live trading from YAML configuration.
    
    Args:
        yaml_path: Path to live trading YAML configuration
        
    Returns:
        WorkflowResult with live trading results
    """
    coordinator = YAMLCoordinator()
    return await coordinator.execute_yaml_workflow(yaml_path)