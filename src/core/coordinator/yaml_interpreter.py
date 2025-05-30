"""YAML configuration interpreter for the Coordinator.

This module bridges YAML configurations to the Coordinator's internal format,
enabling the YAML-driven workflow execution.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging

from .simple_types import WorkflowConfig, WorkflowType
from ..config import ConfigSchemaValidator, ValidationResult


logger = logging.getLogger(__name__)


class YAMLInterpreter:
    """Interprets YAML configurations for the Coordinator."""
    
    def __init__(self):
        """Initialize the YAML interpreter."""
        self.validator = ConfigSchemaValidator()
        self._type_mapping = {
            "backtest": WorkflowType.BACKTEST,
            "optimization": WorkflowType.OPTIMIZATION,
            "live_trading": WorkflowType.LIVE_TRADING,
            "live": WorkflowType.LIVE_TRADING,
            "analysis": WorkflowType.ANALYSIS,
            "validation": WorkflowType.VALIDATION
        }
    
    def load_and_interpret(
        self,
        config_path: Union[str, Path]
    ) -> Tuple[WorkflowConfig, ValidationResult]:
        """Load and interpret a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration
            
        Returns:
            Tuple of (WorkflowConfig, ValidationResult)
        """
        # Validate configuration
        validation_result = self.validator.validate_file(config_path)
        
        if not validation_result.is_valid:
            # Return empty config with validation errors
            return WorkflowConfig(workflow_type=WorkflowType.VALIDATION), validation_result
        
        # Use normalized config
        yaml_config = validation_result.normalized_config
        
        # Interpret into WorkflowConfig
        workflow_config = self._interpret_config(yaml_config)
        
        return workflow_config, validation_result
    
    def interpret_dict(
        self,
        yaml_config: Dict[str, Any]
    ) -> WorkflowConfig:
        """Interpret a YAML configuration dictionary.
        
        Args:
            yaml_config: YAML configuration as dict
            
        Returns:
            WorkflowConfig instance
        """
        # Detect and validate
        schema_type = self.validator._detect_schema(yaml_config)
        validation_result = self.validator.validate(yaml_config, schema_type)
        
        if not validation_result.is_valid:
            raise ValueError(f"Invalid configuration: {validation_result.errors}")
        
        return self._interpret_config(validation_result.normalized_config)
    
    def _interpret_config(self, yaml_config: Dict[str, Any]) -> WorkflowConfig:
        """Interpret normalized YAML config into WorkflowConfig.
        
        Args:
            yaml_config: Normalized YAML configuration
            
        Returns:
            WorkflowConfig instance
        """
        # Determine workflow type
        config_type = yaml_config.get("type", "backtest")
        workflow_type = self._type_mapping.get(config_type, WorkflowType.BACKTEST)
        
        # Base configuration
        workflow_config = WorkflowConfig(
            workflow_type=workflow_type,
            parameters={
                "name": yaml_config.get("name", f"{workflow_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "description": yaml_config.get("description", "")
            }
        )
        
        # Interpret based on workflow type
        if workflow_type == WorkflowType.BACKTEST:
            self._interpret_backtest_config(yaml_config, workflow_config)
        elif workflow_type == WorkflowType.OPTIMIZATION:
            self._interpret_optimization_config(yaml_config, workflow_config)
        elif workflow_type == WorkflowType.LIVE_TRADING:
            self._interpret_live_trading_config(yaml_config, workflow_config)
        
        # Common configuration elements
        self._interpret_common_config(yaml_config, workflow_config)
        
        return workflow_config
    
    def _interpret_backtest_config(
        self,
        yaml_config: Dict[str, Any],
        workflow_config: WorkflowConfig
    ) -> None:
        """Interpret backtest-specific configuration."""
        # Data configuration
        if "data" in yaml_config:
            workflow_config.data_config = {
                "symbols": yaml_config["data"]["symbols"],
                "start_date": yaml_config["data"]["start_date"],
                "end_date": yaml_config["data"]["end_date"],
                "frequency": yaml_config["data"]["frequency"],
                "source": yaml_config["data"].get("source", "yahoo"),
                "format": yaml_config["data"].get("format", "csv"),
                "timezone": yaml_config["data"].get("timezone", "UTC")
            }
        
        # Backtest-specific config
        workflow_config.backtest_config = {
            "execution_mode": yaml_config.get("execution_mode", "vectorized"),
            "portfolio": yaml_config.get("portfolio", {}),
            "strategies": self._interpret_strategies(yaml_config.get("strategies", [])),
            "risk": self._interpret_risk_config(yaml_config.get("risk", {})),
            "analysis": yaml_config.get("analysis", {})
        }
        
        # Infrastructure needs
        workflow_config.infrastructure_config = {
            "components_needed": ["data_loader", "portfolio_manager", "risk_manager"],
            "capabilities_needed": ["backtesting", "performance_analytics"]
        }
    
    def _interpret_optimization_config(
        self,
        yaml_config: Dict[str, Any],
        workflow_config: WorkflowConfig
    ) -> None:
        """Interpret optimization-specific configuration."""
        # Base configuration from embedded backtest
        if "base_config" in yaml_config:
            base = yaml_config["base_config"]
            workflow_config.data_config = {
                "symbols": base["data"]["symbols"],
                "start_date": base["data"]["start_date"],
                "end_date": base["data"]["end_date"],
                "frequency": base["data"]["frequency"],
                "source": base["data"].get("source", "yahoo")
            }
            
            # Base backtest config
            workflow_config.backtest_config = {
                "portfolio": base.get("portfolio", {}),
                "strategies": self._interpret_strategies(base.get("strategies", []))
            }
        
        # Optimization-specific config
        opt_config = yaml_config.get("optimization", {})
        workflow_config.optimization_config = {
            "method": opt_config.get("method", "grid_search"),
            "parameter_space": opt_config.get("parameter_space", {}),
            "constraints": opt_config.get("constraints", []),
            "objectives": opt_config.get("objectives", []),
            "n_trials": opt_config.get("n_trials", 100),
            "n_jobs": opt_config.get("n_jobs", 4),
            "timeout": opt_config.get("timeout"),
            "early_stopping": opt_config.get("early_stopping", {})
        }
        
        # Parallel execution settings
        workflow_config.parallel_execution = yaml_config.get("parallel", True)
        workflow_config.max_workers = yaml_config.get("max_workers", 4)
        
        # Infrastructure needs
        workflow_config.infrastructure_config = {
            "components_needed": ["data_loader", "portfolio_manager", "optimizer"],
            "capabilities_needed": ["backtesting", "optimization", "parallel_execution"]
        }
    
    def _interpret_live_trading_config(
        self,
        yaml_config: Dict[str, Any],
        workflow_config: WorkflowConfig
    ) -> None:
        """Interpret live trading configuration."""
        # Data configuration
        if "data" in yaml_config:
            workflow_config.data_config = {
                "provider": yaml_config["data"]["provider"],
                "symbols": yaml_config["data"]["symbols"],
                "frequency": yaml_config["data"]["frequency"],
                "lookback_days": yaml_config["data"].get("lookback_days", 30),
                "warmup_period": yaml_config["data"].get("warmup_period", 0)
            }
        
        # Live trading config
        workflow_config.live_config = {
            "paper_trading": yaml_config.get("paper_trading", True),
            "broker": yaml_config.get("broker", {}),
            "portfolio": yaml_config.get("portfolio", {}),
            "strategies": self._interpret_strategies(yaml_config.get("strategies", [])),
            "risk": self._interpret_risk_config(yaml_config.get("risk", {})),
            "execution": yaml_config.get("execution", {}),
            "monitoring": yaml_config.get("monitoring", {})
        }
        
        # Infrastructure needs
        workflow_config.infrastructure_config = {
            "components_needed": [
                "data_feed", "broker_connection", "portfolio_manager", 
                "risk_manager", "order_manager", "monitoring"
            ],
            "capabilities_needed": ["live_trading", "real_time_risk", "monitoring"]
        }
    
    def _interpret_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Interpret strategy configurations."""
        interpreted = []
        
        for strategy in strategies:
            interpreted_strategy = {
                "name": strategy["name"],
                "type": strategy["type"],
                "enabled": strategy.get("enabled", True),
                "allocation": strategy.get("allocation", 1.0),
                "parameters": strategy.get("parameters", {}),
                "capabilities": []
            }
            
            # Determine needed capabilities
            if "indicators" in strategy:
                interpreted_strategy["indicators"] = strategy["indicators"]
                interpreted_strategy["capabilities"].append("technical_indicators")
            
            if "rules" in strategy:
                interpreted_strategy["rules"] = strategy["rules"]
                interpreted_strategy["capabilities"].append("rule_engine")
            
            if strategy.get("ml_model"):
                interpreted_strategy["capabilities"].append("machine_learning")
            
            interpreted.append(interpreted_strategy)
        
        return interpreted
    
    def _interpret_risk_config(self, risk_config: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret risk management configuration."""
        interpreted = {
            "position_sizers": [],
            "risk_limits": [],
            "risk_metrics": ["sharpe", "max_drawdown", "var"]
        }
        
        # Position sizers
        for sizer in risk_config.get("position_sizers", []):
            interpreted["position_sizers"].append({
                "name": sizer["name"],
                "type": sizer["type"],
                "parameters": {k: v for k, v in sizer.items() 
                             if k not in ["name", "type"]}
            })
        
        # Risk limits
        for limit in risk_config.get("limits", []):
            interpreted["risk_limits"].append({
                "type": limit["type"],
                "enabled": limit.get("enabled", True),
                "parameters": {k: v for k, v in limit.items() 
                             if k not in ["type", "enabled"]}
            })
        
        # Additional risk settings
        if "stop_loss" in risk_config:
            interpreted["stop_loss"] = risk_config["stop_loss"]
        
        if "emergency_shutdown" in risk_config:
            interpreted["emergency_shutdown"] = risk_config["emergency_shutdown"]
        
        return interpreted
    
    def _interpret_common_config(
        self,
        yaml_config: Dict[str, Any],
        workflow_config: WorkflowConfig
    ) -> None:
        """Interpret common configuration elements."""
        # Resource limits
        if "resources" in yaml_config:
            resources = yaml_config["resources"]
            workflow_config.memory_limit_mb = resources.get("memory_mb")
            workflow_config.cpu_cores = resources.get("cpu_cores")
        
        # Timeout
        if "timeout" in yaml_config:
            workflow_config.timeout_seconds = yaml_config["timeout"]
        
        # Additional parameters
        workflow_config.parameters.update({
            "config_version": yaml_config.get("version", "1.0"),
            "tags": yaml_config.get("tags", []),
            "metadata": yaml_config.get("metadata", {})
        })


class YAMLWorkflowBuilder:
    """Builds workflow execution plans from YAML configurations."""
    
    def __init__(self, interpreter: YAMLInterpreter):
        """Initialize the workflow builder."""
        self.interpreter = interpreter
    
    def build_container_hierarchy(
        self,
        workflow_config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Build container hierarchy specification from workflow config.
        
        Args:
            workflow_config: Interpreted workflow configuration
            
        Returns:
            Container hierarchy specification
        """
        hierarchy = {
            "root": {
                "type": "workflow",
                "id": f"{workflow_config.workflow_type.value}_root",
                "capabilities": ["monitoring", "error_handling", "event_bus"],
                "children": []
            }
        }
        
        # Add containers based on workflow type
        if workflow_config.workflow_type == WorkflowType.BACKTEST:
            hierarchy["root"]["children"].extend(
                self._build_backtest_containers(workflow_config)
            )
        elif workflow_config.workflow_type == WorkflowType.OPTIMIZATION:
            hierarchy["root"]["children"].extend(
                self._build_optimization_containers(workflow_config)
            )
        elif workflow_config.workflow_type == WorkflowType.LIVE_TRADING:
            hierarchy["root"]["children"].extend(
                self._build_live_trading_containers(workflow_config)
            )
        
        return hierarchy
    
    def _build_backtest_containers(
        self,
        workflow_config: WorkflowConfig
    ) -> List[Dict[str, Any]]:
        """Build container specifications for backtest workflow."""
        containers = []
        
        # Data container
        containers.append({
            "type": "data",
            "id": "data_container",
            "capabilities": ["data_loading", "data_validation"],
            "config": workflow_config.data_config
        })
        
        # Strategy containers
        for i, strategy in enumerate(workflow_config.backtest_config["strategies"]):
            containers.append({
                "type": "strategy",
                "id": f"strategy_{strategy['name']}",
                "capabilities": strategy.get("capabilities", []) + ["backtesting"],
                "config": strategy
            })
        
        # Risk & Portfolio container
        containers.append({
            "type": "risk_portfolio",
            "id": "risk_portfolio_container",
            "capabilities": ["risk_portfolio", "position_sizing", "risk_limits"],
            "config": workflow_config.backtest_config.get("risk", {})
        })
        
        # Analysis container
        if workflow_config.backtest_config.get("analysis"):
            containers.append({
                "type": "analysis",
                "id": "analysis_container",
                "capabilities": ["performance_analytics", "reporting"],
                "config": workflow_config.backtest_config["analysis"]
            })
        
        return containers
    
    def _build_optimization_containers(
        self,
        workflow_config: WorkflowConfig
    ) -> List[Dict[str, Any]]:
        """Build container specifications for optimization workflow."""
        # Start with backtest containers as base
        containers = self._build_backtest_containers(workflow_config)
        
        # Add optimization-specific container
        containers.append({
            "type": "optimization",
            "id": "optimization_container",
            "capabilities": ["optimization", "parameter_search", "parallel_execution"],
            "config": workflow_config.optimization_config
        })
        
        return containers
    
    def _build_live_trading_containers(
        self,
        workflow_config: WorkflowConfig
    ) -> List[Dict[str, Any]]:
        """Build container specifications for live trading workflow."""
        containers = []
        
        # Data feed container
        containers.append({
            "type": "data_feed",
            "id": "data_feed_container",
            "capabilities": ["real_time_data", "data_streaming"],
            "config": workflow_config.data_config
        })
        
        # Broker container
        containers.append({
            "type": "broker",
            "id": "broker_container",
            "capabilities": ["order_execution", "position_tracking"],
            "config": workflow_config.live_config["broker"]
        })
        
        # Strategy containers
        for strategy in workflow_config.live_config["strategies"]:
            containers.append({
                "type": "strategy",
                "id": f"strategy_{strategy['name']}",
                "capabilities": strategy.get("capabilities", []) + ["live_trading"],
                "config": strategy
            })
        
        # Risk & Portfolio container with real-time capabilities
        containers.append({
            "type": "risk_portfolio",
            "id": "risk_portfolio_container",
            "capabilities": ["risk_portfolio", "real_time_risk", "emergency_shutdown"],
            "config": workflow_config.live_config.get("risk", {})
        })
        
        # Monitoring container
        if workflow_config.live_config.get("monitoring"):
            containers.append({
                "type": "monitoring",
                "id": "monitoring_container",
                "capabilities": ["monitoring", "alerting", "metrics_export"],
                "config": workflow_config.live_config["monitoring"]
            })
        
        return containers