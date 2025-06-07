"""Integration of configuration validator with the Coordinator.

This module shows how the Coordinator uses the configuration validator
to ensure YAML configurations are valid before execution.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from .simple_validator import SimpleConfigValidator, ValidationResult


logger = logging.getLogger(__name__)


class CoordinatorConfigValidator:
    """Configuration validator for the Coordinator."""
    
    def __init__(self):
        """Initialize the validator."""
        self.validator = SimpleConfigValidator()
        self._validation_cache: Dict[str, ValidationResult] = {}
    
    def validate_workflow_config(
        self,
        config_path: Union[str, Path]
    ) -> ValidationResult:
        """Validate a workflow configuration file.
        
        This is the main entry point for the Coordinator to validate
        configurations before execution.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ValidationResult with details about validation
        """
        config_path = Path(config_path)
        
        # Check cache
        cache_key = str(config_path.absolute())
        if cache_key in self._validation_cache:
            logger.debug(f"Using cached validation for {config_path}")
            return self._validation_cache[cache_key]
        
        # Validate
        logger.info(f"Validating configuration: {config_path}")
        result = self.validator.validate_file(config_path)
        
        # Log results
        if result.is_valid:
            logger.info(f"✓ Configuration valid: {config_path}")
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"  ⚠ {warning}")
        else:
            logger.error(f"✗ Configuration invalid: {config_path}")
            for error in result.errors:
                logger.error(f"  ✗ {error}")
        
        # Cache result
        self._validation_cache[cache_key] = result
        
        return result
    
    def validate_and_load(
        self,
        config_path: Union[str, Path],
        raise_on_error: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Validate and load a configuration file.
        
        Args:
            config_path: Path to YAML configuration
            raise_on_error: Whether to raise exception on validation error
            
        Returns:
            Normalized configuration dict or None if invalid
            
        Raises:
            SchemaValidationError: If validation fails and raise_on_error=True
        """
        result = self.validate_workflow_config(config_path)
        
        if not result.is_valid:
            if raise_on_error:
                result.raise_if_invalid()
            return None
        
        return result.normalized_config
    
    def suggest_fixes(self, result: ValidationResult) -> List[str]:
        """Suggest fixes for validation errors.
        
        Args:
            result: Validation result with errors
            
        Returns:
            List of suggested fixes
        """
        suggestions = []
        
        for error in result.errors:
            if "required field missing" in error:
                field = error.split(".")[1].split(":")[0]
                suggestions.append(f"Add '{field}' field to your configuration")
            
            elif "not in allowed choices" in error:
                # Extract field and choices
                parts = error.split("'")
                if len(parts) >= 3:
                    value = parts[1]
                    choices_str = error.split("[")[1].split("]")[0]
                    suggestions.append(f"Change '{value}' to one of: {choices_str}")
            
            elif "must be before" in error:
                suggestions.append("Check that start_date is before end_date")
            
            elif "min must be less than max" in error:
                param = error.split(".")[2].split(":")[0]
                suggestions.append(f"Ensure {param}.min < {param}.max in parameter_space")
            
            elif "api_key required" in error:
                suggestions.append("Add 'api_key' to broker configuration or set paper_trading: true")
        
        return suggestions
    
    def create_example_config(
        self,
        workflow_type: str,
        output_path: Union[str, Path]
    ) -> None:
        """Create an example configuration file.
        
        Args:
            workflow_type: Type of workflow (backtest, optimization, live_trading)
            output_path: Where to save the example
        """
        examples = {
            "backtest": {
                "name": "Example Backtest",
                "type": "backtest",
                "description": "Example backtest configuration",
                "data": {
                    "symbols": ["SPY", "QQQ"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "frequency": "1d",
                    "source": "yahoo"
                },
                "portfolio": {
                    "initial_capital": 100000,
                    "commission": {"type": "fixed", "value": 1.0}
                },
                "strategies": [{
                    "name": "example_strategy",
                    "type": "moving_average_crossover",
                    "parameters": {
                        "fast_period": 10,
                        "slow_period": 30
                    }
                }],
                "risk": {
                    "limits": [
                        {"type": "position", "max_position": 10000},
                        {"type": "drawdown", "max_drawdown_pct": 20}
                    ]
                }
            },
            "optimization": {
                "name": "Example Optimization",
                "type": "optimization",
                "description": "Example optimization configuration",
                "base_config": {
                    "data": {
                        "symbols": ["SPY"],
                        "start_date": "2022-01-01",
                        "end_date": "2023-12-31",
                        "frequency": "1h"
                    },
                    "portfolio": {"initial_capital": 100000},
                    "strategies": [{
                        "name": "strategy_to_optimize",
                        "type": "moving_average_crossover"
                    }]
                },
                "optimization": {
                    "method": "bayesian",
                    "parameter_space": {
                        "fast_period": {"type": "int", "min": 5, "max": 50},
                        "slow_period": {"type": "int", "min": 20, "max": 200}
                    },
                    "constraints": [{
                        "type": "expression",
                        "expression": "slow_period > fast_period + 10"
                    }],
                    "objectives": [{
                        "metric": "sharpe_ratio",
                        "direction": "maximize"
                    }],
                    "n_trials": 100
                }
            },
            "live_trading": {
                "name": "Example Live Trading",
                "type": "live_trading",
                "description": "Example paper trading configuration",
                "paper_trading": True,
                "broker": {
                    "name": "paper",
                    "timeout": 30
                },
                "data": {
                    "provider": "yahoo",
                    "symbols": ["SPY", "QQQ"],
                    "frequency": "5min",
                    "lookback_days": 30
                },
                "portfolio": {
                    "initial_capital": 100000,
                    "max_positions": 2
                },
                "strategies": [{
                    "name": "live_strategy",
                    "type": "momentum_strategy",
                    "parameters": {
                        "lookback_period": 20,
                        "threshold": 0.02
                    }
                }],
                "risk": {
                    "limits": [
                        {"type": "position", "max_position": 25000},
                        {"type": "daily_loss", "max_daily_loss_pct": 2.0}
                    ]
                }
            }
        }
        
        if workflow_type not in examples:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(examples[workflow_type], f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created example {workflow_type} configuration at {output_path}")
    
    def validate_string(
        self,
        config_str: str,
        schema_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate a YAML configuration string.
        
        Args:
            config_str: YAML configuration as string
            schema_name: Schema to validate against
            
        Returns:
            ValidationResult
        """
        try:
            config = yaml.safe_load(config_str)
            
            if schema_name is None:
                schema_name = self.validator._detect_schema(config)
                if schema_name is None:
                    return ValidationResult(
                        is_valid=False,
                        errors=["Could not determine configuration type"]
                    )
            
            return self.validator.validate(config, schema_name)
            
        except yaml.YAMLError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid YAML: {str(e)}"]
            )


def validate_config_for_coordinator(config_path: Union[str, Path]) -> bool:
    """Convenience function to validate a configuration file.
    
    This is what the Coordinator would call before executing a workflow.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    validator = CoordinatorConfigValidator()
    result = validator.validate_workflow_config(config_path)
    
    if not result.is_valid:
        print("\nConfiguration validation failed!")
        print("\nErrors:")
        for error in result.errors:
            print(f"  ✗ {error}")
        
        suggestions = validator.suggest_fixes(result)
        if suggestions:
            print("\nSuggested fixes:")
            for suggestion in suggestions:
                print(f"  → {suggestion}")
        
        return False
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    return True


# Example usage in Coordinator
class ConfigValidationMixin:
    """Mixin for adding configuration validation to the Coordinator."""
    
    def __init__(self):
        self.config_validator = CoordinatorConfigValidator()
    
    def load_validated_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate a configuration file.
        
        Args:
            config_path: Path to YAML configuration
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            SchemaValidationError: If configuration is invalid
        """
        config = self.config_validator.validate_and_load(config_path, raise_on_error=True)
        
        # Log successful validation
        logger.info(f"Configuration loaded and validated: {Path(config_path).name}")
        logger.debug(f"Workflow type: {config.get('type', 'auto-detected')}")
        logger.debug(f"Strategies: {len(config.get('strategies', []))}")
        
        return config