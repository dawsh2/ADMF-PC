"""YAML configuration schema validator.

This module provides validation for YAML configurations used throughout
the ADMF-PC system. It ensures configurations are valid before the
Coordinator attempts to execute them.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from jsonschema import validate, ValidationError, Draft7Validator
from jsonschema.exceptions import SchemaError


logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_config: Optional[Dict[str, Any]] = None
    
    def raise_if_invalid(self):
        """Raise exception if validation failed."""
        if not self.is_valid:
            raise SchemaValidationError(
                f"Configuration validation failed with {len(self.errors)} errors",
                self.errors
            )


@dataclass 
class ConfigSchema:
    """Configuration schema definition."""
    name: str
    version: str
    schema: Dict[str, Any]
    description: Optional[str] = None
    examples: Optional[List[Dict[str, Any]]] = None


class ConfigSchemaValidator:
    """Validates YAML configurations against schemas."""
    
    def __init__(self):
        """Initialize schema validator."""
        self._schemas: Dict[str, ConfigSchema] = {}
        self._validators: Dict[str, Draft7Validator] = {}
        self._custom_validators: Dict[str, callable] = {}
        
        # Register built-in custom validators
        self._register_custom_validators()
    
    def register_schema(self, schema: ConfigSchema) -> None:
        """Register a configuration schema.
        
        Args:
            schema: Schema to register
        """
        try:
            # Create JSON schema validator
            validator = Draft7Validator(schema.schema)
            validator.check_schema(schema.schema)  # Validate the schema itself
            
            self._schemas[schema.name] = schema
            self._validators[schema.name] = validator
            
            logger.info(f"Registered schema: {schema.name} v{schema.version}")
            
        except SchemaError as e:
            logger.error(f"Invalid schema {schema.name}: {str(e)}")
            raise
    
    def validate_file(
        self,
        config_path: Union[str, Path],
        schema_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate a YAML configuration file.
        
        Args:
            config_path: Path to YAML file
            schema_name: Schema to validate against (auto-detected if None)
            
        Returns:
            Validation result
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"Configuration file not found: {config_path}"],
                warnings=[]
            )
        
        try:
            # Load YAML
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Auto-detect schema if not specified
            if schema_name is None:
                schema_name = self._detect_schema(config)
                if schema_name is None:
                    return ValidationResult(
                        is_valid=False,
                        errors=["Could not determine configuration type"],
                        warnings=[]
                    )
            
            return self.validate(config, schema_name)
            
        except yaml.YAMLError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid YAML: {str(e)}"],
                warnings=[]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Error loading configuration: {str(e)}"],
                warnings=[]
            )
    
    def validate(
        self,
        config: Dict[str, Any],
        schema_name: str
    ) -> ValidationResult:
        """Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            schema_name: Schema to validate against
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check schema exists
        if schema_name not in self._schemas:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown schema: {schema_name}"],
                warnings=[]
            )
        
        schema = self._schemas[schema_name]
        validator = self._validators[schema_name]
        
        # JSON schema validation
        schema_errors = []
        for error in validator.iter_errors(config):
            path = " -> ".join(str(p) for p in error.path) if error.path else "root"
            schema_errors.append(f"{path}: {error.message}")
        
        if schema_errors:
            errors.extend(schema_errors)
        
        # Custom validation
        if schema_name in self._custom_validators:
            custom_errors, custom_warnings = self._custom_validators[schema_name](config)
            errors.extend(custom_errors)
            warnings.extend(custom_warnings)
        
        # Normalize configuration
        normalized_config = self._normalize_config(config, schema_name) if not errors else None
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_config=normalized_config
        )
    
    def _detect_schema(self, config: Dict[str, Any]) -> Optional[str]:
        """Auto-detect schema type from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Detected schema name or None
        """
        # Check for explicit type field
        if "type" in config:
            type_map = {
                "backtest": "backtest",
                "optimization": "optimization", 
                "live_trading": "live_trading",
                "live": "live_trading"
            }
            return type_map.get(config["type"])
        
        # Detect based on structure
        if "backtesting" in config or "historical_data" in config:
            return "backtest"
        elif "optimization" in config or "parameter_space" in config:
            return "optimization"
        elif "live_trading" in config or "broker" in config:
            return "live_trading"
        
        return None
    
    def _normalize_config(
        self,
        config: Dict[str, Any],
        schema_name: str
    ) -> Dict[str, Any]:
        """Normalize configuration with defaults and conversions.
        
        Args:
            config: Configuration to normalize
            schema_name: Schema name
            
        Returns:
            Normalized configuration
        """
        # Create deep copy
        normalized = json.loads(json.dumps(config))
        
        # Apply schema-specific normalization
        if schema_name == "backtest":
            normalized = self._normalize_backtest_config(normalized)
        elif schema_name == "optimization":
            normalized = self._normalize_optimization_config(normalized)
        elif schema_name == "live_trading":
            normalized = self._normalize_live_config(normalized)
        
        return normalized
    
    def _normalize_backtest_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize backtest configuration."""
        # Set defaults
        config.setdefault("name", f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        config.setdefault("execution_mode", "vectorized")
        
        # Normalize data section
        if "data" in config:
            data = config["data"]
            data.setdefault("format", "csv")
            data.setdefault("timezone", "UTC")
        
        # Normalize strategies
        if "strategies" in config:
            for strategy in config["strategies"]:
                strategy.setdefault("enabled", True)
                strategy.setdefault("allocation", 1.0)
        
        return config
    
    def _normalize_optimization_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize optimization configuration."""
        # Set defaults
        config.setdefault("name", f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        config.setdefault("parallel", True)
        config.setdefault("max_workers", 4)
        
        # Normalize optimization section
        if "optimization" in config:
            opt = config["optimization"]
            opt.setdefault("method", "grid_search")
            opt.setdefault("n_trials", 100)
            
            # Normalize objectives
            if "objectives" in opt:
                for obj in opt["objectives"]:
                    obj.setdefault("weight", 1.0)
                    obj.setdefault("direction", "maximize")
        
        return config
    
    def _normalize_live_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize live trading configuration."""
        # Set defaults
        config.setdefault("name", f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        config.setdefault("paper_trading", True)
        
        # Normalize broker section
        if "broker" in config:
            broker = config["broker"]
            broker.setdefault("timeout", 30)
            broker.setdefault("retry_count", 3)
        
        # Normalize risk section
        if "risk" in config:
            risk = config["risk"]
            risk.setdefault("check_frequency", "pre_trade")
            risk.setdefault("max_retry", 3)
        
        return config
    
    def _register_custom_validators(self) -> None:
        """Register custom validation functions."""
        
        def validate_backtest(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
            """Custom validation for backtest configs."""
            errors = []
            warnings = []
            
            # Validate date range
            if "data" in config and "start_date" in config["data"] and "end_date" in config["data"]:
                try:
                    start = datetime.fromisoformat(config["data"]["start_date"])
                    end = datetime.fromisoformat(config["data"]["end_date"])
                    if start >= end:
                        errors.append("start_date must be before end_date")
                except ValueError:
                        errors.append("Invalid date format (use YYYY-MM-DD)")
            
            # Validate strategy allocations
            if "strategies" in config:
                total_allocation = sum(s.get("allocation", 1.0) for s in config["strategies"])
                if abs(total_allocation - 1.0) > 0.001:
                    warnings.append(f"Strategy allocations sum to {total_allocation}, not 1.0")
            
            return errors, warnings
        
        def validate_optimization(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
            """Custom validation for optimization configs."""
            errors = []
            warnings = []
            
            # Validate parameter constraints
            if "optimization" in config and "parameter_space" in config["optimization"]:
                for param_name, param_def in config["optimization"]["parameter_space"].items():
                    if "min" in param_def and "max" in param_def:
                        if param_def["min"] >= param_def["max"]:
                            errors.append(f"Parameter {param_name}: min must be less than max")
                    
                    # Check for common parameter relationships
                    if param_name == "fast_period" and "slow_period" in config["optimization"]["parameter_space"]:
                        warnings.append("Consider adding constraint: fast_period < slow_period")
            
            return errors, warnings
        
        def validate_live(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
            """Custom validation for live trading configs."""
            errors = []
            warnings = []
            
            # Validate broker config
            if "broker" in config:
                if not config.get("paper_trading", True) and not config["broker"].get("api_key"):
                    errors.append("API key required for live trading")
            
            # Validate risk limits for live trading
            if "risk" in config and "limits" in config["risk"]:
                if not any(limit.get("type") == "daily_loss" for limit in config["risk"]["limits"]):
                    warnings.append("No daily loss limit configured for live trading")
            
            return errors, warnings
        
        self._custom_validators["backtest"] = validate_backtest
        self._custom_validators["optimization"] = validate_optimization
        self._custom_validators["live_trading"] = validate_live
    
    def get_schema_info(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered schema.
        
        Args:
            schema_name: Name of schema
            
        Returns:
            Schema information or None
        """
        if schema_name not in self._schemas:
            return None
        
        schema = self._schemas[schema_name]
        return {
            "name": schema.name,
            "version": schema.version,
            "description": schema.description,
            "properties": list(schema.schema.get("properties", {}).keys()),
            "required": schema.schema.get("required", []),
            "examples_count": len(schema.examples) if schema.examples else 0
        }
    
    def list_schemas(self) -> List[str]:
        """List all registered schema names."""
        return list(self._schemas.keys())