"""Simple YAML configuration validator without external dependencies.

This module provides validation for YAML configurations without requiring jsonschema.
It implements a lightweight validation system suitable for the ADMF-PC use case.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging


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
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized_config: Optional[Dict[str, Any]] = None
    
    def raise_if_invalid(self):
        """Raise exception if validation failed."""
        if not self.is_valid:
            raise SchemaValidationError(
                f"Configuration validation failed with {len(self.errors)} errors",
                self.errors
            )


@dataclass
class FieldSpec:
    """Specification for a configuration field."""
    name: str
    type: type
    required: bool = True
    default: Any = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    validator: Optional[callable] = None
    description: Optional[str] = None


class SimpleConfigValidator:
    """Simple configuration validator."""
    
    def __init__(self):
        """Initialize validator."""
        self._schemas: Dict[str, Dict[str, List[FieldSpec]]] = {}
        self._register_built_in_schemas()
    
    def _register_built_in_schemas(self):
        """Register built-in configuration schemas."""
        # Backtest schema
        self._schemas["backtest"] = {
            "root": [
                FieldSpec("name", str, required=True),
                FieldSpec("type", str, required=False, default="backtest", choices=["backtest"]),
                FieldSpec("description", str, required=False),
                FieldSpec("execution_mode", str, required=False, default="vectorized", 
                         choices=["vectorized", "event_driven"]),
                FieldSpec("data", dict, required=True),
                FieldSpec("portfolio", dict, required=True),
                FieldSpec("strategies", list, required=True, min_length=1),
                FieldSpec("risk", dict, required=False),
                FieldSpec("analysis", dict, required=False)
            ],
            "data": [
                FieldSpec("source", str, required=False),
                FieldSpec("symbols", list, required=True, min_length=1),
                FieldSpec("start_date", str, required=True),
                FieldSpec("end_date", str, required=True),
                FieldSpec("frequency", str, required=True, 
                         choices=["1min", "5min", "15min", "30min", "1h", "1d"]),
                FieldSpec("format", str, required=False, default="csv",
                         choices=["csv", "parquet", "hdf5"]),
                FieldSpec("timezone", str, required=False, default="UTC")
            ],
            "portfolio": [
                FieldSpec("initial_capital", (int, float), required=True, min_value=0),
                FieldSpec("currency", str, required=False, default="USD"),
                FieldSpec("commission", dict, required=False),
                FieldSpec("slippage", dict, required=False)
            ],
            "strategy": [
                FieldSpec("name", str, required=True),
                FieldSpec("type", str, required=True),
                FieldSpec("enabled", bool, required=False, default=True),
                FieldSpec("allocation", (int, float), required=False, default=1.0,
                         min_value=0, max_value=1),
                FieldSpec("parameters", dict, required=False),
                FieldSpec("capabilities", list, required=False)
            ],
            "risk_limit": [
                FieldSpec("type", str, required=True,
                         choices=["position", "exposure", "drawdown", "var", 
                                "concentration", "leverage", "daily_loss", "symbol_restriction"]),
                FieldSpec("enabled", bool, required=False, default=True)
            ],
            "position_sizer": [
                FieldSpec("name", str, required=True),
                FieldSpec("type", str, required=True,
                         choices=["fixed", "percentage", "volatility", "kelly", "atr"])
            ]
        }
        
        # Optimization schema
        self._schemas["optimization"] = {
            "root": [
                FieldSpec("name", str, required=True),
                FieldSpec("type", str, required=False, default="optimization", 
                         choices=["optimization"]),
                FieldSpec("description", str, required=False),
                FieldSpec("base_config", dict, required=True),
                FieldSpec("optimization", dict, required=True),
                FieldSpec("parallel", bool, required=False, default=True),
                FieldSpec("max_workers", int, required=False, default=4, min_value=1),
                FieldSpec("output", dict, required=False)
            ],
            "optimization": [
                FieldSpec("method", str, required=True,
                         choices=["grid_search", "random_search", "bayesian", 
                                "genetic", "differential_evolution", "walk_forward"]),
                FieldSpec("parameter_space", dict, required=True),
                FieldSpec("constraints", list, required=False),
                FieldSpec("objectives", list, required=True, min_length=1),
                FieldSpec("n_trials", int, required=False, default=100, min_value=1),
                FieldSpec("n_jobs", int, required=False, min_value=1),
                FieldSpec("timeout", int, required=False, min_value=0),
                FieldSpec("early_stopping", dict, required=False)
            ],
            "objective": [
                FieldSpec("metric", str, required=True),
                FieldSpec("direction", str, required=True, 
                         choices=["minimize", "maximize"]),
                FieldSpec("weight", (int, float), required=False, default=1.0)
            ],
            "parameter": [
                FieldSpec("type", str, required=True, choices=["int", "float", "choice"]),
                FieldSpec("min", (int, float), required=False),
                FieldSpec("max", (int, float), required=False),
                FieldSpec("step", (int, float), required=False),
                FieldSpec("choices", list, required=False)
            ]
        }
        
        # Live trading schema
        self._schemas["live_trading"] = {
            "root": [
                FieldSpec("name", str, required=True),
                FieldSpec("type", str, required=False, default="live_trading",
                         choices=["live_trading", "live"]),
                FieldSpec("description", str, required=False),
                FieldSpec("paper_trading", bool, required=False, default=True),
                FieldSpec("broker", dict, required=True),
                FieldSpec("data", dict, required=True),
                FieldSpec("portfolio", dict, required=True),
                FieldSpec("strategies", list, required=True, min_length=1),
                FieldSpec("risk", dict, required=False),
                FieldSpec("execution", dict, required=False),
                FieldSpec("monitoring", dict, required=False)
            ],
            "broker": [
                FieldSpec("name", str, required=True,
                         choices=["alpaca", "interactive_brokers", "paper"]),
                FieldSpec("api_key", str, required=False),
                FieldSpec("api_secret", str, required=False),
                FieldSpec("base_url", str, required=False),
                FieldSpec("account_id", str, required=False),
                FieldSpec("timeout", int, required=False, default=30, min_value=1),
                FieldSpec("retry_count", int, required=False, default=3, min_value=0)
            ],
            "data": [
                FieldSpec("provider", str, required=True,
                         choices=["broker", "yahoo", "polygon", "alpaca"]),
                FieldSpec("symbols", list, required=True, min_length=1),
                FieldSpec("frequency", str, required=True),
                FieldSpec("lookback_days", int, required=False, min_value=1),
                FieldSpec("warmup_period", int, required=False, min_value=0)
            ]
        }
    
    def validate_file(
        self,
        config_path: Union[str, Path],
        schema_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate a YAML configuration file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"Configuration file not found: {config_path}"]
            )
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if schema_name is None:
                schema_name = self._detect_schema(config)
                if schema_name is None:
                    return ValidationResult(
                        is_valid=False,
                        errors=["Could not determine configuration type"]
                    )
            
            return self.validate(config, schema_name)
            
        except yaml.YAMLError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid YAML: {str(e)}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Error loading configuration: {str(e)}"]
            )
    
    def validate(
        self,
        config: Dict[str, Any],
        schema_name: str
    ) -> ValidationResult:
        """Validate a configuration dictionary."""
        if schema_name not in self._schemas:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown schema: {schema_name}"]
            )
        
        errors = []
        warnings = []
        
        # Validate root fields
        schema = self._schemas[schema_name]
        self._validate_fields(config, schema.get("root", []), "root", errors, warnings)
        
        # Validate nested structures
        if "data" in config and "data" in schema:
            self._validate_fields(config["data"], schema["data"], "data", errors, warnings)
        
        if "portfolio" in config and "portfolio" in schema:
            self._validate_fields(config["portfolio"], schema["portfolio"], "portfolio", errors, warnings)
        
        if "strategies" in config and isinstance(config["strategies"], list):
            for i, strategy in enumerate(config["strategies"]):
                if "strategy" in schema:
                    self._validate_fields(strategy, schema["strategy"], f"strategies[{i}]", errors, warnings)
        
        if "optimization" in config and "optimization" in schema:
            self._validate_fields(config["optimization"], schema["optimization"], "optimization", errors, warnings)
            
            # Validate objectives
            if "objectives" in config["optimization"] and "objective" in schema:
                for i, obj in enumerate(config["optimization"]["objectives"]):
                    self._validate_fields(obj, schema["objective"], f"optimization.objectives[{i}]", errors, warnings)
            
            # Validate parameter space
            if "parameter_space" in config["optimization"] and "parameter" in schema:
                for param_name, param_def in config["optimization"]["parameter_space"].items():
                    self._validate_fields(param_def, schema["parameter"], 
                                        f"optimization.parameter_space.{param_name}", errors, warnings)
        
        if "broker" in config and "broker" in schema:
            self._validate_fields(config["broker"], schema["broker"], "broker", errors, warnings)
        
        if "data" in config and schema_name == "live_trading" and "data" in schema:
            self._validate_fields(config["data"], schema["data"], "data", errors, warnings)
        
        # Custom validation
        if schema_name == "backtest":
            self._validate_backtest_custom(config, errors, warnings)
        elif schema_name == "optimization":
            self._validate_optimization_custom(config, errors, warnings)
        elif schema_name == "live_trading":
            self._validate_live_trading_custom(config, errors, warnings)
        
        # Normalize if valid
        normalized_config = self._normalize_config(config, schema_name) if not errors else None
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_config=normalized_config
        )
    
    def _validate_fields(
        self,
        data: Dict[str, Any],
        specs: List[FieldSpec],
        path: str,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate fields against specifications."""
        spec_map = {spec.name: spec for spec in specs}
        
        # Check required fields
        for spec in specs:
            if spec.required and spec.name not in data:
                errors.append(f"{path}.{spec.name}: required field missing")
        
        # Validate present fields
        for field_name, value in data.items():
            if field_name not in spec_map:
                continue  # Skip unknown fields
            
            spec = spec_map[field_name]
            field_path = f"{path}.{field_name}"
            
            # Type check
            if value is not None:
                if isinstance(spec.type, tuple):
                    if not isinstance(value, spec.type):
                        errors.append(f"{field_path}: expected {spec.type}, got {type(value).__name__}")
                        continue
                else:
                    if not isinstance(value, spec.type):
                        errors.append(f"{field_path}: expected {spec.type.__name__}, got {type(value).__name__}")
                        continue
            
            # Choices validation
            if spec.choices and value not in spec.choices:
                errors.append(f"{field_path}: value '{value}' not in allowed choices {spec.choices}")
            
            # Range validation
            if spec.min_value is not None and value < spec.min_value:
                errors.append(f"{field_path}: value {value} is less than minimum {spec.min_value}")
            
            if spec.max_value is not None and value > spec.max_value:
                errors.append(f"{field_path}: value {value} is greater than maximum {spec.max_value}")
            
            # Length validation
            if hasattr(value, '__len__'):
                if spec.min_length is not None and len(value) < spec.min_length:
                    errors.append(f"{field_path}: length {len(value)} is less than minimum {spec.min_length}")
                
                if spec.max_length is not None and len(value) > spec.max_length:
                    errors.append(f"{field_path}: length {len(value)} is greater than maximum {spec.max_length}")
            
            # Custom validator
            if spec.validator:
                try:
                    is_valid, message = spec.validator(value)
                    if not is_valid:
                        errors.append(f"{field_path}: {message}")
                except Exception as e:
                    errors.append(f"{field_path}: validator error - {str(e)}")
    
    def _validate_backtest_custom(
        self,
        config: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Custom validation for backtest configs."""
        # Validate date range
        if "data" in config and "start_date" in config["data"] and "end_date" in config["data"]:
            try:
                start = datetime.fromisoformat(config["data"]["start_date"])
                end = datetime.fromisoformat(config["data"]["end_date"])
                if start >= end:
                    errors.append("data.start_date must be before data.end_date")
            except ValueError:
                errors.append("Invalid date format in data section (use YYYY-MM-DD)")
        
        # Validate strategy allocations
        if "strategies" in config:
            total_allocation = sum(s.get("allocation", 1.0) for s in config["strategies"])
            if abs(total_allocation - 1.0) > 0.001:
                warnings.append(f"Strategy allocations sum to {total_allocation}, not 1.0")
    
    def _validate_optimization_custom(
        self,
        config: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Custom validation for optimization configs."""
        if "optimization" in config and "parameter_space" in config["optimization"]:
            for param_name, param_def in config["optimization"]["parameter_space"].items():
                if "min" in param_def and "max" in param_def:
                    if param_def["min"] >= param_def["max"]:
                        errors.append(f"optimization.parameter_space.{param_name}: min must be less than max")
                
                # Suggest constraint for common patterns
                if param_name == "fast_period":
                    slow_params = [p for p in config["optimization"]["parameter_space"].keys() 
                                 if "slow" in p.lower()]
                    if slow_params and "constraints" not in config["optimization"]:
                        warnings.append(f"Consider adding constraint: {param_name} < {slow_params[0]}")
    
    def _validate_live_trading_custom(
        self,
        config: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Custom validation for live trading configs."""
        # API key required for live trading
        if not config.get("paper_trading", True):
            if "broker" in config and not config["broker"].get("api_key"):
                errors.append("broker.api_key required for live trading (paper_trading=false)")
        
        # Risk warnings
        if "risk" in config and "limits" in config["risk"]:
            has_daily_loss = any(l.get("type") == "daily_loss" for l in config["risk"]["limits"])
            if not has_daily_loss:
                warnings.append("No daily loss limit configured - recommended for live trading")
    
    def _detect_schema(self, config: Dict[str, Any]) -> Optional[str]:
        """Auto-detect schema type from configuration."""
        # Check explicit type
        if "type" in config:
            type_map = {
                "backtest": "backtest",
                "optimization": "optimization",
                "live_trading": "live_trading",
                "live": "live_trading"
            }
            return type_map.get(config["type"])
        
        # Detect by structure
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
        """Normalize configuration with defaults."""
        import copy
        normalized = copy.deepcopy(config)
        
        # Apply defaults from schema
        schema = self._schemas[schema_name]
        
        # Root level defaults
        for spec in schema.get("root", []):
            if spec.default is not None and spec.name not in normalized:
                normalized[spec.name] = spec.default
        
        # Nested defaults
        if schema_name == "backtest":
            if "data" in normalized:
                for spec in schema.get("data", []):
                    if spec.default is not None and spec.name not in normalized["data"]:
                        normalized["data"][spec.name] = spec.default
            
            if "strategies" in normalized:
                for strategy in normalized["strategies"]:
                    for spec in schema.get("strategy", []):
                        if spec.default is not None and spec.name not in strategy:
                            strategy[spec.name] = spec.default
        
        return normalized