"""
Simple Configuration Validation

Lightweight validation for workflow configurations without external dependencies.
Focuses on the most common validation needs for ADMF-PC configurations.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import logging

from .component_schemas import (
    validate_component_config,
    get_component_schema,
    list_component_types,
    list_components_for_type
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


class ConfigValidator:
    """
    Simple configuration validator for workflow configs.
    
    Provides basic validation without complex schema definitions.
    Focuses on common use cases like required fields, types, and ranges.
    """
    
    def __init__(self):
        self.custom_validators: Dict[str, Callable] = {}
    
    def validate_workflow_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a workflow configuration."""
        errors = []
        warnings = []
        
        # Basic structure validation
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return ValidationResult(False, errors, warnings)
        
        # Validate common workflow fields
        self._validate_strategies(config.get('strategies', []), errors, warnings)
        self._validate_risk_profiles(config.get('risk_profiles', []), errors, warnings)
        self._validate_execution_config(config.get('execution', {}), errors, warnings)
        self._validate_date_range(config, errors, warnings)
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def validate_required_fields(self, config: Dict[str, Any], 
                                required_fields: List[str]) -> ValidationResult:
        """Validate that required fields are present."""
        errors = []
        warnings = []
        
        for field in required_fields:
            if field not in config:
                errors.append(f"Required field missing: {field}")
            elif config[field] is None:
                errors.append(f"Required field cannot be None: {field}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def validate_types(self, config: Dict[str, Any], 
                      type_specs: Dict[str, type]) -> ValidationResult:
        """Validate field types."""
        errors = []
        warnings = []
        
        for field, expected_type in type_specs.items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    errors.append(f"Field '{field}' must be {expected_type.__name__}, got {type(value).__name__}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def validate_ranges(self, config: Dict[str, Any], 
                       range_specs: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """Validate numeric ranges."""
        errors = []
        warnings = []
        
        for field, spec in range_specs.items():
            if field in config:
                value = config[field]
                
                if 'min' in spec and value < spec['min']:
                    errors.append(f"Field '{field}' must be >= {spec['min']}, got {value}")
                
                if 'max' in spec and value > spec['max']:
                    errors.append(f"Field '{field}' must be <= {spec['max']}, got {value}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_strategies(self, strategies: List[Dict], errors: List[str], warnings: List[str]):
        """Validate strategy configurations using component schemas."""
        if not strategies:
            warnings.append("No strategies defined")
            return
        
        for i, strategy in enumerate(strategies):
            if not isinstance(strategy, dict):
                errors.append(f"Strategy {i} must be a dictionary")
                continue
            
            strategy_type = strategy.get('type')
            if not strategy_type:
                errors.append(f"Strategy {i} missing required 'type' field")
                continue
            
            # Use component schema validation
            validation_errors = validate_component_config('strategies', strategy_type, strategy)
            for error in validation_errors:
                errors.append(f"Strategy {i} ({strategy_type}): {error}")
            
            # Legacy parameter validation for backward compatibility
            params = strategy.get('parameters', {})
            if 'lookback_period' in params:
                if not isinstance(params['lookback_period'], int) or params['lookback_period'] < 1:
                    errors.append(f"Strategy {i} lookback_period must be positive integer")
    
    def _validate_risk_profiles(self, risk_profiles: List[Dict], errors: List[str], warnings: List[str]):
        """Validate risk profile configurations using component schemas."""
        if not risk_profiles:
            warnings.append("No risk profiles defined")
            return
        
        for i, profile in enumerate(risk_profiles):
            if not isinstance(profile, dict):
                errors.append(f"Risk profile {i} must be a dictionary")
                continue
            
            profile_type = profile.get('type')
            if not profile_type:
                errors.append(f"Risk profile {i} missing required 'type' field")
                continue
            
            # Use component schema validation
            validation_errors = validate_component_config('risk_profiles', profile_type, profile)
            for error in validation_errors:
                errors.append(f"Risk profile {i} ({profile_type}): {error}")
    
    def _validate_execution_config(self, execution: Dict[str, Any], errors: List[str], warnings: List[str]):
        """Validate execution configuration."""
        if not execution:
            return
        
        # Validate event tracing config if present
        if 'enable_event_tracing' in execution:
            if not isinstance(execution['enable_event_tracing'], bool):
                errors.append("enable_event_tracing must be boolean")
        
        # Validate trace settings
        if 'trace_settings' in execution:
            trace_settings = execution['trace_settings']
            if not isinstance(trace_settings, dict):
                errors.append("trace_settings must be a dictionary")
    
    def _validate_date_range(self, config: Dict[str, Any], errors: List[str], warnings: List[str]):
        """Validate date range configuration."""
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        
        if start_date and end_date:
            # Basic string format check (could be enhanced)
            if not isinstance(start_date, str) or not isinstance(end_date, str):
                errors.append("start_date and end_date must be strings")
                return
            
            # Check date format (YYYY-MM-DD)
            import re
            date_pattern = r'^\d{4}-\d{2}-\d{2}$'
            
            if not re.match(date_pattern, start_date):
                errors.append("start_date must be in YYYY-MM-DD format")
            
            if not re.match(date_pattern, end_date):
                errors.append("end_date must be in YYYY-MM-DD format")
            
            # Check logical order
            if start_date >= end_date:
                errors.append("start_date must be before end_date")
    
    def add_custom_validator(self, name: str, validator: Callable[[Any], bool]):
        """Add a custom validation function."""
        self.custom_validators[name] = validator
    
    def validate_custom(self, value: Any, validator_name: str) -> bool:
        """Run a custom validator."""
        if validator_name in self.custom_validators:
            try:
                return self.custom_validators[validator_name](value)
            except Exception as e:
                logger.error(f"Custom validator '{validator_name}' failed: {e}")
                return False
        return True


# Convenience functions for common validation patterns
def validate_workflow(config: Dict[str, Any]) -> ValidationResult:
    """Quick workflow validation."""
    validator = ConfigValidator()
    return validator.validate_workflow_config(config)


def check_required_fields(config: Dict[str, Any], *fields: str) -> ValidationResult:
    """Quick required field check."""
    validator = ConfigValidator()
    return validator.validate_required_fields(config, list(fields))