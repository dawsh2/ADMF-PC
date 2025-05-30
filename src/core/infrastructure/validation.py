"""
Validation infrastructure for ADMF-PC.

Provides validation rules, validators, and results.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
from datetime import datetime

from ..logging import StructuredLogger


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge with another validation result."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            metadata={**self.metadata, **other.metadata}
        )


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    @abstractmethod
    def validate(self, target: Any) -> ValidationResult:
        """Validate the target."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get rule description."""
        pass


class StateValidationRule(ValidationRule):
    """Validates object state/attributes."""
    
    def __init__(
        self,
        attribute: str,
        validator: Union[str, Callable],
        message: Optional[str] = None
    ):
        """
        Initialize state validation rule.
        
        Args:
            attribute: Attribute to validate
            validator: Validation function or predefined validator name
            message: Custom error message
        """
        self.attribute = attribute
        self.validator = validator
        self.message = message
        self._validators = self._get_validators()
    
    def validate(self, target: Any) -> ValidationResult:
        """Validate attribute on target."""
        if not hasattr(target, self.attribute):
            return ValidationResult(
                is_valid=False,
                errors=[f"Missing attribute: {self.attribute}"]
            )
        
        value = getattr(target, self.attribute)
        
        # Get validator function
        if isinstance(self.validator, str):
            validator_func = self._validators.get(self.validator)
            if not validator_func:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Unknown validator: {self.validator}"]
                )
        else:
            validator_func = self.validator
        
        # Validate
        try:
            is_valid = validator_func(value)
            if not is_valid:
                error_msg = self.message or f"Validation failed for {self.attribute}"
                return ValidationResult(is_valid=False, errors=[error_msg])
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error for {self.attribute}: {str(e)}"]
            )
    
    def get_description(self) -> str:
        """Get rule description."""
        return f"State validation for {self.attribute}"
    
    def _get_validators(self) -> Dict[str, Callable]:
        """Get predefined validators."""
        return {
            'non_negative': lambda x: x >= 0,
            'positive': lambda x: x > 0,
            'non_empty': lambda x: bool(x),
            'is_string': lambda x: isinstance(x, str),
            'is_number': lambda x: isinstance(x, (int, float)),
            'is_list': lambda x: isinstance(x, list),
            'is_dict': lambda x: isinstance(x, dict),
            'is_bool': lambda x: isinstance(x, bool)
        }


class RangeValidationRule(ValidationRule):
    """Validates numeric values are within range."""
    
    def __init__(
        self,
        attribute: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        message: Optional[str] = None
    ):
        self.attribute = attribute
        self.min_value = min_value
        self.max_value = max_value
        self.message = message
    
    def validate(self, target: Any) -> ValidationResult:
        """Validate range."""
        if not hasattr(target, self.attribute):
            return ValidationResult(
                is_valid=False,
                errors=[f"Missing attribute: {self.attribute}"]
            )
        
        value = getattr(target, self.attribute)
        
        if not isinstance(value, (int, float)):
            return ValidationResult(
                is_valid=False,
                errors=[f"{self.attribute} must be numeric"]
            )
        
        errors = []
        
        if self.min_value is not None and value < self.min_value:
            errors.append(
                self.message or f"{self.attribute} ({value}) is below minimum ({self.min_value})"
            )
        
        if self.max_value is not None and value > self.max_value:
            errors.append(
                self.message or f"{self.attribute} ({value}) is above maximum ({self.max_value})"
            )
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def get_description(self) -> str:
        """Get rule description."""
        return f"Range validation for {self.attribute} [{self.min_value}, {self.max_value}]"


class RegexValidationRule(ValidationRule):
    """Validates string values match pattern."""
    
    def __init__(
        self,
        attribute: str,
        pattern: str,
        message: Optional[str] = None
    ):
        self.attribute = attribute
        self.pattern = pattern
        self.regex = re.compile(pattern)
        self.message = message
    
    def validate(self, target: Any) -> ValidationResult:
        """Validate pattern match."""
        if not hasattr(target, self.attribute):
            return ValidationResult(
                is_valid=False,
                errors=[f"Missing attribute: {self.attribute}"]
            )
        
        value = getattr(target, self.attribute)
        
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                errors=[f"{self.attribute} must be a string"]
            )
        
        if not self.regex.match(value):
            error_msg = self.message or f"{self.attribute} does not match pattern {self.pattern}"
            return ValidationResult(is_valid=False, errors=[error_msg])
        
        return ValidationResult(is_valid=True)
    
    def get_description(self) -> str:
        """Get rule description."""
        return f"Pattern validation for {self.attribute}"


class DependencyValidationRule(ValidationRule):
    """Validates dependencies between attributes."""
    
    def __init__(
        self,
        attribute: str,
        depends_on: str,
        condition: Callable[[Any, Any], bool],
        message: Optional[str] = None
    ):
        """
        Initialize dependency validation.
        
        Args:
            attribute: Attribute to validate
            depends_on: Attribute it depends on
            condition: Function(dependent_value, dependency_value) -> bool
            message: Custom error message
        """
        self.attribute = attribute
        self.depends_on = depends_on
        self.condition = condition
        self.message = message
    
    def validate(self, target: Any) -> ValidationResult:
        """Validate dependency."""
        for attr in [self.attribute, self.depends_on]:
            if not hasattr(target, attr):
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Missing attribute: {attr}"]
                )
        
        value = getattr(target, self.attribute)
        dependency = getattr(target, self.depends_on)
        
        try:
            is_valid = self.condition(value, dependency)
            if not is_valid:
                error_msg = self.message or f"{self.attribute} fails dependency check with {self.depends_on}"
                return ValidationResult(is_valid=False, errors=[error_msg])
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Dependency validation error: {str(e)}"]
            )
    
    def get_description(self) -> str:
        """Get rule description."""
        return f"Dependency validation: {self.attribute} depends on {self.depends_on}"


class ComponentValidator:
    """Validates components using rules."""
    
    def __init__(
        self,
        component_name: str,
        rules: Optional[List[ValidationRule]] = None
    ):
        """
        Initialize component validator.
        
        Args:
            component_name: Name of component being validated
            rules: List of validation rules
        """
        self.component_name = component_name
        self.rules = rules or []
        self._logger = StructuredLogger(f"Validator.{component_name}")
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules.append(rule)
    
    def validate(self, target: Any) -> ValidationResult:
        """Validate target against all rules."""
        self._logger.debug(f"Validating {self.component_name}")
        
        result = ValidationResult(is_valid=True)
        
        for rule in self.rules:
            try:
                rule_result = rule.validate(target)
                result = result.merge(rule_result)
                
                if not rule_result.is_valid:
                    self._logger.debug(
                        f"Rule failed: {rule.get_description()}",
                        errors=rule_result.errors
                    )
                    
            except Exception as e:
                self._logger.error(
                    f"Rule execution failed: {rule.get_description()}",
                    error=str(e)
                )
                result.errors.append(f"Rule error: {str(e)}")
                result.is_valid = False
        
        # Log summary
        if result.is_valid:
            self._logger.info(
                f"Validation passed for {self.component_name}",
                rules_checked=len(self.rules)
            )
        else:
            self._logger.warning(
                f"Validation failed for {self.component_name}",
                errors=len(result.errors),
                warnings=len(result.warnings)
            )
        
        return result


class ConfigValidator:
    """Validates configuration dictionaries."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize config validator.
        
        Args:
            schema: Validation schema
        """
        self.schema = schema
        self._logger = StructuredLogger("ConfigValidator")
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration."""
        result = ValidationResult(is_valid=True)
        
        for key, rules in self.schema.items():
            if key not in config:
                if rules.get('required', False):
                    result.errors.append(f"Missing required field: {key}")
                    result.is_valid = False
                continue
            
            value = config[key]
            
            # Type validation
            expected_type = rules.get('type')
            if expected_type:
                if not self._check_type(value, expected_type):
                    result.errors.append(
                        f"{key}: Expected type {expected_type}, got {type(value).__name__}"
                    )
                    result.is_valid = False
                    continue
            
            # Range validation
            if 'min' in rules and value < rules['min']:
                result.errors.append(f"{key}: Value {value} below minimum {rules['min']}")
                result.is_valid = False
            
            if 'max' in rules and value > rules['max']:
                result.errors.append(f"{key}: Value {value} above maximum {rules['max']}")
                result.is_valid = False
            
            # Pattern validation
            if 'pattern' in rules and isinstance(value, str):
                if not re.match(rules['pattern'], value):
                    result.errors.append(f"{key}: Value does not match pattern {rules['pattern']}")
                    result.is_valid = False
            
            # Enum validation
            if 'enum' in rules and value not in rules['enum']:
                result.errors.append(f"{key}: Value must be one of {rules['enum']}")
                result.is_valid = False
            
            # Custom validation
            if 'validator' in rules:
                try:
                    validator_result = rules['validator'](value, config)
                    if not validator_result:
                        result.errors.append(f"{key}: Custom validation failed")
                        result.is_valid = False
                except Exception as e:
                    result.errors.append(f"{key}: Validation error: {str(e)}")
                    result.is_valid = False
        
        return result
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'number': (int, float)
        }
        
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        
        return True  # Unknown type, assume valid