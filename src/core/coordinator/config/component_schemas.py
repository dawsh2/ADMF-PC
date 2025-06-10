"""
Component Configuration Schemas

Enhanced version that integrates existing detailed schemas with new organization.
Preserves all existing schema definitions while providing better structure.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import existing detailed schemas
from .schemas import (
    STRATEGY_COMPONENT_SCHEMA,
    RISK_LIMIT_SCHEMA, 
    POSITION_SIZER_SCHEMA,
    BACKTEST_SCHEMA
)


@dataclass
class ComponentSchema:
    """Schema definition for a component type."""
    name: str
    description: str
    required_fields: List[str]
    optional_fields: Dict[str, Any]  # field_name -> default_value
    field_types: Dict[str, type]
    field_constraints: Dict[str, Dict[str, Any]]  # field_name -> constraints
    examples: List[Dict[str, Any]]
    json_schema: Optional[Dict[str, Any]] = None  # Original JSON schema if available


# Strategy Configuration Schemas (using existing detailed schema)
STRATEGY_SCHEMAS = {
    "strategy_component": ComponentSchema(
        name="strategy_component",
        description="Generic strategy component (supports all strategy types)",
        required_fields=["name", "type"],
        optional_fields={
            "enabled": True,
            "allocation": 1.0,
            "parameters": {},
            "capabilities": []
        },
        field_types={
            "name": str,
            "type": str,
            "enabled": bool,
            "allocation": float,
            "parameters": dict,
            "capabilities": list
        },
        field_constraints={
            "allocation": {"min": 0, "max": 1}
        },
        examples=[
            {
                "name": "momentum_strategy",
                "type": "momentum",
                "allocation": 0.5,
                "parameters": {
                    "fast_period": 10,
                    "slow_period": 30
                }
            }
        ],
        json_schema=STRATEGY_COMPONENT_SCHEMA
    )
}

# Risk Configuration Schemas (using existing detailed schemas)
RISK_SCHEMAS = {
    "risk_limits": ComponentSchema(
        name="risk_limits",
        description="Comprehensive risk limit definitions",
        required_fields=["type"],
        optional_fields={
            "enabled": True,
            "max_position": None,
            "max_exposure_pct": None,
            "max_drawdown_pct": None,
            "reduce_at_pct": None,
            "confidence_level": None,
            "max_var_pct": None,
            "max_position_pct": None,
            "max_sector_pct": None,
            "max_leverage": None,
            "max_daily_loss": None,
            "max_daily_loss_pct": None,
            "allowed_symbols": [],
            "blocked_symbols": []
        },
        field_types={
            "type": str,
            "enabled": bool,
            "max_position": float,
            "max_exposure_pct": float,
            "max_drawdown_pct": float,
            "reduce_at_pct": float,
            "confidence_level": float,
            "max_var_pct": float,
            "max_position_pct": float,
            "max_sector_pct": float,
            "max_leverage": float,
            "max_daily_loss": float,
            "max_daily_loss_pct": float,
            "allowed_symbols": list,
            "blocked_symbols": list
        },
        field_constraints={},
        examples=[
            {
                "type": "position",
                "max_position": 10000,
                "enabled": True
            }
        ],
        json_schema=RISK_LIMIT_SCHEMA
    ),
    
    "position_sizer": ComponentSchema(
        name="position_sizer",
        description="Position sizing algorithms",
        required_fields=["name", "type"],
        optional_fields={
            "size": None,
            "percentage": None,
            "risk_per_trade": None,
            "lookback_period": None,
            "kelly_fraction": None,
            "max_leverage": None,
            "risk_amount": None,
            "atr_multiplier": None
        },
        field_types={
            "name": str,
            "type": str,
            "size": float,
            "percentage": float,
            "risk_per_trade": float,
            "lookback_period": int,
            "kelly_fraction": float,
            "max_leverage": float,
            "risk_amount": float,
            "atr_multiplier": float
        },
        field_constraints={},
        examples=[
            {
                "name": "fixed_sizer",
                "type": "fixed",
                "size": 1000
            }
        ],
        json_schema=POSITION_SIZER_SCHEMA
    )
}

# Data Source Schemas
DATA_SCHEMAS = {
    "csv_file": ComponentSchema(
        name="csv_file",
        description="CSV file data source",
        required_fields=["type", "file_path"],
        optional_fields={
            "symbol_column": "symbol",
            "date_column": "date",
            "ohlcv_columns": {
                "open": "open",
                "high": "high", 
                "low": "low",
                "close": "close",
                "volume": "volume"
            }
        },
        field_types={
            "type": str,
            "file_path": str,
            "symbol_column": str,
            "date_column": str,
            "ohlcv_columns": dict
        },
        field_constraints={},
        examples=[
            {
                "type": "csv_file",
                "file_path": "./data/SPY.csv",
                "symbol_column": "symbol",
                "date_column": "date"
            }
        ]
    )
}

# Execution Schemas
EXECUTION_SCHEMAS = {
    "simulated": ComponentSchema(
        name="simulated",
        description="Simulated execution for backtesting",
        required_fields=["type"],
        optional_fields={
            "slippage": 0.001,
            "commission": 0.005,
            "latency_ms": 0
        },
        field_types={
            "type": str,
            "slippage": float,
            "commission": float,
            "latency_ms": int
        },
        field_constraints={
            "slippage": {"min": 0, "max": 0.1},
            "commission": {"min": 0, "max": 0.1},
            "latency_ms": {"min": 0, "max": 10000}
        },
        examples=[
            {
                "type": "simulated",
                "slippage": 0.001,
                "commission": 0.005
            }
        ]
    )
}

# All schemas registry
ALL_COMPONENT_SCHEMAS = {
    "strategies": STRATEGY_SCHEMAS,
    "risk_profiles": RISK_SCHEMAS,
    "data_sources": DATA_SCHEMAS,
    "execution": EXECUTION_SCHEMAS
}


def get_component_schema(component_type: str, component_name: str) -> Optional[ComponentSchema]:
    """Get schema for a specific component."""
    category_schemas = ALL_COMPONENT_SCHEMAS.get(component_type)
    if category_schemas:
        return category_schemas.get(component_name)
    return None


def get_all_schemas_for_type(component_type: str) -> Dict[str, ComponentSchema]:
    """Get all schemas for a component type."""
    return ALL_COMPONENT_SCHEMAS.get(component_type, {})


def list_component_types() -> List[str]:
    """List all available component types."""
    return list(ALL_COMPONENT_SCHEMAS.keys())


def list_components_for_type(component_type: str) -> List[str]:
    """List all available components for a type."""
    schemas = ALL_COMPONENT_SCHEMAS.get(component_type, {})
    return list(schemas.keys())


def validate_component_config(component_type: str, component_name: str, 
                            config: Dict[str, Any]) -> List[str]:
    """
    Validate a component configuration against its schema.
    
    Returns:
        List of validation errors (empty if valid)
    """
    schema = get_component_schema(component_type, component_name)
    if not schema:
        return [f"Unknown component: {component_type}.{component_name}"]
    
    errors = []
    
    # Check required fields
    for field in schema.required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Check field types and constraints
    for field, value in config.items():
        if field in schema.field_types:
            expected_type = schema.field_types[field]
            if not isinstance(value, expected_type):
                errors.append(f"Field '{field}' must be {expected_type.__name__}, got {type(value).__name__}")
        
        if field in schema.field_constraints:
            constraints = schema.field_constraints[field]
            if 'min' in constraints and value < constraints['min']:
                errors.append(f"Field '{field}' must be >= {constraints['min']}")
            if 'max' in constraints and value > constraints['max']:
                errors.append(f"Field '{field}' must be <= {constraints['max']}")
    
    return errors


# Example usage functions for documentation
def get_example_config(component_type: str, component_name: str) -> Optional[Dict[str, Any]]:
    """Get example configuration for a component."""
    schema = get_component_schema(component_type, component_name)
    if schema and schema.examples:
        return schema.examples[0]
    return None


def generate_documentation() -> str:
    """Generate documentation for all component schemas."""
    docs = []
    docs.append("# Component Configuration Reference\n")
    
    for component_type, schemas in ALL_COMPONENT_SCHEMAS.items():
        docs.append(f"## {component_type.title()}\n")
        
        for name, schema in schemas.items():
            docs.append(f"### {name}")
            docs.append(f"{schema.description}\n")
            
            docs.append("**Required fields:**")
            for field in schema.required_fields:
                field_type = schema.field_types.get(field, "any")
                docs.append(f"- `{field}` ({field_type.__name__ if hasattr(field_type, '__name__') else field_type})")
            
            docs.append("\n**Optional fields:**")
            for field, default in schema.optional_fields.items():
                field_type = schema.field_types.get(field, "any")
                docs.append(f"- `{field}` ({field_type.__name__ if hasattr(field_type, '__name__') else field_type}, default: {default})")
            
            if schema.examples:
                docs.append("\n**Example:**")
                docs.append("```yaml")
                example = schema.examples[0]
                for key, value in example.items():
                    docs.append(f"{key}: {value}")
                docs.append("```\n")
    
    return "\n".join(docs)