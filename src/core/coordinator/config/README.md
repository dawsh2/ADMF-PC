# Configuration Schema System

This directory contains the unified configuration schema system for ADMF-PC components.

## Overview

The schema system provides:
- **Preserves existing detailed schemas** from schemas.py
- **Enhanced organization** with ComponentSchema wrapper
- **Automatic validation** of user configurations  
- **Documentation generation** for configuration options
- **Type safety** and constraint enforcement
- **Backward compatibility** with existing JSON schemas

## Files

- `schemas.py` - **EXISTING** detailed JSON schemas (PRESERVED)
- `component_schemas.py` - Enhanced organization of existing schemas + new ones
- `validator.py` - Configuration validation using schemas
- `pattern_loader.py` - YAML pattern loading for workflows
- `resolver.py` - Template and reference resolution

## Schema Architecture

### **Existing Schemas (PRESERVED)**
The system preserves all existing detailed schemas from `schemas.py`:
- `STRATEGY_COMPONENT_SCHEMA` - Comprehensive strategy definitions
- `RISK_LIMIT_SCHEMA` - Detailed risk limit configurations  
- `POSITION_SIZER_SCHEMA` - Position sizing algorithms
- `BACKTEST_SCHEMA` - Complete backtest workflow schemas

### **Enhanced Organization** 
`component_schemas.py` wraps existing schemas with better organization:
- Provides unified ComponentSchema interface
- Adds validation helpers and documentation
- Maintains full backward compatibility with existing JSON schemas

## Adding a New Component Type

When you add a new component type that users can configure, follow these steps:

### 1. Define the Component Schema

Add your schema to `component_schemas.py`:

```python
# In component_schemas.py

NEW_COMPONENT_SCHEMAS = {
    "my_new_component": ComponentSchema(
        name="my_new_component",
        description="Description of what this component does",
        required_fields=["type", "required_param"],
        optional_fields={
            "optional_param": "default_value",
            "another_param": 42
        },
        field_types={
            "type": str,
            "required_param": str,
            "optional_param": str,
            "another_param": int
        },
        field_constraints={
            "another_param": {"min": 1, "max": 100}
        },
        examples=[
            {
                "type": "my_new_component",
                "required_param": "example_value",
                "optional_param": "custom_value"
            }
        ]
    )
}

# Add to the ALL_COMPONENT_SCHEMAS registry
ALL_COMPONENT_SCHEMAS = {
    # ... existing schemas ...
    "my_component_type": NEW_COMPONENT_SCHEMAS
}
```

### 2. Update Discovery System (if needed)

If your component uses the discovery system, add a decorator in `src/core/components/discovery.py`:

```python
def my_component_type(
    name: Optional[str] = None,
    config_schema: Optional[Dict[str, Any]] = None,
    **metadata
):
    """Decorator for registering new component type."""
    def decorator(func_or_class):
        # Registration logic
        return func_or_class
    return decorator
```

### 3. Update Factory (if needed)

If your component needs factory support, update `src/core/components/factory.py`:

```python
def create_my_component_type(component_type: str, config: Dict[str, Any]):
    """Factory function for new component type."""
    # Component creation logic
    pass
```

### 4. Add Validation Integration

The validator will automatically pick up your schema, but you may want to add specific validation in `validator.py`:

```python
def _validate_my_component_type(self, components: List[Dict], errors: List[str], warnings: List[str]):
    """Validate my component type configurations."""
    for i, component in enumerate(components):
        component_type = component.get('type')
        if component_type:
            validation_errors = validate_component_config('my_component_type', component_type, component)
            for error in validation_errors:
                errors.append(f"Component {i} ({component_type}): {error}")
```

## Example: Adding a New Classifier Type

Let's say you want to add a new "sentiment_classifier" component:

```python
# 1. Add to component_schemas.py
CLASSIFIER_SCHEMAS = {
    "sentiment_classifier": ComponentSchema(
        name="sentiment_classifier", 
        description="Sentiment-based market classification",
        required_fields=["type", "data_source"],
        optional_fields={
            "sentiment_threshold": 0.7,
            "lookback_days": 5,
            "weight": 1.0
        },
        field_types={
            "type": str,
            "data_source": str,
            "sentiment_threshold": float,
            "lookback_days": int,
            "weight": float
        },
        field_constraints={
            "sentiment_threshold": {"min": 0.0, "max": 1.0},
            "lookback_days": {"min": 1, "max": 30},
            "weight": {"min": 0.0, "max": 10.0}
        },
        examples=[
            {
                "type": "sentiment_classifier",
                "data_source": "twitter_feed",
                "sentiment_threshold": 0.8,
                "lookback_days": 3
            }
        ]
    )
}

# Add to ALL_COMPONENT_SCHEMAS
ALL_COMPONENT_SCHEMAS = {
    # ... existing ...
    "classifiers": CLASSIFIER_SCHEMAS  # or merge with existing classifiers
}
```

```python
# 2. Add decorator (if using discovery)
@classifier(name="sentiment_classifier")  
def sentiment_classifier_func(market_data, sentiment_data, params):
    # Implementation
    pass
```

```python
# 3. Users can now configure it:
classifiers:
  - type: sentiment_classifier
    data_source: twitter_feed
    sentiment_threshold: 0.8
    lookback_days: 3
```

## Schema Validation

The system automatically validates:
- **Required fields** are present
- **Field types** are correct  
- **Constraints** are satisfied (min/max values, choices, etc.)
- **Unknown fields** are flagged

## Documentation Generation

Generate documentation for all schemas:

```python
from src.core.coordinator.config import generate_documentation

docs = generate_documentation()
print(docs)  # Markdown documentation for all component types
```

## Best Practices

1. **Keep schemas simple** - Only validate what's necessary
2. **Provide good examples** - Users rely on these heavily
3. **Use descriptive field names** - Self-documenting is better
4. **Set sensible defaults** - Minimize required configuration
5. **Add constraints** - Prevent impossible values early
6. **Update tests** - Add validation tests for new schemas

## Integration Points

The schema system integrates with:
- **Workflow validation** (`validator.py`)
- **Component discovery** (`src/core/components/discovery.py`)
- **Factory systems** (`src/core/components/factory.py`)
- **Configuration resolution** (`resolver.py`)
- **Pattern loading** (`pattern_loader.py`)

This provides a unified, type-safe way to handle all user-configurable components in ADMF-PC.