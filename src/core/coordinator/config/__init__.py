"""
Coordinator Configuration Utilities

Consolidated utilities for pattern loading, configuration resolution, and validation.
"""

from .pattern_loader import PatternLoader
from .resolver import ConfigResolver
from .validator import ConfigValidator, ValidationResult, validate_workflow, check_required_fields
from .clean_syntax_parser import CleanSyntaxParser, parse_clean_config
from .component_schemas import (
    ComponentSchema,
    get_component_schema,
    get_all_schemas_for_type,
    list_component_types,
    list_components_for_type,
    validate_component_config,
    get_example_config,
    generate_documentation
)

# Pydantic models (preferred for new code)
try:
    from .models import (
        WorkflowConfig,
        DataConfig,
        PortfolioConfig,
        StrategyConfig,
        RiskConfig,
        RiskLimitConfig,
        PositionSizerConfig,
        ExecutionConfig,
        validate_workflow_dict,
        get_validation_errors,
        validate_partial_config,
        generate_schema_docs,
        get_example_config as get_pydantic_example
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    WorkflowConfig = None
    DataConfig = None
    PortfolioConfig = None
    StrategyConfig = None
    RiskConfig = None
    RiskLimitConfig = None
    PositionSizerConfig = None
    ExecutionConfig = None
    validate_workflow_dict = None
    get_validation_errors = None
    validate_partial_config = None
    generate_schema_docs = None
    get_pydantic_example = None

__all__ = [
    'PatternLoader',
    'ConfigResolver', 
    'ConfigValidator',
    'ValidationResult',
    'validate_workflow',
    'check_required_fields',
    'CleanSyntaxParser',
    'parse_clean_config',
    # Legacy component schemas
    'ComponentSchema',
    'get_component_schema',
    'get_all_schemas_for_type', 
    'list_component_types',
    'list_components_for_type',
    'validate_component_config',
    'get_example_config',
    'generate_documentation',
    # Pydantic models (if available)
    'PYDANTIC_AVAILABLE',
    'WorkflowConfig',
    'DataConfig',
    'PortfolioConfig', 
    'StrategyConfig',
    'RiskConfig',
    'RiskLimitConfig',
    'PositionSizerConfig',
    'ExecutionConfig',
    'validate_workflow_dict',
    'get_validation_errors',
    'validate_partial_config',
    'generate_schema_docs',
    'get_pydantic_example'
]