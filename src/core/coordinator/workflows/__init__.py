"""
Modular workflow management system.

This module provides a clean, modular approach to workflow management with:
- Pattern detection and configuration building
- Pluggable execution strategies
- Multi-parameter workflow support
- Communication pattern management

ARCHITECTURE: Modular components with clear separation of concerns:
- Pattern detection → config/pattern_detector.py
- Configuration building → config/config_builders.py
- Execution strategies → execution/*.py
- Communication patterns → patterns/communication_patterns.py
- Main orchestration → topology.py
"""

# Canonical workflow manager moved to coordinator level
# from ..topology import WorkflowManager  # Available at coordinator level
from .config import PatternDetector, ParameterAnalyzer, ConfigBuilder
from .execution import ExecutionStrategy, get_executor
from .patterns import get_communication_config

# Import specific execution strategies for direct access
try:
    from .execution.standard_executor import StandardExecutor
except ImportError:
    # StandardExecutor not yet implemented - use placeholder
    StandardExecutor = None

# Re-export result type
from ...types.workflow import WorkflowResult

# Core factory functions (for direct factory access)
from ...containers.factory import (
    get_global_factory,
    get_global_registry,
    compose_pattern,
    register_container_type,
    ContainerPattern
)

from ...communication.factory import AdapterFactory

__all__ = [
    # Main workflow management - canonical implementation (moved to coordinator level)
    # 'WorkflowManager',  # Available at coordinator level
    'WorkflowResult',
    
    # Configuration and analysis
    'PatternDetector',
    'ParameterAnalyzer', 
    'ConfigBuilder',
    
    # Execution strategies
    'ExecutionStrategy',
    'get_executor',
    'StandardExecutor',
    
    # Pattern management
    'get_communication_config',
    
    # Core factory access (for advanced usage)
    'get_global_factory',
    'get_global_registry', 
    'compose_pattern',
    'register_container_type',
    'ContainerPattern',
    'AdapterFactory',
]

# Usage Examples:
#
# EASIEST: Use convenience functions
# ```python
# from .workflows import execute_workflow
# result = await execute_workflow('simple_backtest', config)
# ```
#
# RECOMMENDED: Use unified factory
# ```python
# from .workflows import create_workflow
# manager = create_workflow('simple_backtest', config)
# result = await manager.execute_workflow(workflow_config)
# ```
#
# ADVANCED: Use WorkflowManager directly  
# ```python
# from .workflows import WorkflowManager
# manager = WorkflowManager()
# result = await manager.execute(workflow_config)
# ```
#
# EXPERT: Direct factory access for custom patterns
# ```python
# from .workflows import get_global_factory, AdapterFactory
# container_factory = get_global_factory()
# containers = container_factory.compose_pattern('simple_backtest', config)
# 
# comm_factory = AdapterFactory()  
# adapters = comm_factory.create_adapters_from_config(comm_config, containers)
# ```
#
# DEPRECATED: Legacy workflow classes  
# ```python
# from .workflows import BacktestWorkflow  # Use WorkflowManager instead
# ```