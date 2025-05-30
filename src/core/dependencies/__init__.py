"""
Dependency management system for ADMF-PC.

This package provides dependency injection and resolution for the
containerized architecture, ensuring proper component initialization
order and complete state isolation between containers.

Key Components:
- DependencyGraph: Tracks component dependencies and detects cycles
- DependencyContainer: Manages component instances with DI
- ScopedContainer: Provides isolated execution environments

Example Usage:
    ```python
    # Create a container for a backtest
    container = ScopedContainer("backtest_001")
    
    # Register components
    container.register_type("DataProvider", HistoricalDataProvider)
    container.register_type("Strategy", TrendFollowingStrategy,
                          dependencies=["DataProvider", "EventBus"])
    container.register_type("Portfolio", Portfolio,
                          dependencies=["EventBus"])
    
    # Resolve all components in proper order
    components = container.resolve_all()
    
    # Components are initialized with dependencies injected
    strategy = container.resolve("Strategy")
    # strategy has DataProvider and EventBus already injected
    
    # Teardown when done
    container.teardown()
    ```
"""

from .graph import (
    ComponentNode,
    DependencyGraph,
    DependencyValidator
)

from .container import (
    Registration,
    DependencyContainer,
    ScopedContainer
)


__all__ = [
    # Graph components
    "ComponentNode",
    "DependencyGraph",
    "DependencyValidator",
    
    # Container components
    "Registration",
    "DependencyContainer",
    "ScopedContainer"
]