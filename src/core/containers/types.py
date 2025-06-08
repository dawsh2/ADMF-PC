"""
Type definitions for container module.

This module provides type aliases and TypedDict definitions for better
type safety in container operations.
"""

from typing import (
    TypedDict, NotRequired, Dict, Any, List, Union, Callable, 
    Optional, Protocol, runtime_checkable
)
from .protocols import Container, ContainerRole


# Type aliases for clarity
ContainerFactory = Callable[[Dict[str, Any]], Container]
ComponentFactory = Callable[[Dict[str, Any]], 'ContainerComponent']
EventHandler = Callable[['Event'], None]


@runtime_checkable
class ContainerComponent(Protocol):
    """Protocol that all container components should implement."""
    
    def initialize(self, container: Container) -> None:
        """Initialize component with container reference."""
        ...
    
    def cleanup(self) -> None:
        """Clean up component resources."""
        ...


class ContainerConfigDict(TypedDict):
    """Type definition for container configuration."""
    
    # Core settings
    role: ContainerRole
    name: str
    container_id: NotRequired[str]
    
    # Financial settings
    initial_capital: NotRequired[float]
    base_currency: NotRequired[str]
    
    # Event tracing settings
    event_tracing: NotRequired[List[str]]
    enable_event_tracing: NotRequired[bool]
    retention_policy: NotRequired[str]
    sliding_window_size: NotRequired[int]
    
    # Performance settings
    enable_metrics: NotRequired[bool]
    metrics_config: NotRequired[Dict[str, Any]]
    
    # Features and capabilities
    features: NotRequired[List[str]]
    capabilities: NotRequired[List[str]]
    
    # Component configuration
    components: NotRequired[List['ComponentConfigDict']]
    
    # Results handling
    results_storage: NotRequired[str]  # 'memory', 'disk', 'hybrid'
    results_dir: NotRequired[str]
    
    # Execution settings
    execution: NotRequired['ExecutionConfigDict']
    
    # Custom configuration
    config: NotRequired[Dict[str, Any]]


class ComponentConfigDict(TypedDict):
    """Type definition for component configuration."""
    
    type: str  # Component type name
    config: NotRequired[Dict[str, Any]]  # Component-specific config


class ExecutionConfigDict(TypedDict):
    """Type definition for execution configuration."""
    
    enable_event_tracing: NotRequired[bool]
    trace_settings: NotRequired['TraceSettingsDict']
    track_metrics: NotRequired[bool]
    synchronization: NotRequired['SynchronizationConfigDict']


class TraceSettingsDict(TypedDict):
    """Type definition for trace settings."""
    
    trace_dir: NotRequired[str]
    max_events: NotRequired[int]
    trace_specific: NotRequired[List[str]]
    container_settings: NotRequired[Dict[str, 'ContainerTraceSettingsDict']]


class ContainerTraceSettingsDict(TypedDict):
    """Type definition for per-container trace settings."""
    
    enabled: NotRequired[bool]
    max_events: NotRequired[int]
    events_to_trace: NotRequired[List[str]]


class SynchronizationConfigDict(TypedDict):
    """Type definition for synchronization configuration."""
    
    enabled: NotRequired[bool]
    mode: NotRequired[str]  # 'terminal_events', 'phase_barriers', etc.
    timeout_ms: NotRequired[int]


class MetricsConfigDict(TypedDict):
    """Type definition for metrics configuration."""
    
    initial_capital: float
    retention_policy: NotRequired[str]
    max_events: NotRequired[int]
    collection: NotRequired['MetricsCollectionDict']
    objective_function: NotRequired[Dict[str, Any]]
    annualization_factor: NotRequired[float]
    min_periods: NotRequired[int]
    custom_metrics: NotRequired[List[Dict[str, Any]]]


class MetricsCollectionDict(TypedDict):
    """Type definition for metrics collection settings."""
    
    store_equity_curve: NotRequired[bool]
    store_trades: NotRequired[bool]
    store_positions: NotRequired[bool]
    store_orders: NotRequired[bool]


# Pattern configuration types
class ContainerPatternDict(TypedDict):
    """Type definition for container pattern configuration."""
    
    name: str
    description: NotRequired[str]
    structure: Dict[str, Any]  # Hierarchical structure definition
    default_config: NotRequired[Dict[str, Any]]


# Factory registration types
ComponentRegistryEntry = TypedDict('ComponentRegistryEntry', {
    'factory': ComponentFactory,
    'description': NotRequired[str],
    'config_schema': NotRequired[Dict[str, Any]]
})