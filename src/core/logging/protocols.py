"""
Logging Protocols for Container-Aware Logging System v3
Protocol + Composition based design with complete lifecycle management
"""

from typing import Protocol, runtime_checkable, Dict, Any, Optional, Set
from datetime import datetime


@runtime_checkable
class Loggable(Protocol):
    """
    Protocol for anything that can log messages.
    
    This protocol enables any component to gain logging capability through
    composition without inheritance. Core to the zero-inheritance philosophy
    of the logging system.
    """
    
    def log(self, level: str, message: str, **context) -> None:
        """Log a message with optional context"""
        ...


@runtime_checkable 
class EventTrackable(Protocol):
    """
    Protocol for anything that can track event flows across containers.
    
    This protocol enables cross-container event correlation and flow tracing
    essential for debugging multi-container architectures.
    """
    
    def trace_event(self, event_id: str, source: str, target: str, **context) -> None:
        """Trace an event flow between components"""
        ...


@runtime_checkable
class ContainerAware(Protocol):
    """
    Protocol for anything that knows about container context.
    
    This protocol provides container isolation and identification
    capabilities essential for multi-container logging systems.
    """
    
    @property
    def container_id(self) -> str:
        """Get the container ID"""
        ...
    
    @property
    def component_name(self) -> str:
        """Get the component name within the container"""
        ...


@runtime_checkable
class CorrelationAware(Protocol):
    """
    Protocol for anything that can track correlation across boundaries.
    
    This protocol enables end-to-end request/signal tracing across
    multiple containers and components.
    """
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for tracking"""
        ...
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID"""
        ...


@runtime_checkable
class Debuggable(Protocol):
    """
    Protocol for anything that can be debugged.
    
    This protocol provides state capture and debugging capabilities
    that can be added to any component through composition.
    """
    
    def capture_state(self) -> Dict[str, Any]:
        """Capture current component state for debugging"""
        ...
    
    def enable_tracing(self, enabled: bool) -> None:
        """Enable/disable detailed tracing for this component"""
        ...


@runtime_checkable
class LifecycleManaged(Protocol):
    """
    Protocol for anything that can be managed through its lifecycle.
    
    This protocol enables automatic lifecycle management including
    initialization, cleanup, and status monitoring.
    """
    
    def initialize(self) -> None:
        """Initialize the component"""
        ...
    
    def cleanup(self) -> None:
        """Clean up resources"""
        ...
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        ...


@runtime_checkable
class EventScopeClassifier(Protocol):
    """
    Protocol for anything that can classify event communication scopes.
    
    This protocol enables automatic detection of communication patterns
    (internal bus, external tiers) for proper logging classification.
    """
    
    def detect_scope(self, context: Dict[str, Any]) -> str:
        """Detect the communication scope of an event"""
        ...


@runtime_checkable
class LogRetentionAware(Protocol):
    """
    Protocol for anything that can manage log retention policies.
    
    This protocol enables automated log lifecycle management including
    archiving, compression, and cleanup.
    """
    
    def apply_retention_rules(self, base_log_dir: str) -> Dict[str, Any]:
        """Apply log retention policies and return statistics"""
        ...


@runtime_checkable
class PerformanceOptimized(Protocol):
    """
    Protocol for anything that can optimize performance dynamically.
    
    This protocol enables performance optimization features like
    async writing, batching, and adaptive configuration.
    """
    
    def optimize_for_throughput(self) -> None:
        """Optimize for maximum throughput"""
        ...
    
    def optimize_for_latency(self) -> None:
        """Optimize for minimum latency"""
        ...
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        ...