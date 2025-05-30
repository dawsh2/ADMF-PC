"""
Infrastructure protocols for ADMF-PC.

These protocols define the interfaces for components that support
various infrastructure capabilities.
"""

from typing import Protocol, runtime_checkable, Dict, Any, Optional, List
from abc import abstractmethod
from datetime import datetime


@runtime_checkable
class Loggable(Protocol):
    """Protocol for components that support logging."""
    
    @property
    @abstractmethod
    def logger(self):
        """Get component's logger."""
        ...
    
    @abstractmethod
    def log(self, level: str, message: str, **context) -> None:
        """Log a message with context."""
        ...


@runtime_checkable
class Monitorable(Protocol):
    """Protocol for components that can be monitored."""
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        ...
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health check status."""
        ...


@runtime_checkable
class Measurable(Protocol):
    """Protocol for components that track performance."""
    
    @abstractmethod
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        ...
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        ...


@runtime_checkable
class ErrorAware(Protocol):
    """Protocol for components with error handling."""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle an error, return True if handled."""
        ...
    
    @abstractmethod
    def get_error_policy(self) -> Dict[str, Any]:
        """Get component's error handling policy."""
        ...


@runtime_checkable
class Debuggable(Protocol):
    """Protocol for components that support debugging."""
    
    @abstractmethod
    def capture_state(self) -> Dict[str, Any]:
        """Capture current state for debugging."""
        ...
    
    @abstractmethod
    def enable_tracing(self, trace_config: Dict[str, Any]) -> None:
        """Enable execution tracing."""
        ...


@runtime_checkable
class Validatable(Protocol):
    """Protocol for components that can be validated."""
    
    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """Validate component state/configuration."""
        ...
    
    @abstractmethod
    def get_validation_rules(self) -> List[Dict[str, Any]]:
        """Get component's validation rules."""
        ...