"""
Infrastructure capabilities for ADMF-PC.

This module provides opt-in infrastructure services (logging, monitoring,
error handling, debugging, validation) as capabilities that can be added
to components without inheritance.

Example Usage:
    ```python
    from src.core.infrastructure import (
        MonitoringCapability,
        ErrorHandlingCapability,
        ValidationCapability
    )
    
    # Create component with infrastructure
    component = factory.create_component({
        'class': 'MyStrategy',
        'capabilities': [
            'lifecycle',
            'events', 
            'logging',
            'monitoring',
            'error_handling'
        ],
        'logger_name': 'strategies.my_strategy',
        'track_performance': ['calculate_signal'],
        'error_handling': {
            'retry': {'max_attempts': 3},
            'critical_methods': ['execute_trade']
        }
    })
    
    # Use infrastructure features
    component.log_info("Strategy started")
    component.record_metric("signal_strength", 0.85)
    
    with component.create_error_boundary("trade_execution"):
        component.execute_trade()
    ```
"""

from .protocols import (
    Loggable,
    Monitorable,
    Measurable,
    ErrorAware,
    Debuggable,
    Validatable
)

from .capabilities import (
    LoggingCapability,
    MonitoringCapability,
    ErrorHandlingCapability,
    DebuggingCapability,
    ValidationCapability
)

from .monitoring import (
    MetricsCollector,
    PerformanceTracker,
    HealthCheck,
    ComponentHealthCheck
)

from .error_handling import (
    ErrorPolicy,
    ErrorBoundary,
    RetryPolicy,
    retry
)

from .validation import (
    ValidationResult,
    ValidationRule,
    ComponentValidator,
    ConfigValidator
)


__all__ = [
    # Protocols
    "Loggable",
    "Monitorable", 
    "Measurable",
    "ErrorAware",
    "Debuggable",
    "Validatable",
    
    # Capabilities
    "LoggingCapability",
    "MonitoringCapability",
    "ErrorHandlingCapability",
    "DebuggingCapability",
    "ValidationCapability",
    
    # Monitoring
    "MetricsCollector",
    "PerformanceTracker",
    "HealthCheck",
    "ComponentHealthCheck",
    
    # Error Handling
    "ErrorPolicy",
    "ErrorBoundary",
    "RetryPolicy",
    "retry",
    
    # Validation
    "ValidationResult",
    "ValidationRule",
    "ComponentValidator",
    "ConfigValidator"
]