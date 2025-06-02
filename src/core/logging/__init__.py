"""
Container-Aware Logging and Debugging System v3
Protocol + Composition based logging with complete lifecycle management

This module provides the new v3 logging system that replaces the previous
structured logging with enhanced container-aware logging, lifecycle management,
and zero-inheritance architecture.

Key Features:
- Protocol + Composition design (zero inheritance)
- Container-isolated logging with lifecycle management
- Cross-container event correlation and flow tracing
- Automatic log rotation, archiving, and cleanup
- Performance optimization for high-throughput environments
- Universal component enhancement (works with any component)

Example Usage:
    ```python
    # Basic container-aware logging
    from src.core.logging import ContainerLogger
    
    logger = ContainerLogger("strategy_001", "momentum_strategy")
    logger.info("Strategy initialized", signal_strength=0.8)
    
    # Add logging to any component (no inheritance!)
    from src.core.logging import add_logging_to_any_component
    
    my_component = MyCustomClass()
    enhanced = add_logging_to_any_component(my_component, "container_001", "my_comp")
    enhanced.log_info("Component enhanced with logging")
    
    # Lifecycle management through coordinator
    from src.core.logging import LogManager
    
    log_manager = LogManager("main_coordinator")
    registry = log_manager.register_container("strategy_001")
    component_logger = registry.create_component_logger("momentum")
    
    # Automatic cleanup when coordinator shuts down
    await log_manager.shutdown()  # All logs cleaned up automatically
    ```
"""

# New v3 Logging System
from .protocols import (
    Loggable,
    EventTrackable, 
    ContainerAware,
    CorrelationAware,
    Debuggable,
    LifecycleManaged
)

from .container_logger import ContainerLogger, ProductionContainerLogger
from .log_manager import LogManager, LogRetentionPolicy, ContainerLogRegistry
from .event_flow_tracer import EventFlowTracer, ContainerDebugger
from .capabilities import (
    LoggingCapability,
    EventTracingCapability,
    DebuggingCapability,
    PerformanceMonitoringCapability,
    add_logging_to_any_component,
    enhance_strategy_component,
    enhance_data_component,
    enhance_external_library
)

# Legacy system for backward compatibility (deprecated)
from .structured import (
    LogLevel,
    LogContext,
    StructuredFormatter,
    StructuredLogger,
    trace_method,
    setup_logging,
    logger as legacy_logger
)

__all__ = [
    # New v3 System - Protocols
    'Loggable',
    'EventTrackable', 
    'ContainerAware',
    'CorrelationAware',
    'Debuggable',
    'LifecycleManaged',
    
    # New v3 System - Core Components
    'ContainerLogger',
    'ProductionContainerLogger',
    'LogManager',
    'LogRetentionPolicy',
    'ContainerLogRegistry',
    'EventFlowTracer',
    'ContainerDebugger',
    
    # New v3 System - Capabilities
    'LoggingCapability',
    'EventTracingCapability',
    'DebuggingCapability',
    'PerformanceMonitoringCapability',
    'add_logging_to_any_component',
    'enhance_strategy_component',
    'enhance_data_component',
    'enhance_external_library',
    
    # Legacy System (deprecated - use v3 system instead)
    'LogLevel',
    'LogContext', 
    'StructuredFormatter',
    'StructuredLogger',
    'trace_method',
    'setup_logging',
    'legacy_logger'
]

# Backward compatibility aliases
ContainerLogger_v1 = StructuredLogger  # Legacy alias
logger = legacy_logger  # Legacy alias