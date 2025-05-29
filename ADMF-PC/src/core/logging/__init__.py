"""
Logging infrastructure for ADMF-PC.

This package provides structured logging with JSON output, correlation
tracking, and container-aware logging capabilities.

Example Usage:
    ```python
    # Basic logging
    from src.core.logging import StructuredLogger, setup_logging
    
    setup_logging(level="INFO", json_format=True)
    logger = StructuredLogger("my_component")
    logger.info("Component started", version="1.0")
    
    # Container-aware logging
    from src.core.logging import ContainerLogger
    
    logger = ContainerLogger(
        "strategy",
        container_id="backtest_001",
        component_id="trend_strategy"
    )
    logger.info("Strategy initialized")
    
    # Correlation tracking
    with logger.correlation_context("operation_123"):
        logger.info("Starting operation")
        # All logs in this block have correlation_id="operation_123"
        logger.info("Operation complete")
    
    # Method tracing
    from src.core.logging import trace_method
    
    class MyStrategy:
        def __init__(self):
            self._logger = StructuredLogger("MyStrategy")
        
        @trace_method(include_args=True, include_result=True)
        def calculate_signal(self, data):
            return {"action": "buy"}
    ```
"""

from .structured import (
    LogLevel,
    LogContext,
    StructuredFormatter,
    StructuredLogger,
    ContainerLogger,
    trace_method,
    setup_logging,
    logger
)


__all__ = [
    # Log levels
    "LogLevel",
    
    # Core classes
    "LogContext",
    "StructuredFormatter",
    "StructuredLogger",
    "ContainerLogger",
    
    # Decorators and utilities
    "trace_method",
    "setup_logging",
    
    # Module logger
    "logger"
]