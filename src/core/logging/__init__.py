"""
File: src/core/logging/__init__.py
Status: ACTIVE
Version: 1.0.0
Architecture Ref: COMPLEXITY_CHECKLIST.md#step0-documentation-infrastructure
Dependencies: structured.py
Last Review: 2025-05-31
Next Review: 2025-08-31

Purpose: Public API for ADMF-PC structured logging infrastructure as specified
in COMPLEXITY_CHECKLIST.md Step 0. Provides unified interface for JSON-structured
logging, container-aware logging, correlation tracking, and ComponentLogger
pattern essential for event flow validation and system observability.

Key Concepts:
- Structured logging with JSON output for analysis (COMPLEXITY_CHECKLIST.md#logging)
- Container-aware logging for multi-container isolation validation
- ComponentLogger pattern for event flow validation and debugging
- Correlation tracking for request tracing across component boundaries
- Method tracing for automated performance analysis

Critical Dependencies:
- Implements ComponentLogger pattern required for Step 0+ validation
- Provides logging foundation for all system observability and debugging
- Enables container isolation validation and event flow tracking

Example Usage:
    ```python
    # Basic structured logging
    from src.core.logging import StructuredLogger, setup_logging
    
    setup_logging(level="INFO", json_format=True)
    logger = StructuredLogger("my_component")
    logger.info("Component started", version="1.0")
    
    # Container-aware logging for isolation validation
    from src.core.logging import ContainerLogger
    
    logger = ContainerLogger(
        "strategy",
        container_id="backtest_001",
        component_id="trend_strategy"
    )
    logger.info("Strategy initialized")
    
    # ComponentLogger pattern for event flow validation
    logger.log_event_flow("SIGNAL", "Strategy", "RiskContainer", "BUY SPY")
    logger.log_state_change("IDLE", "PROCESSING", "SIGNAL_EVENT_123")
    
    # Correlation tracking for request tracing
    with logger.correlation_context("operation_123"):
        logger.info("Starting operation")
        # All logs in this block have correlation_id="operation_123"
        logger.info("Operation complete")
    
    # Method tracing for performance analysis
    from src.core.logging import trace_method
    
    class MyStrategy:
        def __init__(self):
            self._logger = StructuredLogger("MyStrategy")
        
        @trace_method(include_args=True, include_result=True)
        def calculate_signal(self, data):
            return {"action": "buy", "strength": 0.8}
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