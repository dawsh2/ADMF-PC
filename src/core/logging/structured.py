"""
File: src/core/logging/structured.py
Status: ACTIVE
Version: 1.0.0
Architecture Ref: COMPLEXITY_CHECKLIST.md#step0-documentation-infrastructure
Dependencies: logging, json, datetime, threading, contextlib
Last Review: 2025-05-31
Next Review: 2025-04-30

Purpose: Implements structured logging system for ADMF-PC as specified in
COMPLEXITY_CHECKLIST.md Step 0. Provides JSON-structured output, correlation
tracking, and container-aware logging capabilities essential for event flow
validation and system observability.

Key Concepts:
- JSON-structured log output for parsing and analysis (COMPLEXITY_CHECKLIST.md#logging)
- Correlation ID tracking for request tracing across components
- Container-aware logging with proper context isolation
- ComponentLogger pattern methods for event flow validation
- Thread-local context management for concurrent operations

Critical Dependencies:
- Must support event flow logging for Step 0+ validation requirements
- Provides foundation for all system logging and observability
- Enables container isolation validation and debugging capabilities
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import logging
import json
import datetime
import socket
import threading
from contextlib import contextmanager
from functools import wraps
import traceback
from enum import IntEnum


# Custom log levels
class LogLevel(IntEnum):
    """
    Extended log levels including TRACE for detailed debugging.
    
    This enum extends Python's standard logging levels with TRACE level
    for ultra-detailed debugging during development and troubleshooting.
    Used throughout ADMF-PC for consistent log level management.
    
    Architecture Context:
        - Part of: Structured Logging Infrastructure (COMPLEXITY_CHECKLIST.md#logging)
        - Supports: Event flow validation and debugging workflows
        - Enables: Fine-grained logging control for complex trading systems
    
    Example:
        logger.log(LogLevel.TRACE, "Detailed execution step")
        if logger.isEnabledFor(LogLevel.TRACE):
            # Expensive trace logging only when needed
    """
    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Add TRACE level to logging
logging.addLevelName(LogLevel.TRACE, "TRACE")


@dataclass
class LogContext:
    """
    Context information for structured logging and correlation tracking.
    
    This dataclass encapsulates all contextual information needed for
    structured logging across container boundaries. Essential for event
    flow validation and debugging in complex multi-container systems.
    
    Architecture Context:
        - Part of: Structured Logging Infrastructure (COMPLEXITY_CHECKLIST.md#logging)
        - Enables: Container-aware logging and event flow tracking
        - Supports: Correlation tracking across component boundaries
        - Used by: All components requiring structured logging context
    
    Example:
        context = LogContext(
            correlation_id="trade_123",
            container_id="backtest_001",
            component_id="momentum_strategy"
        )
        logger = StructuredLogger("my_component", context)
    """
    correlation_id: Optional[str] = None
    container_id: Optional[str] = None
    component_id: Optional[str] = None
    execution_mode: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON logs for system observability.
    
    This formatter ensures all logs are in a consistent, parseable JSON format
    suitable for log aggregation, analysis tools, and automated monitoring.
    Essential for ADMF-PC's structured logging infrastructure and debugging.
    
    Architecture Context:
        - Part of: Structured Logging Infrastructure (COMPLEXITY_CHECKLIST.md#logging)
        - Outputs: JSON-structured logs for parsing and analysis
        - Supports: Event flow validation and container isolation debugging
        - Enables: Automated log processing and monitoring dashboards
    
    Example:
        formatter = StructuredFormatter(hostname="trading-node-01")
        handler.setFormatter(formatter)
        # Outputs: {"timestamp": "2025-01-31T10:30:00Z", "level": "INFO", ...}
    """
    
    def __init__(self, hostname: Optional[str] = None):
        super().__init__()
        self.hostname = hostname or socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.
        
        Converts a Python logging record into a structured JSON format
        with consistent fields for timestamp, level, context, and metadata.
        
        Args:
            record: Python logging record to format
            
        Returns:
            JSON-formatted string ready for output and analysis
            
        Example:
            formatter = StructuredFormatter()
            formatted = formatter.format(logging_record)
            # Returns: '{"timestamp": "2025-05-31T10:30:00Z", "level": "INFO", ...}'
        """
        # Base fields
        log_data = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "hostname": self.hostname,
            "thread": threading.current_thread().name,
            "thread_id": threading.get_ident()
        }
        
        # Add location info
        log_data["location"] = {
            "filename": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "module": record.module
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            context = record.context
            if isinstance(context, LogContext):
                if context.correlation_id:
                    log_data["correlation_id"] = context.correlation_id
                if context.container_id:
                    log_data["container_id"] = context.container_id
                if context.component_id:
                    log_data["component_id"] = context.component_id
                if context.execution_mode:
                    log_data["execution_mode"] = context.execution_mode
                if context.extra:
                    log_data["extra"] = context.extra
        
        # Add any extra fields from the record
        if hasattr(record, 'extra_fields'):
            log_data["extra"] = log_data.get("extra", {})
            log_data["extra"].update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, default=str)


class StructuredLogger:
    """
    Logger wrapper that provides structured logging capabilities and ComponentLogger pattern.
    
    This logger ensures consistent structured output, manages context propagation
    for correlation tracking, and implements the ComponentLogger pattern required
    by COMPLEXITY_CHECKLIST.md Step 0 for event flow validation and system observability.
    
    Architecture Context:
        - Part of: Structured Logging Infrastructure (COMPLEXITY_CHECKLIST.md#logging)
        - Implements: ComponentLogger pattern for event flow validation
        - Provides: Container-aware logging with correlation tracking
        - Enables: Multi-container debugging and performance monitoring
        - Supports: Thread-local context management for concurrent operations
    
    Example:
        context = LogContext(container_id="backtest_001", component_id="strategy")
        logger = StructuredLogger("my_component", context)
        logger.info("Component initialized")
        logger.log_event_flow("SIGNAL", "Strategy", "RiskContainer", "BUY SPY")
    """
    
    def __init__(
        self,
        name: str,
        context: Optional[LogContext] = None
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module or component name)
            context: Default logging context
        """
        self.logger = logging.getLogger(name)
        self.context = context or LogContext()
        
        # Thread-local context storage
        self._local = threading.local()
    
    def trace(self, msg: str, **kwargs) -> None:
        """
        Log at TRACE level for detailed debugging.
        
        Args:
            msg: Log message
            **kwargs: Additional context fields
            
        Returns:
            None
            
        Example:
            logger.trace("Entering detailed calculation", step=1, data_size=1000)
        """
        self._log(LogLevel.TRACE, msg, **kwargs)
    
    def debug(self, msg: str, **kwargs) -> None:
        """
        Log at DEBUG level for development debugging.
        
        Args:
            msg: Log message
            **kwargs: Additional context fields
            
        Returns:
            None
            
        Example:
            logger.debug("Processing order", order_id="123", symbol="SPY")
        """
        self._log(LogLevel.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """
        Log at INFO level for general information.
        
        Args:
            msg: Log message
            **kwargs: Additional context fields
            
        Returns:
            None
            
        Example:
            logger.info("Strategy initialized", strategy="momentum", period=20)
        """
        self._log(LogLevel.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """
        Log at WARNING level for concerning conditions.
        
        Args:
            msg: Log message
            **kwargs: Additional context fields
            
        Returns:
            None
            
        Example:
            logger.warning("High memory usage detected", memory_mb=512, threshold=400)
        """
        self._log(LogLevel.WARNING, msg, **kwargs)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        """
        Log at ERROR level for error conditions.
        
        Args:
            msg: Log message
            exc_info: Whether to include exception traceback
            **kwargs: Additional context fields
            
        Returns:
            None
            
        Example:
            logger.error("Order processing failed", order_id="123", exc_info=True)
        """
        self._log(LogLevel.ERROR, msg, exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        """
        Log at CRITICAL level for critical system failures.
        
        Args:
            msg: Log message
            exc_info: Whether to include exception traceback
            **kwargs: Additional context fields
            
        Returns:
            None
            
        Example:
            logger.critical("Database connection lost", db_host="localhost", exc_info=True)
        """
        self._log(LogLevel.CRITICAL, msg, exc_info=exc_info, **kwargs)
    
    def exception(self, msg: str, **kwargs) -> None:
        """
        Log exception with automatic traceback inclusion.
        
        Args:
            msg: Log message
            **kwargs: Additional context fields
            
        Returns:
            None
            
        Example:
            try:
                risky_operation()
            except Exception:
                logger.exception("Operation failed", operation="signal_calc")
        """
        self._log(LogLevel.ERROR, msg, exc_info=True, **kwargs)
    
    @contextmanager
    def correlation_context(self, correlation_id: str):
        """
        Context manager for correlation tracking.
        
        All logs within this context will have the same correlation ID.
        
        Args:
            correlation_id: Correlation ID for tracking
            
        Returns:
            Context manager for correlation tracking
            
        Example:
            with logger.correlation_context("trade_123"):
                logger.info("Starting trade processing")
                process_trade()
                logger.info("Trade processing complete")
        """
        # Save current context
        old_context = getattr(self._local, 'context', None)
        
        # Create new context with correlation ID
        new_context = LogContext(
            correlation_id=correlation_id,
            container_id=self.context.container_id,
            component_id=self.context.component_id,
            execution_mode=self.context.execution_mode,
            extra=self.context.extra.copy()
        )
        
        self._local.context = new_context
        
        try:
            yield
        finally:
            # Restore old context
            if old_context:
                self._local.context = old_context
            else:
                delattr(self._local, 'context')
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """
        Create a new logger with additional context.
        
        Args:
            **kwargs: Additional context fields
            
        Returns:
            New logger with extended context
            
        Example:
            order_logger = logger.bind(order_id="123", symbol="SPY")
            order_logger.info("Processing order")  # Includes order context
        """
        new_context = LogContext(
            correlation_id=self.context.correlation_id,
            container_id=self.context.container_id,
            component_id=self.context.component_id,
            execution_mode=self.context.execution_mode,
            extra={**self.context.extra, **kwargs}
        )
        return StructuredLogger(self.logger.name, new_context)
    
    def _log(
        self,
        level: int,
        msg: str,
        exc_info: bool = False,
        **kwargs
    ) -> None:
        """Internal log method."""
        # Get effective context (thread-local or default)
        context = getattr(self._local, 'context', self.context)
        
        # Create log record with context
        extra = {
            'context': context,
            'extra_fields': kwargs
        }
        
        self.logger.log(level, msg, exc_info=exc_info, extra=extra)
    
    # ComponentLogger pattern methods for COMPLEXITY_CHECKLIST.md compliance
    
    def log_event_flow(self, event_type: str, source: str, destination: str, payload_summary: str) -> None:
        """
        Log event flow for debugging and validation.
        
        This method implements the ComponentLogger pattern required by
        COMPLEXITY_CHECKLIST.md Step 0 for event flow validation and
        container isolation debugging.
        
        Args:
            event_type: Type of event being logged (e.g., 'SIGNAL', 'ORDER', 'FILL')
            source: Source component or container ID
            destination: Destination component or container ID
            payload_summary: Brief summary of event payload
            
        Architecture Context:
            - Required by: COMPLEXITY_CHECKLIST.md#event-flow-validation
            - Enables: Event bus isolation validation
            - Supports: Multi-container debugging and troubleshooting
        
        Returns:
            None
            
        Example:
            logger.log_event_flow("ORDER_EVENT", "RiskContainer", "BacktestBroker", "SPY BUY 100")
        """
        self.info(
            f"EVENT_FLOW | {source} → {destination} | {event_type}",
            event_type=event_type,
            source=source,
            destination=destination,
            payload_summary=payload_summary,
            log_category="event_flow"
        )
    
    def log_state_change(self, old_state: str, new_state: str, trigger: str) -> None:
        """
        Log component state changes for debugging.
        
        This method tracks component state transitions as required by
        COMPLEXITY_CHECKLIST.md Step 0 for system observability and
        debugging support.
        
        Args:
            old_state: Previous component state
            new_state: New component state  
            trigger: What triggered the state change
            
        Architecture Context:
            - Required by: COMPLEXITY_CHECKLIST.md#logging
            - Enables: Component lifecycle debugging
            - Supports: State machine validation and troubleshooting
        
        Returns:
            None
            
        Example:
            logger.log_state_change("WAITING", "PROCESSING_ORDER", "ORDER_EVENT_123")
        """
        self.info(
            f"STATE_CHANGE | {old_state} → {new_state}",
            old_state=old_state,
            new_state=new_state,
            trigger=trigger,
            log_category="state_change"
        )
    
    def log_performance_metric(self, metric_name: str, value: float, context: Dict[str, Any]) -> None:
        """
        Log performance metrics with context for monitoring.
        
        This method records performance metrics as required by
        COMPLEXITY_CHECKLIST.md Step 0 for system monitoring and
        performance analysis.
        
        Args:
            metric_name: Name of the performance metric
            value: Numeric value of the metric
            context: Additional context information for the metric
            
        Architecture Context:
            - Required by: COMPLEXITY_CHECKLIST.md#logging
            - Enables: Performance monitoring and optimization
            - Supports: System health monitoring and alerting
        
        Example:
            logger.log_performance_metric(
                "signal_processing_time_ms", 
                42.3, 
                {"symbol": "SPY", "strategy": "momentum"}
            )
            
        Returns:
            None
        """
        self.info(
            f"PERFORMANCE | {metric_name} | {value}",
            metric_name=metric_name,
            metric_value=value,
            log_category="performance_metric",
            **context
        )
    
    def log_validation_result(self, test_name: str, passed: bool, details: str) -> None:
        """
        Log validation test results for compliance tracking.
        
        This method records validation results as required by
        COMPLEXITY_CHECKLIST.md Step 0 for automated validation
        tracking and compliance reporting.
        
        Args:
            test_name: Name of the validation test
            passed: Whether the test passed or failed
            details: Additional details about the test result
            
        Architecture Context:
            - Required by: COMPLEXITY_CHECKLIST.md#validation-framework
            - Enables: Automated compliance tracking
            - Supports: Validation result aggregation and reporting
        
        Example:
            logger.log_validation_result(
                "container_isolation_test", 
                True, 
                "All 5 containers properly isolated"
            )
            
        Returns:
            None
        """
        level_method = self.info if passed else self.error
        status = "PASS" if passed else "FAIL"
        level_method(
            f"VALIDATION | {test_name} | {status}",
            test_name=test_name,
            passed=passed,
            details=details,
            log_category="validation_result"
        )


def trace_method(
    name: Optional[str] = None,
    include_args: bool = True,
    include_result: bool = False
):
    """
    Decorator for tracing method execution with structured logging.
    
    This decorator provides automatic method entry/exit logging with execution
    timing and optional argument/result logging. Essential for debugging
    complex trading system workflows and performance optimization.
    
    Args:
        name: Optional name override (defaults to method name)
        include_args: Whether to log method arguments
        include_result: Whether to log return value
        
    Returns:
        Decorated function with automatic trace logging
        
    Architecture Context:
        - Part of: Structured Logging Infrastructure (COMPLEXITY_CHECKLIST.md#logging)
        - Enables: Method-level debugging and performance analysis
        - Supports: Automated instrumentation for trading system workflows
    
    Example:
        @trace_method(include_args=True, include_result=True)
        def calculate_signal(self, data):
            return {"action": "buy", "strength": 0.8}
    """
    def decorator(func):
        """
        Decorator function that applies tracing to a method.
        
        Returns:
            Wrapped function with tracing capabilities
            
        Example:
            # Used internally by @trace_method decorator
            decorated_func = decorator(original_function)
        """
        method_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function that executes method with tracing.
            
            Returns:
                Result of the wrapped method execution
                
            Example:
                # Used internally by @trace_method decorator
                result = wrapper(self, arg1, arg2, kwarg1="value")
            """
            # Get logger from self if available
            if args and hasattr(args[0], '_logger'):
                logger = args[0]._logger
            else:
                logger = StructuredLogger(func.__module__)
            
            # Build trace info
            trace_info = {"method": method_name}
            
            if include_args:
                # Skip 'self' for methods
                func_args = args[1:] if args and hasattr(args[0], '__dict__') else args
                trace_info["args"] = func_args
                trace_info["kwargs"] = kwargs
            
            # Log entry
            start_time = datetime.datetime.utcnow()
            logger.trace(f"Entering {method_name}", **trace_info)
            
            try:
                # Execute method
                result = func(*args, **kwargs)
                
                # Log exit
                duration_ms = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000
                exit_info = {
                    "method": method_name,
                    "duration_ms": duration_ms
                }
                
                if include_result:
                    exit_info["result"] = result
                
                logger.trace(f"Exiting {method_name}", **exit_info)
                
                return result
                
            except Exception as e:
                # Log exception
                duration_ms = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000
                logger.error(
                    f"Exception in {method_name}",
                    exc_info=True,
                    method=method_name,
                    duration_ms=duration_ms
                )
                raise
        
        return wrapper
    return decorator


class ContainerLogger(StructuredLogger):
    """
    Logger specifically for container-aware logging and isolation validation.
    
    This logger extends StructuredLogger with container-specific context
    management essential for multi-container architectures. Provides
    automatic container and component ID tracking as required by
    COMPLEXITY_CHECKLIST.md Step 0 for container isolation validation.
    
    Architecture Context:
        - Part of: Structured Logging Infrastructure (COMPLEXITY_CHECKLIST.md#logging)
        - Enables: Container isolation validation and debugging
        - Supports: Multi-container event flow tracking
        - Required for: Container-based architecture patterns
    
    Example:
        logger = ContainerLogger("strategy", "backtest_001", "momentum_strategy")
        logger.info("Strategy processing signal")
        # Output includes container_id and component_id automatically
    """
    
    def __init__(
        self,
        name: str,
        container_id: str,
        component_id: Optional[str] = None
    ):
        """
        Initialize container-aware logger.
        
        Args:
            name: Logger name
            container_id: Container ID
            component_id: Optional component ID
        """
        context = LogContext(
            container_id=container_id,
            component_id=component_id
        )
        super().__init__(name, context)


def setup_logging(
    level: Union[str, int] = logging.INFO,
    console: bool = True,
    file_path: Optional[str] = None,
    json_format: bool = True
) -> None:
    """
    Configure logging for the application with structured output.
    
    Sets up the Python logging system with structured JSON formatting
    and appropriate handlers for console and/or file output. Essential
    for establishing ADMF-PC logging infrastructure.
    
    Args:
        level: Log level (string like "INFO" or logging constant)
        console: Whether to log to console
        file_path: Optional file path for logging
        json_format: Whether to use JSON format
        
    Returns:
        None
        
    Example:
        # Basic setup with JSON console logging
        setup_logging(level="DEBUG", json_format=True)
        
        # Setup with file output
        setup_logging(level="INFO", file_path="logs/trading.log")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


# Create module logger
logger = StructuredLogger(__name__)