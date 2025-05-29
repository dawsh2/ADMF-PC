"""
Structured logging system for ADMF-PC.

This module provides structured logging capabilities with JSON output,
correlation tracking, and container-aware logging.
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
    """Extended log levels including TRACE."""
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
    """Context information for structured logging."""
    correlation_id: Optional[str] = None
    container_id: Optional[str] = None
    component_id: Optional[str] = None
    execution_mode: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON logs.
    
    This formatter ensures all logs are in a consistent, parseable format
    suitable for log aggregation and analysis tools.
    """
    
    def __init__(self, hostname: Optional[str] = None):
        super().__init__()
        self.hostname = hostname or socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
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
    Logger wrapper that provides structured logging capabilities.
    
    This logger ensures consistent structured output and manages
    context propagation for correlation tracking.
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
        """Log at TRACE level."""
        self._log(LogLevel.TRACE, msg, **kwargs)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, msg, **kwargs)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, msg, exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, msg, exc_info=exc_info, **kwargs)
    
    def exception(self, msg: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log(LogLevel.ERROR, msg, exc_info=True, **kwargs)
    
    @contextmanager
    def correlation_context(self, correlation_id: str):
        """
        Context manager for correlation tracking.
        
        All logs within this context will have the same correlation ID.
        
        Args:
            correlation_id: Correlation ID for tracking
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


def trace_method(
    name: Optional[str] = None,
    include_args: bool = True,
    include_result: bool = False
):
    """
    Decorator for tracing method execution.
    
    Logs method entry, exit, execution time, and optionally arguments/results.
    
    Args:
        name: Optional name override (defaults to method name)
        include_args: Whether to log method arguments
        include_result: Whether to log return value
    """
    def decorator(func):
        method_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
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
    """Logger specifically for container-aware logging."""
    
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
    Configure logging for the application.
    
    Args:
        level: Log level
        console: Whether to log to console
        file_path: Optional file path for logging
        json_format: Whether to use JSON format
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