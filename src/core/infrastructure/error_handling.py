"""
Error handling infrastructure for ADMF-PC.

Provides error policies, boundaries, and retry mechanisms.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
import traceback
from functools import wraps
from enum import Enum
import random

from ..logging import StructuredLogger
from ..events import Event, EventType


class ErrorStrategy(Enum):
    """Error handling strategies."""
    LOG_AND_CONTINUE = "log_and_continue"
    LOG_AND_FAIL = "log_and_fail"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"


class BackoffStrategy(Enum):
    """Retry backoff strategies."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    JITTER = "jitter"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay_ms: float = 100
    max_delay_ms: float = 60000
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if should retry based on exception and attempt."""
        if attempt >= self.max_attempts:
            return False
        
        if not self.retryable_exceptions:
            # Retry all exceptions if no specific ones configured
            return True
        
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
    
    def get_delay_ms(self, attempt: int) -> float:
        """Calculate delay before next retry."""
        if self.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.base_delay_ms
        
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay_ms * attempt
        
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay_ms * (2 ** (attempt - 1))
        
        elif self.backoff_strategy == BackoffStrategy.JITTER:
            # Exponential with jitter
            delay = self.base_delay_ms * (2 ** (attempt - 1))
            delay = delay * (0.5 + random.random() * 0.5)
        
        else:
            delay = self.base_delay_ms
        
        return min(delay, self.max_delay_ms)


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error: Exception
    component: str
    method: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    attempt: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorPolicy:
    """Defines how errors should be handled."""
    
    def __init__(
        self,
        retry_config: Optional[Dict[str, Any]] = None,
        fallback_strategy: str = "log_and_continue",
        error_boundaries: Optional[List[Dict[str, Any]]] = None,
        circuit_breaker_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize error policy.
        
        Args:
            retry_config: Configuration for retry behavior
            fallback_strategy: Strategy when retries exhausted
            error_boundaries: List of error boundary configurations
            circuit_breaker_config: Circuit breaker configuration
        """
        self.retry_policy = RetryPolicy(**retry_config) if retry_config else RetryPolicy(max_attempts=1)
        self.fallback_strategy = ErrorStrategy(fallback_strategy)
        self.error_boundaries = error_boundaries or []
        self.circuit_breaker_config = circuit_breaker_config
        self._logger = StructuredLogger("ErrorPolicy")
    
    def handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Handle an error according to policy.
        
        Returns:
            True if error was handled, False if it should propagate
        """
        error_context = ErrorContext(
            error=error,
            component=context.get('component', 'unknown'),
            method=context.get('method'),
            metadata=context
        )
        
        # Check if we should retry
        if self.retry_policy.should_retry(error, error_context.attempt):
            delay_ms = self.retry_policy.get_delay_ms(error_context.attempt)
            self._logger.info(
                f"Retrying after error",
                error_type=type(error).__name__,
                attempt=error_context.attempt,
                delay_ms=delay_ms
            )
            time.sleep(delay_ms / 1000)
            return False  # Don't handle, let caller retry
        
        # Apply fallback strategy
        if self.fallback_strategy == ErrorStrategy.LOG_AND_CONTINUE:
            self._logger.error(
                "Error occurred, continuing",
                error_type=type(error).__name__,
                error_message=str(error),
                component=error_context.component
            )
            return True  # Handled
        
        elif self.fallback_strategy == ErrorStrategy.LOG_AND_FAIL:
            self._logger.error(
                "Error occurred, failing",
                error_type=type(error).__name__,
                error_message=str(error),
                component=error_context.component
            )
            return False  # Not handled
        
        return False


class ErrorBoundary:
    """Context manager for error handling."""
    
    def __init__(
        self,
        component_name: str,
        boundary_name: Optional[str] = None,
        logger: Optional[StructuredLogger] = None,
        event_bus: Optional[Any] = None,
        policy: Optional[ErrorPolicy] = None,
        reraise: bool = True,
        fallback_value: Any = None
    ):
        """
        Initialize error boundary.
        
        Args:
            component_name: Name of the component
            boundary_name: Name of this boundary
            logger: Logger for error reporting
            event_bus: Event bus for error events
            policy: Error handling policy
            reraise: Whether to reraise unhandled errors
            fallback_value: Value to return on error (if not reraising)
        """
        self.component_name = component_name
        self.boundary_name = boundary_name or 'default'
        self.logger = logger or StructuredLogger(f"ErrorBoundary.{component_name}")
        self.event_bus = event_bus
        self.policy = policy or ErrorPolicy()
        self.reraise = reraise
        self.fallback_value = fallback_value
        self.error_occurred = False
        self.error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True
            self.error = exc_val
            
            # Log error
            self.logger.error(
                f"Error in {self.component_name}:{self.boundary_name}",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                traceback=traceback.format_exc()
            )
            
            # Emit error event if event bus available
            if self.event_bus:
                try:
                    error_event = Event(
                        event_type=EventType.ERROR,
                        payload={
                            'component': self.component_name,
                            'boundary': self.boundary_name,
                            'error_type': exc_type.__name__,
                            'error_message': str(exc_val),
                            'traceback': traceback.format_exc()
                        }
                    )
                    self.event_bus.publish(error_event)
                except Exception as e:
                    self.logger.error(f"Failed to emit error event: {e}")
            
            # Apply error policy
            context = {
                'component': self.component_name,
                'boundary': self.boundary_name,
                'method': self.boundary_name
            }
            
            handled = self.policy.handle(exc_val, context)
            
            # Return True to suppress exception if handled and not reraising
            return handled and not self.reraise
        
        return False


def retry(
    max_attempts: int = 3,
    backoff: Union[str, BackoffStrategy] = "exponential",
    base_delay_ms: float = 100,
    max_delay_ms: float = 60000,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """
    Decorator for retrying functions.
    
    Args:
        max_attempts: Maximum retry attempts
        backoff: Backoff strategy
        base_delay_ms: Base delay between retries
        max_delay_ms: Maximum delay between retries
        retryable_exceptions: List of exceptions to retry
    """
    if isinstance(backoff, str):
        backoff = BackoffStrategy(backoff)
    
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff_strategy=backoff,
        base_delay_ms=base_delay_ms,
        max_delay_ms=max_delay_ms,
        retryable_exceptions=retryable_exceptions or []
    )
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    if not policy.should_retry(e, attempt):
                        raise
                    
                    if attempt < max_attempts:
                        delay_ms = policy.get_delay_ms(attempt)
                        time.sleep(delay_ms / 1000)
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Optional[Type[Exception]] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch (None = all)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self._logger = StructuredLogger(f"CircuitBreaker.{name}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == self.State.OPEN:
            if self._should_attempt_reset():
                self.state = self.State.HALF_OPEN
                self._logger.info("Circuit entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN for {self.name}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            if self.expected_exception and not isinstance(e, self.expected_exception):
                raise
            
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == self.State.HALF_OPEN:
            self.state = self.State.CLOSED
            self.failure_count = 0
            self._logger.info("Circuit recovered, entering CLOSED state")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = self.State.OPEN
            self._logger.error(
                f"Circuit opened after {self.failure_count} failures",
                threshold=self.failure_threshold
            )
        
        elif self.state == self.State.HALF_OPEN:
            self.state = self.State.OPEN
            self._logger.warning("Circuit reopened after failure in HALF_OPEN state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit state."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'is_open': self.state == self.State.OPEN
        }
    
    def reset(self):
        """Manually reset the circuit."""
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self._logger.info("Circuit manually reset")