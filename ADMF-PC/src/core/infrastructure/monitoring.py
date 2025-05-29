"""
Monitoring infrastructure for ADMF-PC.

Provides metrics collection, performance tracking, and health checks.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
from functools import wraps
import numpy as np
from enum import Enum

from ..logging import StructuredLogger


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, component_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Initialize metrics collector.
        
        Args:
            component_name: Name of the component
            tags: Default tags for all metrics
        """
        self.component_name = component_name
        self.tags = tags or {}
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self._lock = threading.RLock()
        self._logger = StructuredLogger(f"MetricsCollector.{component_name}")
    
    def record_value(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags={**self.tags, **(tags or {})}
            ))
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self.record_value(f"{name}.duration_ms", duration_ms, tags)
    
    def record_count(self, name: str, count: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self.record_value(f"{name}.count", count, tags)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics with statistics."""
        with self._lock:
            result = {}
            
            for name, points in self.metrics.items():
                if points:
                    values = [p.value for p in points]
                    result[name] = {
                        'count': len(values),
                        'sum': sum(values),
                        'mean': sum(values) / len(values) if values else 0,
                        'min': min(values),
                        'max': max(values),
                        'p50': self._percentile(values, 50),
                        'p95': self._percentile(values, 95),
                        'p99': self._percentile(values, 99),
                        'latest': values[-1],
                        'latest_timestamp': points[-1].timestamp.isoformat()
                    }
            
            return result
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific metric."""
        all_metrics = self.get_all_metrics()
        return all_metrics.get(name)
    
    def clear_old_metrics(self, max_age: timedelta) -> None:
        """Remove metrics older than max_age."""
        cutoff = datetime.now() - max_age
        
        with self._lock:
            for name in list(self.metrics.keys()):
                self.metrics[name] = [
                    p for p in self.metrics[name]
                    if p.timestamp > cutoff
                ]
                
                if not self.metrics[name]:
                    del self.metrics[name]
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile without numpy."""
        if not values:
            return 0
        
        return np.percentile(values, percentile)


class PerformanceTracker:
    """Tracks performance of methods."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize performance tracker.
        
        Args:
            metrics_collector: Collector to record metrics to
        """
        self.metrics_collector = metrics_collector
        self._call_stack = threading.local()
    
    def track_method(self, method_name: str):
        """Decorator to track method performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Track call depth for nested calls
                if not hasattr(self._call_stack, 'depth'):
                    self._call_stack.depth = 0
                
                self._call_stack.depth += 1
                depth = self._call_stack.depth
                
                start_time = time.perf_counter()
                error_occurred = False
                
                try:
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    error_occurred = True
                    raise
                    
                finally:
                    # Calculate duration
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Record metrics
                    self.metrics_collector.record_timing(
                        f"method.{method_name}",
                        duration_ms,
                        tags={
                            'depth': str(depth),
                            'error': str(error_occurred)
                        }
                    )
                    
                    # Record call count
                    self.metrics_collector.record_count(
                        f"method.{method_name}.calls",
                        tags={'error': str(error_occurred)}
                    )
                    
                    self._call_stack.depth -= 1
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        all_metrics = self.metrics_collector.get_all_metrics()
        
        # Extract method performance stats
        method_stats = {}
        for name, stats in all_metrics.items():
            if name.startswith('method.') and '.duration_ms' in name:
                method_name = name.replace('method.', '').replace('.duration_ms', '')
                method_stats[method_name] = {
                    'avg_duration_ms': stats['mean'],
                    'min_duration_ms': stats['min'],
                    'max_duration_ms': stats['max'],
                    'p95_duration_ms': stats['p95'],
                    'p99_duration_ms': stats['p99'],
                    'call_count': all_metrics.get(f'method.{method_name}.calls.count', {}).get('sum', 0)
                }
        
        return method_stats


class HealthCheck:
    """Base class for health checks."""
    
    def check(self) -> HealthCheckResult:
        """Perform health check."""
        raise NotImplementedError


class ComponentHealthCheck(HealthCheck):
    """Health check for components."""
    
    def __init__(
        self,
        component: Any,
        checks: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize component health check.
        
        Args:
            component: Component to check
            checks: List of check types to perform
            thresholds: Thresholds for various metrics
        """
        self.component = component
        self.checks = checks or ['state', 'metrics']
        self.thresholds = thresholds or {}
        self._logger = StructuredLogger(f"HealthCheck.{getattr(component, 'component_id', 'unknown')}")
    
    def check(self) -> HealthCheckResult:
        """Perform health check."""
        try:
            results = []
            
            # State check
            if 'state' in self.checks:
                state_result = self._check_state()
                results.append(state_result)
            
            # Metrics check
            if 'metrics' in self.checks and hasattr(self.component, 'get_metrics'):
                metrics_result = self._check_metrics()
                results.append(metrics_result)
            
            # Performance check
            if 'performance' in self.checks and hasattr(self.component, 'get_performance_stats'):
                perf_result = self._check_performance()
                results.append(perf_result)
            
            # Aggregate results
            if all(r.status == HealthStatus.HEALTHY for r in results):
                status = HealthStatus.HEALTHY
                message = "All checks passed"
            elif any(r.status == HealthStatus.UNHEALTHY for r in results):
                status = HealthStatus.UNHEALTHY
                unhealthy = [r for r in results if r.status == HealthStatus.UNHEALTHY]
                message = f"Unhealthy: {', '.join(r.message for r in unhealthy)}"
            else:
                status = HealthStatus.DEGRADED
                degraded = [r for r in results if r.status == HealthStatus.DEGRADED]
                message = f"Degraded: {', '.join(r.message for r in degraded)}"
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    'checks': [
                        {
                            'status': r.status.value,
                            'message': r.message,
                            'details': r.details
                        }
                        for r in results
                    ]
                }
            )
            
        except Exception as e:
            self._logger.error("Health check failed", error=str(e))
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Health check error: {str(e)}"
            )
    
    def _check_state(self) -> HealthCheckResult:
        """Check component state."""
        # Check if component has required lifecycle state
        if hasattr(self.component, '_initialized') and hasattr(self.component, '_running'):
            if not self.component._initialized:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Component not initialized"
                )
            
            if hasattr(self.component, '_running') and not self.component._running:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="Component not running"
                )
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="State check passed"
        )
    
    def _check_metrics(self) -> HealthCheckResult:
        """Check component metrics."""
        metrics = self.component.get_metrics()
        
        # Check error rate
        error_rate_threshold = self.thresholds.get('error_rate_max', 0.1)
        error_count = metrics.get('errors.count', {}).get('sum', 0)
        total_count = metrics.get('requests.count', {}).get('sum', 1)  # Avoid division by zero
        
        error_rate = error_count / total_count
        if error_rate > error_rate_threshold:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Error rate {error_rate:.2%} exceeds threshold {error_rate_threshold:.2%}",
                details={'error_rate': error_rate}
            )
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Metrics check passed",
            details={'error_rate': error_rate}
        )
    
    def _check_performance(self) -> HealthCheckResult:
        """Check component performance."""
        perf_stats = self.component.get_performance_stats()
        
        # Check if any method is too slow
        slow_methods = []
        p95_threshold = self.thresholds.get('p95_duration_ms', 1000)
        
        for method, stats in perf_stats.items():
            if stats.get('p95_duration_ms', 0) > p95_threshold:
                slow_methods.append(f"{method} (p95: {stats['p95_duration_ms']:.2f}ms)")
        
        if slow_methods:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message=f"Slow methods: {', '.join(slow_methods)}",
                details={'slow_methods': slow_methods}
            )
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Performance check passed"
        )