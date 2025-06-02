"""
ContainerContext - Container Awareness Component for Logging System v3
Composable component for managing container-specific logging context
"""

from datetime import datetime
from typing import Dict, Any, Optional
from .protocols import ContainerAware


class ContainerContext:
    """
    Container context - composable component for container awareness.
    
    This component encapsulates container-specific information and metrics
    for logging. Designed for composition to add container awareness to
    any logging component.
    
    Features:
    - Container and component identification
    - Creation time tracking
    - Activity metrics collection
    - Error rate monitoring
    """
    
    def __init__(self, container_id: str, component_name: str):
        """
        Initialize container context.
        
        Args:
            container_id: Unique identifier for the container
            component_name: Name of the component within the container
        """
        self.container_id = container_id
        self.component_name = component_name
        self.creation_time = datetime.utcnow()
        self.metrics = {
            'log_count': 0,
            'error_count': 0,
            'warning_count': 0,
            'last_activity': None,
            'last_error': None
        }
    
    def update_metrics(self, level: str, timestamp: Optional[datetime] = None):
        """
        Update context metrics based on log level.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.metrics['log_count'] += 1
        self.metrics['last_activity'] = timestamp
        
        if level in ['ERROR', 'CRITICAL']:
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = timestamp
        elif level == 'WARNING':
            self.metrics['warning_count'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of container context and metrics.
        
        Returns:
            Dictionary with context information and metrics
        """
        uptime_seconds = (datetime.utcnow() - self.creation_time).total_seconds()
        
        return {
            'container_id': self.container_id,
            'component_name': self.component_name,
            'creation_time': self.creation_time.isoformat(),
            'uptime_seconds': uptime_seconds,
            'uptime_hours': uptime_seconds / 3600,
            'metrics': self.metrics.copy(),
            'error_rate': self._calculate_error_rate(),
            'activity_rate': self._calculate_activity_rate()
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate as percentage of total logs."""
        if self.metrics['log_count'] == 0:
            return 0.0
        return (self.metrics['error_count'] / self.metrics['log_count']) * 100
    
    def _calculate_activity_rate(self) -> float:
        """Calculate logs per hour."""
        uptime_hours = (datetime.utcnow() - self.creation_time).total_seconds() / 3600
        if uptime_hours == 0:
            return 0.0
        return self.metrics['log_count'] / uptime_hours
    
    def reset_metrics(self):
        """Reset metrics counters (useful for periodic reporting)."""
        self.metrics = {
            'log_count': 0,
            'error_count': 0,
            'warning_count': 0,
            'last_activity': None,
            'last_error': None
        }


class EnhancedContainerContext(ContainerContext):
    """
    Enhanced container context with additional tracking capabilities.
    
    This extended version adds performance tracking, resource monitoring,
    and advanced metrics collection for production environments.
    """
    
    def __init__(self, container_id: str, component_name: str, 
                 track_performance: bool = True):
        """
        Initialize enhanced container context.
        
        Args:
            container_id: Unique identifier for the container
            component_name: Name of the component within the container
            track_performance: Whether to track performance metrics
        """
        super().__init__(container_id, component_name)
        self.track_performance = track_performance
        self.performance_metrics = {
            'log_processing_times': [],
            'avg_processing_time_ms': 0.0,
            'max_processing_time_ms': 0.0,
            'memory_usage_samples': []
        }
        self.event_types = {}  # Track different event types
    
    def update_metrics(self, level: str, timestamp: Optional[datetime] = None,
                      processing_time_ms: Optional[float] = None,
                      event_type: Optional[str] = None):
        """
        Update enhanced metrics.
        
        Args:
            level: Log level
            timestamp: Optional timestamp
            processing_time_ms: Time taken to process the log entry
            event_type: Type of event being logged
        """
        super().update_metrics(level, timestamp)
        
        # Track performance if enabled
        if self.track_performance and processing_time_ms is not None:
            self.performance_metrics['log_processing_times'].append(processing_time_ms)
            
            # Keep only last 1000 samples to manage memory
            if len(self.performance_metrics['log_processing_times']) > 1000:
                self.performance_metrics['log_processing_times'] = \
                    self.performance_metrics['log_processing_times'][-1000:]
            
            # Update derived metrics
            times = self.performance_metrics['log_processing_times']
            self.performance_metrics['avg_processing_time_ms'] = sum(times) / len(times)
            self.performance_metrics['max_processing_time_ms'] = max(times)
        
        # Track event types
        if event_type:
            self.event_types[event_type] = self.event_types.get(event_type, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get enhanced summary including performance metrics."""
        summary = super().get_summary()
        
        if self.track_performance:
            summary['performance_metrics'] = self.performance_metrics.copy()
        
        summary['event_types'] = self.event_types.copy()
        summary['total_event_types'] = len(self.event_types)
        
        return summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        if not self.track_performance:
            return {'performance_tracking': False}
        
        times = self.performance_metrics['log_processing_times']
        if not times:
            return {'performance_tracking': True, 'samples': 0}
        
        # Calculate percentiles
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        return {
            'performance_tracking': True,
            'samples': n,
            'avg_processing_time_ms': self.performance_metrics['avg_processing_time_ms'],
            'max_processing_time_ms': self.performance_metrics['max_processing_time_ms'],
            'min_processing_time_ms': min(times),
            'p50_processing_time_ms': sorted_times[n // 2],
            'p90_processing_time_ms': sorted_times[int(n * 0.9)],
            'p95_processing_time_ms': sorted_times[int(n * 0.95)],
            'p99_processing_time_ms': sorted_times[int(n * 0.99)]
        }