"""
CorrelationTracker - Cross-Boundary Tracking Component for Logging System v3
Composable component for correlation tracking across containers and components
"""

import threading
from typing import Optional, Dict, List, Set
from datetime import datetime
from .protocols import CorrelationAware


class CorrelationTracker:
    """
    Correlation tracking - composable component for cross-boundary tracking.
    
    This component manages correlation IDs for tracing requests/signals across
    multiple containers and components. Uses thread-local storage for
    concurrent operations.
    
    Features:
    - Thread-local correlation ID storage
    - Correlation chain tracking
    - History management with cleanup
    - Context manager support
    """
    
    def __init__(self):
        """Initialize correlation tracker."""
        self.context = threading.local()
        self.correlation_history: Dict[str, List[str]] = {}
        self.correlation_metadata: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._max_history_size = 10000  # Prevent memory growth
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """
        Set correlation ID for current thread.
        
        Args:
            correlation_id: Correlation ID to set
        """
        self.context.correlation_id = correlation_id
        
        # Initialize correlation metadata
        with self._lock:
            if correlation_id not in self.correlation_metadata:
                self.correlation_metadata[correlation_id] = {
                    'created_at': datetime.utcnow(),
                    'thread_id': threading.get_ident(),
                    'event_count': 0
                }
    
    def get_correlation_id(self) -> Optional[str]:
        """
        Get correlation ID for current thread.
        
        Returns:
            Current correlation ID or None if not set
        """
        return getattr(self.context, 'correlation_id', None)
    
    def track_event(self, event_id: str, component_info: Optional[str] = None):
        """
        Track event in correlation chain.
        
        Args:
            event_id: Unique event identifier
            component_info: Optional component information
        """
        correlation_id = self.get_correlation_id()
        if correlation_id:
            with self._lock:
                # Initialize chain if needed
                if correlation_id not in self.correlation_history:
                    self.correlation_history[correlation_id] = []
                
                # Add event to chain
                event_entry = event_id
                if component_info:
                    event_entry = f"{event_id}@{component_info}"
                
                self.correlation_history[correlation_id].append(event_entry)
                
                # Update metadata
                if correlation_id in self.correlation_metadata:
                    self.correlation_metadata[correlation_id]['event_count'] += 1
                    self.correlation_metadata[correlation_id]['last_event'] = datetime.utcnow()
                
                # Cleanup if history gets too large
                self._cleanup_history_if_needed()
    
    def get_correlation_chain(self, correlation_id: str) -> List[str]:
        """
        Get full event chain for correlation ID.
        
        Args:
            correlation_id: Correlation ID to look up
            
        Returns:
            List of event IDs in the correlation chain
        """
        with self._lock:
            return self.correlation_history.get(correlation_id, []).copy()
    
    def get_correlation_metadata(self, correlation_id: str) -> Optional[Dict]:
        """
        Get metadata for correlation ID.
        
        Args:
            correlation_id: Correlation ID to look up
            
        Returns:
            Metadata dictionary or None if not found
        """
        with self._lock:
            return self.correlation_metadata.get(correlation_id, {}).copy()
    
    def get_active_correlations(self) -> Set[str]:
        """
        Get set of all active correlation IDs.
        
        Returns:
            Set of correlation IDs currently being tracked
        """
        with self._lock:
            return set(self.correlation_history.keys())
    
    def cleanup_correlation(self, correlation_id: str):
        """
        Clean up tracking data for a correlation ID.
        
        Args:
            correlation_id: Correlation ID to clean up
        """
        with self._lock:
            self.correlation_history.pop(correlation_id, None)
            self.correlation_metadata.pop(correlation_id, None)
    
    def _cleanup_history_if_needed(self):
        """Clean up old correlation history if it gets too large."""
        if len(self.correlation_history) > self._max_history_size:
            # Remove oldest correlations (by creation time)
            correlations_by_age = sorted(
                self.correlation_metadata.items(),
                key=lambda x: x[1].get('created_at', datetime.min)
            )
            
            # Remove oldest 20% to make room
            to_remove = len(correlations_by_age) // 5
            for correlation_id, _ in correlations_by_age[:to_remove]:
                self.correlation_history.pop(correlation_id, None)
                self.correlation_metadata.pop(correlation_id, None)
    
    def get_statistics(self) -> Dict:
        """
        Get correlation tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        with self._lock:
            total_correlations = len(self.correlation_history)
            total_events = sum(len(chain) for chain in self.correlation_history.values())
            
            if self.correlation_metadata:
                avg_events_per_correlation = total_events / total_correlations if total_correlations > 0 else 0
                oldest_correlation = min(
                    meta.get('created_at', datetime.utcnow()) 
                    for meta in self.correlation_metadata.values()
                )
                tracking_duration = (datetime.utcnow() - oldest_correlation).total_seconds()
            else:
                avg_events_per_correlation = 0
                tracking_duration = 0
            
            return {
                'total_correlations': total_correlations,
                'total_events': total_events,
                'avg_events_per_correlation': avg_events_per_correlation,
                'tracking_duration_seconds': tracking_duration,
                'max_history_size': self._max_history_size
            }


class CorrelationContext:
    """Context manager for correlation tracking."""
    
    def __init__(self, tracker: CorrelationTracker, correlation_id: str):
        """
        Initialize correlation context.
        
        Args:
            tracker: CorrelationTracker instance
            correlation_id: Correlation ID to set
        """
        self.tracker = tracker
        self.correlation_id = correlation_id
        self.previous_id = None
    
    def __enter__(self):
        """Enter correlation context."""
        self.previous_id = self.tracker.get_correlation_id()
        self.tracker.set_correlation_id(self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit correlation context."""
        if self.previous_id:
            self.tracker.set_correlation_id(self.previous_id)
        else:
            # Clear correlation if there was no previous one
            if hasattr(self.tracker.context, 'correlation_id'):
                delattr(self.tracker.context, 'correlation_id')


class EnhancedCorrelationTracker(CorrelationTracker):
    """
    Enhanced correlation tracker with advanced features.
    
    This extended version adds parent-child correlation relationships,
    correlation metrics, and advanced query capabilities.
    """
    
    def __init__(self):
        """Initialize enhanced correlation tracker."""
        super().__init__()
        self.parent_child_relationships: Dict[str, str] = {}  # child -> parent
        self.correlation_metrics: Dict[str, Dict] = {}
    
    def create_child_correlation(self, parent_correlation_id: str, child_correlation_id: str):
        """
        Create parent-child correlation relationship.
        
        Args:
            parent_correlation_id: Parent correlation ID
            child_correlation_id: Child correlation ID
        """
        with self._lock:
            self.parent_child_relationships[child_correlation_id] = parent_correlation_id
            
            # Initialize child correlation
            self.set_correlation_id(child_correlation_id)
    
    def get_correlation_tree(self, root_correlation_id: str) -> Dict:
        """
        Get full correlation tree starting from root.
        
        Args:
            root_correlation_id: Root correlation ID
            
        Returns:
            Dictionary representing the correlation tree
        """
        with self._lock:
            tree = {
                'correlation_id': root_correlation_id,
                'events': self.get_correlation_chain(root_correlation_id),
                'metadata': self.get_correlation_metadata(root_correlation_id),
                'children': []
            }
            
            # Find children
            children = [
                child_id for child_id, parent_id in self.parent_child_relationships.items()
                if parent_id == root_correlation_id
            ]
            
            # Recursively build child trees
            for child_id in children:
                tree['children'].append(self.get_correlation_tree(child_id))
            
            return tree
    
    def track_event_with_timing(self, event_id: str, processing_time_ms: float,
                               component_info: Optional[str] = None):
        """
        Track event with timing information.
        
        Args:
            event_id: Unique event identifier
            processing_time_ms: Time taken to process the event
            component_info: Optional component information
        """
        correlation_id = self.get_correlation_id()
        if correlation_id:
            # Track the event
            self.track_event(event_id, component_info)
            
            # Track timing metrics
            with self._lock:
                if correlation_id not in self.correlation_metrics:
                    self.correlation_metrics[correlation_id] = {
                        'total_processing_time_ms': 0,
                        'event_timings': []
                    }
                
                self.correlation_metrics[correlation_id]['total_processing_time_ms'] += processing_time_ms
                self.correlation_metrics[correlation_id]['event_timings'].append({
                    'event_id': event_id,
                    'processing_time_ms': processing_time_ms,
                    'timestamp': datetime.utcnow().isoformat()
                })
    
    def get_correlation_performance(self, correlation_id: str) -> Dict:
        """
        Get performance metrics for a correlation.
        
        Args:
            correlation_id: Correlation ID to analyze
            
        Returns:
            Performance metrics dictionary
        """
        with self._lock:
            metrics = self.correlation_metrics.get(correlation_id, {})
            
            if not metrics:
                return {'correlation_id': correlation_id, 'no_metrics': True}
            
            timings = metrics.get('event_timings', [])
            
            return {
                'correlation_id': correlation_id,
                'total_processing_time_ms': metrics.get('total_processing_time_ms', 0),
                'event_count': len(timings),
                'avg_event_time_ms': metrics['total_processing_time_ms'] / len(timings) if timings else 0,
                'max_event_time_ms': max(t['processing_time_ms'] for t in timings) if timings else 0,
                'min_event_time_ms': min(t['processing_time_ms'] for t in timings) if timings else 0
            }