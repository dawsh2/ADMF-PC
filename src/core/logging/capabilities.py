"""
Logging Capabilities - Universal Component Enhancement for Logging System v3
Composable capabilities that can be added to ANY component without inheritance
"""

import json
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Callable

from .container_logger import ContainerLogger, ProductionContainerLogger
from .correlation_tracker import CorrelationTracker
from .protocols import Loggable, EventTrackable, Debuggable, CorrelationAware


class LoggingCapability:
    """
    Adds logging capability to any component.
    Pure composition - no inheritance!
    
    This capability can enhance any component (custom classes, external libraries,
    functions) with full logging functionality without requiring inheritance.
    """
    
    @staticmethod
    def add_to_component(component: Any, container_id: str, component_name: str,
                        log_level: str = 'INFO', use_production_logger: bool = False) -> Any:
        """
        Add logging capability to any component.
        
        Args:
            component: Any component to enhance
            container_id: Container identifier
            component_name: Component name
            log_level: Minimum log level
            use_production_logger: Whether to use production-optimized logger
            
        Returns:
            Enhanced component with logging capabilities
        """
        # Create logger and attach it
        if use_production_logger:
            logger = ProductionContainerLogger(container_id, component_name, log_level)
        else:
            logger = ContainerLogger(container_id, component_name, log_level)
            
        component.logger = logger
        
        # Add convenience methods
        component.log = logger.log
        component.log_info = logger.info
        component.log_error = logger.error
        component.log_debug = logger.debug
        component.log_warning = logger.warning
        component.log_trace = getattr(logger, 'trace', logger.debug)  # Fallback
        component.log_critical = logger.critical
        
        # Add correlation tracking
        component.set_correlation_id = logger.set_correlation_id
        component.get_correlation_id = logger.get_correlation_id
        component.with_correlation_id = logger.with_correlation_id
        
        # Add event-specific logging methods
        component.log_event_flow = logger.log_event_flow
        component.log_state_change = logger.log_state_change
        component.log_performance_metric = logger.log_performance_metric
        component.log_validation_result = logger.log_validation_result
        
        # Add trading-specific event logging
        component.log_bar_event = logger.log_bar_event
        component.log_signal_event = logger.log_signal_event
        component.log_order_event = logger.log_order_event
        component.log_fill_event = logger.log_fill_event
        
        # Add lifecycle management
        original_close = getattr(component, 'close', None)
        def enhanced_close():
            logger.close()
            if original_close and callable(original_close):
                original_close()
        component.close = enhanced_close
        
        # Add summary method
        component.get_logging_summary = logger.get_summary
        
        return component


class EventTracingCapability:
    """
    Adds event tracing capability to any component.
    
    This capability enables components to trace events across container
    boundaries for debugging and performance analysis.
    """
    
    @staticmethod
    def add_to_component(component: Any, coordinator_id: str) -> Any:
        """
        Add event tracing capability to any component.
        
        Args:
            component: Any component to enhance
            coordinator_id: Coordinator identifier for tracing
            
        Returns:
            Enhanced component with event tracing capabilities
        """
        # Create event tracer
        from .event_flow_tracer import EventFlowTracer
        tracer = EventFlowTracer(coordinator_id)
        component.event_tracer = tracer
        
        # Add tracing methods
        component.trace_event = tracer.trace_event
        component.trace_internal_event = tracer.trace_internal_event
        component.trace_external_event = tracer.trace_external_event
        
        # Add convenience wrapper for tracing with automatic component info
        def trace_component_event(event_id: str, target: str, event_type: str = "unknown", **context):
            """Trace event from this component."""
            source = getattr(component, 'container_id', 'unknown')
            tracer.trace_external_event(
                event_id=event_id,
                source_container=source,
                target_container=target,
                tier="standard",
                event_type=event_type,
                **context
            )
        
        component.trace_component_event = trace_component_event
        
        return component


class DebuggingCapability:
    """
    Adds debugging capability to any component.
    
    This capability provides state capture, tracing controls, and debugging
    utilities that can be added to any component.
    """
    
    @staticmethod
    def add_to_component(component: Any, enable_auto_tracing: bool = False) -> Any:
        """
        Add debugging capability to any component.
        
        Args:
            component: Any component to enhance
            enable_auto_tracing: Whether to automatically trace method calls
            
        Returns:
            Enhanced component with debugging capabilities
        """
        # Add state capture
        def capture_state() -> Dict[str, Any]:
            """Capture current component state for debugging."""
            state = {
                'timestamp': datetime.utcnow().isoformat(),
                'component_type': type(component).__name__,
                'thread_id': threading.get_ident(),
                'thread_name': threading.current_thread().name
            }
            
            # Capture public attributes
            for attr in dir(component):
                if not attr.startswith('_') and not callable(getattr(component, attr, None)):
                    try:
                        value = getattr(component, attr)
                        # Try to serialize - if it fails, convert to string
                        try:
                            json.dumps(value, default=str)
                            state[attr] = value
                        except:
                            state[attr] = str(value)
                    except:
                        pass
            
            return state
        
        component.capture_state = capture_state
        
        # Add tracing toggle
        component._tracing_enabled = enable_auto_tracing
        component.enable_tracing = lambda enabled: setattr(component, '_tracing_enabled', enabled)
        component.is_tracing_enabled = lambda: getattr(component, '_tracing_enabled', False)
        
        # Add debug logging helper
        def debug_log(message: str, **context):
            """Log debug message if component has logging capability."""
            if hasattr(component, 'log_debug'):
                component.log_debug(f"[DEBUG] {message}", **context)
            else:
                print(f"[DEBUG] {type(component).__name__}: {message}")
        
        component.debug_log = debug_log
        
        # Add method call tracing if auto-tracing is enabled
        if enable_auto_tracing:
            DebuggingCapability._add_method_tracing(component)
        
        return component
    
    @staticmethod
    def _add_method_tracing(component: Any):
        """Add automatic method call tracing to component."""
        original_methods = {}
        
        # Wrap public methods with tracing
        for attr_name in dir(component):
            if not attr_name.startswith('_'):
                attr = getattr(component, attr_name)
                if callable(attr) and not attr_name.startswith(('log_', 'trace_', 'debug_')):
                    original_methods[attr_name] = attr
                    
                    def create_traced_method(method_name, original_method):
                        def traced_method(*args, **kwargs):
                            if component.is_tracing_enabled():
                                start_time = datetime.utcnow()
                                component.debug_log(
                                    f"Entering {method_name}",
                                    method=method_name,
                                    args_count=len(args),
                                    kwargs_count=len(kwargs)
                                )
                                
                                try:
                                    result = original_method(*args, **kwargs)
                                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                                    component.debug_log(
                                        f"Exiting {method_name}",
                                        method=method_name,
                                        duration_ms=duration_ms,
                                        success=True
                                    )
                                    return result
                                except Exception as e:
                                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                                    component.debug_log(
                                        f"Exception in {method_name}",
                                        method=method_name,
                                        duration_ms=duration_ms,
                                        error=str(e),
                                        success=False
                                    )
                                    raise
                            else:
                                return original_method(*args, **kwargs)
                        
                        return traced_method
                    
                    setattr(component, attr_name, create_traced_method(attr_name, attr))


class PerformanceMonitoringCapability:
    """
    Adds performance monitoring capability to any component.
    
    This capability tracks method execution times, call counts, and
    performance metrics for any component.
    """
    
    @staticmethod
    def add_to_component(component: Any, track_all_methods: bool = False) -> Any:
        """
        Add performance monitoring capability to any component.
        
        Args:
            component: Any component to enhance
            track_all_methods: Whether to track all public methods
            
        Returns:
            Enhanced component with performance monitoring
        """
        # Initialize performance tracking
        component._performance_metrics = {
            'method_calls': {},
            'method_timings': {},
            'total_calls': 0,
            'start_time': datetime.utcnow()
        }
        
        def get_performance_metrics() -> Dict[str, Any]:
            """Get performance metrics for this component."""
            metrics = component._performance_metrics.copy()
            uptime = (datetime.utcnow() - metrics['start_time']).total_seconds()
            
            # Calculate derived metrics
            metrics['uptime_seconds'] = uptime
            metrics['calls_per_second'] = metrics['total_calls'] / uptime if uptime > 0 else 0
            
            # Calculate average timings
            avg_timings = {}
            for method, timings in metrics['method_timings'].items():
                if timings:
                    avg_timings[method] = sum(timings) / len(timings)
            metrics['average_timings_ms'] = avg_timings
            
            return metrics
        
        component.get_performance_metrics = get_performance_metrics
        
        def track_method_call(method_name: str, duration_ms: float):
            """Track a method call with timing."""
            metrics = component._performance_metrics
            
            # Update call count
            metrics['method_calls'][method_name] = metrics['method_calls'].get(method_name, 0) + 1
            metrics['total_calls'] += 1
            
            # Update timings (keep last 100 calls to manage memory)
            if method_name not in metrics['method_timings']:
                metrics['method_timings'][method_name] = []
            
            timings = metrics['method_timings'][method_name]
            timings.append(duration_ms)
            if len(timings) > 100:
                metrics['method_timings'][method_name] = timings[-100:]
        
        component.track_method_call = track_method_call
        
        # Add method decorator for manual performance tracking
        def performance_tracked(func):
            """Decorator to track performance of a method."""
            def wrapper(*args, **kwargs):
                start_time = datetime.utcnow()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    component.track_method_call(func.__name__, duration_ms)
                    return result
                except Exception as e:
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    component.track_method_call(func.__name__, duration_ms)
                    raise
            return wrapper
        
        component.performance_tracked = performance_tracked
        
        # Auto-track all methods if requested
        if track_all_methods:
            PerformanceMonitoringCapability._add_auto_tracking(component)
        
        return component
    
    @staticmethod
    def _add_auto_tracking(component: Any):
        """Add automatic performance tracking to all public methods."""
        for attr_name in dir(component):
            if not attr_name.startswith('_') and not attr_name.startswith('get_performance'):
                attr = getattr(component, attr_name)
                if callable(attr):
                    tracked_method = component.performance_tracked(attr)
                    setattr(component, attr_name, tracked_method)


def add_logging_to_any_component(component: Any, container_id: str, component_name: str,
                                 enable_tracing: bool = True, enable_debugging: bool = True,
                                 enable_performance: bool = False, 
                                 use_production_logger: bool = False) -> Any:
    """
    Add comprehensive logging capabilities to ANY component using pure composition.
    No inheritance required!
    
    Args:
        component: Any component to enhance
        container_id: Container identifier
        component_name: Component name
        enable_tracing: Whether to add event tracing capability
        enable_debugging: Whether to add debugging capability
        enable_performance: Whether to add performance monitoring
        use_production_logger: Whether to use production-optimized logger
        
    Returns:
        Enhanced component with full logging capabilities
    """
    # Add logging capability
    component = LoggingCapability.add_to_component(
        component, container_id, component_name, 
        use_production_logger=use_production_logger
    )
    
    # Add event tracing capability
    if enable_tracing:
        component = EventTracingCapability.add_to_component(component, "main_coordinator")
    
    # Add debugging capability
    if enable_debugging:
        component = DebuggingCapability.add_to_component(component)
    
    # Add performance monitoring capability
    if enable_performance:
        component = PerformanceMonitoringCapability.add_to_component(component)
    
    return component


# Convenience functions for specific use cases
def enhance_strategy_component(strategy: Any, container_id: str) -> Any:
    """Enhance a strategy component with trading-specific logging."""
    return add_logging_to_any_component(
        strategy, container_id, "strategy",
        enable_tracing=True,
        enable_debugging=True,
        enable_performance=True,
        use_production_logger=True
    )


def enhance_data_component(data_component: Any, container_id: str) -> Any:
    """Enhance a data component with high-performance logging."""
    return add_logging_to_any_component(
        data_component, container_id, "data_handler",
        enable_tracing=True,
        enable_debugging=False,  # Skip debugging for performance
        enable_performance=True,
        use_production_logger=True
    )


def enhance_external_library(library_component: Any, container_id: str, component_name: str) -> Any:
    """Enhance an external library component with basic logging."""
    return add_logging_to_any_component(
        library_component, container_id, component_name,
        enable_tracing=False,  # Keep it simple for external libs
        enable_debugging=False,
        enable_performance=False,
        use_production_logger=False
    )