"""
Infrastructure capabilities for ADMF-PC.

These capabilities add infrastructure services (logging, monitoring, 
error handling, etc.) to components without requiring inheritance.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
import threading
import time
from datetime import datetime

from ..components import CapabilityEnhancer
from ..logging import StructuredLogger, ContainerLogger, trace_method
# from .monitoring import MetricsCollector, PerformanceTracker, ComponentHealthCheck
# Temporarily disabled for testing without numpy
MetricsCollector = None
PerformanceTracker = None
ComponentHealthCheck = None
from .error_handling import ErrorPolicy, ErrorBoundary, retry, CircuitBreaker
from .validation import ComponentValidator, ConfigValidator, ValidationResult


class LoggingCapability(CapabilityEnhancer):
    """Adds structured logging to components."""
    
    def can_enhance(self, component: Any, capability: str) -> bool:
        return capability == "logging"
    
    def enhance(self, component: Any, context: Dict[str, Any]) -> Any:
        spec = context
        if not hasattr(component, 'logger'):
            # Create logger
            logger_name = spec.get('logger_name', component.__class__.__name__)
            log_level = spec.get('log_level', 'INFO')
            component_id = spec.get('name', 'unknown')
            
            # Use container logger if container_id provided
            container_id = spec.get('container_id')
            if container_id:
                component.logger = ContainerLogger(
                    logger_name,
                    container_id,
                    component_id
                )
            else:
                component.logger = StructuredLogger(
                    name=logger_name,
                    context={'component_id': component_id}
                )
            
            # Set log level
            if hasattr(component.logger, 'logger'):
                component.logger.logger.setLevel(log_level)
            
            # Add correlation context support
            component.correlation_id = None
            
            # Add convenience logging methods
            def log(level: str, msg: str, **ctx):
                extra_context = {'correlation_id': component.correlation_id} if component.correlation_id else {}
                extra_context.update(ctx)
                component.logger.log(level, msg, **extra_context)
            
            component.log = log
            component.log_info = lambda msg, **ctx: component.log('INFO', msg, **ctx)
            component.log_error = lambda msg, **ctx: component.log('ERROR', msg, **ctx)
            component.log_debug = lambda msg, **ctx: component.log('DEBUG', msg, **ctx)
            component.log_warning = lambda msg, **ctx: component.log('WARNING', msg, **ctx)
            
            # Add method tracing if requested
            trace_methods = spec.get('trace_methods', [])
            if trace_methods:
                self._add_method_tracing(component, trace_methods)
            
            component.logger.info(f"Logging capability added to {component_id}")
        
        return component
    
    def _add_method_tracing(self, component: Any, methods: List[str]) -> None:
        """Add automatic method entry/exit logging."""
        if methods == True:  # Trace all public methods
            methods = [m for m in dir(component) 
                      if not m.startswith('_') and callable(getattr(component, m))]
        
        for method_name in methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                if callable(original_method):
                    # Use the trace_method decorator
                    traced = trace_method(name=method_name)(original_method)
                    # Bind logger
                    traced.__self__ = component
                    setattr(component, method_name, traced)


class MonitoringCapability(CapabilityEnhancer):
    """Adds monitoring and metrics to components."""
    
    def can_enhance(self, component: Any, capability: str) -> bool:
        return capability == "monitoring"
    
    def enhance(self, component: Any, context: Dict[str, Any]) -> Any:
        spec = context
        if not hasattr(component, 'metrics_collector'):
            # Create metrics collector
            component.metrics_collector = MetricsCollector(
                component_name=spec.get('name', component.__class__.__name__),
                tags=spec.get('metric_tags', {})
            )
            
            # Add metric recording methods
            component.record_metric = component.metrics_collector.record_value
            component.record_timing = component.metrics_collector.record_timing
            component.increment_counter = component.metrics_collector.record_count
            
            # Add performance tracking
            component.performance_tracker = PerformanceTracker(component.metrics_collector)
            
            # Auto-instrument specified methods
            methods_to_track = spec.get('track_performance', [])
            self._instrument_methods(component, methods_to_track)
            
            # Add health check
            component.health_check = self._create_health_check(component, spec)
            component.get_health_status = component.health_check.check
            
            # Add metrics getter
            component.get_metrics = component.metrics_collector.get_all_metrics
            component.get_performance_stats = component.performance_tracker.get_stats
            
            # Add memory cleanup
            max_metric_age = spec.get('max_metric_age_minutes', 60)
            if max_metric_age:
                from datetime import timedelta
                component.cleanup_old_metrics = lambda: component.metrics_collector.clear_old_metrics(
                    timedelta(minutes=max_metric_age)
                )
            
            if hasattr(component, 'logger'):
                component.logger.info("Monitoring capability added")
        
        return component
    
    def _instrument_methods(self, component: Any, methods: List[str]) -> None:
        """Add performance tracking to methods."""
        for method_name in methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                if callable(original_method):
                    tracked = component.performance_tracker.track_method(method_name)(original_method)
                    setattr(component, method_name, tracked)
    
    def _create_health_check(self, component: Any, spec: Dict[str, Any]) -> ComponentHealthCheck:
        """Create appropriate health check for component."""
        health_config = spec.get('health_check', {})
        
        return ComponentHealthCheck(
            component=component,
            checks=health_config.get('checks', ['state', 'metrics']),
            thresholds=health_config.get('thresholds', {})
        )


class ErrorHandlingCapability(CapabilityEnhancer):
    """Adds robust error handling to components."""
    
    def can_enhance(self, component: Any, capability: str) -> bool:
        return capability == "error_handling"
    
    def enhance(self, component: Any, context: Dict[str, Any]) -> Any:
        spec = context
        # Create error policy
        error_config = spec.get('error_handling', {})
        component.error_policy = ErrorPolicy(
            retry_config=error_config.get('retry', {}),
            fallback_strategy=error_config.get('fallback', 'log_and_continue'),
            error_boundaries=error_config.get('boundaries', []),
            circuit_breaker_config=error_config.get('circuit_breaker')
        )
        
        # Add error boundary creation
        def create_error_boundary(name: Optional[str] = None, **kwargs):
            return ErrorBoundary(
                component_name=spec.get('name', component.__class__.__name__),
                boundary_name=name,
                logger=getattr(component, 'logger', None),
                event_bus=self._get_event_bus(component),
                policy=component.error_policy,
                **kwargs
            )
        
        component.create_error_boundary = create_error_boundary
        
        # Add retry decorator factory
        def with_retry(**retry_kwargs):
            merged_config = {**component.error_policy.retry_policy.__dict__, **retry_kwargs}
            return retry(**merged_config)
        
        component.with_retry = with_retry
        
        # Add error handling method
        def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
            context = context or {}
            context['component'] = spec.get('name', component.__class__.__name__)
            
            # Log error if logger available
            if hasattr(component, 'logger'):
                component.log_error(
                    "Component error occurred", 
                    error=str(error),
                    error_type=type(error).__name__,
                    context=context
                )
            
            # Record error metric if monitoring available
            if hasattr(component, 'increment_counter'):
                component.increment_counter(
                    'errors',
                    tags={'error_type': type(error).__name__}
                )
            
            # Apply error policy
            return component.error_policy.handle(error, context)
        
        component.handle_error = handle_error
        component.get_error_policy = lambda: component.error_policy
        
        # Add circuit breaker if configured
        if error_config.get('circuit_breaker'):
            cb_config = error_config['circuit_breaker']
            component.circuit_breaker = CircuitBreaker(
                name=f"{spec.get('name', 'component')}_circuit",
                failure_threshold=cb_config.get('failure_threshold', 5),
                recovery_timeout=cb_config.get('recovery_timeout', 60.0)
            )
        
        # Wrap critical methods with error boundaries
        critical_methods = error_config.get('critical_methods', [])
        self._wrap_critical_methods(component, critical_methods)
        
        if hasattr(component, 'logger'):
            component.logger.info("Error handling capability added")
        
        return component
    
    def _wrap_critical_methods(self, component: Any, methods: List[str]) -> None:
        """Wrap methods with error boundaries."""
        for method_name in methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                if callable(original_method):
                    @wraps(original_method)
                    def wrapped_method(*args, **kwargs):
                        with component.create_error_boundary(f"{method_name}_boundary"):
                            return original_method(*args, **kwargs)
                    
                    setattr(component, method_name, wrapped_method)
    
    def _get_event_bus(self, component: Any) -> Optional[Any]:
        """Get event bus from component if available."""
        # Check for events capability
        if hasattr(component, 'publish_event'):
            # Component has events capability
            return component
        
        # Check for _events attribute from EventsCapability
        if hasattr(component, '_events') and hasattr(component._events, 'event_bus'):
            return component._events.event_bus
        
        return None


class DebuggingCapability(CapabilityEnhancer):
    """Adds debugging support to components."""
    
    def can_enhance(self, component: Any, capability: str) -> bool:
        return capability == "debugging"
    
    def enhance(self, component: Any, context: Dict[str, Any]) -> Any:
        spec = context
        debug_config = spec.get('debugging', {})
        
        # Create debug context
        component.debug_enabled = debug_config.get('enabled', False)
        component.trace_enabled = debug_config.get('trace_enabled', False)
        component.capture_state_on_error = debug_config.get('capture_state_on_error', True)
        
        # Add state capture
        def capture_state() -> Dict[str, Any]:
            state = {
                'timestamp': datetime.now().isoformat(),
                'component_name': spec.get('name'),
                'component_type': component.__class__.__name__
            }
            
            # Capture public attributes
            for attr in dir(component):
                if not attr.startswith('_'):
                    try:
                        value = getattr(component, attr)
                        if not callable(value) and self._is_serializable(value):
                            state[attr] = value
                    except:
                        pass
            
            # Add metrics if available
            if hasattr(component, 'get_metrics'):
                state['metrics'] = component.get_metrics()
            
            # Add health status if available
            if hasattr(component, 'get_health_status'):
                health = component.get_health_status()
                state['health'] = {
                    'status': health.status.value,
                    'message': health.message
                }
            
            return state
        
        component.capture_state = capture_state
        
        # Add execution tracing
        def enable_tracing(trace_config: Optional[Dict[str, Any]] = None) -> None:
            trace_config = trace_config or {}
            component.trace_enabled = True
            
            if hasattr(component, 'logger'):
                component.logger.info("Tracing enabled", config=trace_config)
            
            # Add tracing to all public methods
            if trace_config.get('trace_all_methods', False):
                for attr in dir(component):
                    if not attr.startswith('_'):
                        method = getattr(component, attr)
                        if callable(method):
                            self._add_tracing(component, attr, trace_config)
        
        component.enable_tracing = enable_tracing
        
        # Add conditional breakpoint support
        component._breakpoints = {}
        
        def set_breakpoint(location: str, condition: Optional[Callable[[], bool]] = None) -> None:
            component._breakpoints[location] = condition
            if hasattr(component, 'logger'):
                component.logger.debug(f"Breakpoint set at {location}")
        
        def check_breakpoint(location: str) -> bool:
            if location in component._breakpoints:
                condition = component._breakpoints[location]
                if condition is None or condition():
                    if hasattr(component, 'logger'):
                        component.logger.warning(
                            f"Breakpoint hit at {location}",
                            state=component.capture_state() if component.capture_state_on_error else None
                        )
                    return True
            return False
        
        component.set_breakpoint = set_breakpoint
        component.check_breakpoint = check_breakpoint
        
        # Auto-enable tracing if configured
        if debug_config.get('auto_trace', False):
            component.enable_tracing(debug_config.get('trace_config', {}))
        
        if hasattr(component, 'logger'):
            component.logger.info("Debugging capability added")
        
        return component
    
    def _is_serializable(self, value: Any) -> bool:
        """Check if value can be safely serialized."""
        try:
            import json
            json.dumps(value)
            return True
        except:
            # Try string representation
            try:
                str(value)
                return True
            except:
                return False
    
    def _add_tracing(self, component: Any, method_name: str, config: Dict[str, Any]) -> None:
        """Add tracing to a method."""
        original_method = getattr(component, method_name)
        
        @wraps(original_method)
        def traced_method(*args, **kwargs):
            if component.trace_enabled:
                trace_id = f"{method_name}_{threading.get_ident()}_{time.time()}"
                
                if hasattr(component, 'logger'):
                    component.logger.trace(
                        f"Entering {method_name}",
                        trace_id=trace_id,
                        args=args if config.get('include_args', False) else None,
                        kwargs=kwargs if config.get('include_args', False) else None
                    )
                
                start_time = time.perf_counter()
                
                try:
                    result = original_method(*args, **kwargs)
                    
                    if hasattr(component, 'logger'):
                        component.logger.trace(
                            f"Exiting {method_name}",
                            trace_id=trace_id,
                            duration_ms=(time.perf_counter() - start_time) * 1000,
                            result=result if config.get('include_result', False) else None
                        )
                    
                    return result
                    
                except Exception as e:
                    if hasattr(component, 'logger'):
                        component.logger.trace(
                            f"Error in {method_name}",
                            trace_id=trace_id,
                            duration_ms=(time.perf_counter() - start_time) * 1000,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                    raise
            else:
                return original_method(*args, **kwargs)
        
        setattr(component, method_name, traced_method)


class ValidationCapability(CapabilityEnhancer):
    """Adds validation support to components."""
    
    def can_enhance(self, component: Any, capability: str) -> bool:
        return capability == "validation"
    
    def enhance(self, component: Any, context: Dict[str, Any]) -> Any:
        spec = context
        validation_config = spec.get('validation', {})
        
        # Create validator
        component.validator = ComponentValidator(
            component_name=spec.get('name', component.__class__.__name__),
            rules=self._create_validation_rules(component, validation_config)
        )
        
        # Add validation method
        component.validate = component.validator.validate
        component.get_validation_rules = lambda: component.validator.rules
        
        # Add configuration validation if component has configure method
        if hasattr(component, 'configure'):
            original_configure = component.configure
            
            @wraps(original_configure)
            def validated_configure(config: Dict[str, Any]) -> None:
                # Validate configuration
                config_rules = validation_config.get('config_rules', {})
                if config_rules:
                    config_validator = ConfigValidator(config_rules)
                    validation_result = config_validator.validate(config)
                    if not validation_result.is_valid:
                        raise ValueError(f"Invalid configuration: {validation_result.errors}")
                
                # Call original configure
                original_configure(config)
                
                # Validate post-configuration state
                post_config_result = component.validate(component)
                if not post_config_result.is_valid:
                    raise ValueError(f"Invalid state after configuration: {post_config_result.errors}")
            
            component.configure = validated_configure
        
        # Add auto-validation hooks if requested
        if validation_config.get('validate_on_lifecycle', False):
            self._add_lifecycle_validation(component)
        
        if hasattr(component, 'logger'):
            component.logger.info("Validation capability added")
        
        return component
    
    def _create_validation_rules(self, component: Any, config: Dict[str, Any]) -> List[ValidationRule]:
        """Create validation rules for component."""
        from .validation import StateValidationRule, RangeValidationRule, DependencyValidationRule
        
        rules = []
        
        # Add custom rules from config
        for rule_config in config.get('rules', []):
            rule_type = rule_config['type']
            
            if rule_type == 'state':
                rules.append(StateValidationRule(
                    attribute=rule_config['attribute'],
                    validator=rule_config['validator'],
                    message=rule_config.get('message')
                ))
            
            elif rule_type == 'range':
                rules.append(RangeValidationRule(
                    attribute=rule_config['attribute'],
                    min_value=rule_config.get('min'),
                    max_value=rule_config.get('max'),
                    message=rule_config.get('message')
                ))
            
            elif rule_type == 'dependency':
                rules.append(DependencyValidationRule(
                    attribute=rule_config['attribute'],
                    depends_on=rule_config['depends_on'],
                    condition=rule_config['condition'],
                    message=rule_config.get('message')
                ))
        
        return rules
    
    def _add_lifecycle_validation(self, component: Any) -> None:
        """Add validation to lifecycle methods."""
        lifecycle_methods = ['initialize', 'start', 'stop', 'reset']
        
        for method_name in lifecycle_methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                
                @wraps(original_method)
                def validated_method(*args, **kwargs):
                    # Pre-validation (skip for initialize)
                    if method_name != 'initialize':
                        pre_result = component.validate(component)
                        if not pre_result.is_valid:
                            raise ValueError(f"Invalid state before {method_name}: {pre_result.errors}")
                    
                    # Call original method
                    result = original_method(*args, **kwargs)
                    
                    # Post-validation
                    post_result = component.validate(component)
                    if not post_result.is_valid:
                        raise ValueError(f"Invalid state after {method_name}: {post_result.errors}")
                    
                    return result
                
                setattr(component, method_name, validated_method)