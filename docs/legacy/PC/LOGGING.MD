# Infrastructure Services with Protocol + Composition Architecture

## Overview

The infrastructure services (logging, monitoring, error handling, debugging, testing, validation) integrate seamlessly with our Protocol + Composition architecture. Instead of requiring inheritance, these services are added as **capabilities** that components can opt into.

## 1. Infrastructure Protocols

```python
from typing import Protocol, runtime_checkable, Dict, Any, Optional, List
from abc import abstractmethod
from datetime import datetime

# === Logging Protocol ===
@runtime_checkable
class Loggable(Protocol):
    """Protocol for components that support logging"""
    
    @property
    @abstractmethod
    def logger(self) -> 'StructuredLogger':
        """Get component's logger"""
        ...
    
    @abstractmethod
    def log(self, level: str, message: str, **context) -> None:
        """Log a message with context"""
        ...

# === Monitoring Protocols ===
@runtime_checkable
class Monitorable(Protocol):
    """Protocol for components that can be monitored"""
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        ...
    
    @abstractmethod
    def get_health_status(self) -> 'HealthStatus':
        """Get health check status"""
        ...

@runtime_checkable
class Measurable(Protocol):
    """Protocol for components that track performance"""
    
    @abstractmethod
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        ...
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        ...

# === Error Handling Protocol ===
@runtime_checkable
class ErrorAware(Protocol):
    """Protocol for components with error handling"""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle an error, return True if handled"""
        ...
    
    @abstractmethod
    def get_error_policy(self) -> 'ErrorPolicy':
        """Get component's error handling policy"""
        ...

# === Debugging Protocol ===
@runtime_checkable
class Debuggable(Protocol):
    """Protocol for components that support debugging"""
    
    @abstractmethod
    def capture_state(self) -> Dict[str, Any]:
        """Capture current state for debugging"""
        ...
    
    @abstractmethod
    def enable_tracing(self, trace_config: Dict[str, Any]) -> None:
        """Enable execution tracing"""
        ...

# === Validation Protocol ===
@runtime_checkable
class Validatable(Protocol):
    """Protocol for components that can be validated"""
    
    @abstractmethod
    def validate(self) -> 'ValidationResult':
        """Validate component state/configuration"""
        ...
    
    @abstractmethod
    def get_validation_rules(self) -> List['ValidationRule']:
        """Get component's validation rules"""
        ...
```

## 2. Infrastructure Capabilities

### 2.1 Logging Capability

```python
class LoggingCapability(Capability):
    """Adds structured logging to components"""
    
    def get_name(self) -> str:
        return "logging"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, 'logger'):
            # Create logger
            logger_name = spec.get('logger_name', component.__class__.__name__)
            log_level = spec.get('log_level', 'INFO')
            
            component.logger = StructuredLogger(
                name=logger_name,
                level=log_level,
                component_id=spec.get('name', 'unknown')
            )
            
            # Add correlation context support
            component.correlation_id = None
            
            # Add convenience logging methods
            component.log = lambda level, msg, **ctx: component.logger.log(
                level, msg, 
                correlation_id=component.correlation_id,
                **ctx
            )
            
            component.log_info = lambda msg, **ctx: component.log('INFO', msg, **ctx)
            component.log_error = lambda msg, **ctx: component.log('ERROR', msg, **ctx)
            component.log_debug = lambda msg, **ctx: component.log('DEBUG', msg, **ctx)
            component.log_warning = lambda msg, **ctx: component.log('WARNING', msg, **ctx)
            
            # Add method tracing if requested
            if spec.get('trace_methods', False):
                self._add_method_tracing(component, spec)
        
        return component
    
    def _add_method_tracing(self, component: Any, spec: Dict[str, Any]) -> None:
        """Add automatic method entry/exit logging"""
        methods_to_trace = spec.get('trace_methods', [])
        if methods_to_trace == True:  # Trace all public methods
            methods_to_trace = [m for m in dir(component) 
                              if not m.startswith('_') and callable(getattr(component, m))]
        
        for method_name in methods_to_trace:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                
                def traced_method(*args, **kwargs):
                    component.log_debug(f"Entering {method_name}", 
                                      args=args, kwargs=kwargs)
                    try:
                        result = original_method(*args, **kwargs)
                        component.log_debug(f"Exiting {method_name}", 
                                          result=result)
                        return result
                    except Exception as e:
                        component.log_error(f"Error in {method_name}", 
                                          error=str(e), error_type=type(e).__name__)
                        raise
                
                setattr(component, method_name, traced_method)
```

### 2.2 Monitoring Capability

```python
class MonitoringCapability(Capability):
    """Adds monitoring and metrics to components"""
    
    def get_name(self) -> str:
        return "monitoring"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
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
        
        return component
    
    def _instrument_methods(self, component: Any, methods: List[str]) -> None:
        """Add performance tracking to methods"""
        for method_name in methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                
                @component.performance_tracker.track_method(method_name)
                def tracked_method(*args, **kwargs):
                    return original_method(*args, **kwargs)
                
                setattr(component, method_name, tracked_method)
    
    def _create_health_check(self, component: Any, spec: Dict[str, Any]) -> 'HealthCheck':
        """Create appropriate health check for component"""
        health_config = spec.get('health_check', {})
        
        return ComponentHealthCheck(
            component=component,
            checks=health_config.get('checks', ['state', 'metrics']),
            thresholds=health_config.get('thresholds', {})
        )
```

### 2.3 Error Handling Capability

```python
class ErrorHandlingCapability(Capability):
    """Adds robust error handling to components"""
    
    def get_name(self) -> str:
        return "error_handling"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Create error policy
        error_config = spec.get('error_handling', {})
        component.error_policy = ErrorPolicy(
            retry_config=error_config.get('retry', {}),
            fallback_strategy=error_config.get('fallback', 'log_and_continue'),
            error_boundaries=error_config.get('boundaries', [])
        )
        
        # Add error boundary creation
        component.create_error_boundary = lambda name=None, **kwargs: ErrorBoundary(
            component_name=spec.get('name', component.__class__.__name__),
            boundary_name=name,
            logger=getattr(component, 'logger', None),
            event_bus=self._get_event_bus(component),
            policy=component.error_policy,
            **kwargs
        )
        
        # Add retry decorator factory
        component.with_retry = lambda **retry_kwargs: retry(
            **{**component.error_policy.retry_config, **retry_kwargs}
        )
        
        # Add error handling method
        def handle_error(error: Exception, context: Dict[str, Any] = None) -> bool:
            context = context or {}
            context['component'] = spec.get('name', component.__class__.__name__)
            
            # Log error if logger available
            if hasattr(component, 'logger'):
                component.log_error("Component error occurred", 
                                  error=str(error),
                                  error_type=type(error).__name__,
                                  context=context)
            
            # Apply error policy
            return component.error_policy.handle(error, context)
        
        component.handle_error = handle_error
        component.get_error_policy = lambda: component.error_policy
        
        # Wrap critical methods with error boundaries
        critical_methods = error_config.get('critical_methods', [])
        self._wrap_critical_methods(component, critical_methods)
        
        return component
    
    def _wrap_critical_methods(self, component: Any, methods: List[str]) -> None:
        """Wrap methods with error boundaries"""
        for method_name in methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                
                def wrapped_method(*args, **kwargs):
                    with component.create_error_boundary(f"{method_name}_boundary"):
                        return original_method(*args, **kwargs)
                
                setattr(component, method_name, wrapped_method)
    
    def _get_event_bus(self, component: Any) -> Optional[Any]:
        """Get event bus from component if available"""
        if hasattr(component, '_events') and hasattr(component._events, 'event_bus'):
            return component._events.event_bus
        return None
```

### 2.4 Debugging Capability

```python
class DebuggingCapability(Capability):
    """Adds debugging support to components"""
    
    def get_name(self) -> str:
        return "debugging"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        debug_config = spec.get('debugging', {})
        
        # Create debug context
        component.debug_context = DebugContext(
            component_name=spec.get('name', component.__class__.__name__),
            trace_enabled=debug_config.get('trace_enabled', False),
            capture_state_on_error=debug_config.get('capture_state_on_error', True)
        )
        
        # Add state capture
        def capture_state() -> Dict[str, Any]:
            state = {
                'timestamp': datetime.now(),
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
            
            return state
        
        component.capture_state = capture_state
        
        # Add execution tracing
        def enable_tracing(trace_config: Dict[str, Any] = None) -> None:
            trace_config = trace_config or {}
            component.debug_context.trace_enabled = True
            component.debug_context.trace_config = trace_config
            
            # Add execution tracer if needed
            if not hasattr(component, 'execution_tracer'):
                component.execution_tracer = ExecutionTracer(
                    component_name=spec.get('name'),
                    **trace_config
                )
        
        component.enable_tracing = enable_tracing
        
        # Add breakpoint support
        component.set_breakpoint = lambda condition=None: component.debug_context.set_breakpoint(condition)
        
        # Auto-enable tracing if configured
        if debug_config.get('auto_trace', False):
            component.enable_tracing(debug_config.get('trace_config', {}))
        
        return component
    
    def _is_serializable(self, value: Any) -> bool:
        """Check if value can be safely serialized"""
        try:
            import json
            json.dumps(value)
            return True
        except:
            return False
```

### 2.5 Validation Capability

```python
class ValidationCapability(Capability):
    """Adds validation support to components"""
    
    def get_name(self) -> str:
        return "validation"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        validation_config = spec.get('validation', {})
        
        # Create validator
        component.validator = ComponentValidator(
            component_name=spec.get('name', component.__class__.__name__),
            rules=self._create_validation_rules(component, validation_config)
        )
        
        # Add validation method
        component.validate = component.validator.validate
        component.get_validation_rules = lambda: component.validator.rules
        
        # Add configuration validation if component is configurable
        if hasattr(component, 'configure'):
            original_configure = component.configure
            
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
                post_config_result = component.validate()
                if not post_config_result.is_valid:
                    raise ValueError(f"Invalid state after configuration: {post_config_result.errors}")
            
            component.configure = validated_configure
        
        # Add auto-validation hooks if requested
        if validation_config.get('validate_on_lifecycle', False):
            self._add_lifecycle_validation(component)
        
        return component
    
    def _create_validation_rules(self, component: Any, config: Dict[str, Any]) -> List['ValidationRule']:
        """Create validation rules for component"""
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
            elif rule_type == 'method':
                rules.append(MethodValidationRule(
                    method=rule_config['method'],
                    expected=rule_config.get('expected'),
                    message=rule_config.get('message')
                ))
        
        # Add default rules based on component capabilities
        if hasattr(component, '_lifecycle'):
            rules.append(LifecycleValidationRule())
        
        if hasattr(component, '_events'):
            rules.append(EventSystemValidationRule())
        
        return rules
    
    def _add_lifecycle_validation(self, component: Any) -> None:
        """Add validation to lifecycle methods"""
        lifecycle_methods = ['initialize', 'start', 'stop', 'reset']
        
        for method_name in lifecycle_methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                
                def validated_method(*args, **kwargs):
                    # Pre-validation
                    pre_result = component.validate()
                    if not pre_result.is_valid and method_name != 'initialize':
                        raise ValueError(f"Invalid state before {method_name}: {pre_result.errors}")
                    
                    # Call original method
                    result = original_method(*args, **kwargs)
                    
                    # Post-validation
                    post_result = component.validate()
                    if not post_result.is_valid:
                        raise ValueError(f"Invalid state after {method_name}: {post_result.errors}")
                    
                    return result
                
                setattr(component, method_name, validated_method)
```

## 3. Supporting Infrastructure Classes

### 3.1 Structured Logger

```python
class StructuredLogger:
    """Logger that outputs structured JSON logs"""
    
    def __init__(self, name: str, level: str = 'INFO', component_id: str = None):
        self.name = name
        self.level = level
        self.component_id = component_id
        self.correlation_context = threading.local()
        
    def log(self, level: str, message: str, **context) -> None:
        """Log a structured message"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            'component_id': self.component_id,
            'thread': threading.current_thread().name,
            'correlation_id': getattr(self.correlation_context, 'correlation_id', None),
            **context
        }
        
        # Output as JSON (in production, this would go to appropriate handler)
        print(json.dumps(log_entry))
    
    def with_correlation_id(self, correlation_id: str):
        """Context manager for correlation tracking"""
        return CorrelationContext(self, correlation_id)

class CorrelationContext:
    """Context manager for correlation ID tracking"""
    
    def __init__(self, logger: StructuredLogger, correlation_id: str):
        self.logger = logger
        self.correlation_id = correlation_id
        self.previous_id = None
        
    def __enter__(self):
        self.previous_id = getattr(self.logger.correlation_context, 'correlation_id', None)
        self.logger.correlation_context.correlation_id = self.correlation_id
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.correlation_context.correlation_id = self.previous_id
```

### 3.2 Metrics Collector

```python
class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self, component_name: str, tags: Dict[str, str] = None):
        self.component_name = component_name
        self.tags = tags or {}
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self._lock = threading.RLock()
        
    def record_value(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a metric value"""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags={**self.tags, **(tags or {})}
            ))
    
    def record_timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None) -> None:
        """Record a timing metric"""
        self.record_value(f"{name}.duration_ms", duration_ms, tags)
    
    def record_count(self, name: str, count: int = 1, tags: Dict[str, str] = None) -> None:
        """Record a counter metric"""
        self.record_value(f"{name}.count", count, tags)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics with statistics"""
        with self._lock:
            result = {}
            for name, points in self.metrics.items():
                if points:
                    values = [p.value for p in points]
                    result[name] = {
                        'count': len(values),
                        'sum': sum(values),
                        'mean': np.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'p50': np.percentile(values, 50),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99)
                    }
            return result
```

### 3.3 Error Boundary

```python
class ErrorBoundary:
    """Context manager for error handling"""
    
    def __init__(self, component_name: str, boundary_name: str = None,
                 logger: Optional[StructuredLogger] = None,
                 event_bus: Optional[Any] = None,
                 policy: Optional['ErrorPolicy'] = None,
                 reraise: bool = True):
        self.component_name = component_name
        self.boundary_name = boundary_name or 'default'
        self.logger = logger
        self.event_bus = event_bus
        self.policy = policy or ErrorPolicy()
        self.reraise = reraise
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Log error
            if self.logger:
                self.logger.log_error(
                    f"Error in {self.component_name}:{self.boundary_name}",
                    error_type=exc_type.__name__,
                    error_message=str(exc_val),
                    traceback=traceback.format_exc()
                )
            
            # Emit error event
            if self.event_bus:
                self.event_bus.publish(Event(
                    EventType.ERROR,
                    {
                        'component': self.component_name,
                        'boundary': self.boundary_name,
                        'error_type': exc_type.__name__,
                        'error_message': str(exc_val),
                        'timestamp': datetime.now()
                    }
                ))
            
            # Apply error policy
            handled = self.policy.handle(exc_val, {
                'component': self.component_name,
                'boundary': self.boundary_name
            })
            
            # Suppress exception if handled and not reraising
            return handled and not self.reraise
        
        return False
```

## 4. Configuration Examples

### 4.1 Component with All Infrastructure Capabilities

```yaml
components:
  advanced_strategy:
    class: "ComplexTrendStrategy"
    capabilities: 
      - "lifecycle"
      - "events"
      - "logging"
      - "monitoring"
      - "error_handling"
      - "debugging"
      - "validation"
    
    params:
      fast_period: 10
      slow_period: 30
    
    # Logging configuration
    logger_name: "strategies.complex_trend"
    log_level: "DEBUG"
    trace_methods: ["calculate_signal", "update_state"]
    
    # Monitoring configuration
    metric_tags:
      strategy_type: "trend_following"
      version: "2.0"
    track_performance: ["calculate_signal", "on_bar"]
    health_check:
      checks: ["state", "metrics", "performance"]
      thresholds:
        signal_rate_min: 0.1
        error_rate_max: 0.05
    
    # Error handling configuration
    error_handling:
      retry:
        max_attempts: 3
        backoff: "exponential"
        retryable_exceptions: ["NetworkError", "DataError"]
      fallback: "use_previous_signal"
      critical_methods: ["calculate_signal", "execute_trade"]
      boundaries:
        - name: "data_processing"
          methods: ["on_bar", "process_indicators"]
        - name: "signal_generation"
          methods: ["calculate_signal", "validate_signal"]
    
    # Debugging configuration
    debugging:
      trace_enabled: false  # Enable in development
      capture_state_on_error: true
      auto_trace: false
      trace_config:
        include_args: true
        include_results: true
        max_depth: 3
    
    # Validation configuration
    validation:
      validate_on_lifecycle: true
      config_rules:
        fast_period:
          type: "range"
          min: 5
          max: 50
        slow_period:
          type: "range"
          min: 10
          max: 200
          must_be_greater_than: "fast_period"
      rules:
        - type: "state"
          attribute: "position_size"
          validator: "non_negative"
          message: "Position size cannot be negative"
        - type: "method"
          method: "is_ready"
          expected: true
          message: "Strategy must be ready before generating signals"
```

### 4.2 Minimal Component with Selective Capabilities

```yaml
components:
  simple_indicator:
    function: "calculate_sma"
    capabilities: ["logging"]  # Just logging, no overhead
    
    logger_name: "indicators.sma"
    log_level: "INFO"
```

### 4.3 Production Component Profile

```yaml
capability_profiles:
  production_strategy:
    description: "Full production strategy with all infrastructure"
    capabilities: 
      - "lifecycle"
      - "events"
      - "logging"
      - "monitoring"
      - "error_handling"
      - "validation"
    
    defaults:
      log_level: "INFO"
      error_handling:
        retry:
          max_attempts: 3
          backoff: "exponential"
      monitoring:
        health_check:
          checks: ["state", "metrics"]
      validation:
        validate_on_lifecycle: true

# Usage
components:
  my_strategy:
    class: "MyTradingStrategy"
    profile: "production_strategy"  # Gets all production capabilities
    params:
      # strategy-specific params
```

## 5. Benefits of This Approach

### 5.1 Zero Overhead for Simple Components

```python
# Simple function with no infrastructure
def calculate_sma(prices: List[float], period: int) -> float:
    return sum(prices[-period:]) / period

# Can be used directly, no infrastructure overhead
```

### 5.2 Gradual Enhancement

```python
# Start simple
component = SimpleStrategy()

# Add capabilities as needed
component = add_capability(component, LoggingCapability(), {"logger_name": "my_strategy"})
component = add_capability(component, MonitoringCapability(), {"track_performance": ["on_bar"]})
```

### 5.3 Testing Infrastructure

```python
def test_component_with_mocked_infrastructure():
    # Create component with test infrastructure
    component = create_component({
        'class': 'MyStrategy',
        'capabilities': ['logging', 'monitoring'],
        'logger_name': 'test.strategy',
        'log_level': 'DEBUG'
    })
    
    # Infrastructure is automatically available
    assert hasattr(component, 'logger')
    assert hasattr(component, 'metrics_collector')
    
    # Can verify infrastructure interactions
    component.log_info("Test message")
    component.record_metric("test_metric", 42.0)
    
    metrics = component.get_metrics()
    assert 'test_metric' in metrics
```

### 5.4 Runtime Infrastructure Control

```python
# Enable/disable infrastructure at runtime
if config.get('debug_mode'):
    component.enable_tracing({'include_args': True})
    component.logger.level = 'DEBUG'

# Conditional monitoring
if config.get('production_mode'):
    component.health_check.enable()
    component.metrics_collector.start_reporting()
```

## 6. Integration with Container Architecture

The infrastructure capabilities work seamlessly with our container architecture:

```python
class UniversalScopedContainer:
    """Container with infrastructure support"""
    
    def create_component(self, spec: Dict[str, Any]) -> Any:
        """Create component with infrastructure"""
        component = self.component_factory.create_component(spec)
        
        # Infrastructure is added via capabilities
        # No special handling needed!
        
        # All components in container can share infrastructure services
        if 'logging' in spec.get('capabilities', []):
            # Logger automatically gets container context
            component.logger.add_context(container_id=self.container_id)
        
        return component
```

## Summary

By treating infrastructure services as **capabilities** rather than base classes:

1. **Components choose what they need** - no forced inheritance
2. **Zero overhead for simple components** - only pay for what you use
3. **Consistent interface via protocols** - type safety without inheritance
4. **Easy testing** - mock or real infrastructure as needed
5. **Runtime flexibility** - enable/disable features dynamically
6. **Perfect container integration** - infrastructure aware of isolation boundaries

This approach maintains the simplicity and flexibility of the Protocol + Composition architecture while providing enterprise-grade infrastructure support.
