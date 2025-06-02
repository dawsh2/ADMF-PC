# Logging Standards

Comprehensive logging standards for ADMF-PC to ensure consistent, structured, and useful logging across all components.

## Core Logging Principles

1. **Structured Logging**: Use consistent formats for machine parsing
2. **Contextual Information**: Include container ID, component name, and state
3. **Event Flow Tracking**: Log all event publishing and handling
4. **Performance Metrics**: Track key performance indicators
5. **Error Context**: Include full context for debugging

## Logging Infrastructure

### Component Logger Setup

Every component MUST initialize logging:

```python
from src.core.logging import ComponentLogger

class Component:
    def __init__(self, config: Dict[str, Any], container_id: str):
        # Required: Create logger with context
        self.logger = ComponentLogger(
            component_name=self.__class__.__name__,
            container_id=container_id
        )
        
        # Required: Log initialization
        self.logger.log_state_change("created", "initializing", "constructor")
```

### Logger Interface

```python
class ComponentLogger:
    """Standard logger interface for all components"""
    
    def log_event_flow(
        self,
        event_type: str,
        source: str,
        destination: str,
        payload_summary: str
    ) -> None:
        """Log event flow between components"""
    
    def log_state_change(
        self,
        old_state: str,
        new_state: str,
        trigger: str
    ) -> None:
        """Log component state transitions"""
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        context: Dict[str, Any]
    ) -> None:
        """Log performance measurements"""
    
    def log_validation_result(
        self,
        test_name: str,
        passed: bool,
        details: str
    ) -> None:
        """Log validation test results"""
```

## Logging Formats

### Event Flow Format

```python
# Standard format for event flow logging
"EVENT_FLOW | {container_id} | {source} → {destination} | {event_type} | {payload_summary}"

# Example
"EVENT_FLOW | backtest_001 | DataStreamer → IndicatorHub | BAR_DATA | AAPL 2024-01-15 10:30:00"
"EVENT_FLOW | backtest_001 | MomentumStrategy → RiskManager | SIGNAL | BUY AAPL strength=0.8"
```

### State Change Format

```python
# Standard format for state changes
"STATE_CHANGE | {container_id} | {component} | {old_state} → {new_state} | {trigger}"

# Example
"STATE_CHANGE | backtest_001 | MomentumStrategy | initialized → ready | start()"
"STATE_CHANGE | backtest_001 | RiskManager | ready → processing | SIGNAL event"
```

### Performance Metric Format

```python
# Standard format for performance metrics
"PERFORMANCE | {container_id} | {metric_name} | {value} | {context_json}"

# Example
"PERFORMANCE | backtest_001 | signal_generation_time | 0.0023 | {\"strategy\": \"momentum\", \"bars\": 1000}"
"PERFORMANCE | backtest_001 | memory_usage_mb | 45.3 | {\"components\": 5, \"data_points\": 50000}"
```

### Validation Result Format

```python
# Standard format for validation results
"VALIDATION | {container_id} | {test_name} | {PASS|FAIL} | {details}"

# Example
"VALIDATION | backtest_001 | event_isolation | PASS | No event leakage detected"
"VALIDATION | backtest_001 | signal_accuracy | FAIL | Expected 0.65, got 0.45"
```

## What to Log

### Always Log

1. **Component Lifecycle**
```python
def initialize(self, context: Dict[str, Any]) -> None:
    self.logger.log_state_change("created", "initialized", "initialize()")
    
def start(self) -> None:
    self.logger.log_state_change("initialized", "ready", "start()")
    
def stop(self) -> None:
    self.logger.log_state_change("running", "stopped", "stop()")
```

2. **Event Publishing**
```python
def publish_signal(self, signal: Dict[str, Any]) -> None:
    self.event_bus.publish(Event(type="SIGNAL", data=signal))
    
    self.logger.log_event_flow(
        event_type="SIGNAL",
        source=self.component_id,
        destination="RiskManager",
        payload_summary=f"{signal['action']} {signal['symbol']} @ {signal['strength']}"
    )
```

3. **Event Handling**
```python
def handle_bar_data(self, event: Event) -> None:
    self.logger.log_event_flow(
        event_type="BAR_DATA",
        source=event.source,
        destination=self.component_id,
        payload_summary=f"Processing bar for {event.data['symbol']}"
    )
    
    # Process event...
```

4. **Performance Metrics**
```python
def process_batch(self, data: List[Bar]) -> None:
    start_time = time.time()
    
    # Process data...
    
    elapsed = time.time() - start_time
    self.logger.log_performance_metric(
        metric_name="batch_processing_time",
        value=elapsed,
        context={"batch_size": len(data), "avg_per_item": elapsed/len(data)}
    )
```

5. **Errors and Exceptions**
```python
try:
    result = self.process_signal(signal)
except Exception as e:
    self.logger.error(
        f"Signal processing failed | {self.container_id} | "
        f"signal={signal} | error={str(e)} | traceback={traceback.format_exc()}"
    )
    raise
```

### Conditional Logging

1. **Debug Information** (only in debug mode)
```python
if self.logger.isEnabledFor(logging.DEBUG):
    self.logger.debug(
        f"Indicator calculation | {self.container_id} | "
        f"SMA(20)={sma_20} | RSI(14)={rsi_14}"
    )
```

2. **Detailed State** (configurable verbosity)
```python
if self.config.get('verbose_logging', False):
    self.logger.info(
        f"Portfolio state | {self.container_id} | "
        f"positions={len(self.positions)} | value={self.portfolio_value}"
    )
```

## Logging Levels

### Level Guidelines

- **ERROR**: Component failures, unrecoverable errors
- **WARNING**: Degraded performance, non-critical issues
- **INFO**: State changes, event flow, key operations
- **DEBUG**: Detailed calculations, intermediate values

### Examples by Level

```python
# ERROR - Component failure
self.logger.error(f"Failed to connect to data source: {error}")

# WARNING - Degraded operation
self.logger.warning(f"Signal strength {strength} below threshold, skipping")

# INFO - Normal operation
self.logger.info(f"Generated {count} signals in {elapsed:.2f}s")

# DEBUG - Detailed information
self.logger.debug(f"Indicator values: {indicators}")
```

## Performance Logging

### Required Performance Metrics

```python
class PerformanceMetrics:
    """Standard metrics all components should track"""
    
    # Processing time
    PROCESSING_TIME = "processing_time_ms"
    
    # Memory usage
    MEMORY_USAGE_MB = "memory_usage_mb"
    
    # Event counts
    EVENTS_PROCESSED = "events_processed"
    EVENTS_PUBLISHED = "events_published"
    
    # Component specific
    SIGNALS_GENERATED = "signals_generated"
    ORDERS_CREATED = "orders_created"
    FILLS_PROCESSED = "fills_processed"
```

### Performance Logging Example

```python
class Strategy:
    def process_bar(self, bar: Bar) -> None:
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Process bar and generate signal
        signal = self._calculate_signal(bar)
        
        # Log performance
        self.logger.log_performance_metric(
            metric_name=PerformanceMetrics.PROCESSING_TIME,
            value=(time.time() - start_time) * 1000,  # ms
            context={
                "operation": "signal_generation",
                "symbol": bar.symbol
            }
        )
        
        self.logger.log_performance_metric(
            metric_name=PerformanceMetrics.MEMORY_USAGE_MB,
            value=self._get_memory_usage() - start_memory,
            context={"operation": "signal_generation"}
        )
```

## Event Flow Logging

### Complete Event Lifecycle

```python
class EventFlowLogger:
    """Track complete event lifecycle"""
    
    def log_event_lifecycle(self, event: Event) -> None:
        # 1. Event creation
        self.logger.log_event_flow(
            event_type=event.type,
            source=event.source,
            destination="EventBus",
            payload_summary=f"Event created: {event.id}"
        )
        
        # 2. Event routing
        for subscriber in self.get_subscribers(event.type):
            self.logger.log_event_flow(
                event_type=event.type,
                source="EventBus",
                destination=subscriber.component_id,
                payload_summary=f"Routing to {subscriber.component_id}"
            )
        
        # 3. Event handling
        # (Logged by receiving component)
        
        # 4. Event completion
        self.logger.log_event_flow(
            event_type=event.type,
            source="EventBus",
            destination="Complete",
            payload_summary=f"Event {event.id} processed by {len(subscribers)} handlers"
        )
```

## Container Isolation Logging

### Isolation Validation Logging

```python
class IsolationLogger:
    """Log container isolation validation"""
    
    def log_isolation_test(self, test_name: str, result: bool, details: Dict[str, Any]):
        self.logger.log_validation_result(
            test_name=f"isolation_{test_name}",
            passed=result,
            details=json.dumps(details)
        )
        
        # Log specific isolation metrics
        self.logger.log_performance_metric(
            metric_name="container_isolation_score",
            value=details.get('isolation_score', 0.0),
            context={
                "test": test_name,
                "container_count": details.get('container_count', 0)
            }
        )
```

## Log Aggregation

### Structured Log Format for Aggregation

```python
import json
import time

class StructuredLogger:
    """Output logs in JSON format for aggregation"""
    
    def log(self, level: str, message: str, **kwargs):
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "container_id": self.container_id,
            "component": self.component_name,
            "message": message,
            **kwargs
        }
        
        # Output as JSON for log aggregation systems
        print(json.dumps(log_entry))
```

### Log Correlation

```python
class CorrelatedLogger:
    """Add correlation IDs for request tracking"""
    
    def __init__(self, component_name: str, container_id: str):
        self.component_name = component_name
        self.container_id = container_id
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking"""
        self.correlation_id = correlation_id
    
    def log(self, level: str, message: str, **kwargs):
        log_data = {
            "correlation_id": self.correlation_id,
            "container_id": self.container_id,
            "component": self.component_name,
            **kwargs
        }
        # Log with correlation context
```

## Testing and Logging

### Test Logging Requirements

```python
def test_component_logging():
    """Verify component logs required events"""
    
    # Capture logs
    with capture_logs() as log_capture:
        component = Component(config, "test_container")
        component.process(test_data)
    
    # Verify required logs
    assert log_capture.contains("STATE_CHANGE", "initialized → ready")
    assert log_capture.contains("EVENT_FLOW", "SIGNAL")
    assert log_capture.contains("PERFORMANCE", "processing_time")
```

### Log-Based Testing

```python
class LogBasedTester:
    """Use logs to verify behavior"""
    
    def verify_event_flow(self, logs: List[str]) -> bool:
        """Verify correct event flow from logs"""
        
        # Extract event flow logs
        event_logs = [
            log for log in logs 
            if "EVENT_FLOW" in log
        ]
        
        # Verify sequence
        expected_flow = [
            "DataStreamer → IndicatorHub",
            "IndicatorHub → Strategy",
            "Strategy → RiskManager",
            "RiskManager → ExecutionEngine"
        ]
        
        return all(
            any(expected in log for log in event_logs)
            for expected in expected_flow
        )
```

## Configuration

### Logging Configuration

```yaml
logging:
  # Global settings
  level: INFO
  format: structured  # or "text"
  
  # Component-specific settings
  components:
    Strategy:
      level: DEBUG
      include_performance: true
      
    RiskManager:
      level: INFO
      include_validation: true
      
    ExecutionEngine:
      level: WARNING
      include_fills: true
  
  # Output configuration
  outputs:
    - type: console
      format: text
      
    - type: file
      path: logs/admf.log
      format: json
      rotation: daily
      
    - type: elasticsearch
      url: http://localhost:9200
      index: admf-logs
```

## Best Practices

### 1. Consistent Context

Always include container ID and component name:

```python
# Good
self.logger.info(f"Processing signal | {self.container_id} | {self.component_id}")

# Bad
logging.info("Processing signal")  # No context!
```

### 2. Meaningful Messages

```python
# Good
self.logger.info(
    f"Generated BUY signal | {self.container_id} | "
    f"symbol={symbol} | strength={strength:.3f} | "
    f"indicators=SMA({sma:.2f}), RSI({rsi:.2f})"
)

# Bad
self.logger.info("Signal generated")  # Not helpful!
```

### 3. Performance Awareness

```python
# Don't log in tight loops
for i in range(1000000):
    # Bad: Logging in tight loop
    self.logger.debug(f"Processing item {i}")
    
# Good: Summary logging
start = time.time()
for i in range(1000000):
    process_item(i)
self.logger.info(f"Processed 1M items in {time.time()-start:.2f}s")
```

### 4. Error Context

```python
# Good: Full context
try:
    result = process_order(order)
except Exception as e:
    self.logger.error(
        f"Order processing failed | {self.container_id} | "
        f"order={order.to_dict()} | "
        f"portfolio_state={self.get_portfolio_summary()} | "
        f"error={str(e)} | "
        f"traceback={traceback.format_exc()}"
    )

# Bad: No context
except Exception as e:
    self.logger.error(f"Error: {e}")  # Not enough info!
```

## Summary

Effective logging in ADMF-PC:

1. **Uses structured formats** for machine parsing
2. **Includes full context** (container, component, state)
3. **Tracks event flow** through the system
4. **Measures performance** at key points
5. **Provides debugging information** when needed
6. **Enables log-based testing** and validation
7. **Supports aggregation** and correlation

Follow these standards to create logs that are useful for debugging, monitoring, and understanding system behavior.