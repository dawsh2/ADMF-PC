# Communication Module

The communication module provides the event communication infrastructure for ADMF-PC, enabling flexible and configurable communication patterns between containers through pluggable adapters.

## Overview

The communication module solves the circular dependency problem that occurs when container hierarchy conflicts with event flow patterns. It separates container organization (configuration) from event communication (runtime behavior).

## Architecture

```
┌─────────────────────────────────────────┐
│    EventCommunicationFactory            │
│    Creates adapters from config         │
├─────────────────────────────────────────┤
│ + create_communication_layer()          │
│ + register_adapter_type()               │
│ + cleanup_all_adapters()                │
└─────────────────────────────────────────┘
                    │ Creates
                    ▼
┌─────────────────────────────────────────┐
│      CommunicationLayer                 │
│    Manages all active adapters          │
├─────────────────────────────────────────┤
│ + add_adapter()                         │
│ + get_system_metrics()                  │
│ + setup_all_adapters()                  │
│ + cleanup()                             │
└─────────────────────────────────────────┘
                    │ Contains
                    ▼
┌─────────────────────────────────────────┐
│      CommunicationAdapter               │
│      (Abstract Base Class)              │
├─────────────────────────────────────────┤
│ + connect() -> bool                     │
│ + disconnect() -> None                  │
│ + send_event(Event) -> bool            │
│ + process_incoming() -> None           │
└─────────────────────────────────────────┘
                    ▲ Inherits
     ┌──────────────┴──────────────┐
     │                             │
┌────┴─────────────┐      ┌───────┴──────┐
│    Pipeline      │      │  Broadcast   │
│    Adapter       │      │   Adapter    │
│ (Linear flow)    │      │ (One-to-many)│
└──────────────────┘      └──────────────┘
```

## Key Components

### 1. EventCommunicationFactory
Creates and manages communication adapters from YAML configuration.

```python
from src.core.communication import EventCommunicationFactory

factory = EventCommunicationFactory(coordinator_id, log_manager)
comm_layer = factory.create_communication_layer(config, containers)
```

### 2. CommunicationLayer
Manages all active adapters and provides system-wide metrics.

```python
# Get system metrics
metrics = comm_layer.get_system_metrics()
print(f"Total events: {metrics['total_events']}")
print(f"Health: {metrics['overall_health']}")

# Get specific adapter
pipeline = comm_layer.get_adapter('main_pipeline')
```

### 3. Communication Adapters

#### PipelineCommunicationAdapter (Implemented)
Linear event flow through container stages with automatic transformation.

Features:
- Sequential event processing
- Event transformation between stages
- Per-stage metrics tracking
- Correlation ID propagation

#### Future Adapters (Planned)
- **BroadcastAdapter**: One-to-many event distribution
- **HierarchicalAdapter**: Parent-child communication with context
- **SelectiveAdapter**: Rule-based conditional routing

## Usage Examples

### Basic Pipeline Setup

```yaml
# config/simple_pipeline.yaml
communication:
  adapters:
    - type: "pipeline"
      name: "main_flow"
      containers:
        - "data_container"
        - "strategy_container"
        - "risk_container"
        - "execution_container"
```

### Fix Multi-Strategy Circular Dependencies

```yaml
# Problem: Circular dependencies in container hierarchy
# Solution: Use pipeline adapter for linear event flow

communication:
  adapters:
    - type: "pipeline"
      name: "fixed_flow"
      containers:
        - "data_container"
        - "classifier_container"
        - "strategy_container"
        - "risk_container"
        - "execution_container"
```

### Integration with Coordinator

```python
from src.core.coordinator import WorkflowCoordinator
from src.core.communication import EventCommunicationFactory

class EnhancedCoordinator(WorkflowCoordinator):
    def __init__(self, config):
        super().__init__(config)
        self.communication_factory = EventCommunicationFactory(
            self.coordinator_id,
            self.log_manager
        )
    
    async def setup_communication(self, comm_config):
        self.communication_layer = self.communication_factory.create_communication_layer(
            comm_config,
            self.containers
        )
        await self.communication_layer.setup_all_adapters()
```

## Container Requirements

Containers must implement these methods for adapter integration:

```python
class AdapterCompatibleContainer:
    def on_output_event(self, handler: Callable):
        """Register handler for output events"""
        
    def emit_output_event(self, event: Event):
        """Emit event to registered handlers"""
        
    async def receive_event(self, event: Event):
        """Receive event from adapter"""
        
    @property
    def expected_input_type(self) -> Optional[EventType]:
        """Expected input event type for transformations"""
```

The `EnhancedContainer` base class provides these methods by default.

## Pipeline Adapter Details

### Event Transformation
The pipeline adapter automatically transforms events between stages:

```python
# Built-in transformations
BAR → INDICATOR → SIGNAL → ORDER → FILL

# Custom transformations
transformer = pipeline.event_transformer
transformer.add_rule(
    from_type=EventType.CUSTOM_1,
    to_type=EventType.CUSTOM_2,
    transformer=my_transform_function
)
```

### Pipeline Metrics

```python
pipeline = comm_layer.get_adapter('main_pipeline')
metrics = pipeline.get_pipeline_metrics()

# Returns:
{
    'total_stages': 5,
    'stage_metrics': {
        0: {'events_processed': 100, 'average_latency_ms': 1.2},
        1: {'events_processed': 100, 'average_latency_ms': 2.5},
        # ...
    },
    'end_to_end_metrics': {
        'total_events': 100,
        'average_latency_ms': 8.7,
        'error_rate': 0.01
    }
}
```

## Configuration Reference

### Factory Configuration

```yaml
communication:
  pattern: "pipeline"  # Communication pattern name
  adapters:
    - type: "pipeline"
      name: "custom_pipeline"
      containers: ["container1", "container2", "container3"]
      
      # Optional adapter settings
      log_level: "DEBUG"              # Logging level
      retry_attempts: 3               # Retry failed events
      retry_delay_ms: 100            # Delay between retries
      timeout_ms: 5000               # Event timeout
      buffer_size: 1000              # Event buffer size
      enable_compression: false       # Compress events
      enable_encryption: false        # Encrypt events
      
      # Custom settings (adapter-specific)
      custom_settings:
        key: value
```

## Monitoring and Metrics

### System-Wide Metrics

```python
metrics = comm_layer.get_system_metrics()
```

Returns:
- `total_adapters`: Number of configured adapters
- `active_adapters`: Currently running adapters
- `total_events`: Events processed across all adapters
- `events_per_second`: System throughput
- `overall_health`: 'healthy', 'warning', or 'critical'
- `latency_percentiles`: p50, p95, p99 latencies
- `adapters`: Per-adapter detailed metrics

### Health Calculation
- **Healthy**: Error rate < 1% and all adapters connected
- **Warning**: Error rate < 5% or < 25% adapters disconnected
- **Critical**: Error rate ≥ 5% or ≥ 25% adapters disconnected

## Best Practices

1. **Start Simple**
   - Begin with a single pipeline adapter
   - Add complexity only when needed
   - Test thoroughly at each step

2. **Monitor Performance**
   - Watch error rates and latencies
   - Set up alerts for critical metrics
   - Use correlation IDs for debugging

3. **Handle Errors Gracefully**
   - Implement retry logic in containers
   - Log errors with full context
   - Consider dead letter queues

4. **Optimize for Your Use Case**
   - Adjust buffer sizes for throughput
   - Tune timeouts for responsiveness
   - Consider parallel pipelines for scale

## Troubleshooting

### Events Not Flowing
1. Check container names match configuration
2. Verify containers implement required methods
3. Enable DEBUG logging: `log_level: "DEBUG"`
4. Check logs for correlation IDs

### High Latency
1. Profile individual stage processing times
2. Check for blocking operations in containers
3. Consider parallel pipelines
4. Review transformation complexity

### Memory Issues
1. Reduce buffer sizes
2. Implement event batching
3. Add backpressure mechanisms
4. Monitor container memory usage

## Examples

See complete examples in:
- `src/core/communication/test_factory.py` - Basic factory usage
- `src/core/communication/coordinator_integration_example.py` - Coordinator integration
- `config/communication_examples.yaml` - Configuration patterns

## Future Enhancements

1. **Additional Adapters**
   - BroadcastAdapter for data distribution
   - HierarchicalAdapter for parent-child patterns
   - SelectiveAdapter for conditional routing
   - NetworkAdapter for distributed systems

2. **Advanced Features**
   - Event persistence and replay
   - Distributed tracing integration
   - Advanced routing rules
   - Performance optimizations

3. **Network Protocols**
   - WebSocket adapter
   - gRPC adapter
   - ZeroMQ adapter
   - Kafka integration