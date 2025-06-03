# Communication Module

The communication module provides a unified adapter interface for various communication protocols in the ADMF-PC system. It enables seamless integration of different communication mechanisms while maintaining consistency and proper event flow.

## Overview

The communication module implements the adapter pattern to abstract away the details of specific communication protocols (WebSocket, gRPC, ZeroMQ, etc.) and provides a consistent interface for event-based communication across containers.

## Key Components

### CommunicationAdapter (base_adapter.py)
The abstract base class that all communication adapters must implement. It provides:
- Lifecycle management (setup/cleanup)
- Event serialization/deserialization
- Metrics tracking
- Error handling and recovery
- Correlation ID support
- Full logging integration

### AdapterConfig
Configuration dataclass for adapters including:
- Name and type identification
- Retry and timeout settings
- Buffer and performance options
- Custom protocol-specific settings

### AdapterMetrics
Comprehensive metrics tracking:
- Events sent/received/failed
- Bytes transferred
- Error rates and latency
- Connection statistics

## Architecture

```
┌─────────────────────────────────────────┐
│         CommunicationAdapter            │
│         (Abstract Base Class)           │
├─────────────────────────────────────────┤
│ + connect() -> bool                     │
│ + disconnect() -> None                  │
│ + send_event(Event) -> bool            │
│ + process_incoming() -> None           │
│ + setup() -> None                      │
│ + cleanup() -> None                    │
└─────────────────────────────────────────┘
                    ▲
                    │ Inherits
     ┌──────────────┴──────────────┐
     │                             │
┌────┴─────┐              ┌───────┴──────┐
│WebSocket │              │    gRPC      │
│ Adapter  │              │   Adapter    │
└──────────┘              └──────────────┘
```

## Usage Example

```python
from src.core.communication import CommunicationAdapter, AdapterConfig
from src.core.events.types import Event, EventType

# Configure adapter
config = AdapterConfig(
    name="market_data",
    adapter_type="websocket",
    retry_attempts=3,
    timeout_ms=5000,
    custom_settings={
        "url": "wss://market-data.example.com",
        "compression": True
    }
)

# Create adapter (using concrete implementation)
adapter = WebSocketAdapter(config)

# Setup
await adapter.setup()

# Send events
event = Event(
    event_type=EventType.SIGNAL,
    payload={"symbol": "AAPL", "direction": 1},
    source_id="strategy_1",
    metadata={"correlation_id": "trade-123"}
)
success = await adapter.send_event(event)

# Process incoming messages
await adapter.process_incoming()

# Get metrics
metrics = adapter.get_metrics()
print(f"Events sent: {metrics.events_sent}")
print(f"Average latency: {metrics.average_latency_ms}ms")

# Cleanup
await adapter.cleanup()
```

## Implementing a New Adapter

To implement a new communication protocol:

1. **Inherit from CommunicationAdapter**
   ```python
   class MyProtocolAdapter(CommunicationAdapter):
       def __init__(self, config: AdapterConfig):
           super().__init__(config)
           # Protocol-specific initialization
   ```

2. **Implement Abstract Methods**
   - `connect()`: Establish connection to the communication channel
   - `disconnect()`: Close the connection
   - `send_raw()`: Send raw bytes through the channel
   - `receive_raw()`: Receive raw bytes from the channel

3. **Optional: Override Serialization**
   - Override `serialize_event()` and `deserialize_event()` for custom formats
   - Default implementation uses JSON

4. **Add Protocol-Specific Features**
   - Use `config.custom_settings` for protocol-specific configuration
   - Add helper methods as needed
   - Maintain metrics tracking

## Integration with ADMF-PC

The communication adapters integrate with the ADMF-PC event system:

1. **Event Flow**
   - Internal events → Adapter → External protocol
   - External protocol → Adapter → Internal events

2. **Container Isolation**
   - Each adapter runs in its own context
   - Full logging with container awareness
   - Correlation tracking across boundaries

3. **Metrics and Monitoring**
   - All adapters report standardized metrics
   - Integration with system monitoring
   - Performance tracking and optimization

## Best Practices

1. **Error Handling**
   - Always implement retry logic
   - Log errors with full context
   - Graceful degradation on failures

2. **Performance**
   - Use async/await for non-blocking I/O
   - Implement buffering for high throughput
   - Monitor and optimize latency

3. **Security**
   - Support encryption when needed
   - Validate all incoming data
   - Implement proper authentication

4. **Testing**
   - Create mock adapters for testing
   - Test error conditions and recovery
   - Validate metrics accuracy

## Future Enhancements

- Protocol-specific implementations (WebSocket, gRPC, ZeroMQ)
- Message compression and encryption
- Advanced routing and filtering
- Load balancing and failover
- Performance optimizations