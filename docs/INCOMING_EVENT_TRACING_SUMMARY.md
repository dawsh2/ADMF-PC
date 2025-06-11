# Incoming Event Tracing Summary

## Problem
The portfolio container needed to trace incoming SIGNAL and FILL events for analysis, but the original approach of republishing events to its own bus created an infinite loop.

## Solution
We implemented a simple, clean solution by modifying the `Container.receive_event()` method to trace incoming events directly:

```python
def receive_event(self, event: Any) -> None:
    """Receive an event from a child container or route."""
    self._metrics['events_processed'] += 1
    self._metrics['last_activity'] = datetime.now()
    
    # If this container has a tracer and wants to trace incoming events,
    # just call trace_event directly. The tracer will handle the rest.
    if hasattr(self.event_bus, '_tracer') and self.event_bus._tracer:
        # Only trace certain event types for portfolio containers
        if self.container_type == 'portfolio':
            event_type = getattr(event, 'event_type', None)
            if event_type in ['SIGNAL', 'FILL', 'ORDER']:
                self.event_bus._tracer.trace_event(event)
        # Other containers can define their own rules or trace everything
        else:
            self.event_bus._tracer.trace_event(event)
    
    # Publish to internal event bus for components to handle
    self.event_bus.publish(event)
    
    # Forward to all child containers (enables sibling communication)
    for child_id, child in self._child_containers.items():
        # Don't send back to the originating container
        if hasattr(event, 'container_id') and event.container_id == child_id:
            continue
        child.receive_event(event)
```

## Key Benefits

1. **Simplicity**: No complex wrappers or injection patterns needed
2. **Clean Architecture**: Leverages existing EventTracer protocol
3. **No Loops**: Events are traced when received, not republished
4. **Selective Tracing**: Portfolio containers only trace relevant events (SIGNAL, FILL, ORDER)
5. **Protocol + Composition**: Uses existing tracer without inheritance

## Test Results

With this implementation:
- Portfolio containers successfully trace incoming signals
- No infinite loops or duplicate events
- Clean separation of concerns
- Minimal code changes required

## Example Usage

```python
# Enable tracing on portfolio container
portfolio_config = {
    "execution": {
        "enable_event_tracing": True,
        "trace_settings": {
            "trace_id": "portfolio_trace",
            "max_events": 1000,
            "storage_backend": "memory"
        }
    }
}

# Portfolio will automatically trace incoming SIGNAL, FILL, and ORDER events
```

## Architecture Compliance

This solution follows the ADMF-PC principles:
- **Protocol + Composition**: Uses existing EventTracer without modification
- **No Inheritance**: Simple composition of tracer functionality
- **Clean Boundaries**: Container handles routing, tracer handles storage
- **Minimal Complexity**: Solution is just a few lines of code