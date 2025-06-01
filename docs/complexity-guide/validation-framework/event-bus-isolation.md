# Event Bus Isolation Validation

**CRITICAL**: This validation MUST be implemented and passing before any development begins.

## Overview

Event bus isolation ensures that events from one container cannot leak into another container. This is fundamental to:
- Parallel backtest execution
- Container independence
- Reproducible results
- System scalability

## Required Implementation Files

```bash
# 1. Enhanced isolation framework
src/core/events/enhanced_isolation.py

# 2. Comprehensive isolation tests  
src/core/events/isolation_tests.py

# 3. Integration with complexity checklist validation
tests/isolation/test_container_isolation.py
```

## Validation Commands

```bash
# Run basic isolation validation
python -m src.core.events.enhanced_isolation

# Run comprehensive isolation test suite
python -m src.core.events.isolation_tests

# Validate isolation during backtest execution
python -c "from src.core.events.enhanced_isolation import run_isolation_validation; print(run_isolation_validation())"
```

## Isolation Test Requirements

### Basic Isolation
- Events published in Container A must NOT appear in Container B
- Each container must have its own event bus instance
- No shared mutable state between containers

### Strict Mode Enforcement
- Violations must be caught and prevented
- Clear error messages when isolation is breached
- Automatic cleanup of leaked references

### Parallel Container Stress Test
- Run 20+ containers simultaneously
- Verify no event cross-contamination
- Monitor memory usage during parallel execution

### Violation Detection
- Detect wrong container IDs in events
- Log all isolation violations
- Fail fast on any breach

### Resource Cleanup
- All containers properly disposed after use
- No memory leaks from event subscriptions
- Event bus references cleared

### Parent-Child Hierarchy
- Nested containers maintain isolation
- Child containers don't leak to parents
- Sibling containers remain independent

## Implementation Example

```python
class EnhancedIsolationManager:
    """Manages event bus isolation across containers"""
    
    def __init__(self):
        self._container_buses = {}
        self._strict_mode = True
        self._violation_log = []
    
    def create_isolated_bus(self, container_id: str) -> EventBus:
        """Create an isolated event bus for a container"""
        if container_id in self._container_buses:
            raise ValueError(f"Container {container_id} already has a bus")
        
        bus = IsolatedEventBus(container_id, self)
        self._container_buses[container_id] = bus
        return bus
    
    def validate_event_routing(self, event: Event, target_container: str):
        """Validate event is routed to correct container"""
        if hasattr(event, 'container_id') and event.container_id != target_container:
            violation = {
                'event': event,
                'expected_container': target_container,
                'actual_container': event.container_id,
                'timestamp': datetime.now()
            }
            self._violation_log.append(violation)
            
            if self._strict_mode:
                raise IsolationViolation(
                    f"Event from {event.container_id} sent to {target_container}"
                )
```

## Validation Integration

Every step in the complexity guide must include:

```python
# At the start of each step
from src.core.events.enhanced_isolation import get_enhanced_isolation_manager, IsolationTestSuite

isolation_manager = get_enhanced_isolation_manager()
test_suite = IsolationTestSuite(isolation_manager)
isolation_results = test_suite.run_all_tests()

assert isolation_results['overall_passed'], f"Isolation failed: {isolation_results['summary']}"
print("✅ Event bus isolation validated")
```

## Common Isolation Violations

### 1. Shared Event Bus Instance
```python
# ❌ WRONG - Shared bus
event_bus = EventBus()
container_a = BacktestContainer(event_bus)
container_b = BacktestContainer(event_bus)  # Same bus!

# ✅ CORRECT - Isolated buses
container_a = BacktestContainer(isolation_manager.create_isolated_bus("a"))
container_b = BacktestContainer(isolation_manager.create_isolated_bus("b"))
```

### 2. Cross-Container Event Publishing
```python
# ❌ WRONG - Publishing to wrong container
container_a.event_bus.publish(Event(
    event_type="SIGNAL",
    container_id="container_b"  # Wrong container!
))

# ✅ CORRECT - Events stay in their container
container_a.event_bus.publish(Event(
    event_type="SIGNAL",
    container_id="container_a"
))
```

### 3. Leaked Event Handlers
```python
# ❌ WRONG - Handler registered on multiple buses
def my_handler(event):
    pass

container_a.event_bus.subscribe("SIGNAL", my_handler)
container_b.event_bus.subscribe("SIGNAL", my_handler)  # Leak!

# ✅ CORRECT - Separate handlers per container
container_a.event_bus.subscribe("SIGNAL", lambda e: handle_a(e))
container_b.event_bus.subscribe("SIGNAL", lambda e: handle_b(e))
```

## Success Criteria

The event bus isolation validation passes when:
- ✅ All isolation tests pass
- ✅ No violations detected during execution
- ✅ Memory usage is bounded
- ✅ Performance meets requirements
- ✅ Parallel execution works correctly

## Next Steps

After event bus isolation is validated:
1. Implement synthetic data validation
2. Set up logging infrastructure
3. Begin Step 1 of the complexity guide