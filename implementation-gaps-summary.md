# Implementation Gaps in Coordinator Refactor

## Current State

The coordinator refactor has clean architecture but **key execution pieces are mocked**:

### 1. Mock Implementations Found

#### In `sequencer.py`:
```python
def _execute_topology(self, topology, config, context, phase_name):
    """Execute a topology and collect results."""
    # ... comments about what it WOULD do ...
    
    # Simulate event processing time
    time.sleep(1)
    
    # Mock performance metrics
    performance = {
        'sharpe_ratio': 1.5 + (len(containers) * 0.1),
        'total_return': 0.15,
        'max_drawdown': 0.08,
        # ...
    }
```

#### In `sequences/train_test.py`:
```python
def _execute_topology(self, topology, config, context):
    """Execute the topology and return results."""
    # Mock implementation
    import random
    return {
        'metrics': {
            'sharpe_ratio': random.uniform(0.5, 2.0),
            # ...
        }
    }
```

### 2. Inter-Phase Data Management

Looking at the protocols, the system **expects** these components:

1. **DataManagerProtocol** - For storing/retrieving phase outputs
2. **ResultStreamerProtocol** - For streaming results  
3. **CheckpointManagerProtocol** - For workflow state

But based on our earlier discussion, these were supposed to be replaced by:
- **PhaseOutputEvent** - Store phase data as events
- **TraceQuery** - Retrieve phase data from event traces
- **Event-based result extraction** - No separate result streamer

## What Needs to Be Implemented

### 1. Real Topology Execution

The `_execute_topology` method needs to:

```python
def _execute_topology(self, topology, config, context, phase_name):
    """Actually execute the topology."""
    
    # 1. Get containers and adapters
    containers = topology.get('containers', {})
    adapters = topology.get('adapters', [])
    
    # 2. Initialize containers
    for name, container in containers.items():
        if hasattr(container, 'initialize'):
            container.initialize()
    
    # 3. Start adapters (they connect containers)
    for adapter in adapters:
        adapter.start()
    
    # 4. Run the backtest/optimization
    # This is where data flows through containers via events
    # The actual implementation depends on the data source
    
    # 5. Collect metrics from containers
    from ...analytics.metrics_collection import MetricsCollector
    collector = MetricsCollector()
    metrics = collector.collect_from_containers(containers)
    
    # 6. Clean up
    for adapter in adapters:
        adapter.stop()
    
    return {
        'success': True,
        'metrics': metrics.get('aggregate', {}),
        'container_metrics': metrics.get('by_container', {})
    }
```

### 2. Inter-Phase Data Flow

Based on our event-based design, the sequencer should:

```python
# After phase execution, store output as event
from ..events.phase_events import PhaseOutputEvent

output_event = PhaseOutputEvent(
    phase_name=phase_name,
    workflow_id=context['workflow_id'],
    output_data=phase_result
)
self.event_store.store_event(output_event)

# In next phase, retrieve previous outputs
from ..events.trace_query import TraceQuery

trace_query = TraceQuery(self.event_store)
previous_output = trace_query.get_phase_output(
    workflow_id=context['workflow_id'],
    phase_name='training'
)
```

### 3. Optimization Result Collection

For optimization with multiple containers:

```python
def collect_optimization_results(topology):
    """Collect results from optimization topology."""
    
    # Get parameter combinations from topology
    param_combos = topology.get('optimization', {}).get('parameter_combinations', [])
    
    # Get metrics from each portfolio container
    results = []
    for combo in param_combos:
        container_id = combo['container_id']
        container = topology['containers'].get(container_id)
        
        if container:
            # Get metrics (via component or method)
            if hasattr(container, 'get_metrics'):
                metrics = container.get_metrics()
            elif hasattr(container, 'get_component'):
                portfolio_state = container.get_component('portfolio_state')
                metrics = portfolio_state.get_metrics() if portfolio_state else {}
            else:
                metrics = {}
            
            results.append({
                'parameters': combo['parameters'],
                'metrics': metrics
            })
    
    # Find best based on objective
    objective = topology.get('optimization', {}).get('objective', 'sharpe_ratio')
    best = max(results, key=lambda x: x['metrics'].get(objective, -float('inf')))
    
    return {
        'optimal_parameters': best['parameters'],
        'all_results': results
    }
```

## The Real Question

Do we need to:

1. **Implement the mocked methods** to actually run backtests?
2. **Add the event-based phase data flow** (PhaseOutputEvent, TraceQuery)?
3. **Wire up the metrics collection** from containers?

Or is this intentionally left as an integration point where users would plug in their own execution engine?

The architecture is clean, but without these implementations, it can't actually run backtests or optimizations - it just returns random numbers!