# Coordinator Refactor Changes Summary

## Key Requirements
1. Make everything synchronous (remove all async/await)
2. Coordinator is NOT part of the event system (orchestration layer only)
3. Use component/discovery.py for workflow discovery
4. Replace DataFlowManager with event-based phase data using TraceQuery
5. Replace ResultStreamer with event-based result extraction using ResultExtractor

## Core Principles
- Coordinator orchestrates but doesn't touch events
- Sequencer manages phase execution with isolated event systems
- Each phase gets fresh event system that's torn down after execution
- Phase data flows through event traces, not separate infrastructure
- Results extracted from events, not streamed separately

## Detailed Changes Required

### 1. Coordinator Changes (`coordinator.py`)

#### Remove async/await
- Change all `async def` to `def`
- Remove all `await` keywords
- Change `asyncio.sleep()` to `time.sleep()`
- Remove asyncio imports

#### Remove Event System Integration
- Coordinator should NOT have any event bus references
- Remove result streaming callbacks that touch events
- All event interaction happens through Sequencer

#### Use component/discovery.py
Replace custom workflow discovery:
```python
# REMOVE:
from .workflow_discovery import discover_workflows

# ADD:
from ..components.discovery import get_component_registry

# In _initialize_workflow_patterns():
registry = get_component_registry()
workflows = registry.get_components_by_type('workflow')
for workflow_info in workflows:
    workflow_func = workflow_info.factory
    workflow_def = workflow_func()
    self.workflow_patterns[workflow_info.name] = WorkflowPattern(
        name=workflow_info.name,
        phases=workflow_def['phases']
    )
```

#### Remove DataFlowManager and ResultStreamer
```python
# REMOVE these imports and usages:
from .data_management import DataFlowManager
from .result_streaming import ResultStreamer

# REMOVE these attributes:
self.data_manager = None
self.result_streamers: Dict[str, Any] = {}

# REMOVE methods:
- _get_data_manager()
- _create_result_streamer()
- _handle_result_stream()
- _finalize_result_stream()
```

### 2. Sequencer Changes (`sequencer.py`)

#### Remove async/await
- Change all `async def` to `def`
- Remove all `await` keywords
- Remove asyncio imports

#### Add Event System Management
Each phase execution should:
1. Create fresh event bus
2. Execute topology with that bus
3. Extract results from event traces
4. Tear down event bus

```python
def _execute_phase(self, phase_config, base_config, context, phase_index):
    """Execute a single phase with isolated event system."""
    # Create fresh event bus for this phase
    from ..events.event_bus import EventBus
    from ..events.tracing.unified_tracer import UnifiedTracer
    
    event_bus = EventBus()
    tracer = UnifiedTracer(
        trace_id=f"{context['workflow_id']}_{phase_config['name']}",
        enable_event_tracing=True
    )
    event_bus.attach_tracer(tracer)
    
    try:
        # Build and execute topology
        topology = self.topology_builder.build_topology(
            phase_config['topology'],
            phase_execution_config
        )
        
        # Execute with isolated event system
        result = self._execute_topology_with_events(
            topology, 
            phase_execution_config,
            event_bus,
            tracer
        )
        
        # Extract results from events
        extracted_results = self._extract_phase_results(tracer)
        result['extracted_results'] = extracted_results
        
        # Store phase data in event traces for next phases
        if phase_config.get('depends_on'):
            self._store_phase_data_in_events(
                context['workflow_id'],
                phase_config['name'],
                extracted_results,
                event_bus
            )
        
        return result
        
    finally:
        # Tear down event system
        event_bus.shutdown()
```

#### Add Result Extraction
```python
def _extract_phase_results(self, tracer):
    """Extract results from event traces."""
    from ..events.tracing.query_interface import TraceQuery
    from ..events.result_extraction import (
        PortfolioMetricsExtractor,
        SignalExtractor,
        FillExtractor
    )
    
    # Create query interface from tracer
    query = TraceQuery(tracer)
    
    # Extract different result types
    extractors = [
        PortfolioMetricsExtractor(),
        SignalExtractor(),
        FillExtractor()
    ]
    
    results = {}
    for extractor in extractors:
        category_results = []
        for event in query.events:
            if extractor.can_extract(event):
                extracted = extractor.extract(event)
                if extracted:
                    category_results.append(extracted)
        
        if category_results:
            results[extractor.result_category] = category_results
    
    return results
```

#### Add Phase Data Access
```python
def _get_phase_data(self, workflow_id, phase_name):
    """Get data from previous phase using event traces."""
    # Load trace file for previous phase
    trace_file = f"./traces/{workflow_id}_{phase_name}.jsonl"
    
    if os.path.exists(trace_file):
        from ..events.tracing.query_interface import TraceQuery
        query = TraceQuery(trace_file)
        
        # Extract relevant data
        summary = query.get_summary()
        return {
            'events': query.events,
            'summary': summary,
            'extracted_results': self._extract_phase_results(query)
        }
    
    return None
```

### 3. TopologyBuilder Changes (`topology_builder.py`)

#### Remove async
- No async methods needed here
- Already follows single responsibility

#### Add Event System to Topologies
Topologies should include event system configuration:
```python
def _create_backtest_topology(self, config):
    """Create a backtest topology with event system config."""
    topology = create_backtest_topology(config, tracing_enabled)
    
    # Add event system configuration
    topology['event_config'] = {
        'isolation': True,  # Each phase gets isolated event system
        'tracing': {
            'enabled': config.get('tracing', {}).get('enabled', True),
            'output_dir': config.get('tracing', {}).get('output_dir', './traces')
        }
    }
    
    return topology
```

### 4. Remove These Files
- `workflow_discovery.py` - replaced by component/discovery.py
- `data_management.py` - replaced by event traces
- `result_streaming.py` - replaced by result extraction

### 5. Update Protocols (`protocols.py`)

#### Remove async from all protocols
```python
@runtime_checkable
class SequencerProtocol(Protocol):
    def execute_phases(  # Remove async
        self, 
        pattern: Dict[str, Any], 
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...

# Remove ResultStreamerProtocol and DataManagerProtocol entirely
```

### 6. Example Usage After Refactor

```python
# Synchronous workflow execution
coordinator = Coordinator()

# Execute workflow (no await needed)
result = coordinator.execute_workflow({
    'workflow': 'walk_forward',
    'strategy': 'momentum',
    'data_file': 'SPY.csv'
})

# Results are extracted from events, not separate infrastructure
phase_results = result['results']['phase_results']
for phase_name, phase_data in phase_results.items():
    extracted = phase_data['extracted_results']
    
    # Access portfolio metrics
    if 'portfolio_metrics' in extracted:
        for metric in extracted['portfolio_metrics']:
            print(f"Sharpe: {metric['sharpe_ratio']}")
    
    # Access signals
    if 'signals' in extracted:
        print(f"Generated {len(extracted['signals'])} signals")
```

## Implementation Order

1. **First**: Update protocols.py to remove async
2. **Second**: Refactor Sequencer to be synchronous and manage event systems
3. **Third**: Update Coordinator to be synchronous and use component discovery
4. **Fourth**: Remove data_management.py and result_streaming.py
5. **Fifth**: Update TopologyBuilder event configuration
6. **Sixth**: Test with simple workflow to verify changes

## Key Benefits

1. **Simpler Architecture**: No async complexity
2. **True Isolation**: Each phase has isolated event system
3. **Single Source of Truth**: Events are the only data source
4. **No Duplicate Infrastructure**: Remove DataFlowManager and ResultStreamer
5. **Clean Separation**: Coordinator never touches events directly