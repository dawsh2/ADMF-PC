# Result Streaming Architecture in ADMF-PC

## Overview

ADMF-PC implements a sophisticated result streaming architecture that manages data flow between workflow phases, supports distributed execution, and enables real-time monitoring. The architecture is built around three key components:

1. **ResultStreamer** - Real-time result streaming with buffering and aggregation
2. **DataFlowManager** - Inter-phase data management and storage
3. **Coordinator** - Orchestrates result streaming across workflow execution

## Architecture Components

### 1. ResultStreamer (`result_streaming.py`)

The ResultStreamer handles real-time result streaming to disk with intelligent buffering and format support.

**Key Features:**
- **Multiple Output Formats**: JSONL (default), Parquet (efficient storage), CSV (compatibility)
- **Buffered Writing**: Configurable buffer size (default 1000) for efficient I/O
- **Real-time Aggregation**: Tracks metrics per phase/container during execution
- **Compression Support**: Optional compression for Parquet output
- **Performance Tracking**: Monitors write/flush counts

**Usage Pattern:**
```python
# Created by Coordinator for each workflow
result_streamer = ResultStreamer(
    workflow_id="workflow_123",
    output_dir="./results/workflow_123",
    buffer_size=1000,
    format='jsonl',
    compression='snappy'  # for parquet
)

# Stream results during execution
await result_streamer.write_result({
    'phase': 'optimization',
    'container': 'strategy_001',
    'timestamp': datetime.now().isoformat(),
    'result': {
        'metrics': {'sharpe_ratio': 1.5, 'total_return': 0.15},
        'signals_generated': 100
    }
})

# Get aggregated results after execution
aggregated = await result_streamer.get_aggregated_results()
```

### 2. DataFlowManager (`data_management.py`)

The DataFlowManager handles data persistence and retrieval between workflow phases, crucial for distributed execution.

**Key Features:**
- **Smart Storage Strategy**: In-memory cache for small data (<1MB), disk for larger
- **Format Optimization**: Parquet for signal data, pickle for general objects
- **Phase References**: Create references without loading data
- **Data Transfer**: Copy data between workflow phases
- **Cleanup Management**: Automatic workflow data cleanup

**Storage Strategy:**
```python
# Store phase output (automatically chooses storage method)
await data_manager.store_phase_output(
    workflow_id="workflow_123",
    phase_name="signal_generation",
    output={
        'signals': [...]  # List of signal data
        'metadata': {...}  # Phase metadata
    }
)

# Retrieve in later phase
signals = await data_manager.get_phase_output(
    workflow_id="workflow_123",
    phase_name="signal_generation"
)
```

### 3. Distributed Result Collection

For distributed execution across multiple sequencers:

```python
class DistributedResultCollector:
    """Collects results from multiple distributed sequencers"""
    
    def __init__(self, workflow_id: str, num_sources: int):
        # Create result queue per source
        self.result_queues = {
            i: asyncio.Queue() for i in range(num_sources)
        }
```

## Data Flow Patterns

### 1. Sequential Phase Execution

```
Phase 1: Signal Generation
    ↓ (store_phase_output)
    ↓ Results stored to disk/cache
    ↓
Phase 2: Optimization  
    ↓ (get_phase_output from Phase 1)
    ↓ Process signals
    ↓ (store_phase_output)
    ↓
Phase 3: Analysis
    ↓ (get_phase_outputs from Phase 1 & 2)
    ↓ Generate final report
```

### 2. Result Streaming Flow

```
Container Execution
    ↓
Container Results
    ↓
Result Callback (from Sequencer)
    ↓
Coordinator._handle_result_stream()
    ↓
ResultStreamer.write_result()
    ↓
Buffer (in memory)
    ↓ (when buffer full)
Flush to Disk (JSONL/Parquet/CSV)
```

### 3. Aggregation Flow

```
All Phases Complete
    ↓
ResultStreamer.flush()
    ↓
ResultStreamer.get_aggregated_results()
    ↓
Aggregated Metrics:
- Per-phase summaries
- Top performers
- Average metrics
- Best containers
```

## Implementation Details

### Coordinator Integration

The Coordinator manages result streaming at the workflow level:

```python
class Coordinator:
    async def execute_workflow(self, config):
        # Create result streamer
        if self.enable_result_streaming:
            result_streamer = self._create_result_streamer(workflow_id, config)
            self.result_streamers[workflow_id] = result_streamer
        
        # Execute with result callback
        result = await sequencer.execute_phases(
            pattern, config, context,
            result_callback=self._handle_result_stream
        )
        
        # Finalize and aggregate
        if self.enable_result_streaming:
            final_results = await self._finalize_result_stream(workflow_id)
            result['aggregated_results'] = final_results
```

### Sequencer Phase Handling

The Sequencer manages phase transitions and data dependencies:

```python
class Sequencer:
    def _merge_phase_config(self, base_config, phase_config, phase_index):
        """Merge configs and handle phase dependencies"""
        
        # Handle phase dependencies
        if phase_config.get('depends_on'):
            for dep_phase in phase_config['depends_on']:
                if dep_phase in self.phase_results:
                    merged[f'{dep_phase}_results'] = self.phase_results[dep_phase]
        
        # Handle inheritance from previous phases
        if phase_config.get('inherit_from'):
            inherit_phase = phase_config['inherit_from']
            if inherit_phase in self.phase_results:
                inherited_data = self.phase_results[inherit_phase].get('output', {})
                merged['inherited_data'] = inherited_data
```

## Performance Optimizations

### 1. Buffered Writing
- Results buffered in memory before disk writes
- Reduces I/O operations significantly
- Configurable buffer size based on workload

### 2. Format-Specific Optimizations
- **JSONL**: Append-only, line-by-line streaming
- **Parquet**: Columnar storage with compression
- **CSV**: Compatible but less efficient

### 3. Async I/O
- Non-blocking file operations
- Parallel result processing
- Efficient resource utilization

## Use Cases

### 1. Multi-Phase Optimization
```yaml
workflow:
  phases:
    - name: signal_generation
      topology: signal_generation
    - name: parameter_optimization
      topology: optimization
      depends_on: [signal_generation]
    - name: analysis
      topology: analysis
      depends_on: [parameter_optimization]
```

### 2. Distributed Walk-Forward
```python
# Distribute phases across workers
phase_assignments = coordinator._distribute_phases(phases, num_workers=8)

# Each worker stores results independently
await data_manager.store_phase_output(
    workflow_id, phase_name, results
)
```

### 3. Real-Time Monitoring
```python
# Stream results as they're generated
async def monitor_results(workflow_id):
    streamer = result_streamers[workflow_id]
    while workflow_active:
        metrics = streamer.metrics
        print(f"Processed: {streamer.write_count} results")
        await asyncio.sleep(1)
```

## Best Practices

1. **Buffer Size Selection**
   - Small workflows: 100-500 buffer size
   - Large workflows: 1000-5000 buffer size
   - Memory constrained: 50-100 buffer size

2. **Format Selection**
   - Research/debugging: JSONL (human readable)
   - Production/storage: Parquet (compressed, efficient)
   - External tools: CSV (compatibility)

3. **Cleanup Strategy**
   - Always cleanup after workflow completion
   - Implement retention policies for results
   - Monitor disk usage in production

4. **Error Handling**
   - Flush buffers on errors
   - Save partial results
   - Enable checkpoint recovery

## Architecture Benefits

1. **Scalability**: Supports distributed execution across multiple machines
2. **Performance**: Buffered I/O and format optimization
3. **Flexibility**: Multiple output formats and storage strategies
4. **Debugging**: Complete result history with aggregation
5. **Resilience**: Checkpoint support and partial result recovery

The result streaming architecture enables ADMF-PC to handle complex multi-phase workflows efficiently while maintaining data integrity and providing real-time monitoring capabilities.