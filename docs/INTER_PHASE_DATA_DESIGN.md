# Configurable Inter-Phase Data Flow

## Design Overview

Inter-phase data flow can be configured to use different strategies based on workflow needs:

1. **Memory** - Fast, for small datasets
2. **File** - Scalable, for large datasets  
3. **Streaming** - Real-time, for continuous workflows
4. **Hybrid** - Memory with file overflow

## Configuration

```yaml
# In workflow configuration
workflow: optimization_workflow
data_flow:
  mode: file  # Options: memory, file, streaming, hybrid
  
  # File mode settings
  file_settings:
    base_path: ./results
    format: parquet  # Options: parquet, json, hdf5, pickle
    compression: snappy
    partition_by: [date, symbol]  # For large datasets
    
  # Memory mode settings  
  memory_settings:
    max_size_mb: 1000  # Switch to file if larger
    
  # Streaming settings
  streaming_settings:
    buffer_size: 1000
    flush_interval: 60  # seconds
    
  # What to persist
  persistence:
    save_all_phases: true  # Save even if using memory mode
    save_final_only: false
    formats:
      metrics: json
      signals: parquet
      portfolios: hdf5

phases:
  - name: optimization
    topology: backtest
    # Phase can override data flow mode
    data_flow:
      mode: file  # Force file for large optimization
    config:
      parameter_space:
        strategy.threshold: [0.01, 0.02, 0.03, 0.04, 0.05]
        
  - name: analysis
    topology: analysis
    depends_on: [optimization]
    # Use memory for small analysis results
    data_flow:
      mode: memory
```

## Implementation Architecture

```python
# Data flow manager interface
class DataFlowManager(Protocol):
    """Interface for different data flow strategies."""
    
    def save_output(self, phase_name: str, data: Any) -> str:
        """Save phase output and return reference."""
        ...
        
    def load_input(self, phase_name: str) -> Any:
        """Load phase output."""
        ...

# Concrete implementations
class MemoryDataFlow(DataFlowManager):
    """In-memory data passing."""
    def __init__(self):
        self.storage = {}
    
    def save_output(self, phase_name: str, data: Any) -> str:
        self.storage[phase_name] = data
        return f"memory://{phase_name}"

class FileDataFlow(DataFlowManager):
    """File-based data passing."""
    def __init__(self, base_path: str, format: str = 'parquet'):
        self.base_path = Path(base_path)
        self.format = format
    
    def save_output(self, phase_name: str, data: Any) -> str:
        output_path = self.base_path / phase_name / f"output.{self.format}"
        # Use analytics tools to save
        save_results(data, output_path, format=self.format)
        return str(output_path)

class StreamingDataFlow(DataFlowManager):
    """Streaming data between phases."""
    def __init__(self, buffer_size: int = 1000):
        self.buffers = {}
        self.buffer_size = buffer_size
    
    def save_output(self, phase_name: str, data: Any) -> str:
        # Stream to buffer, flush periodically
        if phase_name not in self.buffers:
            self.buffers[phase_name] = StreamBuffer(self.buffer_size)
        self.buffers[phase_name].append(data)
        return f"stream://{phase_name}"

class HybridDataFlow(DataFlowManager):
    """Memory with file overflow."""
    def __init__(self, memory_limit_mb: int = 1000):
        self.memory = MemoryDataFlow()
        self.file = FileDataFlow('./overflow')
        self.memory_limit = memory_limit_mb * 1024 * 1024
    
    def save_output(self, phase_name: str, data: Any) -> str:
        size = get_object_size(data)
        if size < self.memory_limit:
            return self.memory.save_output(phase_name, data)
        else:
            return self.file.save_output(phase_name, data)
```

## Sequencer Integration

```python
class Sequencer:
    def __init__(self, data_flow_config: Optional[Dict[str, Any]] = None):
        # Create data flow manager based on config
        self.data_flow = self._create_data_flow_manager(data_flow_config)
        
    def _create_data_flow_manager(self, config: Optional[Dict[str, Any]]) -> DataFlowManager:
        if not config:
            return MemoryDataFlow()  # Default
            
        mode = config.get('mode', 'memory')
        
        if mode == 'memory':
            return MemoryDataFlow()
        elif mode == 'file':
            settings = config.get('file_settings', {})
            return FileDataFlow(
                base_path=settings.get('base_path', './results'),
                format=settings.get('format', 'parquet')
            )
        elif mode == 'streaming':
            settings = config.get('streaming_settings', {})
            return StreamingDataFlow(
                buffer_size=settings.get('buffer_size', 1000)
            )
        elif mode == 'hybrid':
            settings = config.get('memory_settings', {})
            return HybridDataFlow(
                memory_limit_mb=settings.get('max_size_mb', 1000)
            )
    
    def _execute_phase(self, phase_config, ...):
        # Execute phase
        result = self._execute_topology(...)
        
        # Save output using configured method
        output_ref = self.data_flow.save_output(phase_name, result['output'])
        
        # Track reference for dependencies
        self.phase_outputs[phase_name] = output_ref
        
        return result
    
    def _resolve_dependencies(self, phase_config):
        dependencies = {}
        
        for dep_phase in phase_config.get('depends_on', []):
            if dep_phase in self.phase_outputs:
                # Load using configured method
                output_ref = self.phase_outputs[dep_phase]
                data = self.data_flow.load_input(dep_phase)
                dependencies[f'{dep_phase}_output'] = data
                
        return dependencies
```

## Use Cases

### 1. Small Workflow (Memory)
```yaml
data_flow:
  mode: memory  # Fast, everything in RAM
```

### 2. Large Optimization (File)
```yaml
data_flow:
  mode: file
  file_settings:
    format: parquet
    compression: snappy
```

### 3. Real-time Analysis (Streaming)
```yaml
data_flow:
  mode: streaming
  streaming_settings:
    buffer_size: 100
    flush_interval: 10
```

### 4. Mixed Workflow (Hybrid)
```yaml
data_flow:
  mode: hybrid
  memory_settings:
    max_size_mb: 500  # Use memory up to 500MB
```

## Benefits

1. **Flexibility** - Choose based on use case
2. **Performance** - Memory mode for speed
3. **Scalability** - File mode for large data
4. **Debuggability** - Optional persistence
5. **Backward Compatible** - Default to memory mode

## Container Integration

Containers can also participate in streaming:

```python
class Container:
    def save_results(self, output_config: Dict[str, Any]):
        """Save results based on configuration."""
        if output_config.get('streaming'):
            # Stream results as generated
            self.result_streamer = ResultStreamer(output_config)
            self.event_bus.subscribe(
                EventType.PORTFOLIO_UPDATE,
                self.result_streamer.stream_result
            )
        else:
            # Batch save at end
            results = self.collect_results()
            save_path = output_config.get('output_directory')
            save_results(results, save_path)
```

This design gives users complete control over how data flows between phases while maintaining clean architecture!