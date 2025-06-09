# Per-Container Data Flow Configuration

## Why Per-Container Configuration?

Different containers have vastly different memory profiles:
- **Data containers**: Stream large datasets, minimal state
- **Feature containers**: May cache computed features
- **Portfolio containers**: Track positions and history
- **Execution containers**: Aggregate results from many portfolios

## Design Approach

### 1. Container-Level Configuration

```yaml
# Global default
data_flow:
  default_mode: memory
  default_memory_limit_mb: 100

# Per-container overrides
containers:
  data:
    data_flow:
      mode: streaming  # Stream market data
      streaming_settings:
        buffer_size: 1000
        
  features:
    data_flow:
      mode: hybrid  # Cache recent features
      memory_limit_mb: 500
      file_settings:
        format: parquet
        
  portfolio_*:  # Wildcard for all portfolio containers
    data_flow:
      mode: memory  # Keep in memory during execution
      memory_limit_mb: 50  # Each portfolio is small
      persistence:
        on_completion: true  # Save when done
        
  execution:
    data_flow:
      mode: file  # Aggregate results are large
      file_settings:
        format: parquet
        compression: snappy
```

### 2. Container Implementation

```python
class ContainerWithDataFlow(Container):
    """Container that manages its own data flow."""
    
    def __init__(self, config: ContainerConfig):
        super().__init__(config)
        
        # Get container-specific data flow config
        self.data_flow_config = self._get_data_flow_config()
        self.data_manager = self._create_data_manager()
        
        # Track data based on configuration
        self._setup_data_tracking()
    
    def _get_data_flow_config(self) -> Dict[str, Any]:
        """Get data flow config for this container."""
        # Check for container-specific config
        container_config = self.config.config.get('data_flow', {})
        
        # Fall back to defaults
        global_config = self.config.config.get('global_data_flow', {})
        
        return {
            **global_config.get('defaults', {}),
            **container_config
        }
    
    def _create_data_manager(self) -> DataManager:
        """Create appropriate data manager."""
        mode = self.data_flow_config.get('mode', 'memory')
        
        if mode == 'memory':
            return MemoryDataManager(
                limit_mb=self.data_flow_config.get('memory_limit_mb', 100)
            )
        elif mode == 'streaming':
            return StreamingDataManager(
                buffer_size=self.data_flow_config.get('streaming_settings', {}).get('buffer_size', 1000),
                flush_callback=self._flush_to_storage
            )
        elif mode == 'file':
            return FileDataManager(
                base_path=f"./results/{self.container_id}",
                format=self.data_flow_config.get('file_settings', {}).get('format', 'parquet')
            )
        elif mode == 'hybrid':
            return HybridDataManager(
                memory_limit_mb=self.data_flow_config.get('memory_limit_mb', 100),
                overflow_path=f"./overflow/{self.container_id}"
            )
    
    def store_result(self, key: str, data: Any) -> str:
        """Store result using configured method."""
        return self.data_manager.store(key, data)
    
    def get_stored_result(self, key: str) -> Any:
        """Retrieve stored result."""
        return self.data_manager.retrieve(key)
```

### 3. Smart Memory Management

```python
class HybridDataManager:
    """Manages memory with automatic overflow."""
    
    def __init__(self, memory_limit_mb: int, overflow_path: str):
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.overflow_path = Path(overflow_path)
        self.memory_store = {}
        self.file_references = {}
        self.current_memory_usage = 0
    
    def store(self, key: str, data: Any) -> str:
        """Store with automatic overflow."""
        size = self._estimate_size(data)
        
        if self.current_memory_usage + size > self.memory_limit:
            # Overflow to disk
            return self._store_to_file(key, data)
        else:
            # Keep in memory
            self.memory_store[key] = data
            self.current_memory_usage += size
            return f"memory://{key}"
    
    def _maybe_evict(self):
        """Evict least recently used if needed."""
        if self.current_memory_usage > self.memory_limit * 0.9:
            # Find least recently used
            lru_key = min(self.access_times, key=self.access_times.get)
            self._evict_to_disk(lru_key)
```

### 4. Topology Builder Integration

```python
def create_backtest_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create topology with per-container data flow."""
    
    # Global data flow config
    global_data_flow = config.get('data_flow', {})
    
    # Container-specific overrides
    container_configs = config.get('containers', {})
    
    # Create portfolio containers with optimization check
    if 'parameter_space' in config:
        # Many containers - use minimal memory each
        portfolio_data_flow = {
            'mode': 'hybrid',
            'memory_limit_mb': 10,  # Small limit per container
            'persistence': {
                'on_completion': True,  # Save results
                'format': 'parquet'
            }
        }
    else:
        # Few containers - can use more memory
        portfolio_data_flow = {
            'mode': 'memory',
            'memory_limit_mb': 100
        }
    
    # Apply to portfolio containers
    for i, params in enumerate(param_combinations):
        portfolio_config = {
            'type': 'portfolio',
            'data_flow': portfolio_data_flow,
            # ... other config
        }
```

## Use Cases

### 1. Optimization with 1000 Containers
```yaml
# Each portfolio gets minimal memory
containers:
  portfolio_*:
    data_flow:
      mode: hybrid
      memory_limit_mb: 5  # 5MB each = 5GB total
      persistence:
        on_completion: true
```

### 2. High-Frequency Data Processing
```yaml
containers:
  data:
    data_flow:
      mode: streaming
      streaming_settings:
        buffer_size: 100  # Small buffer, fast flush
        
  features:
    data_flow:
      mode: memory  # Keep computed features in RAM
      memory_limit_mb: 2000
```

### 3. Memory-Constrained Environment
```yaml
# Everything uses minimal memory
data_flow:
  default_mode: hybrid
  default_memory_limit_mb: 50

containers:
  execution:
    data_flow:
      mode: file  # Force file for aggregation
```

## Benefits

1. **Fine-Grained Control**: Each container optimized for its role
2. **Memory Safety**: Automatic overflow prevents OOM
3. **Performance**: Critical containers stay in memory
4. **Scalability**: Can run thousands of containers
5. **Flexibility**: Mix modes within same workflow

## Container Coordination

The Sequencer still manages inter-phase data flow:

```python
class Sequencer:
    def _execute_phase(self, ...):
        # Containers manage their own data
        result = self._execute_topology(...)
        
        # Collect container outputs
        container_outputs = {}
        for name, container in topology['containers'].items():
            if hasattr(container, 'get_final_results'):
                # Container decides format/location
                container_outputs[name] = container.get_final_results()
        
        # Phase output references container results
        phase_output = {
            'container_results': container_outputs,
            'aggregate_metrics': result['metrics']
        }
        
        # Save phase-level summary
        self.data_flow.save_output(phase_name, phase_output)
```

This design gives maximum flexibility while maintaining clean boundaries!