# Inter-Phase Data and Memory Management

## Overview

This document describes how ADMF-PC handles data flow between phases, memory management, and results persistence. The system uses event tracing as the unified metrics system, with configurable retention policies and storage options.

## Core Principles

1. **Event Tracing IS the Metrics System**: We don't maintain separate data tracking - events are processed for metrics then retained/discarded based on policy
2. **Container Lifecycle Writing**: Results are written when containers are destroyed, not at phase completion
3. **Container-Level Tracing**: Each container manages its own event tracing and metrics
4. **Workflow-Defined Settings**: Workflows define optimal settings for their phases, with user override capability
5. **Composable Architecture**: Complex sequences (walk-forward, Monte Carlo) naturally handle memory through container lifecycle

## Understanding Execution Hierarchy

### Phases vs Sequences vs Execution Windows

- **Phase**: High-level workflow step (e.g., "parameter_search", "validation")
- **Sequence**: Execution pattern that implements the phase (e.g., "train_test", "walk_forward")
- **Execution Window**: Actual container lifecycle where topology runs

Example with walk-forward:
```
Walk-Forward Phase
├── Window 0: Train[Jan-Jun] → Test[Jul] (containers created → destroyed → results written)
├── Window 1: Train[Feb-Jul] → Test[Aug] (containers created → destroyed → results written)
└── Window 2: Train[Mar-Aug] → Test[Sep] (containers created → destroyed → results written)
```

### When Results Are Written

Results are written when containers are destroyed, NOT at phase completion:
- **Single-pass**: Write once when containers destroyed
- **Train/Test**: Write after train containers destroyed, again after test
- **Walk-forward**: Write after each window's containers destroyed
- **Monte Carlo**: Write after each simulation batch

This keeps memory usage bounded regardless of sequence complexity.

## User Configuration

Users simply specify the workflow and optional overrides:

```yaml
workflow: train_test_optimization

data:
  symbols: [SPY]
  start: "2020-01-01"
  end: "2023-12-31"

# Optional: custom results directory
results_dir: my_experiment_2024_01

# Optional: override workflow defaults
results_storage: memory  # Override all phases

# Or per-phase overrides
phase_overrides:
  parameter_search:
    results_storage: memory  # Just this phase
    event_tracing: ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL', 'PORTFOLIO_UPDATE']
    retention_policy: sliding_window
```

## Workflow-Level Configuration

Workflows define defaults with override support:

```python
# In train_test_optimization.py
class TrainTestOptimizationWorkflow:
    """Workflow with intelligent override handling."""
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        # Get any phase overrides from user config
        phase_overrides = config.get('phase_overrides', {})
        
        # Apply global override if present
        global_storage = config.get('results_storage')
        
        # Parameter search phase
        param_search_config = {
            **config,  # User config
            # Workflow defaults
            'results_storage': 'disk',  # Large optimization results
            'event_tracing': ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL'],
            'retention_policy': 'trade_complete',
            'data_split': 0.8
        }
        
        # Apply overrides in order of precedence
        if global_storage:
            param_search_config['results_storage'] = global_storage
        if 'parameter_search' in phase_overrides:
            param_search_config.update(phase_overrides['parameter_search'])
        
        # Validation phase
        validation_config = {
            **config,
            # Different defaults for validation
            'results_storage': 'memory',
            'event_tracing': ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL', 'PORTFOLIO_UPDATE'],
            'retention_policy': 'sliding_window',
            'sliding_window_size': 1000
        }
        
        # Apply overrides
        if global_storage:
            validation_config['results_storage'] = global_storage
        if 'validation' in phase_overrides:
            validation_config.update(phase_overrides['validation'])
        
        return {
            "parameter_search": PhaseConfig(
                name="parameter_search",
                sequence="train_test",
                topology="backtest",
                description="Search parameter space on training data",
                config=param_search_config
            ),
            "validation": PhaseConfig(
                name="validation",
                sequence="single_pass",
                topology="backtest", 
                description="Validate on test data with equity tracking",
                config=validation_config
            )
        }
```

### Configuration Options (Set by Workflows)

#### results_storage
- `memory`: Keep results in memory for next phase
- `disk`: Write results to disk at phase completion, free memory

#### event_tracing
- List of event types to trace (e.g., `['POSITION_OPEN', 'POSITION_CLOSE', 'FILL']`)
- Controls what events are captured for metrics and analysis

#### retention_policy
- `trade_complete`: Remove events when position closes (memory efficient)
- `sliding_window`: Keep last N events (for equity curves)
- `minimal`: Only keep active positions

## Implementation Details

### 1. Configuration Flow

The configuration flows through the system as follows:

```
User YAML → Workflow → PhaseConfig.config → Sequencer → TopologyBuilder → Containers
```

### 2. Container-Level Implementation

```python
# In container.py - setup tracing and lifecycle-based writing
def _setup_tracing(self):
    """Setup event tracing based on container config."""
    execution_config = self.config.config.get('execution', {})
    
    # Only setup for portfolio containers
    if self.role != ContainerRole.PORTFOLIO:
        return
    
    # Get configuration from workflow-defined settings
    event_tracing = execution_config.get('event_tracing', ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL'])
    retention_policy = execution_config.get('retention_policy', 'trade_complete')
    sliding_window_size = execution_config.get('sliding_window_size', 1000)
    
    # Determine if we should store equity curve based on events being traced
    store_equity_curve = 'PORTFOLIO_UPDATE' in event_tracing
    
    # Setup metrics tracking
    metrics_config = {
        'initial_capital': self.config.config.get('initial_capital', 100000.0),
        'retention_policy': retention_policy,
        'max_events': sliding_window_size if retention_policy == 'sliding_window' else 10000,
        'collection': {
            'store_equity_curve': store_equity_curve,
            'store_trades': True
        },
        'objective_function': self.config.config.get('objective_function', {'name': 'sharpe_ratio'})
    }
    
    self.streaming_metrics = MetricsEventTracer(metrics_config)
    
    # Subscribe to specified events
    from ..events import EventType
    for event_type_str in event_tracing:
        try:
            event_type = EventType[event_type_str]
            self.event_bus.subscribe(event_type, self.streaming_metrics.trace_event)
        except KeyError:
            logger.warning(f"Unknown event type: {event_type_str}")

async def cleanup(self) -> None:
    """Cleanup resources and save results if configured."""
    # Save results before cleanup if disk storage
    if self.config.config.get('results_storage') == 'disk':
        self._save_results_before_cleanup()
    
    # Continue with normal cleanup
    await super().cleanup()

def _save_results_before_cleanup(self) -> None:
    """Save results during container destruction."""
    if not self.streaming_metrics:
        return
        
    results = self.streaming_metrics.get_results()
    
    # Build path from execution metadata
    metadata = self.config.config.get('metadata', {})
    results_dir = metadata.get('results_dir', './results')
    workflow_id = metadata.get('workflow_id', 'unknown')
    phase_name = metadata.get('phase_name', 'unknown')
    window_id = metadata.get('window_id', '')  # For walk-forward, etc.
    
    # Construct path with window ID if present
    if window_id:
        path = f"{results_dir}/{workflow_id}/{phase_name}/{window_id}"
    else:
        path = f"{results_dir}/{workflow_id}/{phase_name}"
    
    os.makedirs(path, exist_ok=True)
    
    # Write container results
    filepath = f"{path}/{self.container_id}_results.json"
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved results for {self.container_id} to {filepath}")
```

### 3. Composable Phase Configurations (Future Enhancement)

To reduce repetition in workflow definitions, we could later create standard configurations:

```python
# In coordinator_refactor/phase_configs.py (future)
class StandardPhaseConfigs:
    """Reusable phase configurations."""
    
    @staticmethod
    def optimization_phase(name: str, **overrides) -> Dict[str, Any]:
        """Standard config for optimization phases."""
        config = {
            'results_storage': 'disk',
            'event_tracing': ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL'],
            'retention_policy': 'trade_complete',
        }
        config.update(overrides)
        return config
    
    @staticmethod
    def validation_phase(name: str, **overrides) -> Dict[str, Any]:
        """Standard config for validation phases with equity tracking."""
        config = {
            'results_storage': 'memory',
            'event_tracing': ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL', 'PORTFOLIO_UPDATE'],
            'retention_policy': 'sliding_window',
            'sliding_window_size': 1000
        }
        config.update(overrides)
        return config

# Then in workflow:
config={
    **user_config,
    **StandardPhaseConfigs.optimization_phase("parameter_search", data_split=0.8)
}
```
### 4. Sequencer Results Collection

```python
# In sequencer.py - add results collection and storage
def execute_sequence(self, phase_config: PhaseConfig, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute sequence with results collection."""
    sequence_name = phase_config.sequence
    
    if sequence_name not in self.sequences:
        raise ValueError(f"Unknown sequence: {sequence_name}")
    
    sequence = self.sequences[sequence_name]
    
    # Ensure topology builder is available in context
    if not context.get('topology_builder'):
        context['topology_builder'] = self.topology_builder
    
    logger.info(f"Executing sequence '{sequence_name}' for phase '{phase_config.name}'")
    
    try:
        # Build and execute topology
        topology_definition = {
            'mode': phase_config.topology,
            'config': phase_config.config,
            'metadata': {
                'workflow_id': context.get('workflow_name', 'unknown'),
                'phase_name': phase_config.name
            }
        }
        
        topology = self.topology_builder.build_topology(topology_definition)
        
        # Execute the topology (actual implementation would start containers)
        result = self._execute_topology(topology, phase_config, context)
        
        # Collect results from containers
        phase_results = self._collect_phase_results(topology)
        
        # Handle storage based on phase config
        results_storage = phase_config.config.get('results_storage', 'memory')
        
        if results_storage == 'disk':
            results_path = self._save_results_to_disk(phase_results, phase_config, context)
            # Return minimal info to save memory
            result['results_saved'] = True
            result['results_path'] = results_path
            result['summary'] = self._create_summary(phase_results)
        else:
            # Keep in memory for next phase
            result['phase_results'] = phase_results
        
        # Add phase metadata
        result['sequence_name'] = sequence_name
        result['phase_name'] = phase_config.name
        result['success'] = True
        
        # Collect outputs as specified in phase config
        output = {}
        for key, should_collect in phase_config.output.items():
            if should_collect and key in phase_results:
                output[key] = phase_results[key]
        result['output'] = output
        
        return result
        
    except Exception as e:
        logger.error(f"Sequence '{sequence_name}' failed for phase '{phase_config.name}': {e}")
        return {
            'success': False,
            'sequence_name': sequence_name,
            'phase_name': phase_config.name,
            'error': str(e)
        }

def _collect_phase_results(self, topology: Dict[str, Any]) -> Dict[str, Any]:
    """Collect results from all portfolio containers."""
    results = {
        'container_results': {},
        'aggregate_metrics': {},
        'trades': [],
        'equity_curves': {}
    }
    
    containers = topology.get('containers', {})
    portfolio_results = []
    
    for container_id, container in containers.items():
        # Collect from portfolio containers
        if hasattr(container, 'streaming_metrics') and container.streaming_metrics:
            container_results = container.streaming_metrics.get_results()
            results['container_results'][container_id] = container_results
            
            # Aggregate portfolio data
            if container.role == ContainerRole.PORTFOLIO:
                portfolio_results.append(container_results)
                
                # Collect trades if available
                if 'trades' in container_results:
                    results['trades'].extend(container_results['trades'])
                
                # Store equity curve if available
                if 'equity_curve' in container_results:
                    results['equity_curves'][container_id] = container_results['equity_curve']
    
    # Calculate aggregate metrics
    if portfolio_results:
        results['aggregate_metrics'] = self._aggregate_portfolio_metrics(portfolio_results)
    
    return results

def _save_results_to_disk(self, results: Dict[str, Any], 
                          phase_config: PhaseConfig, 
                          context: Dict[str, Any]) -> str:
    """Save results to disk and return path."""
    import json
    import os
    
    # Build path using custom results_dir if provided
    base_results_dir = "./results"
    custom_dir = context.get('results_dir', '')
    workflow_id = context.get('workflow_name', 'unknown')
    phase_name = phase_config.name
    
    if custom_dir:
        results_dir = os.path.join(base_results_dir, custom_dir, phase_name)
    else:
        results_dir = os.path.join(base_results_dir, workflow_id, phase_name)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Save container results separately for easier analysis
    container_dir = os.path.join(results_dir, 'containers')
    os.makedirs(container_dir, exist_ok=True)
    
    for container_id, container_results in results['container_results'].items():
        filepath = os.path.join(container_dir, f"{container_id}_results.json")
        with open(filepath, 'w') as f:
            json.dump(container_results, f, indent=2, default=str)
    
    # Save aggregate results
    aggregate_path = os.path.join(results_dir, 'aggregate_results.json')
    with open(aggregate_path, 'w') as f:
        json.dump({
            'aggregate_metrics': results['aggregate_metrics'],
            'total_trades': len(results['trades']),
            'containers_tracked': len(results['container_results'])
        }, f, indent=2)
    
    # Save trades if present
    if results['trades']:
        trades_path = os.path.join(results_dir, 'all_trades.json')
        with open(trades_path, 'w') as f:
            json.dump(results['trades'], f, indent=2, default=str)
    
    # Save phase summary
    summary = {
        'phase': phase_name,
        'timestamp': datetime.now().isoformat(),
        'containers': list(results['container_results'].keys()),
        'metrics_summary': results['aggregate_metrics'],
        'config': {
            'results_storage': phase_config.config.get('results_storage'),
            'tracing_level': phase_config.config.get('tracing_level')
        }
    }
    
    summary_path = os.path.join(results_dir, 'phase_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved phase results to {results_dir}")
    return results_dir

def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
    """Create minimal summary for memory efficiency."""
    return {
        'best_sharpe': results['aggregate_metrics'].get('best_sharpe_ratio', 0),
        'avg_return': results['aggregate_metrics'].get('avg_total_return', 0),
        'total_trades': len(results.get('trades', [])),
        'containers': len(results['container_results'])
    }

def _aggregate_portfolio_metrics(self, portfolio_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics from multiple portfolios."""
    if not portfolio_results:
        return {}
    
    # Find best performing portfolio
    best_sharpe = -float('inf')
    best_portfolio_metrics = None
    
    all_metrics = []
    for result in portfolio_results:
        metrics = result.get('metrics', {})
        if metrics:
            all_metrics.append(metrics)
            sharpe = metrics.get('sharpe_ratio', -float('inf'))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_portfolio_metrics = metrics
    
    if not all_metrics:
        return {}
    
    # Calculate aggregates
    aggregate = {
        'best_sharpe_ratio': best_sharpe,
        'best_portfolio_metrics': best_portfolio_metrics,
        'portfolio_count': len(portfolio_results)
    }
    
    # Average key metrics
    for metric_name in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']:
        values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
        if values:
            aggregate[f'avg_{metric_name}'] = sum(values) / len(values)
    
    return aggregate
```

### 4. Workflow Integration

```python
# In workflow.py - pass results_dir to context
def _execute_phase(self, phase_config: PhaseConfig, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single phase with proper context."""
    # Add results directory to context if specified
    if 'results_dir' in self.config:
        context['results_dir'] = self.config['results_dir']
    
    # Add workflow metadata
    context['workflow_name'] = context.get('workflow_name', 'unknown')
    
    # Delegate to sequencer
    result = self.sequencer.execute_sequence(phase_config, context)
    
    # Add topology info
    result['topology'] = phase_config.topology
    
    return result
```

## Results Directory Structure

With custom results directory:
```
results/
└── my_experiment_2024_01/        # Custom directory
    ├── train/
    │   ├── containers/
    │   │   ├── portfolio_001_results.json
    │   │   ├── portfolio_002_results.json
    │   │   └── ...
    │   ├── aggregate_results.json
    │   ├── all_trades.json
    │   └── phase_summary.json
    └── test/
        └── (in memory or on disk based on config)
```

Without custom directory:
```
results/
└── train_test_workflow/          # Uses workflow name
    ├── train/
    │   └── ...
    └── test/
        └── ...
```

## Memory Estimates

### Minimal Tracing (Position Events Only)
- ~10-20 active trades per portfolio
- ~5 events per active trade
- 100 portfolios: 10KB * 100 = 1MB

### Equity Tracing (All Portfolio Updates)
- ~100K portfolio updates per portfolio
- Sliding window of 1000 events
- 100 portfolios: 1MB * 100 = 100MB

## Benefits

1. **Simple Configuration**: Just two fields per phase
2. **Memory Efficient**: Write large optimization results to disk
3. **Flexible Organization**: Custom results directories
4. **Unified System**: Event tracing provides both metrics and debug info
5. **Scalable**: Can handle parameter searches with hundreds of portfolios

## Future Enhancements

1. **Streaming to Disk**: For very long backtests or debugging
2. **Compression**: Gzip JSON files to save disk space
3. **Selective Loading**: Load specific container results from disk
4. **Memory Monitoring**: Automatic switch to disk when memory pressure high
5. **Parallel Writing**: Write container results in parallel for speed