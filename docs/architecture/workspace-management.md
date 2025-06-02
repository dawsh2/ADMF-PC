# Workspace Management and File-Based Communication

## Overview

The Coordinator implements a sophisticated workspace management system that enables complex multi-phase workflows through standardized file-based communication patterns. Each workflow execution creates a structured workspace with defined directories for different types of intermediate results: signals, performance metrics, analysis outputs, metadata, and checkpoints.

## Why File-Based Communication?

This file-based communication approach provides several architectural advantages:

1. **Natural Checkpointing**: Any phase can be restarted from its last successful completion without affecting previous phases
2. **Debugging and Analysis**: All intermediate results are inspectable and modifiable
3. **Parallel Execution**: File-based communication naturally supports concurrent readers and writers
4. **Workflow Composition**: Outputs of one workflow can become inputs to another
5. **Manual Intervention**: Researchers can inspect and modify intermediate results between phases

## Workspace Structure

```
./results/workflow_123/
├── signals/              # Trading signals from strategies
│   ├── trial_0.jsonl     # Line-delimited JSON for streaming
│   ├── trial_1.jsonl
│   └── ...
├── performance/          # Performance metrics
│   ├── trial_0.json      # Complete metrics per trial
│   ├── trial_1.json
│   └── summary.json      # Aggregated results
├── analysis/             # Analysis outputs
│   ├── regime_optimal_params.json
│   ├── ensemble_weights.json
│   └── correlation_matrix.csv
├── metadata/             # Workflow metadata
│   ├── config.yaml       # Original configuration
│   ├── execution_log.json
│   └── phase_timing.json
├── checkpoints/          # Resumability data
│   ├── phase_1_complete.checkpoint
│   ├── phase_2_complete.checkpoint
│   └── current_state.json
└── visualizations/       # Optional plots/reports
    ├── performance_chart.png
    └── report.html
```

## Naming Conventions

The workspace management system follows a strict naming convention that encodes workflow identity, phase information, and result types in file paths:

```python
def generate_workspace_path(workflow_id: str, phase: str, result_type: str) -> Path:
    """Generate standardized workspace paths"""
    base = Path(f"./results/{workflow_id}")
    
    # Standard result types have defined locations
    type_dirs = {
        'signals': 'signals',
        'performance': 'performance',
        'analysis': 'analysis',
        'checkpoint': 'checkpoints'
    }
    
    return base / type_dirs.get(result_type, result_type) / f"{phase}_{result_type}.json"
```

## Multi-Phase Data Flow

### Complete Example: 4-Phase Optimization Workflow

```
COORDINATOR CREATES WORKSPACE:
└── ./results/workflow_123/
    ├── signals/
    ├── performance/
    ├── analysis/
    ├── metadata/
    └── checkpoints/

PHASE 1: Parameter Discovery
├── COORDINATOR → OPTIMIZER: "Expand parameters"
├── OPTIMIZER → COORDINATOR: [param_set_1, param_set_2, ...]
├── COORDINATOR → BACKTESTER: "Execute with paths.signals/trial_N.jsonl"
└── OUTPUT: signals/trial_*.jsonl, performance/trial_*.json

PHASE 2: Regime Analysis  
├── COORDINATOR → OPTIMIZER: "Analyze at paths.performance/"
├── OPTIMIZER reads: performance/trial_*.json
├── OPTIMIZER analyzes: Find best params per regime
└── OPTIMIZER writes: analysis/regime_optimal_params.json

PHASE 3: Ensemble Optimization
├── COORDINATOR → OPTIMIZER: "Expand weight space"
├── OPTIMIZER → COORDINATOR: [weight_set_1, weight_set_2, ...]
├── COORDINATOR → BACKTESTER: "Replay signals at paths.signals/"
├── BACKTESTER reads: signals/*, analysis/regime_optimal_params.json
├── BACKTESTER writes: performance/ensemble_trial_*.json
├── COORDINATOR → OPTIMIZER: "Analyze ensemble results"
└── OPTIMIZER writes: analysis/ensemble_weights.json

PHASE 4: Validation
├── COORDINATOR → BACKTESTER: "Full test with all optimizations"
├── BACKTESTER reads: analysis/regime_optimal_params.json, analysis/ensemble_weights.json
└── BACKTESTER writes: performance/validation_results.json
```

## File Formats

### Signal Files (JSONL)
```jsonl
{"timestamp": "2023-01-01T09:30:00", "symbol": "AAPL", "action": "BUY", "strength": 0.8, "strategy": "momentum"}
{"timestamp": "2023-01-01T09:31:00", "symbol": "AAPL", "action": "HOLD", "strength": 0.6, "strategy": "momentum"}
{"timestamp": "2023-01-01T09:32:00", "symbol": "AAPL", "action": "SELL", "strength": 0.9, "strategy": "momentum"}
```

### Performance Files (JSON)
```json
{
  "trial_id": "trial_0",
  "parameters": {
    "lookback": 20,
    "threshold": 0.02
  },
  "metrics": {
    "total_return": 0.156,
    "sharpe_ratio": 1.45,
    "max_drawdown": -0.082,
    "win_rate": 0.58
  },
  "trades": 145,
  "period": {
    "start": "2023-01-01",
    "end": "2023-12-31"
  }
}
```

### Analysis Files (JSON)
```json
{
  "regime_optimal_params": {
    "bull_market": {
      "lookback": 10,
      "threshold": 0.01,
      "performance": 0.234
    },
    "bear_market": {
      "lookback": 30,
      "threshold": 0.03,
      "performance": 0.145
    },
    "neutral_market": {
      "lookback": 20,
      "threshold": 0.02,
      "performance": 0.089
    }
  },
  "analysis_metadata": {
    "method": "regime_classification",
    "trials_analyzed": 18,
    "timestamp": "2024-01-15T14:30:00Z"
  }
}
```

## Checkpointing and Resumability

### Checkpoint Creation
```python
class WorkspaceManager:
    def checkpoint_phase(self, phase_name: str, state: Dict) -> None:
        """Create checkpoint after phase completion"""
        checkpoint = {
            'phase': phase_name,
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'completed_trials': self.get_completed_trials(),
            'workspace_hash': self.compute_workspace_hash()
        }
        
        checkpoint_path = self.workspace / 'checkpoints' / f'{phase_name}.checkpoint'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
```

### Resume from Checkpoint
```python
def resume_workflow(self, workflow_id: str) -> Optional[str]:
    """Resume workflow from last checkpoint"""
    workspace = Path(f"./results/{workflow_id}")
    checkpoints = sorted(workspace.glob("checkpoints/*.checkpoint"))
    
    if not checkpoints:
        return None  # Start from beginning
    
    # Load most recent checkpoint
    last_checkpoint = checkpoints[-1]
    with open(last_checkpoint) as f:
        checkpoint_data = json.load(f)
    
    # Verify workspace integrity
    if self.verify_workspace_integrity(workspace, checkpoint_data):
        return checkpoint_data['phase']
    else:
        raise WorkspaceCorruptedError("Workspace modified since checkpoint")
```

## Parallel Access Patterns

### Concurrent Writing
```python
class SignalWriter:
    """Thread-safe signal writer"""
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.locks = {}  # File-level locks
    
    def write_signal(self, trial_id: str, signal: Dict) -> None:
        """Write signal with file locking"""
        file_path = self.workspace / 'signals' / f'trial_{trial_id}.jsonl'
        
        # Acquire file lock
        if file_path not in self.locks:
            self.locks[file_path] = threading.Lock()
            
        with self.locks[file_path]:
            with open(file_path, 'a') as f:
                f.write(json.dumps(signal) + '\n')
```

### Concurrent Reading
```python
class PerformanceReader:
    """Parallel performance file reader"""
    def read_all_trials(self, workspace: Path) -> List[Dict]:
        """Read all trial results in parallel"""
        performance_files = list(workspace.glob("performance/trial_*.json"))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for file_path in performance_files:
                future = executor.submit(self._read_json, file_path)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to read file: {e}")
                    
        return results
```

## Workspace Lifecycle

### Creation
```python
def create_workspace(self, workflow_config: Dict) -> Path:
    """Create new workspace with standard structure"""
    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    workspace = Path(f"./results/{workflow_id}")
    
    # Create directory structure
    directories = [
        'signals', 'performance', 'analysis',
        'metadata', 'checkpoints', 'visualizations'
    ]
    
    for dir_name in directories:
        (workspace / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Save original configuration
    config_path = workspace / 'metadata' / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(workflow_config, f)
    
    return workspace
```

### Cleanup
```python
def cleanup_workspace(self, workspace: Path, keep_days: int = 7) -> None:
    """Clean up old workspaces"""
    cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
    
    for workspace_dir in Path("./results").iterdir():
        if workspace_dir.is_dir():
            # Check modification time
            mtime = workspace_dir.stat().st_mtime
            
            if mtime < cutoff_time:
                # Archive before deletion
                self.archive_workspace(workspace_dir)
                shutil.rmtree(workspace_dir)
                logger.info(f"Cleaned up old workspace: {workspace_dir}")
```

## Advanced Patterns

### Workspace Composition
```python
class CompositeWorkspace:
    """Combine outputs from multiple workflows"""
    def __init__(self, workspace_ids: List[str]):
        self.workspaces = [Path(f"./results/{wid}") for wid in workspace_ids]
    
    def aggregate_signals(self) -> pd.DataFrame:
        """Combine signals from multiple workflows"""
        all_signals = []
        
        for workspace in self.workspaces:
            signal_files = workspace.glob("signals/*.jsonl")
            for signal_file in signal_files:
                df = pd.read_json(signal_file, lines=True)
                df['workspace'] = workspace.name
                all_signals.append(df)
        
        return pd.concat(all_signals, ignore_index=True)
```

### Incremental Processing
```python
class IncrementalProcessor:
    """Process new files as they appear"""
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.processed = set()
    
    def process_new_files(self, pattern: str, processor: Callable) -> None:
        """Process files matching pattern"""
        while True:
            files = set(self.workspace.glob(pattern))
            new_files = files - self.processed
            
            for file_path in new_files:
                try:
                    processor(file_path)
                    self.processed.add(file_path)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
            
            time.sleep(1)  # Poll interval
```

## Best Practices

### DO:
- Use atomic file operations for consistency
- Include timestamps in all output files
- Validate file contents before processing
- Use file locks for concurrent access
- Clean up old workspaces regularly

### DON'T:
- Modify files after phase completion
- Assume file existence without checking
- Use binary formats for intermediate data
- Store large datasets in workspace
- Ignore file system limitations

## Error Handling

### Corrupted File Recovery
```python
def read_with_recovery(self, file_path: Path) -> Optional[Dict]:
    """Read file with corruption recovery"""
    try:
        # Try normal read
        with open(file_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Try line-by-line recovery for JSONL
        if file_path.suffix == '.jsonl':
            valid_lines = []
            with open(file_path) as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        valid_lines.append(json.loads(line))
                    except:
                        logger.warning(f"Skipping corrupted line {line_no} in {file_path}")
            return {'recovered_data': valid_lines, 'corrupted': True}
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return None
```

## Summary

Workspace management provides:

1. **Reproducibility**: Complete audit trail of workflow execution
2. **Resumability**: Restart from any checkpoint
3. **Debuggability**: Inspect intermediate results
4. **Composability**: Combine workflow outputs
5. **Scalability**: Parallel processing support

The file-based approach enables sophisticated multi-phase workflows while maintaining simplicity and transparency.