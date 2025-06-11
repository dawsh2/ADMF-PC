# CLI Piping Design for Multi-Phase Workflows

## Vision

Enable Unix-style command composition for complex trading system workflows:

```bash
# Complete regime-adaptive optimization pipeline
python main.py --signal-generation grid_search.yaml | \
python main.py --signal-replay --optimize-ensemble | \
python main.py --backtest --validate > results.json
```

## Design Principles

1. **Unix Philosophy**: Each command does one thing well
2. **Streaming JSON**: Pass structured data between phases
3. **Smart Defaults**: Later phases infer from previous output
4. **Progressive Enhancement**: Works with files or pipes

## Implementation Approach

### Phase Output Format

Each phase outputs JSON with:
- `metadata`: Phase info, parameters used
- `artifacts`: Paths to generated files
- `config`: Configuration for next phase
- `results`: Phase-specific results

Example:
```json
{
  "metadata": {
    "phase": "signal_generation",
    "timestamp": "2025-01-06T10:00:00Z",
    "strategies": ["momentum", "mean_reversion"],
    "parameters": {...}
  },
  "artifacts": {
    "signals": "workspaces/signal_gen_abc123/signals/",
    "summary": "workspaces/signal_gen_abc123/summary.json"
  },
  "config": {
    "strategies": [...],  // For next phase
    "features": [...]
  },
  "results": {
    "total_signals": 15420,
    "windows_processed": 12
  }
}
```

### CLI Enhancements

#### 1. Support stdin config
```python
# When config is "-", read from stdin
if args.signal_generation == "-":
    config = json.load(sys.stdin)
```

#### 2. Structured output mode
```python
# Add --output-format flag
parser.add_argument('--output-format', 
    choices=['human', 'json', 'pipe'],
    default='human'
)
```

#### 3. Input chaining
```python
# Add --from-pipe to read previous phase output
parser.add_argument('--from-pipe',
    action='store_true',
    help='Read previous phase output from stdin'
)
```

## Usage Examples

### Example 1: Basic Pipeline
```bash
# Generate signals with grid search
python main.py --signal-generation grid_search.yaml --output-format pipe | \
# Optimize ensemble weights using those signals
python main.py --signal-replay --from-pipe --optimize-ensemble | \
# Final validation
python main.py --backtest --from-pipe --validate
```

### Example 2: With Intermediate Files
```bash
# Phase 1: Generate signals
python main.py --signal-generation grid_search.yaml \
  --output-format json > phase1.json

# Phase 2: Optimize ensemble (reads phase1.json)
python main.py --signal-replay --from-file phase1.json \
  --optimize-ensemble --output-format json > phase2.json

# Phase 3: Final backtest
python main.py --backtest --from-file phase2.json \
  --validate --output-format human
```

### Example 3: Conditional Execution
```bash
# Only proceed if signal generation succeeds
python main.py --signal-generation grid.yaml --output-format pipe | \
jq -e '.results.total_signals > 1000' && \
python main.py --signal-replay --from-pipe --optimize
```

### Example 4: Parallel Processing
```bash
# Run multiple parameter sweeps in parallel
parallel -j 4 'python main.py --signal-generation {} --output-format pipe' \
  ::: configs/momentum_*.yaml | \
python main.py --signal-replay --from-pipe --merge --optimize
```

## Config Inheritance

When using `--from-pipe`, the next phase inherits:
- Strategy configurations
- Feature definitions  
- Data settings (unless overridden)
- Relevant artifacts paths

Example flow:
```yaml
# grid_search.yaml (Phase 1)
data:
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  
strategies:
  - type: momentum
    param_grid:
      fast_period: [5, 10, 20]
      slow_period: [20, 30, 50]
      
walk_forward:
  train_months: 12
  test_months: 3
  step_months: 3
```

Phase 2 automatically inherits strategies and uses signal artifacts:
```bash
# This automatically uses momentum strategies and signals from phase 1
python main.py --signal-replay --from-pipe --optimize-ensemble
```

## Advanced Features

### 1. Workflow Templates
```bash
# Save common pipelines
alias regime-optimize='python main.py --signal-generation - --output-format pipe | \
                      python main.py --signal-replay --from-pipe --optimize-ensemble | \
                      python main.py --backtest --from-pipe --validate'

# Use with any config
cat my_config.yaml | regime-optimize
```

### 2. Progress Monitoring
```bash
# Tee to file while piping
python main.py --signal-generation grid.yaml --output-format pipe | \
tee phase1.json | \
python main.py --signal-replay --from-pipe --progress
```

### 3. Error Handling
```bash
# Use set -e and pipefail for robust pipelines
set -eo pipefail

python main.py --signal-generation grid.yaml --output-format pipe | \
python main.py --signal-replay --from-pipe || {
  echo "Pipeline failed at signal replay"
  exit 1
}
```

## Implementation Phases

### Phase 1: Basic Piping
- [ ] Add `--output-format` flag
- [ ] Support JSON output mode
- [ ] Parse stdin when config is "-"

### Phase 2: Smart Chaining
- [ ] Add `--from-pipe` flag
- [ ] Implement config inheritance
- [ ] Auto-detect artifact paths

### Phase 3: Advanced Features
- [ ] Parallel signal merging
- [ ] Progress reporting in pipe mode
- [ ] Streaming results (don't wait for completion)

## Benefits

1. **Composability**: Mix and match phases as needed
2. **Debugging**: Save intermediate results easily
3. **Parallelization**: Natural parallel processing support
4. **Integration**: Works with standard Unix tools (jq, tee, parallel)
5. **Flexibility**: Use files or pipes as needed