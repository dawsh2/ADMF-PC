# CLI Topology Flags Design

## Current (Chunky) Approach
```bash
python main.py --topology signal_generation --config config.yaml
python main.py --topology backtest --config config.yaml
python main.py --topology signal_replay --config config.yaml
```

## Proposed Clean Approach

### Direct Topology Flags
```bash
# Signal generation
python main.py --signal-generation config.yaml

# Backtesting
python main.py --backtest config.yaml

# Signal replay
python main.py --signal-replay config.yaml

# Live trading (future)
python main.py --live config.yaml
```

### Benefits
1. **Cleaner syntax** - no redundant "--topology" prefix
2. **Self-documenting** - flag name tells you what it does
3. **Pure configs** - configs only contain strategy/data/execution settings
4. **Unix-like** - follows standard CLI patterns

### Config Files Become Pure Business Logic
```yaml
# config.yaml - ONLY business logic, no topology/workflow specs
data:
  symbols: ["SPY", "QQQ"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"

strategies:
  - type: momentum
    params:
      fast_period: 10
      slow_period: 30

execution:
  initial_capital: 100000
  commission: 0.001
```

### Future: Command Composition
```bash
# Generate signals and immediately replay them
python main.py --signal-generation config.yaml | python main.py --signal-replay -

# Backtest multiple configs
for config in configs/*.yaml; do
    python main.py --backtest $config
done

# Parameter sweep via shell
for period in 10 20 30; do
    sed "s/fast_period: .*/fast_period: $period/" base.yaml | \
    python main.py --backtest -
done
```

## Implementation Plan

### 1. Add Topology Action Flags
```python
# In parser.py
topology_group = parser.add_mutually_exclusive_group()

topology_group.add_argument(
    '--signal-generation', '-sg',
    metavar='CONFIG',
    help='Generate trading signals from strategies'
)

topology_group.add_argument(
    '--backtest', '-bt',
    metavar='CONFIG',
    help='Run backtest simulation'
)

topology_group.add_argument(
    '--signal-replay', '-sr',
    metavar='CONFIG',
    help='Replay previously generated signals'
)

topology_group.add_argument(
    '--optimize', '-opt',
    metavar='CONFIG',
    help='Run parameter optimization'
)
```

### 2. Update CLIArgs
```python
@dataclass
class CLIArgs:
    # Topology action flags (only one can be set)
    signal_generation: Optional[str] = None  # Config path
    backtest: Optional[str] = None          # Config path
    signal_replay: Optional[str] = None     # Config path
    optimize: Optional[str] = None          # Config path
    
    # Workflow flag (for complex multi-phase workflows)
    workflow: Optional[str] = None
    
    # Common options
    verbose: bool = False
    dry_run: bool = False
    # ... etc
```

### 3. Update main.py Logic
```python
def main():
    args = parse_arguments()
    
    # Determine config and action
    if args.signal_generation:
        config = load_yaml_config(args.signal_generation)
        result = coordinator.run_topology('signal_generation', config)
        
    elif args.backtest:
        config = load_yaml_config(args.backtest)
        result = coordinator.run_topology('backtest', config)
        
    elif args.signal_replay:
        config = load_yaml_config(args.signal_replay)
        result = coordinator.run_topology('signal_replay', config)
        
    elif args.workflow:
        # Complex workflows still supported
        config = load_yaml_config(args.config)
        result = coordinator.run_workflow(args.workflow, config)
        
    else:
        # Backward compatibility: check config for topology/workflow
        config = load_yaml_config(args.config)
        result = coordinator.run(config)
```

## Migration Path

1. **Phase 1**: Add new flags alongside old ones
2. **Phase 2**: Deprecate `--topology` and `--mode` flags
3. **Phase 3**: Remove `topology:` and `workflow:` from configs
4. **Phase 4**: Remove deprecated flags and config fields

## Examples

### Before
```bash
# Chunky and redundant
python main.py --topology signal_generation --config my_strategy.yaml
```

### After
```bash
# Clean and clear
python main.py --signal-generation my_strategy.yaml
```