# Topology-First CLI Enhancement

## Enhanced CLI Parser

```python
def parse_arguments() -> CLIArgs:
    """Enhanced parser supporting natural topology → sequence → workflow progression."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Adaptive Decision Making Framework',
        epilog='''
        Examples:
          # Direct topology execution (simplest)
          %(prog)s --topology signal_generation --config data.yaml
          
          # Topology with sequence (iteration)
          %(prog)s --topology backtest --sequence walk_forward --config strategy.yaml
          
          # Full workflow (composition)
          %(prog)s --workflow research_pipeline --config research.yaml
        '''
    )
    
    # Execution mode (mutually exclusive group)
    exec_group = parser.add_mutually_exclusive_group()
    
    exec_group.add_argument(
        '--topology', '-t',
        type=str,
        help='Execute a topology directly (e.g., signal_generation, backtest)'
    )
    
    exec_group.add_argument(
        '--workflow', '-w',
        type=str,
        help='Execute a workflow pattern (e.g., research_pipeline)'
    )
    
    # Sequence can be combined with topology
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        help='Apply sequence to topology (e.g., walk_forward, parameter_sweep)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Configuration file (YAML)'
    )
    
    # ... rest of arguments ...
```

## Natural Usage Examples

### 1. Testing a Strategy Signal Generation
```bash
# Just see what signals a strategy generates
python main.py --topology signal_generation --config my_strategy.yaml

# Config just needs data and strategy info:
# data:
#   path: SPY.csv
#   start: 2023-01-01
#   end: 2023-12-31
# strategies:
#   - type: momentum
#     params:
#       fast: 10
#       slow: 30
```

### 2. Quick Backtest
```bash
# Run a simple backtest
python main.py --topology backtest --config backtest.yaml

# Config adds execution details:
# data: ...
# strategies: ...
# execution:
#   initial_capital: 100000
#   commission: 0.001
```

### 3. Walk-Forward Analysis
```bash
# Add walk-forward sequencing
python main.py --topology backtest --sequence walk_forward --config walk_forward.yaml

# Config adds sequence parameters:
# data: ...
# strategies: ...
# execution: ...
# sequence:
#   train_window: 252
#   test_window: 63
#   step_size: 21
```

### 4. Full Research Pipeline
```bash
# Compose multiple phases
python main.py --workflow research_pipeline --config research.yaml

# Now using full workflow capabilities
```

## Implementation in main.py

```python
def main():
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args)
    
    # Load configuration
    config = load_yaml_config(args.config)
    
    # Create coordinator
    coordinator = Coordinator()
    
    # Route to appropriate execution method
    if args.topology:
        # Direct topology execution
        if args.sequence:
            # Topology + sequence
            result = coordinator.run_sequence(
                sequence=args.sequence,
                topology=args.topology,
                config=config
            )
        else:
            # Just topology
            result = coordinator.run_topology(args.topology, config)
            
    elif args.workflow:
        # Full workflow execution
        result = coordinator.run_workflow(args.workflow, config)
        
    else:
        # Config-driven (check config for topology/workflow)
        result = coordinator.run(config)
    
    # Display results
    display_results(result, args)
```

## Benefits for Users

### 1. Progressive Complexity
```bash
# Start simple
python main.py -t signal_generation -c config.yaml

# Add iteration when needed
python main.py -t backtest -s parameter_sweep -c config.yaml

# Graduate to workflows when ready
python main.py -w optimization_pipeline -c config.yaml
```

### 2. Clear Mental Model
- **Topology**: How components are wired (the graph)
- **Sequence**: How to iterate over data/parameters
- **Workflow**: How to compose multiple topology+sequence combinations

### 3. Better Debugging
```bash
# Test just the topology wiring
python main.py -t my_custom_topology -c test.yaml --dry-run

# Test with single data window
python main.py -t backtest -c single_window.yaml

# Test sequence logic
python main.py -t backtest -s walk_forward -c test.yaml --max-iterations 2
```

## Config File Simplification

### For Topology Execution
```yaml
# Simple topology config - just what's needed
data:
  source: file
  path: SPY.csv
  
strategies:
  - type: momentum
    params:
      period: 20
```

### For Sequence Execution
```yaml
# Add sequence-specific config
sequence:
  window_size: 252
  step_size: 21
  
# Rest of config...
```

### For Workflow Execution
```yaml
# Full workflow config with phases
workflow:
  phases:
    - name: optimization
      topology: backtest
      sequence: parameter_sweep
      config:
        # phase-specific overrides
```

This approach makes the system much more approachable and follows the principle of "make simple things simple, and complex things possible."