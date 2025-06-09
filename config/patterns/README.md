# ADMF-PC Pattern System

This directory contains declarative patterns that define how the system operates without requiring code changes.

## Location

Patterns are now located in `config/patterns/` to make them more accessible as user-configurable components.

## Directory Structure

```
config/patterns/
├── topologies/      # HOW components are connected
│   ├── backtest.yaml
│   ├── signal_generation.yaml
│   └── signal_replay.yaml
├── sequences/       # HOW phases are executed
│   ├── single_pass.yaml
│   ├── walk_forward.yaml
│   ├── monte_carlo.yaml
│   ├── train_test.yaml
│   └── k_fold.yaml
├── workflows/       # Multi-phase workflow patterns
│   ├── simple_backtest.yaml
│   ├── adaptive_ensemble.yaml
│   └── ...
└── examples/        # Example patterns for reference
    ├── custom_sequence.yaml
    └── custom_topology.yaml
```

## Topology Patterns

Topology patterns define the structure of a trading system:
- What containers to create (data, features, portfolio, execution)
- How to wire them together (routes and behaviors)
- What components to use (strategies, risk validators)

### Example: Backtest Topology
```yaml
name: backtest
containers:
  - name_template: "{symbol}_{timeframe}_data"
    type: data
    foreach:
      symbol: {from_config: symbols}
      timeframe: {from_config: timeframes}
routes:
  - type: risk_service
    source_pattern: "portfolio_*"
```

## Sequence Patterns

Sequence patterns define execution strategies:
- How many times to run (iterations)
- How to modify configuration between runs
- How to aggregate results

### Example: Walk Forward Sequence
```yaml
name: walk_forward
iterations:
  type: windowed
  window_generator:
    type: rolling
    train_periods: 252
    test_periods: 63
aggregation:
  type: statistical
  operations: [mean, std, min, max]
```

## Pattern Composition

Patterns can be composed to create complex workflows:

1. **Workflow** selects patterns:
   ```yaml
   phases:
     - name: optimization
       topology: backtest      # Use backtest topology
       sequence: walk_forward  # With walk-forward execution
   ```

2. **Sequence** controls execution:
   - Generates iterations (windows, parameter sets, etc.)
   - Modifies configuration for each iteration
   - Aggregates results

3. **Topology** defines structure:
   - Creates containers and components
   - Sets up communication routes
   - Implements the trading logic

## Value Resolution

Patterns support various value resolution mechanisms:

### Template Strings
```yaml
name_template: "{symbol}_{timeframe}_data"
```

### Configuration References
```yaml
train_periods:
  from_config: walk_forward.train_periods
  default: 252
```

### Context References
```yaml
target: "$root_event_bus"
parameters: "{train.optimal_parameters}"
```

### Foreach Loops
```yaml
foreach:
  symbol: {from_config: symbols}
  timeframe: {from_config: timeframes}
```

## Creating Custom Patterns

To create a custom pattern:

1. **Choose the right type**:
   - Topology: For new system structures
   - Sequence: For new execution strategies

2. **Define the pattern**:
   ```yaml
   name: my_custom_pattern
   description: What this pattern does
   # ... pattern definition
   ```

3. **Place in correct directory**:
   - `config/patterns/topologies/` for topology patterns
   - `config/patterns/sequences/` for sequence patterns
   - `config/patterns/workflows/` for workflow patterns

4. **Use in configuration**:
   ```yaml
   phases:
     - topology: my_custom_pattern
       sequence: single_pass
   ```

## Benefits

1. **No Code Required**: Define complex systems in YAML
2. **Reusable**: Patterns can be shared and versioned
3. **Composable**: Mix and match patterns
4. **Extensible**: Add new patterns without code changes
5. **Testable**: Validate patterns before execution