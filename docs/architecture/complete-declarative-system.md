# Complete Declarative System Architecture

## Overview

ADMF-PC now supports a complete declarative system where trading strategies, execution patterns, and workflows are all defined in YAML rather than code. This creates a three-layer declarative architecture:

1. **Workflows** - Define WHAT business process to execute
2. **Sequences** - Define HOW MANY times to execute phases  
3. **Topologies** - Define HOW components are connected

## Architecture Layers

```
┌─────────────────────────────────────────┐
│          Workflow Layer                 │
│   (Business Process Orchestration)      │
│                                         │
│  - Multi-phase processes                │
│  - Dependencies and conditions          │
│  - Input/output data flow               │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│          Sequence Layer                 │
│      (Execution Strategies)             │
│                                         │
│  - Iterations (single, windowed, grid)  │
│  - Config modifications                 │
│  - Result aggregation                   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│          Topology Layer                 │
│       (System Structure)                │
│                                         │
│  - Containers and components            │
│  - Communication routes                 │
│  - Event flow patterns                  │
└─────────────────────────────────────────┘
```

## Pattern Organization

```
patterns/
├── workflows/              # Business processes
│   ├── adaptive_ensemble.yaml
│   ├── research_pipeline.yaml
│   └── simple_backtest.yaml
├── sequences/              # Execution patterns
│   ├── single_pass.yaml
│   ├── walk_forward.yaml
│   ├── monte_carlo.yaml
│   └── parameter_sweep.yaml
└── topologies/            # System structures
    ├── backtest.yaml
    ├── signal_generation.yaml
    └── signal_replay.yaml
```

## Workflow Patterns

### Structure
```yaml
name: workflow_name
description: What this workflow does

phases:
  - name: phase1
    topology: backtest      # Which topology pattern
    sequence: walk_forward  # Which sequence pattern
    config: {}              # Phase configuration
    depends_on: []          # Dependencies
    conditions: []          # Execution conditions
    inputs: {}              # Input data/files
    outputs: {}             # Output data/files

outputs:                    # Workflow-level outputs
settings:                   # Workflow settings
```

### Features

#### 1. Phase Dependencies
```yaml
phases:
  - name: optimization
  - name: validation
    depends_on: optimization
  - name: deployment
    depends_on: [optimization, validation]
```

#### 2. Conditional Execution
```yaml
conditions:
  - type: metric_threshold
    phase: optimization
    metric: sharpe_ratio
    operator: ">"
    threshold: 1.5
  - type: expression
    expression: "results['optimization']['success'] == True"
```

#### 3. Input/Output Flow
```yaml
phases:
  - name: phase1
    outputs:
      signals: "./signals/{date}/"
  - name: phase2
    inputs:
      signals: "{phase1.outputs.signals}"
```

## Sequence Patterns

### Structure
```yaml
name: sequence_name
description: Execution strategy

iterations:                 # How to iterate
config_modifiers:          # Config changes per iteration
sub_phases:                # Multiple phases per iteration
aggregation:               # Result combination
```

### Iteration Types

#### 1. Single Pass
```yaml
iterations:
  type: single
  count: 1
```

#### 2. Windowed (Walk-Forward)
```yaml
iterations:
  type: windowed
  window_generator:
    type: rolling
    train_periods: 252
    test_periods: 63
    step_size: 21
```

#### 3. Parameter Grid
```yaml
iterations:
  type: parameter_grid
  parameters:
    param1: [1, 2, 3]
    param2: [10, 20, 30]
```

#### 4. Monte Carlo
```yaml
iterations:
  type: repeated
  count: 100
config_modifiers:
  - type: add_seed
    random_seed: "{iteration_index}"
```

## Topology Patterns

### Structure
```yaml
name: topology_name
description: System structure

components:                # Stateless components
containers:                # Container definitions
routes:                    # Communication paths
behaviors:                 # Special logic
```

### Container Creation

#### 1. Simple Containers
```yaml
containers:
  - name: execution
    type: execution
    config:
      mode: backtest
```

#### 2. Dynamic Containers (Foreach)
```yaml
containers:
  - name_template: "{symbol}_{timeframe}_data"
    type: data
    foreach:
      symbol: {from_config: symbols}
      timeframe: {from_config: timeframes}
```

### Routing Patterns

#### 1. Direct Routes
```yaml
routes:
  - type: direct
    source: container1
    target: container2
```

#### 2. Pattern-Based Routes
```yaml
routes:
  - type: broadcast
    source: execution
    targets: "portfolio_*"
```

## Value Resolution

### 1. Template Strings
```yaml
name: "{symbol}_{timeframe}_data"
path: "./results/{phase_name}/{date}/"
```

### 2. Configuration References
```yaml
train_periods:
  from_config: walk_forward.train_periods
  default: 252
```

### 3. Context References
```yaml
parameters: "{train.optimal_parameters}"
start_date: "{previous_phase.end_date}"
```

### 4. Calculations
```yaml
split_date:
  type: calculate
  formula: "start_date + (end_date - start_date) * 0.7"
```

## Complete Example

### User Configuration
```yaml
# config/my_strategy.yaml
symbols: [SPY, QQQ]
strategies:
  - type: momentum
    fast_period: 10
risk_profiles:
  - type: conservative
```

### Workflow Execution
```python
# Python code - minimal!
manager = DeclarativeWorkflowManager()
results = manager.run_workflow({
    'workflow': 'research_pipeline',
    'config': 'config/my_strategy.yaml'
})
```

### What Happens

1. **Workflow Manager** loads `patterns/workflows/research_pipeline.yaml`
2. For each phase:
   - **Topology Builder** loads topology pattern (e.g., `backtest.yaml`)
   - **Sequencer** loads sequence pattern (e.g., `walk_forward.yaml`)
3. System executes according to patterns:
   - Creates containers per topology
   - Iterates per sequence
   - Flows data per workflow

## Benefits

### 1. Complete Separation of Concerns
- **What** (Workflow) vs **How Many** (Sequence) vs **How** (Topology)
- Each layer can be modified independently
- Patterns can be mixed and matched

### 2. No Code Required
- Users work entirely in YAML
- Complex workflows without programming
- Easy to understand and modify

### 3. Reusability
- Share patterns across projects
- Build pattern libraries
- Version control friendly

### 4. Extensibility
- Add new patterns without code changes
- Plugin system for custom behaviors
- Override and extend existing patterns

## Migration Path

### From Code to Declarative

1. **Identify Pattern Type**
   - Business process → Workflow
   - Execution strategy → Sequence
   - System structure → Topology

2. **Extract Configuration**
   - Parameters → YAML config
   - Logic → Pattern definition
   - Calculations → Value resolution

3. **Create Pattern File**
   - Place in appropriate directory
   - Define structure declaratively
   - Test with same inputs

### Example Migration

**Before (Python)**:
```python
class MyWorkflow:
    def run(self):
        # 500 lines of orchestration code
        results1 = self.run_optimization()
        if results1.sharpe > 1.5:
            results2 = self.run_validation()
        # ... more logic
```

**After (YAML)**:
```yaml
phases:
  - name: optimization
    topology: backtest
    sequence: parameter_sweep
  - name: validation
    depends_on: optimization
    conditions:
      - metric_threshold:
          phase: optimization
          metric: sharpe_ratio
          threshold: 1.5
```

## Future Enhancements

1. **Visual Workflow Designer**
   - Drag-and-drop pattern composition
   - Real-time validation
   - Pattern marketplace

2. **Pattern Inheritance**
   ```yaml
   extends: base_workflow
   overrides:
     phases:
       - name: optimization
         sequence: monte_carlo  # Override sequence
   ```

3. **Dynamic Pattern Generation**
   - Patterns that generate patterns
   - Meta-patterns for common structures
   - AI-assisted pattern creation

4. **Cloud Execution**
   - Distribute phases across clusters
   - Serverless execution per phase
   - Result streaming and aggregation

## Conclusion

The complete declarative system transforms ADMF-PC from a code-based framework to a pattern-based platform. Users can:

- Define complex trading workflows in YAML
- Run sophisticated analyses without programming
- Share and reuse patterns across teams
- Focus on strategy rather than implementation

This represents a fundamental shift in how trading systems are built and managed, making advanced quantitative techniques accessible to a much broader audience.