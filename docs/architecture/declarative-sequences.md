# Declarative Sequence System

## Overview

The declarative sequence system allows defining execution patterns in YAML rather than Python code. This makes it easier to create, modify, and share different execution strategies.

## Architecture

### Traditional Approach (Code-Based)
```python
class WalkForwardSequence:
    def execute(self, phase_config, context):
        # 300+ lines of hardcoded logic
        windows = self._generate_windows(...)
        for window in windows:
            train_result = self._train(...)
            test_result = self._test(...)
        return self._aggregate(...)
```

### Declarative Approach (Pattern-Based)
```yaml
name: walk_forward
iterations:
  type: windowed
  window_generator:
    train_periods: 252
    test_periods: 63
sub_phases: [train, test]
aggregation:
  type: statistical
```

## Core Concepts

### 1. Iterations
Define how many times and how to execute:

```yaml
# Single execution
iterations:
  type: single
  count: 1

# Multiple with randomization
iterations:
  type: repeated
  count: 100

# Rolling windows
iterations:
  type: windowed
  window_generator:
    type: rolling
    train_periods: 252
    test_periods: 63
```

### 2. Config Modifiers
Transform configuration for each iteration:

```yaml
config_modifiers:
  - type: set_dates
    train_start: "{window.train_start}"
    train_end: "{window.train_end}"
    
  - type: add_seed
    random_seed: "{iteration_index}"
    
  - type: parameter_noise
    noise_level: 0.1
```

### 3. Sub-Phases
Execute multiple phases per iteration:

```yaml
sub_phases:
  - name: train
    config_override:
      phase: train
      
  - name: test
    depends_on: train  # Use train results
    config_override:
      phase: test
      parameters: "{train.optimal_parameters}"
```

### 4. Aggregation
Combine results across iterations:

```yaml
# Statistical aggregation
aggregation:
  type: statistical
  source: test.metrics
  operations: [mean, std, min, max]

# Distribution analysis
aggregation:
  type: distribution
  metrics: [sharpe_ratio, max_drawdown]
  percentiles: [5, 25, 50, 75, 95]

# Custom aggregation
aggregation:
  type: custom
  function: my_aggregation_function
```

## Built-in Sequence Patterns

### Single Pass
```yaml
name: single_pass
iterations:
  type: single
aggregation:
  type: none
```

### Walk Forward
```yaml
name: walk_forward
iterations:
  type: windowed
  window_generator:
    train_periods: 252
    test_periods: 63
sub_phases: [train, test]
aggregation:
  type: statistical
```

### Monte Carlo
```yaml
name: monte_carlo
iterations:
  type: repeated
  count: 100
config_modifiers:
  - type: add_seed
aggregation:
  type: distribution
```

### Train/Test Split
```yaml
name: train_test
data_split:
  type: percentage
  train_ratio: 0.7
sub_phases: [train, test]
aggregation:
  type: comparison
```

### K-Fold Cross Validation
```yaml
name: k_fold
iterations:
  type: k_fold
  folds: 5
aggregation:
  type: cross_validation
```

## Creating Custom Sequences

### 1. Define Iteration Strategy
```yaml
iterations:
  type: custom
  generator: my_iteration_generator
  config:
    # Custom parameters
```

### 2. Add Config Modifiers
```yaml
config_modifiers:
  - type: custom
    function: modify_for_market_conditions
    params:
      volatility_threshold: 0.2
```

### 3. Define Sub-Phases
```yaml
sub_phases:
  - name: optimization
    topology: optimization
    
  - name: validation
    depends_on: optimization
    topology: backtest
    config_override:
      parameters: "{optimization.best_params}"
```

### 4. Configure Aggregation
```yaml
aggregation:
  type: multi_level
  levels:
    - group_by: market_regime
      aggregate: mean
    - group_by: parameter_set
      aggregate: best
```

## Advanced Features

### Conditional Execution
```yaml
sub_phases:
  - name: initial_test
    
  - name: deep_optimization
    condition: "{initial_test.sharpe_ratio} > 1.0"
    
  - name: risk_analysis
    condition: "{deep_optimization.success} == true"
```

### Dynamic Window Generation
```yaml
iterations:
  type: adaptive_windowed
  base_window: 252
  adjustments:
    high_volatility:
      window: 126
    trending_market:
      window: 504
```

### Nested Sequences
```yaml
sub_phases:
  - name: outer_loop
    sequence: walk_forward
    config:
      sub_phases:
        - name: inner_optimization
          sequence: monte_carlo
```

## Integration with Workflows

Sequences are referenced in workflow definitions:

```yaml
workflow:
  phases:
    - name: research
      topology: backtest
      sequence: walk_forward  # Use walk-forward sequence
      
    - name: validation
      topology: backtest
      sequence: monte_carlo   # Use Monte Carlo sequence
```

## Benefits

1. **Flexibility**: Easy to create new execution patterns
2. **Reusability**: Share sequences across projects
3. **Clarity**: Execution logic is visible and understandable
4. **Extensibility**: Add new pattern types without code changes
5. **Testability**: Validate patterns before execution

## Migration Guide

To convert existing sequence code to declarative:

1. Identify the pattern (iterations, modifications, aggregation)
2. Create YAML pattern file
3. Test with same inputs
4. Remove old Python class

Example:
```python
# OLD: 300 lines of Python
class CustomSequence:
    def execute(self, ...):
        # Complex logic
```

```yaml
# NEW: 30 lines of YAML
name: custom_sequence
iterations:
  type: windowed
aggregation:
  type: statistical
```