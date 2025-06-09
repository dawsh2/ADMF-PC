# Complete Guide to YAML Composition Features

## 1. Sequence Composition

### Basic Composition: Walk-Forward from Train/Test
```yaml
# walk_forward is composed of repeated train/test sequences
name: walk_forward
iterations:
  type: windowed
  window_generator:
    type: rolling

sub_sequences:
  - name: train
    sequence: single_pass  # Reuse existing sequence
    config_override:
      start_date: "{window.train_start}"
      
  - name: test
    sequence: single_pass  # Reuse again
    depends_on: train
    config_override:
      parameters: "{train.results.optimal_parameters}"
```

### Nested Composition: Monte Carlo of Walk-Forwards
```yaml
# Each Monte Carlo iteration runs a complete walk-forward
name: robust_walk_forward
iterations:
  type: repeated
  count: 100

sub_sequences:
  - name: walk_forward_iteration
    sequence: walk_forward  # Entire sequence as sub-sequence!
    config_override:
      random_seed: "{iteration_index}"
```

### Mixed Composition: Sequences and Phases
```yaml
# Sequence can have both sub-sequences and sub-phases
name: complex_analysis
sub_sequences:
  - name: data_prep
    sequence: data_validation
    
sub_phases:
  - name: analysis
    depends_on: data_prep
    config_override:
      validated_data: "{data_prep.outputs.clean_data}"
```

## 2. Workflow Composition

### Iterative Workflows
```yaml
# Workflow that iterates until convergence
name: iterative_optimization
iteration:
  max_iterations: 10
  continue_condition:
    type: expression
    expression: "results.improvement > 0.01"
  config_modifier:
    type: gradient_based
    learning_rate: 0.1

phases:
  - name: optimize
    config:
      parameters: "{iteration_config.parameters}"
```

### Conditional Branching
```yaml
# Different paths based on results
phases:
  - name: initial_test
    
  - name: strategy_selection
    type: conditional_branch
    depends_on: initial_test
    branches:
      - condition:
          expression: "results['initial_test']['strategy_type'] == 'trend'"
        workflow: trend_optimization
        
      - condition:
          expression: "results['initial_test']['strategy_type'] == 'mean_reversion'"
        workflow: mean_reversion_optimization
        
      - condition:
          default: true
        workflow: generic_optimization
```

### Sub-Workflows
```yaml
# Workflows containing other workflows
name: master_workflow
phases:
  - name: phase1
    
sub_workflows:
  trend_optimization:
    phases:
      - name: trend_specific_analysis
      - name: trend_validation
      
  mean_reversion_optimization:
    phases:
      - name: volatility_analysis
      - name: threshold_optimization
```

### Workflow Modules and Imports
```yaml
# Reusable workflow modules
imports:
  - standard_validation: ./modules/validation.yaml
  - risk_analysis: ./modules/risk.yaml

phases:
  - name: custom_optimization
    
  - name: validation
    type: workflow_module
    module: standard_validation
    config:
      override_defaults: true
      
  - name: risk
    type: workflow_module
    module: risk_analysis
    depends_on: validation
```

## 3. Advanced Composition Features

### Dynamic Phase Injection
```yaml
composition:
  inject_phases:
    # Add phases based on conditions
    - condition: 
        config_contains: enable_ml
      phases:
        - name: ml_enhancement
          topology: ml_training
          sequence: cross_validation
          
    # Add debugging on failure
    on_failure:
      - name: debug_analysis
        topology: analysis
        config:
          debug_mode: true
```

### Parallel Execution Groups
```yaml
composition:
  parallel_groups:
    # These phases can run in parallel
    - [robustness_test, regime_analysis, correlation_check]
    - [strategy1_test, strategy2_test, strategy3_test]
    
  # Dependencies still respected
  synchronization_points:
    - after: [robustness_test, regime_analysis]
      before: final_validation
```

### Template Patterns
```yaml
composition:
  templates:
    # Define reusable patterns
    standard_validation:
      sequence: [single_pass, monte_carlo, walk_forward]
      config:
        apply_standard_metrics: true
        
    robust_optimization:
      phases:
        - parameter_sweep
        - sensitivity_analysis
        - stability_check

# Use templates
phases:
  - name: strategy1_validation
    template: standard_validation
    config:
      strategy: strategy1
      
  - name: strategy2_validation
    template: standard_validation
    config:
      strategy: strategy2
```

### Resource Hints and Adaptation
```yaml
features:
  # Resource allocation hints
  resource_hints:
    optimization:
      parallelism: high
      memory: medium
      gpu: optional
      
    walk_forward:
      parallelism: medium
      memory: high
      disk: high
      
  # Adaptive behavior
  adaptive_resources:
    - condition: "data_size > 1GB"
      apply:
        all_phases:
          memory: high
          use_disk_cache: true
```

### Result Aggregation Across Composition
```yaml
# Aggregate across all composition levels
outputs:
  final_results:
    type: hierarchical_aggregation
    levels:
      - name: iteration_level
        aggregate: all_iterations
        method: best_by_metric
        
      - name: branch_level
        aggregate: all_branches
        method: weighted_average
        
      - name: sequence_level
        aggregate: all_sequences
        method: statistical
        
  convergence_analysis:
    type: iteration_analysis
    metrics:
      - improvement_rate
      - convergence_speed
      - final_performance
```

## 4. Complete Example: Research Pipeline with Full Composition

```yaml
name: complete_research_pipeline
description: Shows all composition features

# Workflow-level iteration
iteration:
  max_iterations: 5
  continue_condition:
    expression: "not converged and iteration < max"

# Import modules
imports:
  - validation: ./modules/validation.yaml
  - ml_enhancement: ./modules/ml.yaml

# Main phases
phases:
  # Discovery phase with branching
  - name: discovery
    type: conditional_branch
    branches:
      - condition: {config_contains: use_ml}
        workflow: ml_discovery
      - condition: {default: true}
        workflow: traditional_discovery
        
  # Composed validation
  - name: validation
    depends_on: discovery
    type: composed_phase
    
    # This phase runs multiple sequences
    sequences:
      - name: monte_carlo_validation
        sequence: monte_carlo
        config:
          iterations: 100
          
      - name: walk_forward_validation
        sequence: composed_walk_forward  # Uses composed sequence
        config:
          windows: 12
          
    # Aggregate sequence results
    aggregation:
      type: must_pass_all
      thresholds:
        monte_carlo: {sharpe_95_percentile: 0.5}
        walk_forward: {average_sharpe: 1.0}

# Sub-workflows
sub_workflows:
  ml_discovery:
    phases:
      - name: feature_engineering
        sequence: feature_discovery
        
      - name: model_training
        sequence: cross_validation
        iterations:  # Nested iteration!
          type: model_search
          models: [lstm, transformer, ensemble]
          
  traditional_discovery:
    phases:
      - name: parameter_optimization
        sequence: differential_evolution
        
# Composition features
composition:
  # Parallel where possible
  parallel_groups:
    - [feature_engineering, parameter_optimization]
    
  # Dynamic injection
  inject_phases:
    on_success:
      - name: production_prep
        template: production_template
        
  # Resource optimization
  adaptive_execution:
    - condition: "phase == 'model_training'"
      require: gpu
    - condition: "sequence == 'walk_forward'"
      prefer: high_memory

# Multi-level outputs
outputs:
  strategy_recommendation:
    type: best_path_analysis
    consider:
      - all_iterations
      - all_branches
      - all_sequences
    optimize_for: risk_adjusted_sharpe
    
  research_report:
    type: hierarchical_report
    sections:
      - iteration_summary
      - branch_comparison
      - sequence_performance
      - convergence_analysis
```

## 5. Benefits of Composition

### 1. Reusability
- Sequences can be composed from other sequences
- Workflows can include other workflows
- Templates provide reusable patterns

### 2. Modularity
- Each piece can be developed/tested independently
- Easy to swap components
- Clear separation of concerns

### 3. Flexibility
- Mix and match patterns
- Override at any level
- Dynamic composition based on results

### 4. Maintainability
- Changes propagate through composition
- Single source of truth for patterns
- Version control friendly

## 6. Implementation Notes

The declarative system supports composition through:

1. **Recursive Execution**: Sequences can execute other sequences
2. **Context Propagation**: Results flow through composition levels
3. **Override Mechanism**: Each level can override configuration
4. **Dependency Resolution**: Works across composition boundaries
5. **Aggregation Hierarchy**: Results aggregate up the composition tree

This enables building complex workflows from simple, reusable components - true composability!