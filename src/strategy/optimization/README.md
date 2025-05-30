# ADMF-PC Optimization Framework

The optimization framework provides a Protocol + Composition based approach to making ANY component optimizable without requiring inheritance. This allows for flexible, efficient parameter optimization across the entire ADMF-PC system.

## Key Features

- **Protocol-Based**: Components implement the `Optimizable` protocol without inheritance
- **Capability System**: Add optimization to any component via `OptimizationCapability`
- **Multiple Optimizers**: Grid search, Bayesian optimization, genetic algorithms
- **Flexible Objectives**: Single or composite objectives (Sharpe, returns, drawdown)
- **Constraints**: Relational, range, discrete, and custom constraints
- **Container Isolation**: Each optimization trial runs in isolation
- **Workflow Management**: Sequential and regime-based optimization workflows

## Quick Start

### 1. Making a Component Optimizable

```python
from src.strategy.optimization import OptimizationCapability

# Any existing component
class MyStrategy:
    def __init__(self, fast_period=10, slow_period=30):
        self.fast_period = fast_period
        self.slow_period = slow_period

# Make it optimizable
capability = OptimizationCapability()
strategy = capability.apply(MyStrategy(), {
    'parameter_space': {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50]
    }
})

# Now it has optimization methods
params = strategy.get_parameters()
strategy.set_parameters({'fast_period': 15, 'slow_period': 40})
```

### 2. Running Optimization

```python
from src.strategy.optimization import GridOptimizer, SharpeObjective

# Create optimizer and objective
optimizer = GridOptimizer()
objective = SharpeObjective()

# Define evaluation function
def evaluate(params):
    strategy.set_parameters(params)
    results = run_backtest(strategy)  # Your backtest function
    return objective.calculate(results)

# Optimize
best_params = optimizer.optimize(
    evaluate,
    parameter_space=strategy.get_parameter_space()
)

print(f"Best parameters: {best_params}")
print(f"Best Sharpe: {optimizer.get_best_score()}")
```

### 3. Using Constraints

```python
from src.strategy.optimization import RelationalConstraint

# Ensure fast_period < slow_period
constraint = RelationalConstraint('fast_period', '<', 'slow_period')

def evaluate_constrained(params):
    if not constraint.is_satisfied(params):
        return float('-inf')
    return evaluate(params)
```

### 4. Composite Objectives

```python
from src.strategy.optimization import CompositeObjective, MinDrawdownObjective

# 70% Sharpe, 30% Drawdown minimization
objective = CompositeObjective([
    (SharpeObjective(), 0.7),
    (MinDrawdownObjective(), 0.3)
])
```

## Advanced Usage

### Sequential Optimization Workflow

```python
from src.strategy.optimization import SequentialOptimizationWorkflow

stages = [
    {
        'name': 'coarse_search',
        'optimizer': {'type': 'grid'},
        'objective': {'type': 'sharpe'},
        'component': {
            'class': 'MyStrategy',
            'parameter_space': {
                'fast_period': [5, 15, 25],
                'slow_period': [30, 45, 60]
            }
        },
        'n_trials': 9
    },
    {
        'name': 'fine_tuning',
        'optimizer': {'type': 'bayesian'},
        'objective': {'type': 'sharpe'},
        'component': {
            'class': 'MyStrategy',
            'parameter_space': {
                'fast_period': (10, 20),  # Continuous
                'slow_period': (35, 55)   # Continuous
            }
        },
        'n_trials': 50,
        'use_previous_best': True  # Start from stage 1 best
    }
]

workflow = SequentialOptimizationWorkflow(stages)
results = workflow.run()
```

### Regime-Based Optimization

```python
from src.strategy.optimization import RegimeBasedOptimizationWorkflow

workflow = RegimeBasedOptimizationWorkflow(
    regime_detector_config={'volatility_window': 20},
    component_config={
        'class': 'MyStrategy',
        'parameter_space': {...}
    },
    optimizer_config={
        'type': 'bayesian',
        'objective': 'sharpe',
        'n_trials_per_regime': 30
    }
)

results = workflow.run()
# Returns optimized parameters for each detected regime
```

### Container-Isolated Optimization

```python
from src.strategy.optimization import ContainerizedComponentOptimizer

# Each trial runs in complete isolation
optimizer = ContainerizedComponentOptimizer(
    GridOptimizer(),
    SharpeObjective(),
    use_containers=True
)

results = optimizer.optimize_component(
    component_spec,
    backtest_runner,
    n_trials=100
)
```

## Architecture

### Protocols

- **Optimizable**: Components that can be optimized
- **Optimizer**: Optimization algorithms (grid, Bayesian, genetic)
- **Objective**: Functions to maximize/minimize
- **Constraint**: Parameter constraints

### Key Components

1. **capabilities.py**: `OptimizationCapability` that adds optimization support
2. **optimizers.py**: Grid, Bayesian, and genetic optimization algorithms
3. **objectives.py**: Sharpe, return, drawdown objectives
4. **constraints.py**: Various constraint types
5. **containers.py**: Isolation for optimization trials
6. **workflows.py**: Multi-stage optimization workflows

## Best Practices

1. **Start Simple**: Begin with grid search before moving to Bayesian
2. **Use Constraints**: Prevent invalid parameter combinations
3. **Composite Objectives**: Balance multiple goals (Sharpe vs drawdown)
4. **Container Isolation**: Always use containers for production
5. **Sequential Workflows**: Coarse search → fine tuning
6. **Monitor Progress**: Use optimization history and statistics

## Examples

See `example.py` for complete working examples of:
- Basic grid search optimization
- Bayesian optimization with constraints
- Sequential multi-stage workflows
- Regime-based optimization

## Testing

Run the test suite:
```bash
python test_optimization_workflow.py  # Comprehensive workflow tests
python test_optimization_final.py     # Feature verification tests
```

## Integration

The optimization framework integrates seamlessly with:
- Component system (any component can be optimized)
- Container system (isolation for each trial)
- Event system (optimization events)
- Monitoring (track optimization progress)

No changes to existing components are required - optimization is added as a capability!