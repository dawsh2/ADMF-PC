# ADMF-PC Optimization Framework

## Overview

The ADMF-PC Optimization Framework implements a sophisticated multi-phase optimization system that adheres to Protocol + Composition (PC) architecture with ZERO inheritance. It provides comprehensive parameter optimization, regime analysis, and walk-forward validation capabilities.

## Critical Architectural Decisions

Based on TEST_WORKFLOW.MD, the optimization framework implements six critical decisions:

1. **Clear Phase Transitions with Data Flow**
2. **Consistent Container Naming**  
3. **Result Streaming to Avoid Memory Issues**
4. **Cross-Regime Strategy Tracking**
5. **Checkpointing for Resumability**
6. **Walk-Forward Validation Support**

## Architecture

### Core Components

```
src/strategy/optimization/
├── protocols.py      # Protocols (NO inheritance)
├── optimizers.py     # Optimization algorithms
├── objectives.py     # Objective functions
├── constraints.py    # Parameter constraints (NO inheritance!)
├── containers.py     # Isolation containers
├── workflows.py      # Multi-phase workflows
└── capabilities.py   # Optimization capability
```

### Protocol-Based Design

All components implement protocols without inheritance:

```python
@runtime_checkable
class Optimizer(Protocol):
    """Optimization algorithm protocol."""
    
    def optimize(self, evaluate_func: Callable, 
                parameter_space: Dict[str, Any],
                n_trials: Optional[int] = None) -> Dict[str, Any]:
        """Run optimization."""
        ...

@runtime_checkable
class Objective(Protocol):
    """Objective function protocol."""
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate objective value."""
        ...

@runtime_checkable
class Constraint(Protocol):
    """Parameter constraint protocol."""
    
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """Check constraint satisfaction."""
        ...
```

## Multi-Phase Optimization Workflow

### Phase 1: Parameter Optimization

```python
# Parallel optimization across regimes and classifiers
Phase 1: Grid Search
├── HMM Classifier
│   ├── TRENDING_UP regime
│   ├── HIGH_VOLATILITY regime
│   └── TRENDING_DOWN regime
└── Pattern Classifier
    ├── BREAKOUT regime
    ├── RANGE_BOUND regime
    └── REVERSAL regime
```

### Phase 2: Regime Analysis

```python
# Analyze performance across regimes
Phase 2: Analysis
├── Compare regime performance
├── Identify parameter stability
└── Select best overall parameters
```

### Phase 3: Weight Optimization

```python
# Optimize ensemble weights using signal replay
Phase 3: Weights
├── Load Phase 1 signals
├── Optimize weights per regime
└── Create adaptive configuration
```

### Phase 4: Walk-Forward Validation

```python
# Validate on out-of-sample data
Phase 4: Validation
├── Create walk-forward periods
├── Test adaptive strategy
└── Calculate final performance
```

## Container Architecture

### Optimization Container Hierarchy

```
Coordinator
    │
    └── Optimization Container (Phase-specific)
        │
        ├── Regime Container (HMM_TRENDING_UP)
        │   ├── Trial Container 1
        │   ├── Trial Container 2
        │   └── Trial Container N
        │
        └── Regime Container (PATTERN_BREAKOUT)
            ├── Trial Container 1
            ├── Trial Container 2
            └── Trial Container N
```

### Container Naming Strategy

```python
container_id = f"{phase}_{classifier}_{regime}_{strategy}_{timestamp}"

# Examples:
"phase1_hmm_trending_up_momentum_20240115_143052"
"phase1_pattern_breakout_arbitrage_20240115_143053"
```

## Optimization Algorithms

### Grid Search Optimizer

```python
class GridOptimizer:
    """Exhaustive grid search - NO inheritance!"""
    
    def optimize(self, evaluate_func, parameter_space, n_trials=None):
        # Generate all combinations
        combinations = self._generate_grid(parameter_space)
        
        # Evaluate each
        best_score = -float('inf')
        best_params = None
        
        for params in combinations:
            score = evaluate_func(params)
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params
```

### Bayesian Optimizer

```python
class BayesianOptimizer:
    """Bayesian optimization with acquisition functions."""
    
    def __init__(self, acquisition_function='expected_improvement'):
        self.acquisition = acquisition_function
        self.gaussian_process = None
```

### Genetic Optimizer

```python
class GeneticOptimizer:
    """Genetic algorithm optimization."""
    
    def __init__(self, population_size=50, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
```

## Objective Functions

### Single Objectives

```python
class SharpeObjective:
    """Maximize Sharpe ratio."""
    
    def calculate(self, results: Dict[str, Any]) -> float:
        returns = results['returns']
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
            
        return mean_return / std_return * np.sqrt(252)
```

### Composite Objectives

```python
class CompositeObjective:
    """Combine multiple objectives."""
    
    def __init__(self, components: List[Tuple[Any, float]]):
        self.components = components  # (objective, weight) pairs
    
    def calculate(self, results: Dict[str, Any]) -> float:
        total = 0.0
        for objective, weight in self.components:
            total += weight * objective.calculate(results)
        return total
```

## Constraints (NO Inheritance!)

All constraint classes are standalone implementations:

```python
class RangeConstraint:
    """Parameter range constraint - NO inheritance!"""
    
    def __init__(self, param_name: str, min_value=None, max_value=None):
        self.param_name = param_name
        self.min_value = min_value
        self.max_value = max_value
        self.description = f"{param_name} in [{min_value}, {max_value}]"
    
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        if self.param_name not in params:
            return True
        
        value = params[self.param_name]
        if self.min_value and value < self.min_value:
            return False
        if self.max_value and value > self.max_value:
            return False
        
        return True
```

## Signal Replay System

### Signal Capture

```python
class SignalCapture:
    """Capture signals during optimization."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.signal_buffer = []
    
    def capture(self, signal: Dict[str, Any], metadata: Dict[str, Any]):
        self.signal_buffer.append({
            'signal': signal,
            'metadata': metadata,
            'timestamp': datetime.now()
        })
```

### Signal Replay

```python
class SignalReplayer:
    """Replay captured signals efficiently."""
    
    def replay_with_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Apply weights to historical signals."""
        weighted_signals = []
        
        for signal_data in self.signals:
            strategy_id = signal_data['metadata']['strategy_id']
            weight = weights.get(strategy_id, 0.0)
            
            weighted_signal = signal_data['signal'].copy()
            weighted_signal['strength'] *= weight
            weighted_signals.append(weighted_signal)
        
        return self._aggregate_signals(weighted_signals)
```

## Result Streaming

### Memory-Efficient Result Handling

```python
class ResultAggregator:
    """Stream results to disk, keep only top performers in memory."""
    
    def __init__(self, output_dir: Path, top_k: int = 10):
        self.output_dir = output_dir
        self.top_k = top_k
        self.top_results = []  # Heap of top k results
        self.result_count = 0
    
    def handle_container_result(self, container_id: str, result: Dict[str, Any]):
        # Stream to disk
        result_file = self.output_dir / f"{container_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f)
        
        # Update top results
        score = result.get('score', -float('inf'))
        if len(self.top_results) < self.top_k:
            heapq.heappush(self.top_results, (score, container_id, result))
        elif score > self.top_results[0][0]:
            heapq.heapreplace(self.top_results, (score, container_id, result))
```

## Strategy Identity Tracking

### Cross-Regime Strategy Management

```python
class StrategyIdentity:
    """Track strategy instances across regimes."""
    
    def __init__(self, strategy_class: str, base_params: Dict[str, Any]):
        self.strategy_class = strategy_class
        self.base_params = base_params
        self.canonical_id = self._generate_id()
        self.regime_instances = {}  # regime -> container_id
    
    def add_regime_instance(self, regime: str, container_id: str):
        self.regime_instances[regime] = container_id
```

## Checkpointing

### Workflow State Persistence

```python
class CheckpointManager:
    """Save and restore workflow state."""
    
    def save_checkpoint(self, workflow_id: str, phase: str, state: Dict[str, Any]):
        checkpoint = {
            'workflow_id': workflow_id,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'state': state
        }
        
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}_{phase}.checkpoint"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def restore_checkpoint(self, workflow_id: str, phase: str) -> Optional[Dict[str, Any]]:
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}_{phase}.checkpoint"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None
```

## Walk-Forward Validation

### Period Generation

```python
class WalkForwardValidator:
    """Create and manage walk-forward periods."""
    
    def create_periods(self, data_length: int, 
                      train_size: int, 
                      test_size: int,
                      step_size: int) -> List[Dict[str, Any]]:
        periods = []
        
        start = 0
        while start + train_size + test_size <= data_length:
            periods.append({
                'train_start': start,
                'train_end': start + train_size,
                'test_start': start + train_size,
                'test_end': start + train_size + test_size
            })
            start += step_size
        
        return periods
```

## Usage Examples

### Basic Optimization

```python
# Create optimizer
optimizer = GridOptimizer()
objective = SharpeObjective()

# Define parameter space
parameter_space = {
    'lookback': [10, 20, 30],
    'threshold': [0.01, 0.02, 0.03]
}

# Optimize
best_params = optimizer.optimize(
    evaluate_func=lambda p: objective.calculate(backtest(strategy, p)),
    parameter_space=parameter_space
)
```

### Phase-Aware Workflow

```python
# Configure workflow
config = {
    'workflow_id': 'optimization_001',
    'phases': {
        'phase1': {
            'parameter_space': {...},
            'regime_classifiers': ['hmm', 'pattern'],
            'strategies': [...]
        },
        'phase2': {...},
        'phase3': {...},
        'phase4': {...}
    }
}

# Create and run workflow
workflow = PhaseAwareOptimizationWorkflow(coordinator, config)
results = await workflow.run()
```

### Regime-Based Optimization

```python
workflow = RegimeBasedOptimizationWorkflow(
    regime_detector_config={'type': 'hmm'},
    component_config={'class': 'MomentumStrategy'},
    optimizer_config={'type': 'bayesian'}
)

results = workflow.run()
# Returns optimal parameters for each regime
```

## Performance Considerations

1. **Container Pooling**: Reuse containers for similar trials
2. **Signal Caching**: Cache computed signals for replay
3. **Parallel Execution**: Run regime optimizations in parallel
4. **Result Streaming**: Stream to disk, keep minimal in memory
5. **Lazy Evaluation**: Compute indicators only when needed

## Best Practices

1. **Start with Grid Search**: Understand parameter space
2. **Use Constraints**: Prevent invalid combinations
3. **Monitor Progress**: Track optimization metrics
4. **Validate Results**: Always run walk-forward validation
5. **Save Checkpoints**: Enable resumability
6. **Stream Results**: Avoid memory issues
7. **Track Identity**: Maintain strategy tracking across regimes

## Testing

The optimization framework includes comprehensive tests:

```bash
# Unit tests
python -m pytest tests/test_optimizers.py
python -m pytest tests/test_objectives.py
python -m pytest tests/test_constraints.py

# Integration tests
python test_optimization_workflow.py
python test_optimization_final.py

# Performance tests
python test_optimization_performance.py
```

## Conclusion

The ADMF-PC Optimization Framework demonstrates that complex multi-phase optimization can be implemented without ANY inheritance, using only protocols and composition. This provides:

- Complete flexibility in optimization strategies
- Clean separation of concerns
- Natural parallelization through containers
- Memory-efficient result handling
- Comprehensive regime analysis
- Production-ready checkpointing and validation

The framework seamlessly integrates with the broader ADMF-PC architecture while maintaining the core principle: **Protocol + Composition, NO Inheritance!**