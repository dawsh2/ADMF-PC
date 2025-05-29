# Optimization Framework with Protocol + Composition Architecture

## Overview

The optimization framework integrates seamlessly with our Protocol + Composition architecture. Instead of inheritance hierarchies, optimization capabilities are added through protocols and composition, making ANY component optimizable without forcing base class inheritance.

## 1. Optimization Protocols

```python
from typing import Protocol, runtime_checkable, Dict, Any, List, Optional, Tuple, Callable
from abc import abstractmethod

# === Core Optimization Protocol ===
@runtime_checkable
class Optimizable(Protocol):
    """Protocol for components that can be optimized"""
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return parameter space for optimization"""
        ...
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameter values"""
        ...
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        ...
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameter values"""
        ...

# === Optimizer Protocols ===
@runtime_checkable
class Optimizer(Protocol):
    """Protocol for optimization algorithms"""
    
    @abstractmethod
    def optimize(self, evaluate_func: Callable[[Dict[str, Any]], float], 
                n_trials: int = None, **kwargs) -> Dict[str, Any]:
        """Run optimization and return best parameters"""
        ...
    
    @abstractmethod
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get best parameters found"""
        ...
    
    @abstractmethod
    def get_best_score(self) -> float:
        """Get best score achieved"""
        ...

@runtime_checkable
class Objective(Protocol):
    """Protocol for optimization objectives"""
    
    @abstractmethod
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate objective value from results"""
        ...
    
    @abstractmethod
    def get_direction(self) -> str:
        """Get optimization direction ('maximize' or 'minimize')"""
        ...

@runtime_checkable
class Constraint(Protocol):
    """Protocol for parameter constraints"""
    
    @abstractmethod
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """Check if parameters satisfy constraint"""
        ...
    
    @abstractmethod
    def validate_and_adjust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and optionally adjust parameters"""
        ...

# === Workflow Protocols ===
@runtime_checkable
class OptimizationWorkflow(Protocol):
    """Protocol for optimization workflows"""
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the workflow and return results"""
        ...
    
    @abstractmethod
    def get_stages(self) -> List[str]:
        """Get list of workflow stages"""
        ...
```

## 2. Optimization Capability

```python
class OptimizationCapability(Capability):
    """Adds optimization support to any component"""
    
    def get_name(self) -> str:
        return "optimization"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Check if component already has optimization methods
        has_methods = all(hasattr(component, method) for method in 
                         ['get_parameter_space', 'set_parameters', 
                          'get_parameters', 'validate_parameters'])
        
        if not has_methods:
            # Add default implementations for components without parameters
            component.get_parameter_space = lambda: {}
            component.set_parameters = lambda params: None
            component.get_parameters = lambda: {}
            component.validate_parameters = lambda params: (True, "")
        
        # Add optimization metadata
        component._optimization_metadata = {
            'optimizable': True,
            'last_optimization': None,
            'parameter_history': []
        }
        
        # Add parameter tracking
        original_set_params = component.set_parameters
        
        def tracked_set_parameters(params: Dict[str, Any]) -> None:
            # Validate first
            valid, error = component.validate_parameters(params)
            if not valid:
                raise ValueError(f"Invalid parameters: {error}")
            
            # Apply parameters
            original_set_params(params)
            
            # Track history
            component._optimization_metadata['parameter_history'].append({
                'timestamp': datetime.now(),
                'parameters': params.copy()
            })
        
        component.set_parameters = tracked_set_parameters
        
        # Add optimization helpers
        component.reset_to_defaults = lambda: component.set_parameters({})
        component.get_parameter_history = lambda: component._optimization_metadata['parameter_history']
        
        return component
```

## 3. Container-Aware Optimization

### 3.1 Optimization Container

```python
class OptimizationContainer(UniversalScopedContainer):
    """Specialized container for optimization runs"""
    
    def __init__(self, container_id: str, base_config: Dict[str, Any]):
        super().__init__(container_id)
        self.base_config = base_config
        self.trial_count = 0
        self.results_collector = OptimizationResultsCollector()
    
    def create_trial_instance(self, parameters: Dict[str, Any]) -> str:
        """Create a new instance for parameter trial"""
        trial_id = f"{self.container_id}_trial_{self.trial_count}"
        self.trial_count += 1
        
        # Create component with trial parameters
        component_spec = self.base_config.copy()
        component_spec['params'].update(parameters)
        
        # Create isolated component instance
        component = self.create_component(component_spec)
        
        return trial_id, component
    
    def run_trial(self, parameters: Dict[str, Any], 
                  backtest_runner: Callable) -> Dict[str, Any]:
        """Run a single optimization trial"""
        trial_id, component = self.create_trial_instance(parameters)
        
        try:
            # Initialize component
            self.initialize_component(component)
            
            # Run backtest
            results = backtest_runner(component)
            
            # Collect results
            self.results_collector.add_result(trial_id, parameters, results)
            
            return results
            
        finally:
            # Clean up trial instance
            if hasattr(component, 'teardown'):
                component.teardown()
```

### 3.2 Component Optimizer with Containers

```python
class ContainerizedComponentOptimizer:
    """Optimizes components using container isolation"""
    
    def __init__(self, optimizer: Optimizer, objective: Objective,
                 use_containers: bool = True):
        self.optimizer = optimizer
        self.objective = objective
        self.use_containers = use_containers
        self.container_manager = ContainerLifecycleManager()
    
    def optimize_component(self, component_spec: Dict[str, Any],
                          backtest_runner: Callable,
                          n_trials: int = None) -> Dict[str, Any]:
        """Optimize a component with full isolation"""
        
        if self.use_containers:
            # Create optimization container
            container_id = self.container_manager.create_and_start_container(
                "optimization",
                {'base_config': component_spec}
            )
            
            try:
                container = self.container_manager.active_containers[container_id]
                
                # Define evaluation function
                def evaluate(params: Dict[str, Any]) -> float:
                    results = container.run_trial(params, backtest_runner)
                    return self.objective.calculate(results)
                
                # Run optimization
                best_params = self.optimizer.optimize(evaluate, n_trials)
                
                return {
                    'best_parameters': best_params,
                    'best_score': self.optimizer.get_best_score(),
                    'all_results': container.results_collector.get_all_results()
                }
                
            finally:
                self.container_manager.stop_and_destroy_container(container_id)
        
        else:
            # Direct optimization without containers (for simple cases)
            return self._optimize_direct(component_spec, backtest_runner, n_trials)
```

## 4. Optimization Implementations

### 4.1 Grid Optimizer

```python
class GridOptimizer:
    """Grid search optimization"""
    
    def __init__(self):
        self.best_params = None
        self.best_score = float('-inf')
        self.all_results = []
    
    def optimize(self, evaluate_func: Callable[[Dict[str, Any]], float],
                n_trials: int = None, **kwargs) -> Dict[str, Any]:
        """Run grid search optimization"""
        parameter_space = kwargs.get('parameter_space', {})
        
        # Generate all combinations
        param_combinations = self._generate_combinations(parameter_space)
        
        # Limit trials if specified
        if n_trials:
            param_combinations = param_combinations[:n_trials]
        
        # Evaluate each combination
        for params in param_combinations:
            try:
                score = evaluate_func(params)
                self.all_results.append({'params': params, 'score': score})
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
            except Exception as e:
                print(f"Error evaluating {params}: {e}")
                continue
        
        return self.best_params
    
    def _generate_combinations(self, space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        import itertools
        
        keys = list(space.keys())
        values = [space[key] for key in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def get_best_parameters(self) -> Dict[str, Any]:
        return self.best_params
    
    def get_best_score(self) -> float:
        return self.best_score
```

### 4.2 Bayesian Optimizer

```python
class BayesianOptimizer:
    """Bayesian optimization using Gaussian Process"""
    
    def __init__(self, acquisition_function: str = 'expected_improvement'):
        self.acquisition_function = acquisition_function
        self.gp_model = None
        self.observations = []
        self.best_params = None
        self.best_score = float('-inf')
    
    def optimize(self, evaluate_func: Callable[[Dict[str, Any]], float],
                n_trials: int = 100, **kwargs) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        parameter_space = kwargs.get('parameter_space', {})
        
        # Initial random exploration
        n_initial = min(10, n_trials // 4)
        initial_params = self._random_sample(parameter_space, n_initial)
        
        # Evaluate initial points
        for params in initial_params:
            score = evaluate_func(params)
            self._update_observations(params, score)
        
        # Bayesian optimization loop
        for i in range(n_trials - n_initial):
            # Fit GP model
            self._fit_gp_model()
            
            # Select next point using acquisition function
            next_params = self._select_next_point(parameter_space)
            
            # Evaluate
            score = evaluate_func(next_params)
            self._update_observations(next_params, score)
        
        return self.best_params
    
    def _update_observations(self, params: Dict[str, Any], score: float):
        """Update observations and best parameters"""
        self.observations.append({'params': params, 'score': score})
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
```

## 5. Optimization Workflows

### 5.1 Sequential Workflow

```python
class SequentialOptimizationWorkflow:
    """Multi-stage optimization workflow"""
    
    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        self.results = {}
        self.container_manager = ContainerLifecycleManager()
    
    def run(self) -> Dict[str, Any]:
        """Execute sequential optimization stages"""
        
        for stage_idx, stage_config in enumerate(self.stages):
            stage_name = stage_config.get('name', f'stage_{stage_idx}')
            
            print(f"Running optimization stage: {stage_name}")
            
            # Create optimizer and objective for this stage
            optimizer = self._create_optimizer(stage_config['optimizer'])
            objective = self._create_objective(stage_config['objective'])
            
            # Get component configuration
            component_config = self._prepare_component_config(
                stage_config, 
                self.results  # Previous results available
            )
            
            # Run optimization
            component_optimizer = ContainerizedComponentOptimizer(
                optimizer, objective
            )
            
            stage_results = component_optimizer.optimize_component(
                component_config,
                self._create_backtest_runner(stage_config),
                n_trials=stage_config.get('n_trials', 100)
            )
            
            # Store results
            self.results[stage_name] = stage_results
            
            # Apply results to next stage if needed
            if stage_config.get('feed_forward', True):
                self._apply_results_to_config(stage_results, stage_idx + 1)
        
        return self.results
    
    def get_stages(self) -> List[str]:
        return [s.get('name', f'stage_{i}') for i, s in enumerate(self.stages)]
```

### 5.2 Regime-Based Workflow

```python
class RegimeBasedOptimizationWorkflow:
    """Optimize separately for each market regime"""
    
    def __init__(self, regime_detector_config: Dict[str, Any],
                 component_config: Dict[str, Any],
                 optimizer_config: Dict[str, Any]):
        self.regime_detector_config = regime_detector_config
        self.component_config = component_config
        self.optimizer_config = optimizer_config
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """Run regime-specific optimization"""
        
        # Step 1: Detect regimes in training data
        regimes = self._detect_regimes()
        
        # Step 2: Optimize for each regime
        for regime_name, regime_data in regimes.items():
            print(f"Optimizing for regime: {regime_name}")
            
            # Create regime-specific container
            container_id = f"regime_opt_{regime_name}"
            
            # Filter data for this regime
            regime_runner = self._create_regime_backtest_runner(regime_data)
            
            # Optimize
            optimizer = self._create_optimizer(self.optimizer_config)
            objective = self._create_objective(self.optimizer_config)
            
            component_optimizer = ContainerizedComponentOptimizer(
                optimizer, objective
            )
            
            regime_results = component_optimizer.optimize_component(
                self.component_config,
                regime_runner,
                n_trials=self.optimizer_config.get('n_trials_per_regime', 50)
            )
            
            self.results[regime_name] = regime_results
        
        # Step 3: Create adaptive strategy configuration
        self.results['adaptive_config'] = self._create_adaptive_config()
        
        return self.results
    
    def _create_adaptive_config(self) -> Dict[str, Any]:
        """Create configuration for regime-adaptive strategy"""
        return {
            'regime_detector': self.regime_detector_config,
            'regime_parameters': {
                regime: results['best_parameters']
                for regime, results in self.results.items()
                if regime != 'adaptive_config'
            }
        }
```

## 6. Integration with Strategy Components

### 6.1 Making Any Component Optimizable

```python
# Simple function can be made optimizable
def calculate_ma_signal(prices: List[float], fast: int = 10, slow: int = 30) -> float:
    if len(prices) < slow:
        return 0.0
    fast_ma = sum(prices[-fast:]) / fast
    slow_ma = sum(prices[-slow:]) / slow
    return (fast_ma - slow_ma) / slow_ma

# Wrap with optimization capability
ma_component = create_component({
    'function': calculate_ma_signal,
    'capabilities': ['optimization'],
    'params': {'fast': 10, 'slow': 30},
    'parameter_space': {
        'fast': [5, 10, 15, 20],
        'slow': [20, 30, 40, 50]
    }
})

# Now it's fully optimizable!
```

### 6.2 Complex Strategy with Built-in Optimization

```python
class TrendFollowingStrategy:
    """Strategy with built-in optimization support"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 signal_threshold: float = 0.02):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_threshold = signal_threshold
        
        # Create indicators
        self.fast_ma = SimpleMovingAverage(fast_period)
        self.slow_ma = SimpleMovingAverage(slow_period)
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Define optimizable parameters"""
        return {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 40, 50],
            'signal_threshold': [0.01, 0.02, 0.03, 0.05]
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply optimization parameters"""
        if 'fast_period' in params:
            self.fast_period = params['fast_period']
            self.fast_ma = SimpleMovingAverage(self.fast_period)
        
        if 'slow_period' in params:
            self.slow_period = params['slow_period']
            self.slow_ma = SimpleMovingAverage(self.slow_period)
        
        if 'signal_threshold' in params:
            self.signal_threshold = params['signal_threshold']
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameter constraints"""
        fast = params.get('fast_period', self.fast_period)
        slow = params.get('slow_period', self.slow_period)
        
        if fast >= slow:
            return False, "Fast period must be less than slow period"
        
        threshold = params.get('signal_threshold', self.signal_threshold)
        if threshold <= 0 or threshold >= 1:
            return False, "Signal threshold must be between 0 and 1"
        
        return True, ""
```

## 7. Optimization Manager Integration

```python
class ProtocolAwareOptimizerManager:
    """Optimizer manager that works with protocol-based components"""
    
    def __init__(self):
        self.registered_optimizers = {}
        self.registered_workflows = {}
        self.container_manager = ContainerLifecycleManager()
    
    def optimize_component(self, component_or_spec: Any,
                          optimizer_name: str = 'grid',
                          objective_name: str = 'sharpe',
                          **kwargs) -> Dict[str, Any]:
        """Optimize any component that implements Optimizable protocol"""
        
        # Check if component is optimizable
        if not isinstance(component_or_spec, Optimizable):
            if isinstance(component_or_spec, dict):
                # Try to create component from spec
                component = ComponentFactory().create_component(component_or_spec)
            else:
                raise ValueError("Component must implement Optimizable protocol")
        else:
            component = component_or_spec
        
        # Get parameter space
        param_space = component.get_parameter_space()
        if not param_space:
            return {
                'message': 'Component has no parameters to optimize',
                'component': component.__class__.__name__
            }
        
        # Create optimizer and objective
        optimizer = self.registered_optimizers[optimizer_name]
        objective = self._create_objective(objective_name)
        
        # Run optimization
        component_optimizer = ContainerizedComponentOptimizer(
            optimizer, objective,
            use_containers=kwargs.get('use_containers', True)
        )
        
        return component_optimizer.optimize_component(
            self._component_to_spec(component),
            kwargs.get('backtest_runner'),
            n_trials=kwargs.get('n_trials', 100)
        )
```

## 8. Configuration-Driven Optimization

```yaml
optimization:
  # Component definitions
  components:
    trend_strategy:
      class: "TrendFollowingStrategy"
      capabilities: ["lifecycle", "events", "optimization"]
      parameter_space:
        fast_period: [5, 10, 15, 20]
        slow_period: [20, 30, 40, 50]
        signal_threshold: [0.01, 0.02, 0.03]
    
    regime_detector:
      class: "RegimeDetector"
      capabilities: ["lifecycle", "events", "optimization"]
      parameter_space:
        volatility_window: [15, 20, 25]
        trend_threshold: [0.01, 0.015, 0.02]
  
  # Workflow definition
  workflow:
    type: "sequential"
    stages:
      - name: "optimize_regime_detector"
        component: "regime_detector"
        optimizer:
          type: "grid"
        objective:
          type: "classification_accuracy"
        n_trials: 100
      
      - name: "optimize_strategy_by_regime"
        type: "regime_based"
        component: "trend_strategy"
        regime_detector: "{results.optimize_regime_detector.best_component}"
        optimizer:
          type: "bayesian"
          acquisition: "expected_improvement"
        objective:
          type: "composite"
          components:
            - type: "sharpe"
              weight: 0.7
            - type: "max_drawdown"
              weight: 0.3
              direction: "minimize"
        n_trials_per_regime: 50
  
  # Execution settings
  execution:
    use_containers: true
    parallel: true
    max_workers: 4
    memory_limit_per_container: "2GB"
```

## 9. Benefits of Protocol-Based Optimization

### 9.1 Universal Optimizability

```python
# Any component can be optimized
simple_indicator = lambda prices, period=20: sum(prices[-period:]) / period

# Make it optimizable
optimizable_indicator = add_capability(
    simple_indicator,
    OptimizationCapability(),
    {
        'parameter_space': {'period': [10, 20, 30, 40]},
        'default_params': {'period': 20}
    }
)
```

### 9.2 Clean Separation of Concerns

```python
# Strategy focuses on trading logic
class MyStrategy:
    def calculate_signal(self, data):
        # Pure trading logic
        pass

# Optimization is added separately
strategy = add_capabilities(MyStrategy(), [
    'optimization',
    'lifecycle',
    'events'
])
```

### 9.3 Container-Based Isolation

```python
# Each optimization trial runs in complete isolation
# No state leakage between parameter tests
# Perfect for parallel execution
```

### 9.4 Type Safety

```python
def optimize_any_component(component: Optimizable, 
                          optimizer: Optimizer,
                          objective: Objective) -> Dict[str, Any]:
    """Type-safe optimization of any component"""
    param_space = component.get_parameter_space()
    # ... optimization logic
```

## Summary

The optimization framework with Protocol + Composition:

1. **Makes ANY component optimizable** - Functions, classes, or complex strategies
2. **Zero inheritance overhead** - Components remain simple
3. **Perfect container integration** - Each trial runs in isolation
4. **Type-safe protocols** - Clear contracts without base classes
5. **Configuration-driven** - Complex workflows defined in YAML
6. **Scalable execution** - Natural parallelization through containers

This approach maintains the simplicity of component design while providing industrial-strength optimization capabilities. Whether optimizing a simple indicator or a complex multi-regime strategy, the same clean architecture applies.
