# Step 9: Parameter Expansion

**Status**: Multi-Phase Integration Step
**Complexity**: High
**Prerequisites**: [Step 8.5: Statistical Validation](../03-signal-capture-replay/step-08.5-monte-carlo.md) completed
**Architecture Ref**: [MULTIPHASE_OPTIMIZATION.md](../../legacy/MULTIPHASE_OPTIMIZATION.md)

## 🎯 Objective

Implement systematic parameter space exploration:
- Grid search for exhaustive coverage
- Random search for high-dimensional spaces
- Bayesian optimization for efficient search
- Parameter importance and sensitivity analysis
- Constraint handling and validation

## 📋 Required Reading

Before starting:
1. [Optimization Theory](../references/optimization-theory.md)
2. [Parameter Search Methods](../references/parameter-search.md)
3. [Bayesian Optimization](../references/bayesian-optimization.md)

## 🏗️ Implementation Tasks

### 1. Parameter Expansion Framework

```python
# src/optimization/parameter_expansion.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import uniform, norm
import optuna
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

@dataclass
class ParameterDefinition:
    """Defines a parameter for optimization"""
    name: str
    param_type: str  # 'continuous', 'integer', 'categorical'
    bounds: Union[Tuple[float, float], List[Any]]
    default_value: Any
    
    # Advanced properties
    distribution: Optional[str] = 'uniform'  # 'uniform', 'normal', 'log-uniform'
    constraints: Optional[List[Callable]] = None
    importance_prior: float = 1.0  # Prior belief about importance
    
    def validate_value(self, value: Any) -> bool:
        """Validate parameter value"""
        if self.param_type == 'continuous':
            return self.bounds[0] <= value <= self.bounds[1]
        elif self.param_type == 'integer':
            return self.bounds[0] <= int(value) <= self.bounds[1]
        elif self.param_type == 'categorical':
            return value in self.bounds
        return False
    
    def sample(self, n_samples: int = 1, random_state: Optional[int] = None) -> List[Any]:
        """Sample values from parameter distribution"""
        if random_state:
            np.random.seed(random_state)
        
        if self.param_type == 'continuous':
            if self.distribution == 'uniform':
                return uniform.rvs(self.bounds[0], 
                                 self.bounds[1] - self.bounds[0], 
                                 size=n_samples).tolist()
            elif self.distribution == 'log-uniform':
                log_bounds = np.log(self.bounds)
                return np.exp(uniform.rvs(log_bounds[0], 
                                        log_bounds[1] - log_bounds[0], 
                                        size=n_samples)).tolist()
        elif self.param_type == 'integer':
            return np.random.randint(self.bounds[0], 
                                   self.bounds[1] + 1, 
                                   size=n_samples).tolist()
        elif self.param_type == 'categorical':
            return np.random.choice(self.bounds, size=n_samples).tolist()

class ParameterExpander:
    """
    Base class for parameter expansion strategies.
    Generates parameter combinations for optimization.
    """
    
    def __init__(self, parameter_definitions: Dict[str, ParameterDefinition]):
        self.param_defs = parameter_definitions
        self.expansion_history: List[Dict] = []
        self.logger = ComponentLogger("ParameterExpander", "optimization")
    
    @abstractmethod
    def expand(self, n_configs: Optional[int] = None, 
              constraints: Optional[List[Callable]] = None) -> List[Dict[str, Any]]:
        """Generate parameter configurations"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate a parameter configuration"""
        # Check individual parameter constraints
        for param_name, value in config.items():
            if param_name in self.param_defs:
                if not self.param_defs[param_name].validate_value(value):
                    return False
        
        # Check global constraints
        if hasattr(self, 'global_constraints'):
            for constraint in self.global_constraints:
                if not constraint(config):
                    return False
        
        return True
    
    def apply_constraints(self, configs: List[Dict[str, Any]], 
                        constraints: List[Callable]) -> List[Dict[str, Any]]:
        """Filter configurations by constraints"""
        valid_configs = []
        
        for config in configs:
            valid = True
            for constraint in constraints:
                if not constraint(config):
                    valid = False
                    break
            
            if valid and self.validate_config(config):
                valid_configs.append(config)
        
        self.logger.info(
            f"Applied constraints: {len(configs)} -> {len(valid_configs)} configs"
        )
        
        return valid_configs

class GridSearchExpander(ParameterExpander):
    """
    Exhaustive grid search parameter expansion.
    Best for low-dimensional spaces.
    """
    
    def __init__(self, parameter_definitions: Dict[str, ParameterDefinition],
                 resolution: Optional[Dict[str, int]] = None):
        super().__init__(parameter_definitions)
        self.resolution = resolution or {}
    
    def expand(self, n_configs: Optional[int] = None,
              constraints: Optional[List[Callable]] = None) -> List[Dict[str, Any]]:
        """Generate grid of parameter combinations"""
        param_grids = {}
        
        for param_name, param_def in self.param_defs.items():
            if param_def.param_type == 'continuous':
                # Use resolution or default to 10 points
                n_points = self.resolution.get(param_name, 10)
                param_grids[param_name] = np.linspace(
                    param_def.bounds[0], param_def.bounds[1], n_points
                ).tolist()
            
            elif param_def.param_type == 'integer':
                # All integers in range
                param_grids[param_name] = list(
                    range(param_def.bounds[0], param_def.bounds[1] + 1)
                )
            
            elif param_def.param_type == 'categorical':
                # All categories
                param_grids[param_name] = param_def.bounds
        
        # Generate all combinations
        all_configs = []
        for values in product(*param_grids.values()):
            config = dict(zip(param_grids.keys(), values))
            all_configs.append(config)
        
        self.logger.info(
            f"Grid search generated {len(all_configs)} configurations"
        )
        
        # Apply constraints if provided
        if constraints:
            all_configs = self.apply_constraints(all_configs, constraints)
        
        # Limit if requested
        if n_configs and len(all_configs) > n_configs:
            self.logger.warning(
                f"Limiting grid from {len(all_configs)} to {n_configs} configs"
            )
            # Sample uniformly from grid
            indices = np.linspace(0, len(all_configs) - 1, n_configs, dtype=int)
            all_configs = [all_configs[i] for i in indices]
        
        self.expansion_history.extend(all_configs)
        return all_configs

class RandomSearchExpander(ParameterExpander):
    """
    Random search parameter expansion.
    Efficient for high-dimensional spaces.
    """
    
    def __init__(self, parameter_definitions: Dict[str, ParameterDefinition],
                 random_seed: Optional[int] = None):
        super().__init__(parameter_definitions)
        self.random_seed = random_seed
        if random_seed:
            np.random.seed(random_seed)
    
    def expand(self, n_configs: Optional[int] = None,
              constraints: Optional[List[Callable]] = None) -> List[Dict[str, Any]]:
        """Generate random parameter configurations"""
        n_configs = n_configs or 100  # Default to 100 random configs
        configs = []
        attempts = 0
        max_attempts = n_configs * 10  # Avoid infinite loop with constraints
        
        while len(configs) < n_configs and attempts < max_attempts:
            # Sample each parameter
            config = {}
            for param_name, param_def in self.param_defs.items():
                config[param_name] = param_def.sample(1)[0]
            
            # Check constraints
            if constraints:
                valid = all(constraint(config) for constraint in constraints)
                if not valid:
                    attempts += 1
                    continue
            
            if self.validate_config(config):
                configs.append(config)
            
            attempts += 1
        
        if len(configs) < n_configs:
            self.logger.warning(
                f"Could only generate {len(configs)} valid configs "
                f"(requested {n_configs})"
            )
        
        self.expansion_history.extend(configs)
        return configs
```

### 2. Bayesian Optimization

```python
# src/optimization/bayesian_optimizer.py
class BayesianOptimizer(ParameterExpander):
    """
    Bayesian optimization for efficient parameter search.
    Uses Gaussian Processes to model objective function.
    """
    
    def __init__(self, parameter_definitions: Dict[str, ParameterDefinition],
                 objective_function: Callable,
                 n_initial_points: int = 10,
                 acquisition_function: str = 'EI'):
        super().__init__(parameter_definitions)
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        
        # Scikit-optimize space
        self.space = self._create_skopt_space()
        
        # Results storage
        self.X_observed = []
        self.y_observed = []
    
    def _create_skopt_space(self) -> List:
        """Convert parameter definitions to skopt space"""
        space = []
        self.param_names = []
        
        for param_name, param_def in self.param_defs.items():
            self.param_names.append(param_name)
            
            if param_def.param_type == 'continuous':
                if param_def.distribution == 'log-uniform':
                    space.append(Real(param_def.bounds[0], 
                                    param_def.bounds[1], 
                                    prior='log-uniform'))
                else:
                    space.append(Real(param_def.bounds[0], 
                                    param_def.bounds[1]))
            
            elif param_def.param_type == 'integer':
                space.append(Integer(param_def.bounds[0], 
                                   param_def.bounds[1]))
            
            elif param_def.param_type == 'categorical':
                space.append(Categorical(param_def.bounds))
        
        return space
    
    def optimize(self, n_calls: int = 100, 
                n_parallel: int = 1,
                callback: Optional[Callable] = None) -> OptimizationResult:
        """Run Bayesian optimization"""
        
        def objective_wrapper(params):
            # Convert list to dict
            config = dict(zip(self.param_names, params))
            
            # Evaluate objective (minimize by default)
            try:
                result = self.objective_function(config)
                return -result if hasattr(self, 'maximize') and self.maximize else result
            except Exception as e:
                self.logger.error(f"Objective evaluation failed: {e}")
                return float('inf')
        
        # Run optimization
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.space,
            n_calls=n_calls,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition_function,
            n_jobs=n_parallel,
            callback=callback,
            random_state=42
        )
        
        # Convert results
        best_config = dict(zip(self.param_names, result.x))
        
        return OptimizationResult(
            best_config=best_config,
            best_value=result.fun,
            all_configs=[dict(zip(self.param_names, x)) for x in result.x_iters],
            all_values=result.func_vals,
            convergence=result.func_vals,
            model=result.models[-1] if result.models else None
        )
    
    def suggest_next(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest next parameters to evaluate"""
        if len(self.X_observed) < self.n_initial_points:
            # Still in random exploration phase
            return RandomSearchExpander(self.param_defs).expand(n_suggestions)
        
        # Use acquisition function to suggest
        suggestions = []
        
        # This is simplified - real implementation would use the GP model
        # to find points that maximize the acquisition function
        
        return suggestions
```

### 3. Parameter Importance Analysis

```python
# src/optimization/parameter_importance.py
class ParameterImportanceAnalyzer:
    """
    Analyzes parameter importance using various methods.
    Helps identify which parameters matter most.
    """
    
    def __init__(self, optimization_results: List[Dict[str, Any]]):
        self.results = optimization_results
        self.logger = ComponentLogger("ParameterImportance", "analysis")
    
    def analyze_importance(self, target_metric: str = 'sharpe_ratio') -> ImportanceReport:
        """Comprehensive parameter importance analysis"""
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Separate parameters and metrics
        param_columns = [col for col in df.columns 
                        if col not in ['sharpe_ratio', 'total_return', 'max_drawdown']]
        
        importance_scores = {}
        
        # Method 1: Correlation analysis
        correlations = {}
        for param in param_columns:
            if df[param].dtype in [np.float64, np.int64]:
                corr = df[param].corr(df[target_metric])
                correlations[param] = abs(corr)
        
        importance_scores['correlation'] = correlations
        
        # Method 2: Mutual information
        mutual_info = self._calculate_mutual_information(df, param_columns, target_metric)
        importance_scores['mutual_information'] = mutual_info
        
        # Method 3: Random forest importance
        rf_importance = self._random_forest_importance(df, param_columns, target_metric)
        importance_scores['random_forest'] = rf_importance
        
        # Method 4: Variance-based sensitivity
        sobol_indices = self._calculate_sobol_indices(df, param_columns, target_metric)
        importance_scores['sobol_indices'] = sobol_indices
        
        # Aggregate scores
        aggregated = self._aggregate_importance_scores(importance_scores)
        
        return ImportanceReport(
            parameter_scores=aggregated,
            method_scores=importance_scores,
            top_parameters=self._get_top_parameters(aggregated, n=5),
            interaction_effects=self._detect_interactions(df, param_columns, target_metric)
        )
    
    def _calculate_mutual_information(self, df: pd.DataFrame, 
                                    params: List[str], 
                                    target: str) -> Dict[str, float]:
        """Calculate mutual information between parameters and target"""
        from sklearn.feature_selection import mutual_info_regression
        
        # Prepare data
        X = df[params].fillna(0)
        y = df[target]
        
        # Handle categorical variables
        X_encoded = pd.get_dummies(X)
        
        # Calculate MI
        mi_scores = mutual_info_regression(X_encoded, y, random_state=42)
        
        # Map back to parameter names
        mi_dict = {}
        for i, col in enumerate(X_encoded.columns):
            # Find original parameter
            for param in params:
                if col.startswith(param):
                    if param not in mi_dict:
                        mi_dict[param] = 0
                    mi_dict[param] += mi_scores[i]
        
        return mi_dict
    
    def _random_forest_importance(self, df: pd.DataFrame,
                                params: List[str],
                                target: str) -> Dict[str, float]:
        """Use Random Forest to estimate feature importance"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        X = df[params].copy()
        y = df[target]
        
        # Encode categorical variables
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('missing'))
                encoders[col] = le
        
        X = X.fillna(0)
        
        # Train random forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get importance scores
        importance_dict = dict(zip(params, rf.feature_importances_))
        
        return importance_dict
    
    def _calculate_sobol_indices(self, df: pd.DataFrame,
                               params: List[str],
                               target: str) -> Dict[str, float]:
        """Calculate Sobol sensitivity indices"""
        # Simplified version - real implementation would use SALib
        # This is a variance-based approximation
        
        total_variance = df[target].var()
        sobol_indices = {}
        
        for param in params:
            if df[param].dtype in [np.float64, np.int64]:
                # Group by parameter values and calculate conditional variance
                grouped = df.groupby(pd.qcut(df[param], q=10, duplicates='drop'))[target]
                conditional_variance = grouped.var().mean()
                
                # First-order Sobol index
                sobol_indices[param] = 1 - (conditional_variance / total_variance)
        
        return sobol_indices
```

### 4. Constraint Handling

```python
# src/optimization/constraints.py
class ConstraintHandler:
    """
    Manages parameter constraints for optimization.
    Supports various constraint types.
    """
    
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.logger = ComponentLogger("ConstraintHandler", "optimization")
    
    def add_linear_constraint(self, 
                            coefficients: Dict[str, float],
                            operator: str,
                            value: float):
        """Add linear constraint: sum(coef[i] * param[i]) operator value"""
        def constraint_func(config: Dict[str, Any]) -> bool:
            total = sum(coefficients.get(k, 0) * config.get(k, 0) 
                       for k in coefficients)
            
            if operator == '<=':
                return total <= value
            elif operator == '>=':
                return total >= value
            elif operator == '==':
                return abs(total - value) < 1e-6
            else:
                raise ValueError(f"Unknown operator: {operator}")
        
        self.constraints.append(Constraint(
            name=f"linear_{len(self.constraints)}",
            func=constraint_func,
            constraint_type='linear'
        ))
    
    def add_nonlinear_constraint(self, 
                               constraint_func: Callable[[Dict], bool],
                               name: str):
        """Add custom nonlinear constraint"""
        self.constraints.append(Constraint(
            name=name,
            func=constraint_func,
            constraint_type='nonlinear'
        ))
    
    def add_dependency_constraint(self,
                                param1: str,
                                param2: str,
                                relationship: str):
        """Add parameter dependency constraint"""
        def constraint_func(config: Dict[str, Any]) -> bool:
            v1 = config.get(param1)
            v2 = config.get(param2)
            
            if v1 is None or v2 is None:
                return True
            
            if relationship == 'requires':
                # If param1 is set, param2 must be set
                return not (v1 and not v2)
            elif relationship == 'excludes':
                # param1 and param2 cannot both be set
                return not (v1 and v2)
            elif relationship == '<':
                return v1 < v2
            elif relationship == '>':
                return v1 > v2
            else:
                raise ValueError(f"Unknown relationship: {relationship}")
        
        self.constraints.append(Constraint(
            name=f"{param1}_{relationship}_{param2}",
            func=constraint_func,
            constraint_type='dependency'
        ))
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration against all constraints"""
        violations = []
        
        for constraint in self.constraints:
            try:
                if not constraint.func(config):
                    violations.append(constraint.name)
            except Exception as e:
                self.logger.error(
                    f"Error checking constraint {constraint.name}: {e}"
                )
                violations.append(f"{constraint.name} (error)")
        
        return len(violations) == 0, violations
    
    def repair_config(self, config: Dict[str, Any], 
                     param_defs: Dict[str, ParameterDefinition]) -> Optional[Dict[str, Any]]:
        """Attempt to repair constraint violations"""
        # Simple repair strategy - more sophisticated methods possible
        repaired = config.copy()
        
        # Try small perturbations
        for _ in range(10):
            valid, violations = self.validate_config(repaired)
            if valid:
                return repaired
            
            # Perturb violated parameters
            for param_name, param_def in param_defs.items():
                if param_name in repaired:
                    if param_def.param_type == 'continuous':
                        # Small random perturbation
                        delta = (param_def.bounds[1] - param_def.bounds[0]) * 0.01
                        repaired[param_name] += np.random.uniform(-delta, delta)
                        repaired[param_name] = np.clip(
                            repaired[param_name], 
                            param_def.bounds[0], 
                            param_def.bounds[1]
                        )
        
        return None  # Could not repair
```

### 5. Integration with Replay Engine

```python
# src/optimization/replay_parameter_optimizer.py
class ReplayParameterOptimizer:
    """
    Integrates parameter expansion with signal replay.
    Enables fast parameter optimization.
    """
    
    def __init__(self, replay_engine: SignalReplayEngine,
                 param_definitions: Dict[str, ParameterDefinition]):
        self.replay_engine = replay_engine
        self.param_defs = param_definitions
        self.logger = ComponentLogger("ReplayParameterOptimizer", "optimization")
    
    def optimize_with_method(self, method: str = 'bayesian',
                           n_iterations: int = 100,
                           objective: str = 'sharpe_ratio',
                           constraints: Optional[List[Callable]] = None) -> OptimizationResult:
        """Optimize parameters using specified method"""
        
        # Create objective function
        def objective_function(config: Dict[str, Any]) -> float:
            # Create replay config from parameters
            replay_config = self._params_to_replay_config(config)
            
            # Run replay
            try:
                result = self.replay_engine.replay(replay_config)
                
                # Extract objective value
                if objective == 'sharpe_ratio':
                    return result.sharpe_ratio
                elif objective == 'total_return':
                    return result.total_return
                elif objective == 'calmar_ratio':
                    return result.total_return / result.max_drawdown \
                           if result.max_drawdown > 0 else 0
                else:
                    raise ValueError(f"Unknown objective: {objective}")
                    
            except Exception as e:
                self.logger.error(f"Replay failed for config {config}: {e}")
                return -np.inf  # Penalize failures
        
        # Select optimization method
        if method == 'grid':
            expander = GridSearchExpander(self.param_defs)
            configs = expander.expand(n_configs=n_iterations, constraints=constraints)
            return self._evaluate_configs(configs, objective_function, objective)
            
        elif method == 'random':
            expander = RandomSearchExpander(self.param_defs)
            configs = expander.expand(n_configs=n_iterations, constraints=constraints)
            return self._evaluate_configs(configs, objective_function, objective)
            
        elif method == 'bayesian':
            optimizer = BayesianOptimizer(
                self.param_defs, 
                objective_function,
                n_initial_points=min(10, n_iterations // 5)
            )
            optimizer.maximize = True  # We want to maximize Sharpe
            return optimizer.optimize(n_calls=n_iterations)
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _evaluate_configs(self, configs: List[Dict[str, Any]], 
                        objective_function: Callable,
                        objective_name: str) -> OptimizationResult:
        """Evaluate list of configurations"""
        results = []
        
        for i, config in enumerate(configs):
            self.logger.info(f"Evaluating config {i+1}/{len(configs)}")
            
            value = objective_function(config)
            results.append({
                'config': config,
                'value': value
            })
        
        # Find best
        best_idx = np.argmax([r['value'] for r in results])
        
        return OptimizationResult(
            best_config=results[best_idx]['config'],
            best_value=results[best_idx]['value'],
            all_configs=[r['config'] for r in results],
            all_values=[r['value'] for r in results],
            objective=objective_name,
            method=self.method
        )
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step9_parameter_expansion.py`:

```python
class TestParameterDefinition:
    """Test parameter definition and sampling"""
    
    def test_continuous_parameter(self):
        """Test continuous parameter handling"""
        param = ParameterDefinition(
            name='risk_level',
            param_type='continuous',
            bounds=(0.01, 0.05),
            default_value=0.02
        )
        
        # Test validation
        assert param.validate_value(0.03)
        assert not param.validate_value(0.0)
        assert not param.validate_value(0.1)
        
        # Test sampling
        samples = param.sample(100, random_state=42)
        assert all(0.01 <= s <= 0.05 for s in samples)
        assert len(samples) == 100

class TestGridSearchExpander:
    """Test grid search expansion"""
    
    def test_basic_grid_expansion(self):
        """Test basic grid generation"""
        param_defs = {
            'param1': ParameterDefinition('param1', 'continuous', (0, 1), 0.5),
            'param2': ParameterDefinition('param2', 'integer', (1, 3), 2)
        }
        
        expander = GridSearchExpander(
            param_defs, 
            resolution={'param1': 3}
        )
        
        configs = expander.expand()
        
        # Should have 3 * 3 = 9 configurations
        assert len(configs) == 9
        
        # Check all combinations present
        param1_values = [0.0, 0.5, 1.0]
        param2_values = [1, 2, 3]
        
        for p1 in param1_values:
            for p2 in param2_values:
                assert any(c['param1'] == p1 and c['param2'] == p2 
                          for c in configs)

class TestBayesianOptimizer:
    """Test Bayesian optimization"""
    
    def test_optimization_convergence(self):
        """Test that Bayesian optimization improves"""
        # Define simple quadratic objective
        def objective(config):
            x = config['x']
            return -(x - 0.7)**2  # Maximum at x=0.7
        
        param_defs = {
            'x': ParameterDefinition('x', 'continuous', (0, 1), 0.5)
        }
        
        optimizer = BayesianOptimizer(param_defs, objective)
        result = optimizer.optimize(n_calls=20)
        
        # Should find near-optimal value
        assert abs(result.best_config['x'] - 0.7) < 0.1
        assert result.best_value > -0.01  # Close to maximum of 0
```

### Integration Tests

Create `tests/integration/test_step9_optimization_integration.py`:

```python
def test_parameter_expansion_with_replay():
    """Test parameter optimization with replay engine"""
    # Setup replay engine with captured signals
    replay_engine = setup_test_replay_engine()
    
    # Define parameters to optimize
    param_defs = {
        'max_position_size': ParameterDefinition(
            'max_position_size', 'continuous', (0.05, 0.20), 0.10
        ),
        'risk_per_trade': ParameterDefinition(
            'risk_per_trade', 'continuous', (0.01, 0.03), 0.02
        ),
        'position_sizing_method': ParameterDefinition(
            'position_sizing_method', 'categorical', 
            ['fixed', 'risk_based', 'volatility_adjusted'], 
            'fixed'
        )
    }
    
    # Create optimizer
    optimizer = ReplayParameterOptimizer(replay_engine, param_defs)
    
    # Run optimization
    result = optimizer.optimize_with_method(
        method='random',
        n_iterations=50,
        objective='sharpe_ratio'
    )
    
    # Verify optimization found improvement
    assert result.best_value > 1.0  # Reasonable Sharpe
    assert 0.05 <= result.best_config['max_position_size'] <= 0.20
    assert result.best_config['position_sizing_method'] in \
           ['fixed', 'risk_based', 'volatility_adjusted']

def test_constraint_handling():
    """Test parameter constraints in optimization"""
    param_defs = {
        'leverage': ParameterDefinition('leverage', 'continuous', (0, 3), 1),
        'risk_limit': ParameterDefinition('risk_limit', 'continuous', (0.01, 0.1), 0.05),
        'max_positions': ParameterDefinition('max_positions', 'integer', (1, 20), 10)
    }
    
    # Add constraints
    constraint_handler = ConstraintHandler()
    
    # Linear constraint: leverage * risk_limit <= 0.15
    constraint_handler.add_linear_constraint(
        {'leverage': 1, 'risk_limit': 1}, '<=', 0.15
    )
    
    # Dependency: high leverage requires low max_positions
    constraint_handler.add_nonlinear_constraint(
        lambda cfg: not (cfg['leverage'] > 2 and cfg['max_positions'] > 5),
        'leverage_position_limit'
    )
    
    # Generate configurations
    expander = RandomSearchExpander(param_defs)
    configs = expander.expand(n_configs=100, 
                            constraints=constraint_handler.constraints)
    
    # Verify all configs satisfy constraints
    for config in configs:
        valid, violations = constraint_handler.validate_config(config)
        assert valid, f"Config {config} violates {violations}"
```

### System Tests

Create `tests/system/test_step9_full_optimization.py`:

```python
def test_multi_method_optimization_comparison():
    """Compare different optimization methods"""
    # Setup
    replay_engine = create_production_replay_engine()
    param_defs = create_comprehensive_param_definitions()
    
    methods = ['grid', 'random', 'bayesian']
    results = {}
    
    # Run each method
    for method in methods:
        optimizer = ReplayParameterOptimizer(replay_engine, param_defs)
        
        start_time = time.time()
        result = optimizer.optimize_with_method(
            method=method,
            n_iterations=100,
            objective='sharpe_ratio'
        )
        elapsed = time.time() - start_time
        
        results[method] = {
            'result': result,
            'time': elapsed,
            'efficiency': result.best_value / elapsed  # Sharpe per second
        }
    
    # Compare results
    best_sharpes = {m: r['result'].best_value for m, r in results.items()}
    
    # Bayesian should be most efficient
    assert results['bayesian']['efficiency'] > results['random']['efficiency']
    
    # All methods should find reasonable solutions
    assert all(sharpe > 0.5 for sharpe in best_sharpes.values())
    
    # Log comparison
    print("\nOptimization Method Comparison:")
    for method, result in results.items():
        print(f"{method:10} - Sharpe: {result['result'].best_value:.3f}, "
              f"Time: {result['time']:.1f}s, "
              f"Efficiency: {result['efficiency']:.3f}")

def test_parameter_importance_workflow():
    """Test complete parameter importance analysis"""
    # Run optimization first
    optimizer = create_test_optimizer()
    opt_result = optimizer.optimize_with_method(
        method='random',
        n_iterations=200,
        objective='sharpe_ratio'
    )
    
    # Analyze parameter importance
    analyzer = ParameterImportanceAnalyzer([
        {'config': cfg, 'sharpe_ratio': val}
        for cfg, val in zip(opt_result.all_configs, opt_result.all_values)
    ])
    
    importance_report = analyzer.analyze_importance('sharpe_ratio')
    
    # Verify analysis completeness
    assert importance_report.parameter_scores is not None
    assert len(importance_report.top_parameters) > 0
    
    # Most important parameters should have high scores
    top_param = importance_report.top_parameters[0]
    assert importance_report.parameter_scores[top_param] > 0.5
    
    # Check interaction effects detected
    assert importance_report.interaction_effects is not None
```

## ✅ Validation Checklist

### Parameter Expansion
- [ ] Grid search generates all combinations
- [ ] Random search respects bounds
- [ ] Bayesian optimization converges
- [ ] Constraints properly enforced
- [ ] Parameter validation working

### Optimization Integration
- [ ] Replay engine integration smooth
- [ ] Objective functions evaluated correctly
- [ ] Results tracked properly
- [ ] Performance acceptable

### Analysis Tools
- [ ] Parameter importance calculated
- [ ] Sensitivity analysis working
- [ ] Constraint validation accurate
- [ ] Results reproducible

## 📊 Performance Optimization

### Parallel Evaluation
```python
class ParallelParameterOptimizer:
    """Evaluate multiple configurations in parallel"""
    
    def parallel_evaluate(self, configs: List[Dict], n_workers: int = None):
        n_workers = n_workers or mp.cpu_count() - 1
        
        with mp.Pool(n_workers) as pool:
            results = pool.map(self.evaluate_single, configs)
        
        return results
```

### Caching Results
```python
class CachedOptimizer:
    """Cache evaluation results to avoid redundant calculations"""
    
    def __init__(self):
        self.cache = {}
    
    def evaluate_with_cache(self, config: Dict) -> float:
        config_key = self._config_to_key(config)
        
        if config_key in self.cache:
            return self.cache[config_key]
        
        result = self.evaluate(config)
        self.cache[config_key] = result
        
        return result
```

## 🐛 Common Issues

1. **Parameter Scaling**
   - Normalize parameters for optimization
   - Use log scale for wide ranges
   - Consider parameter interactions

2. **Local Optima**
   - Use multiple random starts
   - Increase exploration in Bayesian
   - Try different methods

3. **Constraint Violations**
   - Implement repair mechanisms
   - Use penalty methods
   - Consider soft constraints

## 🎯 Success Criteria

Step 9 is complete when:
1. ✅ All parameter expansion methods working
2. ✅ Optimization integrated with replay
3. ✅ Constraints properly handled
4. ✅ Parameter importance analysis functional
5. ✅ All test tiers pass

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10: End-to-End Workflow](step-10-end-to-end-workflow.md)

## 📚 Additional Resources

- [Optimization Methods Comparison](../references/optimization-comparison.md)
- [Bayesian Optimization Tutorial](../references/bayesian-tutorial.md)
- [Constraint Handling Techniques](../references/constraint-methods.md)