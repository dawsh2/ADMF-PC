"""
Optimizer implementations for ADMF-PC.

These optimizers implement the Optimizer protocol without inheritance.
They can be used to optimize any component that implements Optimizable.
"""

from typing import Dict, Any, List, Optional, Callable
import itertools
import random
from datetime import datetime

from .protocols import Optimizer


class GridOptimizer:
    """
    Grid search optimizer.
    
    Exhaustively searches all parameter combinations.
    Best for smaller parameter spaces or when you need
    to understand the full parameter landscape.
    """
    
    def __init__(self):
        """Initialize grid optimizer."""
        self.history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self._current_trial = 0
        
    def optimize(self, 
                evaluate_func: Callable[[Dict[str, Any]], float],
                parameter_space: Dict[str, Any],
                n_trials: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Args:
            evaluate_func: Function that evaluates parameters
            parameter_space: Space to search
            n_trials: Max trials (uses all combinations if None)
            **kwargs: Additional options
            
        Returns:
            Best parameters found
        """
        # Generate all combinations
        combinations = self._generate_combinations(parameter_space)
        
        # Limit trials if specified
        if n_trials and n_trials < len(combinations):
            combinations = combinations[:n_trials]
        
        # Reset state
        self.history.clear()
        self.best_params = None
        self.best_score = None
        self._current_trial = 0
        
        # Evaluate each combination
        for params in combinations:
            self._current_trial += 1
            
            try:
                # Evaluate parameters
                score = evaluate_func(params)
                
                # Record trial
                trial_result = {
                    'trial_number': self._current_trial,
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': datetime.now()
                }
                self.history.append(trial_result)
                
                # Update best if improved
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
                # Progress callback if provided
                if 'progress_callback' in kwargs:
                    kwargs['progress_callback'](
                        self._current_trial, 
                        len(combinations),
                        self.best_score
                    )
                    
            except Exception as e:
                # Record failed trial
                trial_result = {
                    'trial_number': self._current_trial,
                    'parameters': params.copy(),
                    'score': None,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
                self.history.append(trial_result)
        
        return self.best_params or {}
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        return self.best_params.copy() if self.best_params else None
    
    def get_best_score(self) -> Optional[float]:
        """Get best score achieved."""
        return self.best_score
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all trials."""
        return self.history.copy()
    
    def _generate_combinations(self, parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from space."""
        # Extract parameter names and values
        param_names = []
        param_values = []
        
        for name, space_def in parameter_space.items():
            param_names.append(name)
            
            if isinstance(space_def, list):
                # Discrete values
                param_values.append(space_def)
            elif isinstance(space_def, tuple) and len(space_def) == 2:
                # Range - generate some values
                # For grid search, we need discrete values
                min_val, max_val = space_def
                steps = 5  # Default number of steps
                if isinstance(min_val, int) and isinstance(max_val, int):
                    step_size = max(1, (max_val - min_val) // (steps - 1))
                    values = [min_val + i * step_size for i in range(steps)]
                    values[-1] = max_val  # Ensure we include max
                else:
                    step_size = (max_val - min_val) / (steps - 1)
                    values = [min_val + i * step_size for i in range(steps)]
                param_values.append(values)
            elif isinstance(space_def, dict):
                # Complex definition
                if 'values' in space_def:
                    param_values.append(space_def['values'])
                elif 'min' in space_def and 'max' in space_def:
                    # Generate values based on step
                    min_val = space_def['min']
                    max_val = space_def['max']
                    step = space_def.get('step', (max_val - min_val) / 4)
                    values = []
                    val = min_val
                    while val <= max_val:
                        values.append(val)
                        val += step
                    param_values.append(values)
                else:
                    param_values.append([space_def.get('default', 0)])
            else:
                # Single value
                param_values.append([space_def])
        
        # Generate all combinations
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations


class RandomOptimizer:
    """
    Random search optimizer.
    
    Randomly samples parameter space. Often more efficient
    than grid search for high-dimensional spaces.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random optimizer.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            
        self.history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self._current_trial = 0
        
    def optimize(self, 
                evaluate_func: Callable[[Dict[str, Any]], float],
                parameter_space: Dict[str, Any],
                n_trials: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """Run random search optimization."""
        # Default number of trials
        if n_trials is None:
            n_trials = 100
            
        # Reset state
        self.history.clear()
        self.best_params = None
        self.best_score = None
        self._current_trial = 0
        
        # Run trials
        for _ in range(n_trials):
            self._current_trial += 1
            
            # Sample parameters
            params = self._sample_parameters(parameter_space)
            
            try:
                # Evaluate
                score = evaluate_func(params)
                
                # Record trial
                trial_result = {
                    'trial_number': self._current_trial,
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': datetime.now()
                }
                self.history.append(trial_result)
                
                # Update best
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
            except Exception as e:
                # Record failed trial
                trial_result = {
                    'trial_number': self._current_trial,
                    'parameters': params.copy(),
                    'score': None,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
                self.history.append(trial_result)
        
        return self.best_params or {}
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        return self.best_params.copy() if self.best_params else None
    
    def get_best_score(self) -> Optional[float]:
        """Get best score achieved."""
        return self.best_score
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all trials."""
        return self.history.copy()
    
    def _sample_parameters(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from space."""
        params = {}
        
        for name, space_def in parameter_space.items():
            if isinstance(space_def, list):
                # Choose from discrete values
                params[name] = random.choice(space_def)
            elif isinstance(space_def, tuple) and len(space_def) == 2:
                # Sample from range
                min_val, max_val = space_def
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[name] = random.randint(min_val, max_val)
                else:
                    params[name] = random.uniform(min_val, max_val)
            elif isinstance(space_def, dict):
                # Complex definition
                if 'values' in space_def:
                    params[name] = random.choice(space_def['values'])
                elif 'min' in space_def and 'max' in space_def:
                    min_val = space_def['min']
                    max_val = space_def['max']
                    if space_def.get('type') == 'int':
                        params[name] = random.randint(min_val, max_val)
                    else:
                        params[name] = random.uniform(min_val, max_val)
                else:
                    params[name] = space_def.get('default', 0)
            else:
                # Single value
                params[name] = space_def
        
        return params


# Factory functions
def create_grid_optimizer() -> GridOptimizer:
    """Create a grid search optimizer."""
    return GridOptimizer()


def create_random_optimizer(seed: Optional[int] = None) -> RandomOptimizer:
    """Create a random search optimizer."""
    return RandomOptimizer(seed)