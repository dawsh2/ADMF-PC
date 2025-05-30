"""
Optimization protocols for ADMF-PC strategy module.

These protocols define contracts for optimization components without
requiring inheritance, enabling flexible composition.
"""

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional, Tuple, Callable


@runtime_checkable
class Optimizer(Protocol):
    """
    Protocol for optimization algorithms.
    
    Optimizers search parameter spaces to find optimal values
    according to an objective function.
    """
    
    def optimize(self, 
                evaluate_func: Callable[[Dict[str, Any]], float],
                parameter_space: Dict[str, Any],
                n_trials: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Run optimization and return best parameters.
        
        Args:
            evaluate_func: Function that takes parameters and returns score
            parameter_space: Parameter space definition
            n_trials: Number of trials to run
            **kwargs: Additional optimizer-specific arguments
            
        Returns:
            Best parameters found
        """
        ...
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        ...
    
    def get_best_score(self) -> Optional[float]:
        """Get best score achieved."""
        ...
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all trials.
        
        Returns:
            List of dicts with 'parameters', 'score', and metadata
        """
        ...


@runtime_checkable
class Objective(Protocol):
    """
    Protocol for optimization objectives.
    
    Objectives define what metric to optimize and how to calculate it
    from backtest or evaluation results.
    """
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate objective value from results.
        
        Args:
            results: Evaluation results (e.g., backtest metrics)
            
        Returns:
            Objective value (higher is better by convention)
        """
        ...
    
    def get_direction(self) -> str:
        """
        Get optimization direction.
        
        Returns:
            'maximize' or 'minimize'
        """
        ...
    
    def get_requirements(self) -> List[str]:
        """
        Get required fields in results dict.
        
        Returns:
            List of required result fields
        """
        ...
    
    def is_better(self, score1: float, score2: float) -> bool:
        """
        Compare two scores.
        
        Returns:
            True if score1 is better than score2
        """
        ...


@runtime_checkable
class Constraint(Protocol):
    """
    Protocol for parameter constraints.
    
    Constraints ensure parameter combinations are valid and
    can adjust invalid parameters to satisfy requirements.
    """
    
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """Check if parameters satisfy constraint."""
        ...
    
    def validate_and_adjust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and optionally adjust parameters.
        
        Args:
            params: Parameters to validate/adjust
            
        Returns:
            Adjusted parameters (or original if valid)
        """
        ...
    
    def get_description(self) -> str:
        """Get human-readable description of constraint."""
        ...


@runtime_checkable
class ParameterSpace(Protocol):
    """
    Protocol for parameter space definitions.
    
    Parameter spaces define the searchable space for optimization,
    including types, ranges, and valid values.
    """
    
    def get_dimensions(self) -> Dict[str, Any]:
        """
        Get parameter dimensions.
        
        Returns:
            Dict mapping parameter names to their definitions
        """
        ...
    
    def sample(self, n_samples: int, method: str = 'random') -> List[Dict[str, Any]]:
        """
        Sample parameter combinations.
        
        Args:
            n_samples: Number of samples
            method: Sampling method ('random', 'grid', 'latin_hypercube')
            
        Returns:
            List of parameter combinations
        """
        ...
    
    def is_valid(self, params: Dict[str, Any]) -> bool:
        """Check if parameters are within space."""
        ...
    
    def get_size(self) -> Optional[int]:
        """
        Get total size of parameter space.
        
        Returns:
            Number of combinations (None if infinite)
        """
        ...


@runtime_checkable
class OptimizationWorkflow(Protocol):
    """
    Protocol for optimization workflows.
    
    Workflows orchestrate complex optimization processes,
    potentially with multiple stages and components.
    """
    
    async def run(self) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Returns:
            Workflow results including best parameters, metrics, etc.
        """
        ...
    
    def get_stages(self) -> List[str]:
        """Get list of workflow stages."""
        ...
    
    def get_current_stage(self) -> Optional[str]:
        """Get currently executing stage."""
        ...
    
    def cancel(self) -> None:
        """Cancel the running workflow."""
        ...
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get workflow progress.
        
        Returns:
            Dict with 'stage', 'progress', 'estimated_time_remaining'
        """
        ...


@runtime_checkable
class RegimeAnalyzer(Protocol):
    """
    Protocol for regime-based performance analysis.
    
    Analyzes trading results to determine optimal parameters
    for each market regime.
    """
    
    def analyze_trades(self, 
                      trades: List[Dict[str, Any]], 
                      regime_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trades with regime context.
        
        Args:
            trades: List of completed trades
            regime_history: List of regime classifications over time
            
        Returns:
            Analysis results with regime-specific metrics
        """
        ...
    
    def get_regime_metrics(self, regime: str) -> Dict[str, float]:
        """Get performance metrics for specific regime."""
        ...
    
    def get_best_parameters_by_regime(self) -> Dict[str, Dict[str, Any]]:
        """Get optimal parameters for each regime."""
        ...
    
    def get_regime_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each regime.
        
        Returns:
            Dict with trade counts, durations, transitions, etc.
        """
        ...


@runtime_checkable
class OptimizationContainer(Protocol):
    """
    Protocol for optimization-specific containers.
    
    These containers manage isolated environments for running
    optimization trials with proper state management.
    """
    
    def create_trial_instance(self, 
                            parameters: Dict[str, Any],
                            trial_id: str) -> Tuple[str, Any]:
        """
        Create isolated instance for parameter trial.
        
        Args:
            parameters: Trial parameters
            trial_id: Unique trial identifier
            
        Returns:
            Tuple of (instance_id, component)
        """
        ...
    
    def run_trial(self,
                 parameters: Dict[str, Any],
                 evaluator: Callable) -> Dict[str, Any]:
        """
        Run optimization trial in isolation.
        
        Args:
            parameters: Trial parameters
            evaluator: Function to evaluate the trial
            
        Returns:
            Trial results with metrics
        """
        ...
    
    def get_trial_results(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Get results for specific trial."""
        ...
    
    def cleanup_trial(self, trial_id: str) -> None:
        """Clean up resources for completed trial."""
        ...


@runtime_checkable
class ParameterSampler(Protocol):
    """
    Protocol for parameter sampling strategies.
    
    Samplers generate parameter combinations for optimization
    using various strategies.
    """
    
    def sample(self, 
              parameter_space: Dict[str, Any],
              n_samples: int,
              constraints: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Sample parameters from space.
        
        Args:
            parameter_space: Space definition
            n_samples: Number of samples
            constraints: Optional constraints to satisfy
            
        Returns:
            List of parameter combinations
        """
        ...
    
    def get_next_sample(self,
                       parameter_space: Dict[str, Any],
                       history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get next sample based on history.
        
        Args:
            parameter_space: Space definition
            history: Previous trials and results
            
        Returns:
            Next parameter combination
        """
        ...