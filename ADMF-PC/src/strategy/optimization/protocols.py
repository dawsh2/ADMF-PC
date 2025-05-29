"""
Optimization protocols for ADMF-PC.

These protocols define the contracts for optimization components without
requiring inheritance.
"""

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional, Tuple, Callable
from abc import abstractmethod


@runtime_checkable
class Optimizable(Protocol):
    """Protocol for components that can be optimized"""
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """
        Return parameter space for optimization.
        
        Returns:
            Dict mapping parameter names to their possible values or ranges.
            Examples:
                {'period': [10, 20, 30]}  # Discrete values
                {'threshold': (0.0, 1.0)}  # Continuous range
                {'method': ['SMA', 'EMA']}  # Categorical
        """
        ...
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Apply parameter values to the component.
        
        Args:
            params: Parameter values to apply
        """
        ...
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Current parameter values
        """
        ...
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameter values.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for optimization algorithms"""
    
    @abstractmethod
    def optimize(self, evaluate_func: Callable[[Dict[str, Any]], float], 
                n_trials: int = None, **kwargs) -> Dict[str, Any]:
        """
        Run optimization and return best parameters.
        
        Args:
            evaluate_func: Function that takes parameters and returns score
            n_trials: Number of trials to run (optional)
            **kwargs: Additional optimizer-specific arguments
            
        Returns:
            Best parameters found
        """
        ...
    
    @abstractmethod
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get best parameters found so far"""
        ...
    
    @abstractmethod
    def get_best_score(self) -> float:
        """Get best score achieved"""
        ...
    
    @abstractmethod
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all trials.
        
        Returns:
            List of dicts with 'params', 'score', and other trial info
        """
        ...


@runtime_checkable
class Objective(Protocol):
    """Protocol for optimization objectives"""
    
    @abstractmethod
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate objective value from results.
        
        Args:
            results: Backtest or evaluation results
            
        Returns:
            Objective value (higher is better)
        """
        ...
    
    @abstractmethod
    def get_direction(self) -> str:
        """
        Get optimization direction.
        
        Returns:
            'maximize' or 'minimize'
        """
        ...
    
    @abstractmethod
    def get_requirements(self) -> List[str]:
        """
        Get required fields in results dict.
        
        Returns:
            List of required result fields
        """
        ...


@runtime_checkable
class Constraint(Protocol):
    """Protocol for parameter constraints"""
    
    @abstractmethod
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """
        Check if parameters satisfy constraint.
        
        Args:
            params: Parameters to check
            
        Returns:
            True if constraint is satisfied
        """
        ...
    
    @abstractmethod
    def validate_and_adjust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and optionally adjust parameters to satisfy constraint.
        
        Args:
            params: Parameters to validate/adjust
            
        Returns:
            Adjusted parameters (or original if valid)
        """
        ...
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of constraint"""
        ...


@runtime_checkable
class OptimizationWorkflow(Protocol):
    """Protocol for optimization workflows"""
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute the workflow and return results.
        
        Returns:
            Workflow results including best parameters, scores, etc.
        """
        ...
    
    @abstractmethod
    def get_stages(self) -> List[str]:
        """
        Get list of workflow stages.
        
        Returns:
            Ordered list of stage names
        """
        ...
    
    @abstractmethod
    def get_current_stage(self) -> Optional[str]:
        """
        Get currently executing stage.
        
        Returns:
            Current stage name or None
        """
        ...
    
    @abstractmethod
    def cancel(self) -> None:
        """Cancel the running workflow"""
        ...


@runtime_checkable
class ParameterSampler(Protocol):
    """Protocol for parameter sampling strategies"""
    
    @abstractmethod
    def sample(self, parameter_space: Dict[str, Any], 
              n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample parameters from the space.
        
        Args:
            parameter_space: Parameter space definition
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter combinations
        """
        ...
    
    @abstractmethod
    def get_next_sample(self, parameter_space: Dict[str, Any],
                       history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get next sample based on history.
        
        Args:
            parameter_space: Parameter space definition
            history: Previous trials and their results
            
        Returns:
            Next parameter combination to try
        """
        ...