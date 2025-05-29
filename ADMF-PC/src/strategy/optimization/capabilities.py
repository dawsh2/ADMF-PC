"""
Optimization capability for ADMF-PC.

This capability adds optimization support to any component without
requiring inheritance.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from ...core.components.protocols import Capability


class OptimizationCapability(Capability):
    """Adds optimization support to any component"""
    
    def get_name(self) -> str:
        return "optimization"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply optimization capability to component"""
        
        # Check if component already has optimization methods
        has_methods = all(hasattr(component, method) for method in 
                         ['get_parameter_space', 'set_parameters', 
                          'get_parameters', 'validate_parameters'])
        
        if not has_methods:
            # Add default implementations
            self._add_default_optimization_methods(component, spec)
        
        # Add optimization metadata
        component._optimization_metadata = {
            'optimizable': True,
            'last_optimization': None,
            'parameter_history': [],
            'best_parameters': None,
            'best_score': None
        }
        
        # Add parameter tracking
        self._add_parameter_tracking(component)
        
        # Add optimization helpers
        self._add_optimization_helpers(component)
        
        # Initialize with default parameters if provided
        default_params = spec.get('default_params', {})
        if default_params:
            component.set_parameters(default_params)
        
        return component
    
    def _add_default_optimization_methods(self, component: Any, spec: Dict[str, Any]) -> None:
        """Add default optimization methods for components without them"""
        
        # Store parameter space from spec (check metadata too)
        parameter_space = spec.get('parameter_space', {})
        if not parameter_space and 'metadata' in spec:
            parameter_space = spec['metadata'].get('parameter_space', {})
        
        def get_parameter_space() -> Dict[str, Any]:
            """Get parameter space for optimization"""
            return parameter_space
        
        def set_parameters(params: Dict[str, Any]) -> None:
            """Set parameters on component"""
            # Store parameters as attributes
            for name, value in params.items():
                if hasattr(component, name):
                    setattr(component, name, value)
                else:
                    # Store in a parameters dict if attribute doesn't exist
                    if not hasattr(component, '_parameters'):
                        component._parameters = {}
                    component._parameters[name] = value
        
        def get_parameters() -> Dict[str, Any]:
            """Get current parameters"""
            params = {}
            
            # Get from parameter space keys
            for param_name in parameter_space.keys():
                if hasattr(component, param_name):
                    params[param_name] = getattr(component, param_name)
                elif hasattr(component, '_parameters') and param_name in component._parameters:
                    params[param_name] = component._parameters[param_name]
            
            return params
        
        def validate_parameters(params: Dict[str, Any]) -> tuple[bool, str]:
            """Validate parameters"""
            # Check all required parameters are present
            for required_param in parameter_space.keys():
                if required_param not in params:
                    return False, f"Missing required parameter: {required_param}"
            
            # Check parameter values are valid
            for param_name, param_value in params.items():
                if param_name in parameter_space:
                    valid_values = parameter_space[param_name]
                    
                    # Handle different parameter space types
                    if isinstance(valid_values, list):
                        # Discrete values
                        if param_value not in valid_values:
                            return False, f"Invalid value for {param_name}: {param_value}"
                    elif isinstance(valid_values, tuple) and len(valid_values) == 2:
                        # Range (min, max)
                        if not (valid_values[0] <= param_value <= valid_values[1]):
                            return False, f"{param_name} must be between {valid_values[0]} and {valid_values[1]}"
            
            return True, ""
        
        # Attach methods
        component.get_parameter_space = get_parameter_space
        component.set_parameters = set_parameters
        component.get_parameters = get_parameters
        component.validate_parameters = validate_parameters
    
    def _add_parameter_tracking(self, component: Any) -> None:
        """Add parameter history tracking"""
        
        original_set_params = component.set_parameters
        
        def tracked_set_parameters(params: Dict[str, Any]) -> None:
            """Set parameters with tracking"""
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
    
    def _add_optimization_helpers(self, component: Any) -> None:
        """Add helper methods for optimization"""
        
        def reset_to_defaults() -> None:
            """Reset parameters to defaults"""
            default_params = {}
            param_space = component.get_parameter_space()
            
            # Use first value for discrete, middle for ranges
            for param_name, param_def in param_space.items():
                if isinstance(param_def, list) and param_def:
                    default_params[param_name] = param_def[0]
                elif isinstance(param_def, tuple) and len(param_def) == 2:
                    default_params[param_name] = (param_def[0] + param_def[1]) / 2
            
            component.set_parameters(default_params)
        
        def get_parameter_history() -> List[Dict[str, Any]]:
            """Get parameter change history"""
            return component._optimization_metadata['parameter_history'].copy()
        
        def update_best_parameters(params: Dict[str, Any], score: float) -> None:
            """Update best parameters if score is better"""
            metadata = component._optimization_metadata
            
            if metadata['best_score'] is None or score > metadata['best_score']:
                metadata['best_parameters'] = params.copy()
                metadata['best_score'] = score
                metadata['last_optimization'] = datetime.now()
        
        def get_optimization_stats() -> Dict[str, Any]:
            """Get optimization statistics"""
            metadata = component._optimization_metadata
            return {
                'total_trials': len(metadata['parameter_history']),
                'best_score': metadata['best_score'],
                'best_parameters': metadata['best_parameters'],
                'last_optimization': metadata['last_optimization']
            }
        
        # Attach helper methods
        component.reset_to_defaults = reset_to_defaults
        component.get_parameter_history = get_parameter_history
        component.update_best_parameters = update_best_parameters
        component.get_optimization_stats = get_optimization_stats
        
        # Add parameter constraint support
        if not hasattr(component, '_parameter_constraints'):
            component._parameter_constraints = []
        
        def add_parameter_constraint(constraint: Any) -> None:
            """Add a parameter constraint"""
            component._parameter_constraints.append(constraint)
        
        def apply_constraints(params: Dict[str, Any]) -> Dict[str, Any]:
            """Apply all constraints to parameters"""
            adjusted_params = params.copy()
            
            for constraint in component._parameter_constraints:
                if hasattr(constraint, 'validate_and_adjust'):
                    adjusted_params = constraint.validate_and_adjust(adjusted_params)
            
            return adjusted_params
        
        component.add_parameter_constraint = add_parameter_constraint
        component.apply_constraints = apply_constraints