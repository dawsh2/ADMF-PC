"""
Optimization capability for strategy components.

This capability adds optimization support to any component through
composition, without requiring inheritance.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

from ...core.components.protocols import Capability
from ..protocols import Optimizable


logger = logging.getLogger(__name__)


class OptimizationCapability(Capability):
    """
    Adds optimization support to any component.
    
    This capability provides:
    - Parameter space management
    - Parameter validation and application
    - Optimization history tracking
    - Constraint handling
    """
    
    def get_name(self) -> str:
        return "optimization"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply optimization capability to component."""
        
        # Check if component already implements Optimizable
        has_methods = isinstance(component, Optimizable)
        
        if not has_methods:
            # Add default implementations
            self._add_optimization_methods(component, spec)
        
        # Add optimization metadata
        component._optimization_metadata = {
            'optimizable': True,
            'last_optimization': None,
            'parameter_history': [],
            'best_parameters': None,
            'best_score': None,
            'constraints': [],
            'parameter_space': spec.get('parameter_space', {})
        }
        
        # Enhance existing methods with tracking
        self._enhance_with_tracking(component)
        
        # Add optimization helpers
        self._add_optimization_helpers(component)
        
        # Add constraints if specified
        constraints = spec.get('constraints', [])
        for constraint in constraints:
            component.add_constraint(constraint)
        
        # Initialize with default parameters if provided
        default_params = spec.get('default_params', {})
        if default_params:
            component.set_parameters(default_params)
        
        return component
    
    def _add_optimization_methods(self, component: Any, spec: Dict[str, Any]) -> None:
        """Add default optimization methods for components without them."""
        
        # Get parameter space from spec
        parameter_space = spec.get('parameter_space', {})
        
        def get_parameter_space() -> Dict[str, Any]:
            """Get parameter space for optimization."""
            return component._optimization_metadata['parameter_space']
        
        def set_parameters(params: Dict[str, Any]) -> None:
            """Apply parameters to component."""
            # Apply constraints
            adjusted_params = component.apply_constraints(params)
            
            # Set parameters as attributes
            for name, value in adjusted_params.items():
                if hasattr(component, name):
                    setattr(component, name, value)
                else:
                    # Store in internal dict if attribute doesn't exist
                    if not hasattr(component, '_parameters'):
                        component._parameters = {}
                    component._parameters[name] = value
            
            # If component has reinitialize method, call it
            if hasattr(component, 'reinitialize'):
                component.reinitialize()
        
        def get_parameters() -> Dict[str, Any]:
            """Get current parameter values."""
            params = {}
            param_space = component._optimization_metadata['parameter_space']
            
            for param_name in param_space.keys():
                if hasattr(component, param_name):
                    params[param_name] = getattr(component, param_name)
                elif hasattr(component, '_parameters') and param_name in component._parameters:
                    params[param_name] = component._parameters[param_name]
            
            return params
        
        def validate_parameters(params: Dict[str, Any]) -> Tuple[bool, str]:
            """Validate parameter values."""
            param_space = component._optimization_metadata['parameter_space']
            
            # Check required parameters
            for required_param in param_space.keys():
                if required_param not in params:
                    return False, f"Missing required parameter: {required_param}"
            
            # Validate parameter values
            for param_name, param_value in params.items():
                if param_name not in param_space:
                    continue  # Skip unknown parameters
                
                space_def = param_space[param_name]
                
                # Handle different space definitions
                if isinstance(space_def, list):
                    # Discrete values
                    if param_value not in space_def:
                        return False, f"Invalid value for {param_name}: {param_value} not in {space_def}"
                
                elif isinstance(space_def, tuple) and len(space_def) == 2:
                    # Range (min, max)
                    if not (space_def[0] <= param_value <= space_def[1]):
                        return False, f"{param_name} must be between {space_def[0]} and {space_def[1]}"
                
                elif isinstance(space_def, dict):
                    # Complex definition
                    if 'type' in space_def:
                        param_type = space_def['type']
                        if param_type == 'int' and not isinstance(param_value, int):
                            return False, f"{param_name} must be an integer"
                        elif param_type == 'float' and not isinstance(param_value, (int, float)):
                            return False, f"{param_name} must be a number"
                    
                    if 'min' in space_def and param_value < space_def['min']:
                        return False, f"{param_name} must be >= {space_def['min']}"
                    
                    if 'max' in space_def and param_value > space_def['max']:
                        return False, f"{param_name} must be <= {space_def['max']}"
            
            # Check constraints
            for constraint in component._optimization_metadata['constraints']:
                if hasattr(constraint, 'is_satisfied') and not constraint.is_satisfied(params):
                    desc = constraint.get_description() if hasattr(constraint, 'get_description') else str(constraint)
                    return False, f"Constraint not satisfied: {desc}"
            
            return True, ""
        
        # Attach methods
        component.get_parameter_space = get_parameter_space
        component.set_parameters = set_parameters
        component.get_parameters = get_parameters
        component.validate_parameters = validate_parameters
    
    def _enhance_with_tracking(self, component: Any) -> None:
        """Enhance parameter setting with history tracking."""
        
        original_set_params = component.set_parameters
        
        def tracked_set_parameters(params: Dict[str, Any]) -> None:
            """Set parameters with tracking."""
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
            
            # Limit history size
            if len(component._optimization_metadata['parameter_history']) > 1000:
                component._optimization_metadata['parameter_history'].pop(0)
        
        component.set_parameters = tracked_set_parameters
    
    def _add_optimization_helpers(self, component: Any) -> None:
        """Add helper methods for optimization."""
        
        def update_best_parameters(params: Dict[str, Any], score: float) -> None:
            """Update best parameters if score improved."""
            metadata = component._optimization_metadata
            
            if metadata['best_score'] is None or score > metadata['best_score']:
                metadata['best_parameters'] = params.copy()
                metadata['best_score'] = score
                metadata['last_optimization'] = datetime.now()
                logger.info(f"New best score: {score} with parameters: {params}")
        
        def get_best_parameters() -> Optional[Dict[str, Any]]:
            """Get best parameters found."""
            return component._optimization_metadata['best_parameters']
        
        def get_best_score() -> Optional[float]:
            """Get best score achieved."""
            return component._optimization_metadata['best_score']
        
        def get_parameter_history() -> List[Dict[str, Any]]:
            """Get parameter change history."""
            return component._optimization_metadata['parameter_history'].copy()
        
        def get_optimization_stats() -> Dict[str, Any]:
            """Get optimization statistics."""
            metadata = component._optimization_metadata
            return {
                'total_trials': len(metadata['parameter_history']),
                'best_score': metadata['best_score'],
                'best_parameters': metadata['best_parameters'],
                'last_optimization': metadata['last_optimization'],
                'parameter_space_size': self._calculate_space_size(metadata['parameter_space'])
            }
        
        def reset_to_defaults() -> None:
            """Reset parameters to defaults."""
            param_space = component._optimization_metadata['parameter_space']
            default_params = {}
            
            for param_name, space_def in param_space.items():
                if isinstance(space_def, list) and space_def:
                    # Use first value for discrete
                    default_params[param_name] = space_def[0]
                elif isinstance(space_def, tuple) and len(space_def) == 2:
                    # Use middle value for range
                    default_params[param_name] = (space_def[0] + space_def[1]) / 2
                elif isinstance(space_def, dict):
                    # Use default if specified
                    if 'default' in space_def:
                        default_params[param_name] = space_def['default']
                    elif 'min' in space_def and 'max' in space_def:
                        default_params[param_name] = (space_def['min'] + space_def['max']) / 2
            
            component.set_parameters(default_params)
        
        def add_constraint(constraint: Any) -> None:
            """Add parameter constraint."""
            component._optimization_metadata['constraints'].append(constraint)
        
        def apply_constraints(params: Dict[str, Any]) -> Dict[str, Any]:
            """Apply all constraints to parameters."""
            adjusted_params = params.copy()
            
            for constraint in component._optimization_metadata['constraints']:
                if hasattr(constraint, 'validate_and_adjust'):
                    adjusted_params = constraint.validate_and_adjust(adjusted_params)
            
            return adjusted_params
        
        def update_parameter_space(param_name: str, space_def: Any) -> None:
            """Update parameter space definition."""
            component._optimization_metadata['parameter_space'][param_name] = space_def
        
        # Attach helper methods
        component.update_best_parameters = update_best_parameters
        component.get_best_parameters = get_best_parameters
        component.get_best_score = get_best_score
        component.get_parameter_history = get_parameter_history
        component.get_optimization_stats = get_optimization_stats
        component.reset_to_defaults = reset_to_defaults
        component.add_constraint = add_constraint
        component.apply_constraints = apply_constraints
        component.update_parameter_space = update_parameter_space
    
    def _calculate_space_size(self, parameter_space: Dict[str, Any]) -> Optional[int]:
        """Calculate total size of parameter space."""
        if not parameter_space:
            return 0
        
        size = 1
        for space_def in parameter_space.values():
            if isinstance(space_def, list):
                size *= len(space_def)
            else:
                # Continuous or complex spaces are infinite
                return None
        
        return size