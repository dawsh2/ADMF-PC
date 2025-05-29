"""
Parameter constraint implementations.
"""

from typing import Dict, Any, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class RelationalConstraint:
    """Constraint based on relationship between parameters"""
    
    def __init__(self, param1: str, relation: str, param2: str, 
                 description: str = None):
        """
        Initialize relational constraint.
        
        Args:
            param1: First parameter name
            relation: Relationship ('<', '>', '<=', '>=', '==', '!=')
            param2: Second parameter name
            description: Optional description
        """
        self.param1 = param1
        self.relation = relation
        self.param2 = param2
        
        # Create description if not provided
        if description is None:
            description = f"{param1} {relation} {param2}"
        
        self.description = description
        
        # Map relations to functions
        self.relation_funcs = {
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b
        }
        
        if relation not in self.relation_funcs:
            raise ValueError(f"Invalid relation: {relation}")
    
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """Check if relational constraint is satisfied"""
        if self.param1 not in params or self.param2 not in params:
            return True  # Skip if parameters not present
        
        val1 = params[self.param1]
        val2 = params[self.param2]
        
        try:
            return self.relation_funcs[self.relation](val1, val2)
        except Exception as e:
            logger.error(f"Error checking constraint {self.description}: {e}")
            return False
    
    def validate_and_adjust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters to satisfy constraint if possible"""
        if self.is_satisfied(params):
            return params
        
        adjusted = params.copy()
        
        # Simple adjustment strategies
        if self.param1 in params and self.param2 in params:
            val1 = params[self.param1]
            val2 = params[self.param2]
            
            # For < and <=, ensure param1 is less than param2
            if self.relation in ['<', '<=']:
                if val1 >= val2:
                    # Make param1 slightly less than param2
                    if isinstance(val1, int) and isinstance(val2, int):
                        adjusted[self.param1] = val2 - 1
                    else:
                        adjusted[self.param1] = val2 * 0.9
            
            # For > and >=, ensure param1 is greater than param2
            elif self.relation in ['>', '>=']:
                if val1 <= val2:
                    # Make param1 slightly more than param2
                    if isinstance(val1, int) and isinstance(val2, int):
                        adjusted[self.param1] = val2 + 1
                    else:
                        adjusted[self.param1] = val2 * 1.1
        
        return adjusted
    
    def get_description(self) -> str:
        """Get human-readable description"""
        return self.description


class RangeConstraint:
    """Constraint that parameter must be within range"""
    
    def __init__(self, param_name: str, min_value: Optional[float] = None,
                 max_value: Optional[float] = None, description: str = None):
        """
        Initialize range constraint.
        
        Args:
            param_name: Parameter to constrain
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            description: Optional description
        """
        self.param_name = param_name
        self.min_value = min_value
        self.max_value = max_value
        
        # Create description if not provided
        if description is None:
            if min_value is not None and max_value is not None:
                description = f"{param_name} in [{min_value}, {max_value}]"
            elif min_value is not None:
                description = f"{param_name} >= {min_value}"
            elif max_value is not None:
                description = f"{param_name} <= {max_value}"
            else:
                description = f"{param_name} range constraint"
        
        self.description = description
    
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """Check if parameter is within range"""
        if self.param_name not in params:
            return True  # Skip if parameter not present
        
        value = params[self.param_name]
        
        if self.min_value is not None and value < self.min_value:
            return False
        
        if self.max_value is not None and value > self.max_value:
            return False
        
        return True
    
    def validate_and_adjust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clip parameter to valid range"""
        if self.is_satisfied(params):
            return params
        
        adjusted = params.copy()
        
        if self.param_name in params:
            value = params[self.param_name]
            
            # Clip to range
            if self.min_value is not None:
                value = max(value, self.min_value)
            
            if self.max_value is not None:
                value = min(value, self.max_value)
            
            adjusted[self.param_name] = value
        
        return adjusted
    
    def get_description(self) -> str:
        """Get human-readable description"""
        return self.description


class DiscreteConstraint:
    """Constraint that parameter must be one of allowed values"""
    
    def __init__(self, param_name: str, allowed_values: List[Any],
                 description: str = None):
        """
        Initialize discrete constraint.
        
        Args:
            param_name: Parameter to constrain
            allowed_values: List of allowed values
            description: Optional description
        """
        self.param_name = param_name
        self.allowed_values = allowed_values
        
        # Create description if not provided
        if description is None:
            description = f"{param_name} in {allowed_values}"
        
        self.description = description
    
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """Check if parameter has allowed value"""
        if self.param_name not in params:
            return True
        
        return params[self.param_name] in self.allowed_values
    
    def validate_and_adjust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust to nearest allowed value"""
        if self.is_satisfied(params):
            return params
        
        adjusted = params.copy()
        
        if self.param_name in params:
            current_value = params[self.param_name]
            
            # Find closest allowed value
            if isinstance(current_value, (int, float)):
                # For numeric values, find closest
                closest = min(self.allowed_values, 
                            key=lambda x: abs(x - current_value) if isinstance(x, (int, float)) else float('inf'))
            else:
                # For non-numeric, use first allowed value
                closest = self.allowed_values[0] if self.allowed_values else current_value
            
            adjusted[self.param_name] = closest
        
        return adjusted
    
    def get_description(self) -> str:
        """Get human-readable description"""
        return self.description


class FunctionalConstraint:
    """Constraint defined by a custom function"""
    
    def __init__(self, constraint_func: Callable[[Dict[str, Any]], bool],
                 description: str = "Custom constraint",
                 adjust_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None):
        """
        Initialize functional constraint.
        
        Args:
            constraint_func: Function that returns True if constraint satisfied
            description: Description of the constraint
            adjust_func: Optional function to adjust parameters
        """
        self.description = description
        self.constraint_func = constraint_func
        self.adjust_func = adjust_func
    
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """Check constraint using custom function"""
        try:
            return self.constraint_func(params)
        except Exception as e:
            logger.error(f"Error in functional constraint: {e}")
            return False
    
    def validate_and_adjust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust using custom function if provided"""
        if self.is_satisfied(params):
            return params
        
        if self.adjust_func:
            try:
                return self.adjust_func(params)
            except Exception as e:
                logger.error(f"Error in constraint adjustment: {e}")
        
        return params
    
    def get_description(self) -> str:
        """Get human-readable description"""
        return self.description


class CompositeConstraint:
    """Combine multiple constraints with AND/OR logic"""
    
    def __init__(self, constraints: List[Any], 
                 logic: str = 'AND', description: str = None):
        """
        Initialize composite constraint.
        
        Args:
            constraints: List of constraints to combine
            logic: 'AND' or 'OR'
            description: Optional description
        """
        self.constraints = constraints
        self.logic = logic.upper()
        
        if self.logic not in ['AND', 'OR']:
            raise ValueError("Logic must be 'AND' or 'OR'")
        
        # Create description if not provided
        if description is None:
            constraint_descs = [c.get_description() for c in constraints]
            description = f" {logic} ".join(constraint_descs)
        
        self.description = description
    
    def is_satisfied(self, params: Dict[str, Any]) -> bool:
        """Check if composite constraint is satisfied"""
        if self.logic == 'AND':
            return all(c.is_satisfied(params) for c in self.constraints)
        else:  # OR
            return any(c.is_satisfied(params) for c in self.constraints)
    
    def validate_and_adjust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all constraints in sequence"""
        adjusted = params.copy()
        
        for constraint in self.constraints:
            adjusted = constraint.validate_and_adjust(adjusted)
        
        return adjusted
    
    def get_description(self) -> str:
        """Get human-readable description"""
        return self.description