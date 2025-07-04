"""
Improved FeatureSpec design to eliminate duplication for multi-output features.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class ImprovedFeatureSpec:
    """
    Enhanced FeatureSpec that supports requesting multiple outputs efficiently.
    
    Examples:
        # Single output (backward compatible)
        FeatureSpec('rsi', {'period': 14})
        
        # Multiple outputs - OLD WAY (duplicated)
        [
            FeatureSpec('bollinger_bands', {'period': 20, 'num_std': 2.0}, 'upper'),
            FeatureSpec('bollinger_bands', {'period': 20, 'num_std': 2.0}, 'middle'),
            FeatureSpec('bollinger_bands', {'period': 20, 'num_std': 2.0}, 'lower')
        ]
        
        # Multiple outputs - NEW WAY (no duplication!)
        FeatureSpec('bollinger_bands', {'period': 20, 'num_std': 2.0}, 
                   outputs=['upper', 'middle', 'lower'])
    """
    
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Union[str, List[str]]] = None
    
    def __post_init__(self):
        """Normalize outputs to always be a list internally."""
        if self.outputs is None:
            self.outputs = []
        elif isinstance(self.outputs, str):
            self.outputs = [self.outputs]
    
    def get_canonical_names(self) -> List[str]:
        """Get all canonical feature names this spec represents."""
        base_name = self._get_base_canonical_name()
        
        if not self.outputs:
            return [base_name]
        
        # Return one canonical name per output
        return [f"{base_name}_{output}" for output in self.outputs]
    
    def _get_base_canonical_name(self) -> str:
        """Get the base canonical name without output suffix."""
        if not self.params:
            return self.name
        
        # Sort parameters for consistent naming
        param_parts = []
        for key, value in sorted(self.params.items()):
            if isinstance(value, float):
                param_parts.append(str(value))
            else:
                param_parts.append(str(value))
        
        return f"{self.name}_{'_'.join(param_parts)}"


# Example: Improved strategy decorators
def improved_keltner_strategy():
    """Example of how strategies would use the improved FeatureSpec."""
    
    # OLD WAY - with duplication
    old_features = lambda params: [
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'upper'),
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'middle'),
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'lower')
    ]
    
    # NEW WAY - no duplication!
    new_features = lambda params: [
        ImprovedFeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, outputs=['upper', 'middle', 'lower'])
    ]
    
    # Even cleaner with a helper
    def keltner_features(params):
        return [
            ImprovedFeatureSpec(
                'keltner_channel',
                params={'period': params.get('period', 20),
                       'multiplier': params.get('multiplier', 2.0)},
                outputs=['upper', 'middle', 'lower']
            )
        ]
    
    return new_features


# Alternative approach: Feature builder pattern
class FeatureBuilder:
    """Fluent API for building feature specifications."""
    
    def __init__(self, name: str):
        self.name = name
        self.params = {}
        self.outputs = []
    
    def with_params(self, **params) -> 'FeatureBuilder':
        """Add parameters."""
        self.params.update(params)
        return self
    
    def with_outputs(self, *outputs) -> 'FeatureBuilder':
        """Specify outputs to retrieve."""
        self.outputs.extend(outputs)
        return self
    
    def build(self) -> ImprovedFeatureSpec:
        """Build the feature spec."""
        return ImprovedFeatureSpec(
            name=self.name,
            params=self.params,
            outputs=self.outputs if self.outputs else None
        )


# Example usage with builder
def keltner_with_builder(params):
    """Even cleaner with builder pattern."""
    return [
        FeatureBuilder('keltner_channel')
        .with_params(
            period=params.get('period', 20),
            multiplier=params.get('multiplier', 2.0)
        )
        .with_outputs('upper', 'middle', 'lower')
        .build()
    ]


# Strategy decorator with improved feature handling
def strategy_v2(name: str, 
               features: callable,
               parameter_space: Dict[str, Dict[str, Any]],
               **metadata):
    """
    Improved strategy decorator that handles multi-output features elegantly.
    """
    def decorator(func):
        # Process feature discovery
        def enhanced_feature_discovery(params):
            feature_specs = features(params)
            
            # Expand multi-output specs into individual canonical names
            all_features = []
            for spec in feature_specs:
                if isinstance(spec, ImprovedFeatureSpec):
                    all_features.extend(spec.get_canonical_names())
                else:
                    # Backward compatibility
                    all_features.append(spec.canonical_name)
            
            return all_features
        
        # Attach metadata
        func._strategy_metadata = {
            'name': name,
            'feature_discovery': enhanced_feature_discovery,
            'parameter_space': parameter_space,
            **metadata
        }
        
        return func
    
    return decorator