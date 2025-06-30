"""
Ensemble Strategy Function

Provides a functional interface to the ensemble strategy.
"""

from typing import Dict, Any, Optional
import pandas as pd

from ...types import Signal
from ....core.components.discovery import strategy
from ..ensemble.simple_ensemble import SimpleEnsemble

# Create a singleton instance for feature discovery
_ensemble_instance = None


@strategy('ensemble', 
    description="Combines multiple strategies with weighted voting",
    parameters={
        'strategies': 'List of strategy configurations',
        'combination_method': 'How to combine signals (weighted_vote, majority, unanimous)',
        'threshold': 'Minimum signal value or expression to generate trade'
    })
def ensemble(features: Dict[str, float], bar: pd.Series, params: Dict[str, Any]) -> Optional[Signal]:
    """
    Ensemble strategy that combines multiple strategies.
    
    Parameters:
        strategies: List of strategy configurations with types, params, and weights
        combination_method: How to combine signals ('weighted_vote', 'majority', 'unanimous')
        threshold: Minimum weighted signal value to generate a trade (can be expression)
    """
    global _ensemble_instance
    
    # Create or update ensemble instance
    if _ensemble_instance is None or _ensemble_instance.get_parameters() != params:
        _ensemble_instance = SimpleEnsemble(params)
    
    # Generate signal
    return _ensemble_instance.generate_signal(features, bar)