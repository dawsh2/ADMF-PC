"""
Ensemble Strategy Implementations.

These strategies combine multiple indicator strategies to create
more sophisticated trading logic with confirmation and filtering.
"""

from .trend_momentum_composite import (
    trend_momentum_composite,
    multi_indicator_voting
)
from .duckdb_ensemble import (
    duckdb_ensemble,
    create_custom_ensemble,
    CONSERVATIVE_ENSEMBLE,
    AGGRESSIVE_ENSEMBLE
)
from .two_layer_ensemble import (
    two_layer_ensemble,
    create_two_layer_config,
    CONSERVATIVE_TWO_LAYER,
    AGGRESSIVE_TWO_LAYER,
    BALANCED_TWO_LAYER
)
from .two_layer_ensemble_debug import (
    two_layer_ensemble_debug,
    reset_debug_state
)
from .two_layer_ensemble_enhanced_debug import (
    two_layer_ensemble_enhanced_debug,
    reset_enhanced_debug_state
)
from .simple_ensemble import SimpleEnsemble

__all__ = [
    'trend_momentum_composite',
    'multi_indicator_voting',
    'duckdb_ensemble',
    'create_custom_ensemble',
    'CONSERVATIVE_ENSEMBLE',
    'AGGRESSIVE_ENSEMBLE',
    'two_layer_ensemble',
    'create_two_layer_config',
    'CONSERVATIVE_TWO_LAYER',
    'AGGRESSIVE_TWO_LAYER',
    'BALANCED_TWO_LAYER',
    'two_layer_ensemble_debug',
    'reset_debug_state',
    'two_layer_ensemble_enhanced_debug',
    'reset_enhanced_debug_state',
    'SimpleEnsemble'
]