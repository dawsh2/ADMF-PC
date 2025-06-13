"""
Ensemble Strategy Implementations.

These strategies combine multiple indicator strategies to create
more sophisticated trading logic with confirmation and filtering.
"""

from .trend_momentum_composite import (
    trend_momentum_composite,
    multi_indicator_voting
)

__all__ = [
    'trend_momentum_composite',
    'multi_indicator_voting'
]