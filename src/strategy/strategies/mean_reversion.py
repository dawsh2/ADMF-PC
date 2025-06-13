"""
Mean reversion trading strategy.
"""

# Re-export the simple version
from .mean_reversion_simple import MeanReversionStrategy, mean_reversion_strategy

__all__ = ['MeanReversionStrategy', 'mean_reversion_strategy']