"""
Market classification components.

This module provides classifiers and containers for adaptive trading
based on market conditions. Classification is a general concept that
includes regime detection, market state identification, and other
forms of market categorization.
"""

from .classifier import (
    TrendVolatilityClassifier,
    MultiIndicatorClassifier
)

from .classifier_container import (
    ClassifierContainer,
    AdaptiveWeightContainer
)

__all__ = [
    # Classifiers
    'TrendVolatilityClassifier',
    'MultiIndicatorClassifier',
    
    # Containers
    'ClassifierContainer',
    'AdaptiveWeightContainer'
]