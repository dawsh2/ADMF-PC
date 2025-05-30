"""
Market classification components.

This module provides classifiers and containers for adaptive trading
based on market conditions. Classification is a general concept that
includes regime detection, market state identification, and other
forms of market categorization.
"""

from .classifier import (
    TrendVolatilityClassifier,
    MultiIndicatorClassifier,
    RegimeClassifier,
    RegimeState,
    RegimeContext
)

from .classifier_container import (
    ClassifierContainer,
    AdaptiveWeightContainer
)

from .hmm_classifier import HMMClassifier, HMMRegimeState, HMMParameters
from .pattern_classifier import PatternClassifier, PatternRegimeState, PatternParameters
from .enhanced_classifier_container import (
    EnhancedClassifierContainer,
    create_classifier_hierarchy
)

__all__ = [
    # Base classes
    'RegimeClassifier',
    'RegimeState',
    'RegimeContext',
    
    # Classifiers
    'TrendVolatilityClassifier',
    'MultiIndicatorClassifier',
    'HMMClassifier',
    'HMMRegimeState',
    'HMMParameters',
    'PatternClassifier',
    'PatternRegimeState',
    'PatternParameters',
    
    # Containers
    'ClassifierContainer',
    'AdaptiveWeightContainer',
    'EnhancedClassifierContainer',
    'create_classifier_hierarchy'
]