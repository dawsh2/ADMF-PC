"""
Market classification components for Step 3.

This module provides classifiers and containers for regime detection
and adaptive trading based on market conditions.
"""

# Step 3 Components
from .regime_types import (
    MarketRegime,
    RegimeChangeEvent,
    RegimeState,
    ClassificationFeatures,
    ClassifierConfig
)

from .classifier import (
    ClassifierProtocol,
    BaseClassifier,
    DummyClassifier
)

from .pattern_classifier import PatternClassifier

from .classifier_container import (
    ClassifierContainer,
    create_test_classifier_container,
    create_conservative_classifier_container
)

__all__ = [
    # Types and Events
    'MarketRegime',
    'RegimeChangeEvent', 
    'RegimeState',
    'ClassificationFeatures',
    'ClassifierConfig',
    
    # Base Classes
    'ClassifierProtocol',
    'BaseClassifier',
    'DummyClassifier',
    
    # Implementations
    'PatternClassifier',
    
    # Containers
    'ClassifierContainer',
    'create_test_classifier_container',
    'create_conservative_classifier_container'
]