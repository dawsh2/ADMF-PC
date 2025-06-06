"""
Market classification components for Step 3.

This module provides classifiers and containers for regime detection
and adaptive trading based on market conditions.

Now includes pure function classifiers decorated with @classifier
for automatic discovery.
"""

# Step 3 Components
from .regime_types import (
    RegimeChangeEvent,
    RegimeState,
    ClassificationFeatures,
    ClassifierConfig
)

# Import pure function classifiers for discovery
from .classifiers import (
    MarketRegime,
    trend_classifier,
    volatility_classifier,
    momentum_regime_classifier
)

# from .classifier import (
#     ClassifierProtocol,
#     BaseClassifier,
#     DummyClassifier
# )

# from .pattern_classifier import PatternClassifier

# from .classifier_container import (
#     ClassifierContainer,
#     create_test_classifier_container,
#     create_conservative_classifier_container
# )

__all__ = [
    # Types and Events
    'MarketRegime',
    'RegimeChangeEvent', 
    'RegimeState',
    'ClassificationFeatures',
    'ClassifierConfig',
    
    # Base Classes
    # 'ClassifierProtocol',
    # 'BaseClassifier',
    # 'DummyClassifier',
    
    # Implementations (removed - using pure functions now)
    # 'PatternClassifier',
    
    # Pure function classifiers (decorated)
    'trend_classifier',
    'volatility_classifier',
    'momentum_regime_classifier',
    
    # Containers
    # 'ClassifierContainer',
    # 'create_test_classifier_container',
    # 'create_conservative_classifier_container'
]