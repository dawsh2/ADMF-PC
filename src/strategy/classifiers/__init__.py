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

# Old class-based classifiers moved to tmp/deprecated_classifiers/
# Now using pure function classifiers with @classifier decorator

__all__ = [
    # Types and Events
    'MarketRegime',
    'RegimeChangeEvent', 
    'RegimeState',
    'ClassificationFeatures',
    'ClassifierConfig',
    
    # Pure function classifiers (decorated with @classifier)
    'trend_classifier',
    'volatility_classifier',
    'momentum_regime_classifier'
]