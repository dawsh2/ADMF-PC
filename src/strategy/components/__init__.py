"""
Strategy components for ADMF-PC.

These are building blocks for strategies - features, classifiers,
rules, and signal processing components. All implement protocols
without inheritance.
"""

# Features have been moved to strategy.features module
# Import from there if needed:
# from ..features import FeatureHub, Feature, FeatureState, etc.

# Classifiers moved to strategy.classifiers module
# from .classifiers import (
#     MarketRegime,
#     TrendClassifier,
#     VolatilityClassifier,
#     CompositeClassifier,
#     create_market_regime_classifier
# )

# Signal replay moved elsewhere
# from .signal_replay import (
#     CapturedSignal,
#     SignalCapture,
#     SignalReplayer,
#     WeightedSignalAggregator
# )

# Feature inference moved to core.coordinator.topology
# This is now a topology builder responsibility

from .signal_aggregation import (
    InternalSignal,
    SignalCombiner,
    SignalFilter
)

# Note: indicators.py merged into features.py - FeatureHub now in features.py


__all__ = [
    # Signal Aggregation
    "InternalSignal",
    "SignalCombiner", 
    "SignalFilter"
]