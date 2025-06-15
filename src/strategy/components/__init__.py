"""
Strategy components for ADMF-PC.

These are building blocks for strategies - features, classifiers,
rules, and signal processing components. All implement protocols
without inheritance.
"""

from .features import (
    # Stateful feature computation engine (canonical)
    FeatureHub,
    create_feature_hub,
    FEATURE_REGISTRY,
    DEFAULT_MOMENTUM_FEATURES,
    DEFAULT_MEAN_REVERSION_FEATURES,
    DEFAULT_VOLATILITY_FEATURES,
    DEFAULT_STRUCTURE_FEATURES,
    DEFAULT_VOLUME_FEATURES,
    # Protocol definitions
    Feature,
    FeatureState,
    # Feature registries
    TREND_FEATURES,
    OSCILLATOR_FEATURES,
    VOLATILITY_FEATURES,
    VOLUME_FEATURES,
    MOMENTUM_FEATURES,
    STRUCTURE_FEATURES
)

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
    # Stateful feature computation engine (canonical)
    "FeatureHub",
    "create_feature_hub",
    "FEATURE_REGISTRY",
    "Feature",
    "FeatureState",
    
    # Feature registries
    "TREND_FEATURES",
    "OSCILLATOR_FEATURES",
    "VOLATILITY_FEATURES", 
    "VOLUME_FEATURES",
    "MOMENTUM_FEATURES",
    "STRUCTURE_FEATURES",
    
    # Default configurations
    "DEFAULT_MOMENTUM_FEATURES",
    "DEFAULT_MEAN_REVERSION_FEATURES",
    "DEFAULT_VOLATILITY_FEATURES",
    "DEFAULT_STRUCTURE_FEATURES",
    "DEFAULT_VOLUME_FEATURES",
    
    # Signal Aggregation
    "InternalSignal",
    "SignalCombiner", 
    "SignalFilter"
]