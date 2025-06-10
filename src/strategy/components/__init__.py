"""
Strategy components for ADMF-PC.

These are building blocks for strategies - features, classifiers,
rules, and signal processing components. All implement protocols
without inheritance.
"""

from .features import (
    # Stateless feature functions
    sma_feature,
    ema_feature,
    rsi_feature,
    macd_feature,
    bollinger_bands_feature,
    atr_feature,
    stochastic_feature,
    williams_r_feature,
    cci_feature,
    adx_feature,
    volume_features,
    momentum_features,
    price_action_features,
    compute_feature,
    compute_multiple_features,
    FEATURE_REGISTRY,
    # Stateful feature computation engine
    FeatureHub,
    create_feature_hub,
    DEFAULT_MOMENTUM_FEATURES,
    DEFAULT_MEAN_REVERSION_FEATURES,
    DEFAULT_VOLATILITY_FEATURES
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
    # Stateless feature functions
    "sma_feature",
    "ema_feature",
    "rsi_feature",
    "macd_feature",
    "bollinger_bands_feature",
    "atr_feature",
    "stochastic_feature",
    "williams_r_feature",
    "cci_feature",
    "adx_feature",
    "volume_features",
    "momentum_features",
    "price_action_features",
    "compute_feature",
    "compute_multiple_features",
    "FEATURE_REGISTRY",
    
    # Stateful feature computation engine
    "FeatureHub",
    "create_feature_hub",
    "DEFAULT_MOMENTUM_FEATURES",
    "DEFAULT_MEAN_REVERSION_FEATURES",
    "DEFAULT_VOLATILITY_FEATURES",
    
    # Classifiers - moved to strategy.classifiers
    # "MarketRegime",
    # "TrendClassifier", 
    # "VolatilityClassifier",
    # "CompositeClassifier",
    # "create_market_regime_classifier",
    
    # Signal Replay - moved elsewhere
    # "CapturedSignal",
    # "SignalCapture",
    # "SignalReplayer",
    # "WeightedSignalAggregator",
    
    # Feature Inference - moved to core.coordinator.feature_inference
    
    # Signal Aggregation
    "InternalSignal",
    "SignalCombiner", 
    "SignalFilter"
]