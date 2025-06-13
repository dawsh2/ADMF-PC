"""
Strategy components for ADMF-PC.

These are building blocks for strategies - features, classifiers,
rules, and signal processing components. All implement protocols
without inheritance.
"""

from .features import (
    # Stateless feature functions - now organized by category
    sma_feature,
    ema_feature,
    dema_feature,
    tema_feature,
    rsi_feature,
    stochastic_feature,
    williams_r_feature,
    cci_feature,
    macd_feature,
    adx_feature,
    momentum_feature,
    vortex_feature,
    atr_feature,
    bollinger_bands_feature,
    keltner_channel_feature,
    donchian_channel_feature,
    volatility_feature,
    volume_feature,
    volume_sma_feature,
    volume_ratio_feature,
    ichimoku_feature,
    # Utilities
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
    # Stateless feature functions - organized by category
    "sma_feature",
    "ema_feature",
    "dema_feature",
    "tema_feature",
    "rsi_feature",
    "stochastic_feature",
    "williams_r_feature",
    "cci_feature",
    "macd_feature",
    "adx_feature",
    "momentum_feature",
    "vortex_feature",
    "atr_feature",
    "bollinger_bands_feature",
    "keltner_channel_feature",
    "donchian_channel_feature",
    "volatility_feature",
    "volume_feature",
    "volume_sma_feature",
    "volume_ratio_feature",
    "ichimoku_feature",
    "compute_feature",
    "compute_multiple_features",
    "FEATURE_REGISTRY",
    
    # Stateful feature computation engine
    "FeatureHub",
    "create_feature_hub",
    "DEFAULT_MOMENTUM_FEATURES",
    "DEFAULT_MEAN_REVERSION_FEATURES",
    "DEFAULT_VOLATILITY_FEATURES",
    
    # Signal Aggregation
    "InternalSignal",
    "SignalCombiner", 
    "SignalFilter"
]