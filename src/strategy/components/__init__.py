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

from .classifiers import (
    MarketRegime,
    TrendClassifier,
    VolatilityClassifier,
    CompositeClassifier,
    create_market_regime_classifier
)

from .signal_replay import (
    CapturedSignal,
    SignalCapture,
    SignalReplayer,
    WeightedSignalAggregator
)

from .indicator_inference import (
    infer_indicators_from_strategies,
    get_strategy_requirements,
    validate_strategy_configuration
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
    
    # Classifiers
    "MarketRegime",
    "TrendClassifier",
    "VolatilityClassifier",
    "CompositeClassifier",
    "create_market_regime_classifier",
    
    # Signal Replay
    "CapturedSignal",
    "SignalCapture",
    "SignalReplayer",
    "WeightedSignalAggregator",
    
    # Indicator Inference
    "infer_indicators_from_strategies",
    "get_strategy_requirements",
    "validate_strategy_configuration"
]