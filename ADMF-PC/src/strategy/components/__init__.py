"""
Strategy components for ADMF-PC.

These are building blocks for strategies - indicators, classifiers,
rules, and signal processing components. All implement protocols
without inheritance.
"""

from .indicators import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    RSI,
    ATR,
    create_sma,
    create_ema,
    create_rsi,
    create_atr
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


__all__ = [
    # Indicators
    "SimpleMovingAverage",
    "ExponentialMovingAverage", 
    "RSI",
    "ATR",
    "create_sma",
    "create_ema",
    "create_rsi",
    "create_atr",
    
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
    "WeightedSignalAggregator"
]