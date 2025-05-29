"""
Strategy module for ADMF-PC.

Provides a comprehensive framework for building trading strategies with:
- Protocol-based design for flexibility
- Built-in optimization support
- Regime-aware containers
- Feature extraction
- Trading rules
- Risk management
"""

# Core protocols
from .protocols import (
    Strategy,
    Indicator,
    FeatureExtractor,
    Rule,
    SignalGenerator,
    RiskManager,
    RegimeClassifier,
    Optimizable
)

# Base implementations
from .base import (
    StrategyBase,
    IndicatorBase,
    SignalGeneratorBase
)

# Indicators
from .indicators import (
    IndicatorHub,
    SimpleMovingAverage,
    ExponentialMovingAverage,
    RSI,
    MACD,
    ATR,
    BollingerBands
)

# Regime components
from .regime import (
    TrendVolatilityClassifier,
    MultiIndicatorClassifier,
    RegimeStrategyContainer,
    AdaptiveEnsembleContainer
)

# Features
from .features import (
    PriceFeatureExtractor,
    IndicatorFeatureExtractor,
    TechnicalFeatureExtractor,
    MarketMicrostructureExtractor
)

# Rules
from .rules import (
    ThresholdRule,
    CrossoverRule,
    StopLossRule,
    TakeProfitRule,
    PercentEquityRule,
    MaxPositionRule
)

# Concrete strategies
from .strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    MarketMakingStrategy,
    ArbitrageStrategy
)

__all__ = [
    # Protocols
    'Strategy',
    'Indicator',
    'FeatureExtractor',
    'Rule',
    'SignalGenerator',
    'RiskManager',
    'RegimeClassifier',
    'Optimizable',
    
    # Base classes
    'StrategyBase',
    'IndicatorBase',
    'SignalGeneratorBase',
    
    # Indicators
    'IndicatorHub',
    'SimpleMovingAverage',
    'ExponentialMovingAverage',
    'RSI',
    'MACD',
    'ATR',
    'BollingerBands',
    
    # Regime
    'TrendVolatilityClassifier',
    'MultiIndicatorClassifier',
    'RegimeStrategyContainer',
    'AdaptiveEnsembleContainer',
    
    # Features
    'PriceFeatureExtractor',
    'IndicatorFeatureExtractor',
    'TechnicalFeatureExtractor',
    'MarketMicrostructureExtractor',
    
    # Rules
    'ThresholdRule',
    'CrossoverRule',
    'StopLossRule',
    'TakeProfitRule',
    'PercentEquityRule',
    'MaxPositionRule',
    
    # Strategies
    'MomentumStrategy',
    'MeanReversionStrategy',
    'TrendFollowingStrategy',
    'MarketMakingStrategy',
    'ArbitrageStrategy'
]