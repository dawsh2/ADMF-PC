"""
Feature calculation and management for ADMF-PC.

This module contains the organized incremental feature system:
- Protocol + Composition architecture for O(1) features
- FeatureHub as the canonical stateful component (Tier 2)
- All features organized by type for maintainability

No inheritance - pure protocol + composition architecture.
"""

# Import canonical FeatureHub and factory
from .hub import (
    FeatureHub,
    create_feature_hub,
    FEATURE_REGISTRY,
    DEFAULT_MOMENTUM_FEATURES,
    DEFAULT_MEAN_REVERSION_FEATURES,
    DEFAULT_VOLATILITY_FEATURES,
    DEFAULT_STRUCTURE_FEATURES,
    DEFAULT_VOLUME_FEATURES,
)

# Import protocols for type hints
from .protocols import Feature, FeatureState

# Import all indicators from organized subdirectory
from .indicators import (
    # Feature registries
    TREND_FEATURES,
    OSCILLATOR_FEATURES,
    VOLATILITY_FEATURES,
    VOLUME_FEATURES,
    MOMENTUM_FEATURES,
    STRUCTURE_FEATURES,
    ALL_INDICATOR_FEATURES,
    
    # Trend indicators
    SMA, EMA, DEMA, TEMA, WMA, HMA, VWMA,
    
    # Oscillator indicators
    RSI, StochasticOscillator, WilliamsR, CCI, StochasticRSI, MFI,
    
    # Volatility indicators
    ATR, BollingerBands, KeltnerChannel, DonchianChannel, Volatility, SuperTrend, VWAP,
    
    # Volume indicators
    VolumeSMA, VolumeRatio, OBV, VPT, ChaikinMoneyFlow, AccDistLine, VROC, VolumeMomentum,
    
    # Momentum indicators
    MACD, Momentum, ROC, ADX, Aroon, Vortex,
    
    # Structure indicators
    PivotPoints, SupportResistance, SwingPoints, LinearRegression, FibonacciRetracement, TrendLines,
)

# Export everything needed
__all__ = [
    # Core components
    'FeatureHub',
    'create_feature_hub',
    'Feature',
    'FeatureState',
    'FEATURE_REGISTRY',
    
    # Feature registries
    'TREND_FEATURES',
    'OSCILLATOR_FEATURES', 
    'VOLATILITY_FEATURES',
    'VOLUME_FEATURES',
    'MOMENTUM_FEATURES',
    'STRUCTURE_FEATURES',
    
    # Trend features
    'SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 'HMA', 'VWMA',
    
    # Oscillator features
    'RSI', 'StochasticOscillator', 'WilliamsR', 'CCI', 'StochasticRSI', 'MFI',
    
    # Volatility features
    'ATR', 'BollingerBands', 'KeltnerChannel', 'DonchianChannel', 'Volatility', 'SuperTrend', 'VWAP',
    
    # Volume features
    'VolumeSMA', 'VolumeRatio', 'OBV', 'VPT', 'ChaikinMoneyFlow', 'AccDistLine', 'VROC', 'VolumeMomentum',
    
    # Momentum features
    'MACD', 'Momentum', 'ROC', 'ADX', 'Aroon', 'Vortex',
    
    # Structure features
    'PivotPoints', 'SupportResistance', 'SwingPoints', 'LinearRegression', 'FibonacciRetracement', 'TrendLines',
    
    # Default configurations
    'DEFAULT_MOMENTUM_FEATURES',
    'DEFAULT_MEAN_REVERSION_FEATURES',
    'DEFAULT_VOLATILITY_FEATURES',
    'DEFAULT_STRUCTURE_FEATURES',
    'DEFAULT_VOLUME_FEATURES',
]