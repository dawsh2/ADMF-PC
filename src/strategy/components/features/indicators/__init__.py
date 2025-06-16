"""
Technical indicator features organized by category.

This subdirectory contains all technical analysis indicators used for feature computation.
All indicators implement the Feature protocol using composition patterns.

No inheritance - pure protocol + composition architecture.
"""

# Import all trend indicators
from .trend import (
    SMA, EMA, DEMA, TEMA, WMA, HMA, VWMA, IchimokuCloud, ParabolicSAR,
    TREND_FEATURES
)

# Import all oscillator indicators  
from .oscillators import (
    RSI, StochasticOscillator, WilliamsR, CCI, StochasticRSI, MFI, UltimateOscillator,
    OSCILLATOR_FEATURES
)

# Import all volatility indicators
from .volatility import (
    ATR, BollingerBands, KeltnerChannel, DonchianChannel, Volatility, SuperTrend, VWAP,
    VOLATILITY_FEATURES
)

# Import all volume indicators
from .volume import (
    Volume, VolumeSMA, VolumeRatio, OBV, VPT, ChaikinMoneyFlow, AccDistLine, VROC, VolumeMomentum,
    VOLUME_FEATURES
)

# Import all momentum indicators
from .momentum import (
    MACD, Momentum, ROC, ADX, Aroon, Vortex,
    MOMENTUM_FEATURES
)

# Import all structure indicators
from .structure import (
    PivotPoints, SupportResistance, SwingPoints, LinearRegression, FibonacciRetracement, TrendLines,
    STRUCTURE_FEATURES
)

# Consolidated indicator registry
ALL_INDICATOR_FEATURES = {
    **TREND_FEATURES,
    **OSCILLATOR_FEATURES, 
    **VOLATILITY_FEATURES,
    **VOLUME_FEATURES,
    **MOMENTUM_FEATURES,
    **STRUCTURE_FEATURES,
}

# Export everything needed
__all__ = [
    # Feature registries
    'TREND_FEATURES',
    'OSCILLATOR_FEATURES', 
    'VOLATILITY_FEATURES',
    'VOLUME_FEATURES',
    'MOMENTUM_FEATURES',
    'STRUCTURE_FEATURES',
    'ALL_INDICATOR_FEATURES',
    
    # Trend indicators
    'SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 'HMA', 'VWMA', 'IchimokuCloud', 'ParabolicSAR',
    
    # Oscillator indicators
    'RSI', 'StochasticOscillator', 'WilliamsR', 'CCI', 'StochasticRSI', 'MFI', 'UltimateOscillator',
    
    # Volatility indicators
    'ATR', 'BollingerBands', 'KeltnerChannel', 'DonchianChannel', 'Volatility', 'SuperTrend', 'VWAP',
    
    # Volume indicators
    'Volume', 'VolumeSMA', 'VolumeRatio', 'OBV', 'VPT', 'ChaikinMoneyFlow', 'AccDistLine', 'VROC', 'VolumeMomentum',
    
    # Momentum indicators
    'MACD', 'Momentum', 'ROC', 'ADX', 'Aroon', 'Vortex',
    
    # Structure indicators
    'PivotPoints', 'SupportResistance', 'SwingPoints', 'LinearRegression', 'FibonacciRetracement', 'TrendLines',
]