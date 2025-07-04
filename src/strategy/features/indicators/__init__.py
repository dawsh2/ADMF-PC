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
from .volatility_percentile import VolatilityPercentile

# Add volatility percentile to the registry
VOLATILITY_FEATURES['volatility_percentile'] = VolatilityPercentile

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

# Import all divergence indicators
from .divergence import (
    BollingerRSIDivergence,
    DIVERGENCE_FEATURES
)
from .rsi_divergence import RSIDivergence

# Import BB RSI tracker
from .bb_rsi_tracker import BollingerRSITracker
from .bb_rsi_divergence_proper import BollingerRSIDivergenceProper
from .bb_rsi_dependent_debug import BollingerRSIDependentFeatureDebug as BollingerRSIDependentFeature
from .bb_rsi_divergence_exact import BollingerRSIDivergenceExact
from .bb_rsi_divergence_self_contained import BollingerRSIDivergenceSelfContained
from .bb_rsi_immediate_divergence import BollingerRSIImmediateDivergence

# Add BB RSI features to divergence features
DIVERGENCE_FEATURES['bb_rsi_tracker'] = BollingerRSITracker
DIVERGENCE_FEATURES['bb_rsi_divergence_proper'] = BollingerRSIDivergenceProper
DIVERGENCE_FEATURES['bb_rsi_dependent'] = BollingerRSIDependentFeature
DIVERGENCE_FEATURES['bb_rsi_divergence_exact'] = BollingerRSIDivergenceExact
DIVERGENCE_FEATURES['bb_rsi_divergence_self'] = BollingerRSIDivergenceSelfContained
DIVERGENCE_FEATURES['bb_rsi_immediate_divergence'] = BollingerRSIImmediateDivergence
DIVERGENCE_FEATURES['rsi_divergence'] = RSIDivergence

# Consolidated indicator registry
ALL_INDICATOR_FEATURES = {
    **TREND_FEATURES,
    **OSCILLATOR_FEATURES, 
    **VOLATILITY_FEATURES,
    **VOLUME_FEATURES,
    **MOMENTUM_FEATURES,
    **STRUCTURE_FEATURES,
    **DIVERGENCE_FEATURES,
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
    'DIVERGENCE_FEATURES',
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
    
    # Divergence indicators
    'BollingerRSIDivergence',
    'RSIDivergence',
]