"""
Feature calculation and management for ADMF-PC.

This module contains both:
1. Stateless feature calculation functions (Tier 1) - pure functions for parallelization
2. FeatureHub stateful component (Tier 2) - manages incremental feature computation

Following Protocol + Composition architecture:
- Pure functions for batch computation
- Stateful hub for streaming/incremental updates
- No inheritance, just composition
- Maximum parallelization potential for stateless functions

Now organized into logical groups for better maintainability.
"""

# Import all feature functions for discovery
from .trend import (
    sma_feature,
    ema_feature,
    dema_feature,
    tema_feature
)

from .oscillators import (
    rsi_feature,
    stochastic_feature,
    williams_r_feature,
    cci_feature
)

from .momentum import (
    macd_feature,
    adx_feature,
    momentum_feature,
    vortex_feature
)

from .volatility import (
    atr_feature,
    bollinger_bands_feature,
    keltner_channel_feature,
    donchian_channel_feature,
    volatility_feature
)

from .volume import (
    volume_feature,
    volume_sma_feature,
    volume_ratio_feature
)

from .complex import (
    ichimoku_feature
)

from .price import (
    high_feature,
    low_feature,
    atr_sma_feature,
    volatility_sma_feature
)

# Import stateful components and utilities
from .hub import (
    FeatureHub,
    create_feature_hub,
    compute_feature,
    compute_multiple_features,
    FEATURE_REGISTRY,
    DEFAULT_MOMENTUM_FEATURES,
    DEFAULT_MEAN_REVERSION_FEATURES,
    DEFAULT_VOLATILITY_FEATURES
)

# Export everything
__all__ = [
    # Trend features
    'sma_feature',
    'ema_feature', 
    'dema_feature',
    'tema_feature',
    
    # Oscillator features
    'rsi_feature',
    'stochastic_feature',
    'williams_r_feature',
    'cci_feature',
    
    # Momentum features
    'macd_feature',
    'adx_feature',
    'momentum_feature',
    'vortex_feature',
    
    # Volatility features
    'atr_feature',
    'bollinger_bands_feature',
    'keltner_channel_feature',
    'donchian_channel_feature',
    'volatility_feature',
    
    # Volume features
    'volume_feature',
    'volume_sma_feature',
    'volume_ratio_feature',
    
    # Complex features
    'ichimoku_feature',
    
    # Price features
    'high_feature',
    'low_feature',
    'atr_sma_feature',
    'volatility_sma_feature',
    
    # Stateful components
    'FeatureHub',
    'create_feature_hub',
    
    # Utilities
    'compute_feature',
    'compute_multiple_features',
    'FEATURE_REGISTRY',
    
    # Default configurations
    'DEFAULT_MOMENTUM_FEATURES',
    'DEFAULT_MEAN_REVERSION_FEATURES',
    'DEFAULT_VOLATILITY_FEATURES'
]