"""
Volume-based feature calculations.

Indicators analyzing volume and price-volume relationships.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ....core.components.discovery import feature


@feature(
    name='volume',
    params=['period'],
    min_history='period',
    input_type='volume'
)
def volume_feature(volume: pd.Series, close: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate volume-based features.
    
    Pure function for volume analysis.
    
    Args:
        volume: Volume series
        close: Close price series
        period: Period for calculations
        
    Returns:
        Dict with volume-based features
    """
    # Volume moving average
    volume_ma = volume.rolling(window=period).mean()
    
    # Volume ratio (current vs average)
    volume_ratio = volume / volume_ma
    
    # On Balance Volume
    price_change = close.diff()
    obv_direction = np.where(price_change > 0, volume, 
                   np.where(price_change < 0, -volume, 0))
    obv = pd.Series(obv_direction, index=volume.index).cumsum()
    
    # Volume Price Trend
    vpt = ((close.diff() / close.shift(1)) * volume).cumsum()
    
    return {
        "volume_ma": volume_ma,
        "volume_ratio": volume_ratio,
        "obv": obv,
        "vpt": vpt
    }


@feature(
    name='volume_sma',
    params=['period'],
    min_history='period',
    input_type='volume'
)
def volume_sma_feature(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Volume Simple Moving Average feature.
    
    Pure function for volume smoothing.
    
    Args:
        volume: Volume series
        period: Number of periods for average
        
    Returns:
        Series with volume SMA values
    """
    return volume.rolling(window=period, min_periods=period).mean()


@feature(
    name='volume_ratio',
    params=['period'],
    min_history='period',
    input_type='volume',
    dependencies=['volume_sma']
)
def volume_ratio_feature(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Volume Ratio feature.
    
    Pure function measuring current volume vs average volume.
    
    Args:
        volume: Volume series
        period: Number of periods for average calculation
        
    Returns:
        Series with volume ratio values (>1 = above average)
    """
    volume_avg = volume_sma_feature(volume, period)
    volume_ratio = volume / volume_avg
    return volume_ratio