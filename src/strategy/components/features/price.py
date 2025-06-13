"""
Basic price feature extractions.

Simple features that extract price components from OHLCV data.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
from typing import Union, Dict, Any
from ....core.components.discovery import feature


@feature(
    name='high',
    params=['period'],
    min_history='period',
    description='Rolling high price'
)
def high_feature(data: Union[pd.DataFrame, pd.Series], period: int = 20) -> pd.Series:
    """
    Calculate rolling high price.
    
    Args:
        data: Price data (DataFrame with 'high' column or Series)
        period: Lookback period for high
        
    Returns:
        Series with rolling high values
    """
    if isinstance(data, pd.DataFrame):
        if 'high' not in data.columns:
            # If no high column, use close
            return data['close'].rolling(window=period).max()
        return data['high'].rolling(window=period).max()
    else:
        # If Series, assume it's price data
        return data.rolling(window=period).max()


@feature(
    name='low', 
    params=['period'],
    min_history='period',
    description='Rolling low price'
)
def low_feature(data: Union[pd.DataFrame, pd.Series], period: int = 20) -> pd.Series:
    """
    Calculate rolling low price.
    
    Args:
        data: Price data (DataFrame with 'low' column or Series)
        period: Lookback period for low
        
    Returns:
        Series with rolling low values
    """
    if isinstance(data, pd.DataFrame):
        if 'low' not in data.columns:
            # If no low column, use close
            return data['close'].rolling(window=period).min()
        return data['low'].rolling(window=period).min()
    else:
        # If Series, assume it's price data
        return data.rolling(window=period).min()


@feature(
    name='atr_sma',
    params=['atr_period', 'sma_period'],
    min_history='atr_period + sma_period',
    dependencies=['atr', 'sma'],
    description='SMA of ATR values'
)
def atr_sma_feature(data: pd.DataFrame, atr_period: int = 14, sma_period: int = 20) -> pd.Series:
    """
    Calculate SMA of ATR values.
    
    This is a composite feature that smooths ATR values.
    
    Args:
        data: OHLC DataFrame
        atr_period: Period for ATR calculation
        sma_period: Period for SMA of ATR
        
    Returns:
        Series with smoothed ATR values
    """
    # Import here to avoid circular imports
    from .volatility import atr_feature
    from .trend import sma_feature
    
    # Calculate ATR first (needs OHLC data)
    if isinstance(data, pd.DataFrame) and all(col in data.columns for col in ['high', 'low', 'close']):
        atr_values = atr_feature(data['high'], data['low'], data['close'], period=atr_period)
    else:
        # If not proper OHLC data, return a dummy series
        return pd.Series(index=data.index if hasattr(data, 'index') else range(len(data)), dtype=float)
    
    # Then apply SMA
    return sma_feature(atr_values, period=sma_period)


@feature(
    name='volatility_sma',
    params=['vol_period', 'sma_period'],
    min_history='vol_period + sma_period',
    dependencies=['volatility', 'sma'],
    description='SMA of volatility values'
)
def volatility_sma_feature(data: pd.DataFrame, vol_period: int = 20, sma_period: int = 20) -> pd.Series:
    """
    Calculate SMA of volatility values.
    
    This is a composite feature that smooths volatility values.
    
    Args:
        data: Price DataFrame
        vol_period: Period for volatility calculation
        sma_period: Period for SMA of volatility
        
    Returns:
        Series with smoothed volatility values
    """
    # Import here to avoid circular imports
    from .volatility import volatility_feature
    from .trend import sma_feature
    
    # Calculate volatility first
    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
        vol_values = volatility_feature(data['close'], period=vol_period)
    elif isinstance(data, pd.Series):
        vol_values = volatility_feature(data, period=vol_period)
    else:
        # If not proper data, return a dummy series
        return pd.Series(index=data.index if hasattr(data, 'index') else range(len(data)), dtype=float)
    
    # Then apply SMA
    return sma_feature(vol_values, period=sma_period)


# Export all features
__all__ = [
    'high_feature',
    'low_feature', 
    'atr_sma_feature',
    'volatility_sma_feature'
]