"""
Volatility-based feature calculations.

Indicators measuring price volatility and channel breakouts.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ....core.components.discovery import feature
from .trend import sma_feature, ema_feature


@feature(
    name='atr',
    params=['period'],
    min_history='period + 1',  # Need one extra for shift
    input_type='ohlc'  # Indicates it needs OHLC data, not just close
)
def atr_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range feature.
    
    Pure function measuring volatility using true range.
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        period: Number of periods for ATR calculation
        
    Returns:
        Series with ATR values
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


@feature(
    name='bollinger_bands',
    params=['period', 'std_dev'],
    min_history='period',
    dependencies=['sma']
)
def bollinger_bands_feature(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands feature components.
    
    Pure function returning all Bollinger Band components.
    
    Args:
        prices: Price series
        period: Period for moving average and standard deviation
        std_dev: Number of standard deviations for bands
        
    Returns:
        Dict with 'middle', 'upper', and 'lower' band series
    """
    middle = sma_feature(prices, period)
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return {
        "middle": middle,
        "upper": upper,
        "lower": lower
    }


@feature(
    name='keltner_channel',
    params=['period', 'multiplier'],
    min_history='period + 1',
    input_type='ohlc',
    dependencies=['ema', 'atr']
)
def keltner_channel_feature(high: pd.Series, low: pd.Series, close: pd.Series, 
                           period: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Keltner Channel feature components.
    
    Pure function for volatility-based channel.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for EMA and ATR calculation
        multiplier: Multiplier for ATR in channel calculation
        
    Returns:
        Dict with 'middle', 'upper', and 'lower' channel series
    """
    middle = ema_feature(close, period)
    atr = atr_feature(high, low, close, period)
    
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    
    return {
        "middle": middle,
        "upper": upper,
        "lower": lower
    }


@feature(
    name='donchian_channel',
    params=['period'],
    min_history='period',
    input_type='ohlc'
)
def donchian_channel_feature(high: pd.Series, low: pd.Series, close: pd.Series, 
                            period: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate Donchian Channel feature components.
    
    Pure function for highest/lowest channel breakout system.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series (used for middle calculation)
        period: Period for highest/lowest calculation
        
    Returns:
        Dict with 'upper', 'lower', and 'middle' channel series
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return {
        "upper": upper,
        "lower": lower,
        "middle": middle
    }


@feature(
    name='volatility',
    params=['period'],
    min_history='period + 1'
)
def volatility_feature(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Price Volatility feature.
    
    Pure function measuring price volatility using standard deviation of returns.
    
    Args:
        prices: Price series
        period: Number of periods for volatility calculation
        
    Returns:
        Series with volatility values (annualized if desired)
    """
    returns = prices.pct_change()
    volatility = returns.rolling(window=period).std()
    # Optionally annualize: volatility * np.sqrt(252) for daily data
    return volatility