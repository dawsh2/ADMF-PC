"""
Trend-based feature calculations.

Moving averages and trend-following indicators.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
from typing import Dict, Any
from ....core.components.discovery import feature


@feature(
    name='sma',
    params=['period'],
    min_history='period'
)
def sma_feature(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average feature.
    
    Pure function - no state, just computation.
    
    Args:
        prices: Price series (typically close prices)
        period: Number of periods for average
        
    Returns:
        Series with SMA values
    """
    return prices.rolling(window=period, min_periods=period).mean()


@feature(
    name='ema',
    params=['period', 'smoothing'],
    min_history='period'
)
def ema_feature(prices: pd.Series, period: int, smoothing: float = 2.0) -> pd.Series:
    """
    Calculate Exponential Moving Average feature.
    
    Pure function using pandas built-in EMA calculation.
    
    Args:
        prices: Price series
        period: Number of periods for average
        smoothing: Smoothing factor (typically 2)
        
    Returns:
        Series with EMA values
    """
    alpha = smoothing / (period + 1)
    return prices.ewm(alpha=alpha, adjust=False).mean()


@feature(
    name='dema',
    params=['period'],
    min_history='period * 2',
    dependencies=['ema']
)
def dema_feature(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Double Exponential Moving Average feature.
    
    DEMA = 2 * EMA - EMA(EMA) - reduces lag compared to simple EMA.
    
    Args:
        prices: Price series
        period: Number of periods for DEMA calculation
        
    Returns:
        Series with DEMA values
    """
    ema1 = ema_feature(prices, period)
    ema2 = ema_feature(ema1, period)
    dema = 2 * ema1 - ema2
    return dema


@feature(
    name='tema',
    params=['period'],
    min_history='period * 3',
    dependencies=['ema']
)
def tema_feature(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Triple Exponential Moving Average feature.
    
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)) - further reduces lag.
    
    Args:
        prices: Price series
        period: Number of periods for TEMA calculation
        
    Returns:
        Series with TEMA values
    """
    ema1 = ema_feature(prices, period)
    ema2 = ema_feature(ema1, period)
    ema3 = ema_feature(ema2, period)
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema