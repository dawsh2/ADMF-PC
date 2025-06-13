"""
Oscillator-based feature calculations.

Bounded indicators that oscillate between fixed ranges.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
from typing import Dict, Any
from ....core.components.discovery import feature


@feature(
    name='rsi',
    params=['period'],
    min_history='period + 1'  # Need one extra for diff
)
def rsi_feature(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index feature.
    
    Pure function - RSI calculation using pandas operations.
    
    Args:
        prices: Price series
        period: Number of periods for RSI calculation (default 14)
        
    Returns:
        Series with RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@feature(
    name='stochastic',
    params=['k_period', 'd_period'],
    min_history='k_period + d_period',
    input_type='ohlc'
)
def stochastic_feature(high: pd.Series, low: pd.Series, close: pd.Series, 
                      k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator feature.
    
    Pure function for momentum indicator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: Period for %K calculation
        d_period: Period for %D smoothing
        
    Returns:
        Dict with 'k' and 'd' series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        "k": k_percent,
        "d": d_percent
    }


@feature(
    name='williams_r',
    params=['period'],
    min_history='period',
    input_type='ohlc'
)
def williams_r_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R feature.
    
    Pure function for momentum oscillator (inverse of Stochastic %K).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods for calculation
        
    Returns:
        Series with Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r


@feature(
    name='cci',
    params=['period'],
    min_history='period',
    input_type='ohlc'
)
def cci_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index feature.
    
    Pure function for momentum oscillator measuring deviation from statistical mean.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods for calculation
        
    Returns:
        Series with CCI values (typically -100 to +100, but unbounded)
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: (x - x.mean()).abs().mean(), raw=False
    )
    
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci