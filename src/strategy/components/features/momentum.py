"""
Momentum-based feature calculations.

Indicators measuring rate of change and momentum.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ....core.components.discovery import feature
from .trend import ema_feature


@feature(
    name='macd',
    params=['fast', 'slow', 'signal'],
    min_history='slow + signal',
    dependencies=['ema']
)
def macd_feature(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD feature components.
    
    Pure function returning all MACD components.
    
    Args:
        prices: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        
    Returns:
        Dict with 'macd', 'signal', and 'histogram' series
    """
    ema_fast = ema_feature(prices, fast)
    ema_slow = ema_feature(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema_feature(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


@feature(
    name='adx',
    params=['period'],
    min_history='period * 2',  # Need extra for smoothing
    input_type='ohlc'
)
def adx_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate Average Directional Index (ADX) feature.
    
    Pure function measuring trend strength.
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        period: Number of periods for ADX calculation
        
    Returns:
        Dict with 'adx', 'di_plus', and 'di_minus' series
    """
    # Calculate True Range and Directional Movement
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    dm_plus = ((high - prev_high) > (prev_low - low)) * np.maximum(high - prev_high, 0)
    dm_minus = ((prev_low - low) > (high - prev_high)) * np.maximum(prev_low - low, 0)
    
    # Smooth TR and DM
    tr_smooth = true_range.rolling(window=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # Calculate ADX
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return {
        "adx": adx,
        "di_plus": di_plus,
        "di_minus": di_minus
    }


@feature(
    name='momentum',
    params=['period'],
    min_history='period + 1'
)
def momentum_feature(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Price Momentum feature.
    
    Pure function measuring rate of change.
    
    Args:
        prices: Price series
        period: Number of periods for momentum calculation
        
    Returns:
        Series with momentum values (current price / price N periods ago)
    """
    momentum = prices / prices.shift(period)
    return momentum


@feature(
    name='vortex',
    params=['period'],
    min_history='period + 1',
    input_type='ohlc'
)
def vortex_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate Vortex Indicator feature.
    
    Pure function measuring vortex movement of prices.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods for vortex calculation
        
    Returns:
        Dict with 'vi_plus' and 'vi_minus' series
    """
    prev_close = close.shift(1)
    
    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Vortex Movement
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    
    # Sum over period
    tr_sum = true_range.rolling(window=period).sum()
    vm_plus_sum = vm_plus.rolling(window=period).sum()
    vm_minus_sum = vm_minus.rolling(window=period).sum()
    
    # Vortex Indicators
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    return {
        "vi_plus": vi_plus,
        "vi_minus": vi_minus
    }