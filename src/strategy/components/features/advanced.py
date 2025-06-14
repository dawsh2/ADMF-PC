"""
Advanced technical indicator features.

Complex and specialized indicators for advanced trading strategies.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ....core.components.discovery import feature


@feature(
    name='ultimate_oscillator',
    params=['period1', 'period2', 'period3'],
    min_history='max(period1, period2, period3)',
    input_type='ohlc'
)
def ultimate_oscillator_feature(high: pd.Series, low: pd.Series, close: pd.Series,
                               period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """
    Calculate Ultimate Oscillator feature.
    
    Combines multiple timeframes to reduce false signals.
    
    Args:
        high: High price series
        low: Low price series 
        close: Close price series
        period1: Short period (default 7)
        period2: Medium period (default 14)
        period3: Long period (default 28)
        
    Returns:
        Series with Ultimate Oscillator values (0-100)
    """
    # Calculate True Low and Buying Pressure
    prev_close = close.shift(1)
    true_low = np.minimum(low, prev_close)
    buying_pressure = close - true_low
    true_range = np.maximum(high, prev_close) - true_low
    
    # Calculate averages for each period
    bp_sum1 = buying_pressure.rolling(period1).sum()
    tr_sum1 = true_range.rolling(period1).sum()
    avg1 = bp_sum1 / tr_sum1
    
    bp_sum2 = buying_pressure.rolling(period2).sum()
    tr_sum2 = true_range.rolling(period2).sum()
    avg2 = bp_sum2 / tr_sum2
    
    bp_sum3 = buying_pressure.rolling(period3).sum()
    tr_sum3 = true_range.rolling(period3).sum()
    avg3 = bp_sum3 / tr_sum3
    
    # Calculate Ultimate Oscillator
    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
    
    return uo


@feature(
    name='mfi',
    params=['period'],
    min_history='period + 1',
    input_type='ohlcv'
)
def mfi_feature(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index feature.
    
    Volume-weighted RSI that identifies overbought/oversold conditions.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: Period for calculation (default 14)
        
    Returns:
        Series with MFI values (0-100)
    """
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate money flow
    money_flow = typical_price * volume
    
    # Calculate positive and negative money flow
    price_diff = typical_price.diff()
    positive_flow = money_flow.where(price_diff > 0, 0)
    negative_flow = money_flow.where(price_diff < 0, 0)
    
    # Calculate money flow ratio
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    
    money_ratio = positive_mf / negative_mf
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi


@feature(
    name='obv',
    params=[],
    min_history='1',
    input_type='cv'
)
def obv_feature(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume feature.
    
    Momentum indicator using volume flow to predict price changes.
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        Series with OBV values
    """
    price_change = close.diff()
    
    # OBV calculation
    obv = volume.copy()
    obv.iloc[0] = volume.iloc[0]  # First value
    
    for i in range(1, len(close)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


@feature(
    name='roc',
    params=['period'],
    min_history='period + 1'
)
def roc_feature(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Rate of Change feature.
    
    Momentum oscillator measuring percentage change over period.
    
    Args:
        prices: Price series
        period: Period for calculation (default 10)
        
    Returns:
        Series with ROC values (percentage)
    """
    roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
    return roc


@feature(
    name='cmf',
    params=['period'],
    min_history='period',
    input_type='ohlcv'
)
def cmf_feature(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                period: int = 20) -> pd.Series:
    """
    Calculate Chaikin Money Flow feature.
    
    Volume-weighted indicator measuring buying/selling pressure.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: Period for calculation (default 20)
        
    Returns:
        Series with CMF values (-1 to +1)
    """
    # Calculate Money Flow Multiplier
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.fillna(0)  # Handle division by zero
    
    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * volume
    
    # Calculate CMF
    cmf = mf_volume.rolling(period).sum() / volume.rolling(period).sum()
    
    return cmf


@feature(
    name='ad',
    params=[],
    min_history='1',
    input_type='ohlcv'
)
def ad_feature(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line feature.
    
    Volume-based indicator showing money flow.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        
    Returns:
        Series with A/D Line values
    """
    # Calculate Money Flow Multiplier
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.fillna(0)  # Handle division by zero
    
    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * volume
    
    # Calculate cumulative A/D Line
    ad_line = mf_volume.cumsum()
    
    return ad_line


@feature(
    name='aroon',
    params=['period'],
    min_history='period',
    input_type='hl'
)
def aroon_feature(high: pd.Series, low: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate Aroon indicator feature.
    
    Trend-following indicator identifying trend changes.
    
    Args:
        high: High price series
        low: Low price series
        period: Period for calculation (default 14)
        
    Returns:
        Dict with 'up' and 'down' series
    """
    def aroon_calc(series, period):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(period, len(series)):
            window = series.iloc[i-period+1:i+1]
            days_since_extreme = period - 1 - window.idxmax() + window.index[0]
            result.iloc[i] = ((period - days_since_extreme) / period) * 100
        return result
    
    aroon_up = aroon_calc(high, period)
    aroon_down = aroon_calc(low, period)
    
    return {
        "up": aroon_up,
        "down": aroon_down
    }


@feature(
    name='vwap',
    params=[],
    min_history='1',
    input_type='hlcv'
)
def vwap_feature(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Weighted Average Price feature.
    
    Average price weighted by volume.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        
    Returns:
        Series with VWAP values
    """
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate cumulative totals
    cum_vol = volume.cumsum()
    cum_vol_price = (typical_price * volume).cumsum()
    
    # Calculate VWAP
    vwap = cum_vol_price / cum_vol
    
    return vwap