"""Trend regime analysis."""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


def add_trend_regime(
    df: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    price_col: str = 'close',
    method: str = 'sma_cross',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Add trend regime classification to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    fast_period : int
        Fast moving average period
    slow_period : int
        Slow moving average period
    price_col : str
        Column to use for trend calculation
    method : str
        Method for trend detection ('sma_cross', 'ema_cross', 'linear')
    inplace : bool
        Whether to modify df in place
        
    Returns
    -------
    pd.DataFrame
        DataFrame with trend regime column
    """
    if not inplace:
        df = df.copy()
    
    if method in ['sma_cross', 'ema_cross']:
        # Calculate moving averages
        if method == 'sma_cross':
            df[f'ma_{fast_period}'] = df[price_col].rolling(fast_period).mean()
            df[f'ma_{slow_period}'] = df[price_col].rolling(slow_period).mean()
        else:  # ema_cross
            df[f'ma_{fast_period}'] = df[price_col].ewm(span=fast_period).mean()
            df[f'ma_{slow_period}'] = df[price_col].ewm(span=slow_period).mean()
        
        # Determine trend regime
        df['trend_regime'] = np.where(
            df[f'ma_{fast_period}'] > df[f'ma_{slow_period}'],
            'uptrend',
            'downtrend'
        )
        
    elif method == 'linear':
        # Use linear regression slope
        def calculate_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            return slope
        
        df['trend_slope'] = df[price_col].rolling(slow_period).apply(calculate_slope)
        df['trend_regime'] = np.where(df['trend_slope'] > 0, 'uptrend', 'downtrend')
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


def calculate_trend_strength(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 20
) -> pd.Series:
    """
    Calculate trend strength using ADX-like measure.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    price_col : str
        Price column to use
    window : int
        Window for calculation
        
    Returns
    -------
    pd.Series
        Trend strength values (0-100)
    """
    # Calculate directional movement
    df = df.copy()
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = -df['low'].diff()
    
    df['dm_plus'] = np.where(
        (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
        df['high_diff'],
        0
    )
    
    df['dm_minus'] = np.where(
        (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
        df['low_diff'],
        0
    )
    
    # Calculate ATR
    atr = calculate_atr(df, window)
    
    # Calculate directional indicators
    di_plus = 100 * df['dm_plus'].rolling(window).mean() / atr
    di_minus = 100 * df['dm_minus'].rolling(window).mean() / atr
    
    # Calculate ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window).mean()
    
    return adx


def identify_trend_changes(
    df: pd.DataFrame,
    trend_col: str = 'trend_regime',
    min_duration: int = 5
) -> pd.DataFrame:
    """
    Identify trend change points.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trend regime
    trend_col : str
        Name of trend column
    min_duration : int
        Minimum periods for trend to be valid
        
    Returns
    -------
    pd.DataFrame
        DataFrame with trend change information
    """
    df = df.copy()
    
    # Identify changes
    df['trend_change'] = df[trend_col] != df[trend_col].shift(1)
    
    # Assign trend IDs
    df['trend_id'] = df['trend_change'].cumsum()
    
    # Calculate trend statistics
    trend_changes = []
    
    for trend_id in df['trend_id'].unique():
        trend_data = df[df['trend_id'] == trend_id]
        
        if len(trend_data) >= min_duration:
            stats = {
                'trend_id': trend_id,
                'trend_type': trend_data[trend_col].iloc[0],
                'start_idx': trend_data.index[0],
                'end_idx': trend_data.index[-1],
                'duration': len(trend_data),
                'start_price': trend_data['close'].iloc[0] if 'close' in df else np.nan,
                'end_price': trend_data['close'].iloc[-1] if 'close' in df else np.nan,
                'return': (trend_data['close'].iloc[-1] / trend_data['close'].iloc[0] - 1) if 'close' in df else np.nan
            }
            
            trend_changes.append(stats)
    
    return pd.DataFrame(trend_changes)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Helper function to calculate ATR."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()