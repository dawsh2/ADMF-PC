"""General helper utilities."""

import pandas as pd
import numpy as np
from typing import Union, Optional


def format_large_number(num: float) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Parameters
    ----------
    num : float
        Number to format
        
    Returns
    -------
    str
        Formatted string
        
    Examples
    --------
    >>> format_large_number(1234567)
    '1.2M'
    >>> format_large_number(1234)
    '1.2K'
    """
    if pd.isna(num):
        return 'N/A'
    
    abs_num = abs(num)
    sign = '-' if num < 0 else ''
    
    if abs_num >= 1e9:
        return f"{sign}{abs_num/1e9:.1f}B"
    elif abs_num >= 1e6:
        return f"{sign}{abs_num/1e6:.1f}M"
    elif abs_num >= 1e3:
        return f"{sign}{abs_num/1e3:.1f}K"
    else:
        return f"{sign}{abs_num:.0f}"


def calculate_returns(
    prices: pd.Series,
    method: str = 'simple',
    log_returns: bool = False
) -> pd.Series:
    """
    Calculate returns from price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    method : str
        'simple' for pct_change or 'log' for log returns
    log_returns : bool
        Deprecated, use method='log' instead
        
    Returns
    -------
    pd.Series
        Returns series
    """
    if log_returns or method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def resample_ohlc(
    df: pd.DataFrame,
    target_freq: str,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Resample OHLC data to a different frequency.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp index and OHLC columns
    target_freq : str
        Target frequency (e.g., '5min', '1H', '1D')
    price_col : str
        Column to use for single price resampling
        
    Returns
    -------
    pd.DataFrame
        Resampled OHLC data
    """
    # Define aggregation rules for OHLC
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Filter to only columns that exist
    rules = {col: rule for col, rule in agg_rules.items() if col in df.columns}
    
    # If no OHLC columns, just resample the price column
    if not rules and price_col in df.columns:
        rules = {price_col: 'last'}
    
    return df.resample(target_freq).agg(rules)


def align_signals_with_prices(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    signal_col: str = 'signal',
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Align signals with price data using merge_asof.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with signals and timestamp column
    prices_df : pd.DataFrame
        DataFrame with prices and timestamp index
    signal_col : str
        Name of signal column
    price_col : str
        Name of price column
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with aligned signals and prices
    """
    # Ensure we have timestamp columns
    if 'timestamp' not in signals_df.columns:
        raise ValueError("signals_df must have 'timestamp' column")
    
    # Sort by timestamp
    signals_sorted = signals_df.sort_values('timestamp')
    
    # Reset index if prices_df has timestamp as index
    if prices_df.index.name == 'timestamp':
        prices_reset = prices_df.reset_index()
    else:
        prices_reset = prices_df.copy()
    
    prices_sorted = prices_reset.sort_values('timestamp')
    
    # Merge asof to align signals with prices
    merged = pd.merge_asof(
        signals_sorted,
        prices_sorted[['timestamp', price_col]],
        on='timestamp',
        direction='backward'
    )
    
    return merged


def filter_business_hours(
    df: pd.DataFrame,
    start_time: str = '09:30',
    end_time: str = '16:00',
    timezone: str = 'America/New_York'
) -> pd.DataFrame:
    """
    Filter data to only include business hours.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp index or column
    start_time : str
        Start time in HH:MM format
    end_time : str
        End time in HH:MM format
    timezone : str
        Timezone for business hours
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    # Get timestamp series
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    elif 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("DataFrame must have timestamp index or column")
    
    # Convert to timezone if needed
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(timezone)
    else:
        timestamps = timestamps.tz_convert(timezone)
    
    # Filter by time
    time_mask = (timestamps.time >= pd.to_datetime(start_time).time()) & \
                (timestamps.time <= pd.to_datetime(end_time).time())
    
    return df[time_mask]


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Returns series
    window : int
        Rolling window size
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling metrics
    """
    if min_periods is None:
        min_periods = window // 2
    
    metrics = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    metrics['rolling_return'] = returns.rolling(window, min_periods=min_periods).sum()
    
    # Rolling volatility
    metrics['rolling_volatility'] = returns.rolling(window, min_periods=min_periods).std()
    
    # Rolling Sharpe (assuming 252 trading days)
    metrics['rolling_sharpe'] = (
        metrics['rolling_return'] / metrics['rolling_volatility'] * np.sqrt(252)
    )
    
    # Rolling max drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.rolling(window, min_periods=min_periods).max()
    metrics['rolling_max_dd'] = (cum_returns - rolling_max) / rolling_max
    
    return metrics