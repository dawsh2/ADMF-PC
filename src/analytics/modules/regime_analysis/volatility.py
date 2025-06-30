"""Volatility regime analysis."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List


def calculate_atr(
    df: pd.DataFrame,
    window: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data
    window : int
        ATR window period
    high_col : str
        Name of high column
    low_col : str
        Name of low column
    close_col : str
        Name of close column
        
    Returns
    -------
    pd.Series
        ATR values
    """
    # Calculate True Range
    high_low = df[high_col] - df[low_col]
    high_close = abs(df[high_col] - df[close_col].shift(1))
    low_close = abs(df[low_col] - df[close_col].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(window=window).mean()
    
    return atr


def add_volatility_regime(
    df: pd.DataFrame,
    window: int = 14,
    method: str = 'atr',
    n_regimes: int = 3,
    labels: Optional[List[str]] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Add volatility regime classification to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data
    window : int
        Window for volatility calculation
    method : str
        'atr' or 'std' for volatility measurement
    n_regimes : int
        Number of volatility regimes
    labels : list, optional
        Custom labels for regimes
    inplace : bool
        Whether to modify df in place
        
    Returns
    -------
    pd.DataFrame
        DataFrame with volatility regime column
    """
    if not inplace:
        df = df.copy()
    
    # Calculate volatility measure
    if method == 'atr':
        df['volatility'] = calculate_atr(df, window)
    elif method == 'std':
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)  # Annualized
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Define regime labels
    if labels is None:
        if n_regimes == 2:
            labels = ['low_vol', 'high_vol']
        elif n_regimes == 3:
            labels = ['low_vol', 'normal_vol', 'high_vol']
        else:
            labels = [f'vol_regime_{i}' for i in range(n_regimes)]
    
    # Calculate quantiles for regime boundaries
    quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
    boundaries = df['volatility'].quantile(quantiles).values
    
    # Create bins
    bins = [-np.inf] + list(boundaries) + [np.inf]
    
    # Classify into regimes
    df['volatility_regime'] = pd.cut(
        df['volatility'],
        bins=bins,
        labels=labels
    )
    
    return df


def analyze_volatility_clusters(
    df: pd.DataFrame,
    volatility_col: str = 'volatility',
    threshold_percentile: float = 75
) -> pd.DataFrame:
    """
    Identify volatility clusters (periods of sustained high volatility).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volatility data
    volatility_col : str
        Name of volatility column
    threshold_percentile : float
        Percentile threshold for high volatility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cluster information
    """
    df = df.copy()
    
    # Define high volatility threshold
    threshold = df[volatility_col].quantile(threshold_percentile / 100)
    
    # Identify high volatility periods
    df['high_vol'] = df[volatility_col] > threshold
    
    # Find clusters
    df['cluster_id'] = (df['high_vol'] != df['high_vol'].shift()).cumsum()
    
    # Only keep high volatility clusters
    df.loc[~df['high_vol'], 'cluster_id'] = np.nan
    
    # Calculate cluster statistics
    cluster_stats = []
    
    for cluster_id in df['cluster_id'].dropna().unique():
        cluster_data = df[df['cluster_id'] == cluster_id]
        
        stats = {
            'cluster_id': int(cluster_id),
            'start_time': cluster_data.index[0] if isinstance(df.index, pd.DatetimeIndex) else cluster_data.index[0],
            'end_time': cluster_data.index[-1] if isinstance(df.index, pd.DatetimeIndex) else cluster_data.index[-1],
            'duration_periods': len(cluster_data),
            'avg_volatility': cluster_data[volatility_col].mean(),
            'max_volatility': cluster_data[volatility_col].max(),
            'total_return': cluster_data['close'].iloc[-1] / cluster_data['close'].iloc[0] - 1 if 'close' in df else np.nan
        }
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)


def calculate_volatility_persistence(
    df: pd.DataFrame,
    volatility_col: str = 'volatility',
    lag_periods: List[int] = [1, 5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate volatility persistence (autocorrelation).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volatility data
    volatility_col : str
        Name of volatility column
    lag_periods : list
        Lag periods to calculate autocorrelation
        
    Returns
    -------
    pd.DataFrame
        Autocorrelation values for each lag
    """
    results = {}
    
    for lag in lag_periods:
        correlation = df[volatility_col].corr(df[volatility_col].shift(lag))
        results[f'lag_{lag}'] = correlation
    
    return pd.DataFrame([results])


def volatility_regime_performance(
    df: pd.DataFrame,
    signal_col: str = 'signal',
    regime_col: str = 'volatility_regime'
) -> pd.DataFrame:
    """
    Calculate strategy performance by volatility regime.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with signals and regime data
    signal_col : str
        Name of signal column
    regime_col : str
        Name of regime column
        
    Returns
    -------
    pd.DataFrame
        Performance metrics by regime
    """
    df = df.copy()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['returns'] * df[signal_col].shift(1)
    
    # Group by regime
    regime_performance = df.groupby(regime_col).agg({
        'strategy_returns': ['count', 'mean', 'std', 'sum'],
        'returns': ['mean', 'std']
    })
    
    # Calculate Sharpe ratio for each regime
    regime_performance['sharpe'] = (
        regime_performance[('strategy_returns', 'mean')] / 
        regime_performance[('strategy_returns', 'std')] * 
        np.sqrt(252)
    )
    
    # Calculate information ratio
    regime_performance['info_ratio'] = (
        (regime_performance[('strategy_returns', 'mean')] - regime_performance[('returns', 'mean')]) /
        regime_performance[('strategy_returns', 'std')] * 
        np.sqrt(252)
    )
    
    return regime_performance