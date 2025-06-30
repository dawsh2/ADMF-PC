"""Volume regime analysis."""

import pandas as pd
import numpy as np
from typing import Optional, List


def add_volume_regime(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = 1.5,
    method: str = 'relative',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Add volume regime classification to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume data
    window : int
        Window for volume average
    threshold : float
        Threshold multiplier for high volume
    method : str
        'relative' or 'percentile'
    inplace : bool
        Whether to modify df in place
        
    Returns
    -------
    pd.DataFrame
        DataFrame with volume regime column
    """
    if not inplace:
        df = df.copy()
    
    if 'volume' not in df.columns:
        print("Warning: No volume data found")
        df['volume_regime'] = 'unknown'
        return df
    
    if method == 'relative':
        # Calculate relative volume
        df['volume_sma'] = df['volume'].rolling(window).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma']
        
        # Classify regime
        df['volume_regime'] = np.where(
            df['relative_volume'] > threshold,
            'high_volume',
            'normal_volume'
        )
        
    elif method == 'percentile':
        # Use percentile-based classification
        df['volume_pct'] = df['volume'].rolling(window).rank(pct=True)
        
        conditions = [
            df['volume_pct'] < 0.33,
            df['volume_pct'] < 0.67,
            df['volume_pct'] >= 0.67
        ]
        choices = ['low_volume', 'normal_volume', 'high_volume']
        
        df['volume_regime'] = np.select(conditions, choices)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


def calculate_relative_volume(
    df: pd.DataFrame,
    window: int = 20,
    volume_col: str = 'volume'
) -> pd.Series:
    """
    Calculate relative volume (RVOL).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume data
    window : int
        Window for average volume
    volume_col : str
        Name of volume column
        
    Returns
    -------
    pd.Series
        Relative volume values
    """
    avg_volume = df[volume_col].rolling(window).mean()
    relative_volume = df[volume_col] / avg_volume
    
    return relative_volume


def analyze_volume_patterns(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> pd.DataFrame:
    """
    Analyze volume patterns and their relationship with price.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and volume data
    price_col : str
        Name of price column
    volume_col : str
        Name of volume column
        
    Returns
    -------
    pd.DataFrame
        Volume pattern analysis
    """
    df = df.copy()
    
    # Calculate price changes
    df['price_change'] = df[price_col].pct_change()
    df['price_direction'] = np.sign(df['price_change'])
    
    # Calculate volume metrics
    df['volume_change'] = df[volume_col].pct_change()
    df['relative_volume'] = calculate_relative_volume(df, volume_col=volume_col)
    
    # Identify volume patterns
    patterns = []
    
    # Volume spike on up move
    volume_spike_up = (df['relative_volume'] > 2) & (df['price_direction'] > 0)
    if volume_spike_up.any():
        patterns.append({
            'pattern': 'volume_spike_up',
            'count': volume_spike_up.sum(),
            'avg_price_change': df[volume_spike_up]['price_change'].mean(),
            'avg_relative_volume': df[volume_spike_up]['relative_volume'].mean()
        })
    
    # Volume spike on down move
    volume_spike_down = (df['relative_volume'] > 2) & (df['price_direction'] < 0)
    if volume_spike_down.any():
        patterns.append({
            'pattern': 'volume_spike_down',
            'count': volume_spike_down.sum(),
            'avg_price_change': df[volume_spike_down]['price_change'].mean(),
            'avg_relative_volume': df[volume_spike_down]['relative_volume'].mean()
        })
    
    # Low volume consolidation
    low_volume = df['relative_volume'] < 0.5
    if low_volume.any():
        patterns.append({
            'pattern': 'low_volume_consolidation',
            'count': low_volume.sum(),
            'avg_price_change': df[low_volume]['price_change'].mean(),
            'avg_relative_volume': df[low_volume]['relative_volume'].mean()
        })
    
    # Volume divergence (price up, volume down)
    divergence_up = (df['price_direction'] > 0) & (df['volume_change'] < -0.2)
    if divergence_up.any():
        patterns.append({
            'pattern': 'bearish_divergence',
            'count': divergence_up.sum(),
            'avg_price_change': df[divergence_up]['price_change'].mean(),
            'avg_volume_change': df[divergence_up]['volume_change'].mean()
        })
    
    # Volume divergence (price down, volume down)
    divergence_down = (df['price_direction'] < 0) & (df['volume_change'] < -0.2)
    if divergence_down.any():
        patterns.append({
            'pattern': 'bullish_divergence',
            'count': divergence_down.sum(),
            'avg_price_change': df[divergence_down]['price_change'].mean(),
            'avg_volume_change': df[divergence_down]['volume_change'].mean()
        })
    
    return pd.DataFrame(patterns)