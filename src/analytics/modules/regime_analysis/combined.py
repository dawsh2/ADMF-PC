"""Combined regime analysis functionality."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from .volatility import add_volatility_regime
from .trend import add_trend_regime
from .volume import add_volume_regime


def add_regime_indicators(
    df: pd.DataFrame,
    regimes: List[str] = ['volatility', 'trend', 'volume', 'vwap'],
    volatility_window: int = 14,
    trend_fast: int = 20,
    trend_slow: int = 50,
    volume_window: int = 20,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Add multiple regime indicators to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    regimes : list
        List of regime types to add
    volatility_window : int
        Window for volatility calculation
    trend_fast : int
        Fast MA period for trend
    trend_slow : int
        Slow MA period for trend
    volume_window : int
        Window for volume regime
    inplace : bool
        Whether to modify df in place
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regime indicators
        
    Examples
    --------
    >>> df_with_regimes = add_regime_indicators(df, regimes=['volatility', 'trend'])
    """
    if not inplace:
        df = df.copy()
    
    if 'volatility' in regimes:
        df = add_volatility_regime(df, window=volatility_window, inplace=True)
    
    if 'trend' in regimes:
        df = add_trend_regime(df, fast_period=trend_fast, slow_period=trend_slow, inplace=True)
    
    if 'volume' in regimes:
        df = add_volume_regime(df, window=volume_window, inplace=True)
    
    if 'vwap' in regimes:
        # VWAP regime
        if 'volume' in df.columns:
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_regime'] = np.where(df['close'] > df['vwap'], 'above_vwap', 'below_vwap')
        else:
            print("Warning: No volume data for VWAP calculation")
    
    return df


def analyze_by_regime(
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    regime_column: str,
    metrics: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze trade performance by market regime.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    signals_df : pd.DataFrame
        DataFrame with signals and regime data
    regime_column : str
        Name of regime column to analyze
    metrics : list, optional
        Metrics to calculate
        
    Returns
    -------
    tuple
        (regime_stats, trades_with_regime)
        
    Examples
    --------
    >>> stats, trades = analyze_by_regime(trades_df, signals_df, 'volatility_regime')
    >>> print(stats)
    """
    if metrics is None:
        metrics = ['count', 'mean', 'std', 'sum', 'min', 'max']
    
    # Ensure timestamps are datetime
    trades_df = trades_df.copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    
    # Sort signals by timestamp
    signals_df = signals_df.sort_values('timestamp')
    
    # Assign regime to each trade based on entry time
    trades_with_regime = []
    
    for _, trade in trades_df.iterrows():
        # Find regime at entry
        mask = signals_df['timestamp'] <= trade['entry_time']
        
        if 'symbol' in trade and 'symbol' in signals_df.columns:
            mask &= signals_df['symbol'] == trade['symbol']
        
        matching_signals = signals_df[mask]
        
        if not matching_signals.empty and regime_column in matching_signals.columns:
            entry_regime = matching_signals[regime_column].iloc[-1]
        else:
            entry_regime = 'unknown'
        
        trade_dict = trade.to_dict()
        trade_dict[regime_column] = entry_regime
        trades_with_regime.append(trade_dict)
    
    trades_with_regime_df = pd.DataFrame(trades_with_regime)
    
    # Calculate statistics by regime
    regime_groups = trades_with_regime_df.groupby(regime_column)
    
    # Basic performance metrics
    performance_stats = regime_groups['pct_return'].agg(metrics).round(4)
    performance_stats.columns = [f'return_{m}' for m in metrics]
    
    # Additional metrics
    additional_stats = pd.DataFrame({
        'win_rate': regime_groups.apply(lambda x: (x['pct_return'] > 0).mean()),
        'avg_win': regime_groups.apply(lambda x: x[x['pct_return'] > 0]['pct_return'].mean()),
        'avg_loss': regime_groups.apply(lambda x: x[x['pct_return'] <= 0]['pct_return'].mean()),
        'profit_factor': regime_groups.apply(
            lambda x: x[x['pct_return'] > 0]['pct_return'].sum() / 
                     abs(x[x['pct_return'] <= 0]['pct_return'].sum()) 
                     if (x['pct_return'] <= 0).any() else np.inf
        ),
        'avg_duration_min': regime_groups['duration_minutes'].mean() if 'duration_minutes' in trades_with_regime_df else np.nan
    })
    
    # Combine stats
    regime_stats = pd.concat([performance_stats, additional_stats], axis=1)
    
    # Add direction breakdown if available
    if 'direction' in trades_with_regime_df.columns:
        direction_stats = trades_with_regime_df.groupby([regime_column, 'direction'])['pct_return'].agg(['count', 'mean'])
        direction_stats = direction_stats.unstack(fill_value=0)
        direction_stats.columns = [f'{dir}_{stat}' for stat, dir in direction_stats.columns]
        regime_stats = pd.concat([regime_stats, direction_stats], axis=1)
    
    return regime_stats, trades_with_regime_df


def create_composite_regime(
    df: pd.DataFrame,
    regime_columns: List[str],
    name: str = 'composite_regime'
) -> pd.Series:
    """
    Create a composite regime from multiple regime indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime columns
    regime_columns : list
        List of regime column names to combine
    name : str
        Name for composite regime column
        
    Returns
    -------
    pd.Series
        Composite regime labels
        
    Examples
    --------
    >>> df['market_regime'] = create_composite_regime(
    ...     df, ['trend_regime', 'volatility_regime']
    ... )
    """
    # Combine regime values into single label
    composite = df[regime_columns[0]].astype(str)
    
    for col in regime_columns[1:]:
        composite = composite + '_' + df[col].astype(str)
    
    return composite


def plot_regime_performance(
    regime_stats: pd.DataFrame,
    trades_with_regime: pd.DataFrame,
    regime_column: str,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Create comprehensive regime performance plots.
    
    Parameters
    ----------
    regime_stats : pd.DataFrame
        Statistics by regime
    trades_with_regime : pd.DataFrame
        Trades with regime labels
    regime_column : str
        Name of regime column
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Performance Analysis by {regime_column}', fontsize=16)
    
    # 1. Average returns by regime
    ax = axes[0, 0]
    regime_stats['return_mean'].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title('Average Return by Regime')
    ax.set_ylabel('Return')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # 2. Win rate by regime
    ax = axes[0, 1]
    regime_stats['win_rate'].plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
    ax.set_title('Win Rate by Regime')
    ax.set_ylabel('Win Rate')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    
    # 3. Trade count by regime
    ax = axes[0, 2]
    regime_stats['return_count'].plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.set_title('Trade Count by Regime')
    ax.set_ylabel('Number of Trades')
    
    # 4. Return distribution by regime
    ax = axes[1, 0]
    regimes = trades_with_regime[regime_column].unique()
    for regime in regimes:
        regime_trades = trades_with_regime[trades_with_regime[regime_column] == regime]
        if len(regime_trades) > 0:
            ax.hist(regime_trades['pct_return'] * 100, alpha=0.5, label=regime, bins=30)
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution by Regime')
    ax.legend()
    
    # 5. Profit factor by regime
    ax = axes[1, 1]
    profit_factors = regime_stats['profit_factor'].replace([np.inf, -np.inf], np.nan)
    profit_factors.plot(kind='bar', ax=ax, color='gold', edgecolor='black')
    ax.set_title('Profit Factor by Regime')
    ax.set_ylabel('Profit Factor')
    ax.axhline(1, color='red', linestyle='--', alpha=0.5)
    
    # 6. Average duration by regime (if available)
    ax = axes[1, 2]
    if 'avg_duration_min' in regime_stats.columns:
        regime_stats['avg_duration_min'].plot(kind='bar', ax=ax, color='purple', edgecolor='black')
        ax.set_title('Average Trade Duration by Regime')
        ax.set_ylabel('Duration (minutes)')
    else:
        ax.text(0.5, 0.5, 'Duration data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Average Trade Duration by Regime')
    
    plt.tight_layout()
    plt.show()


def regime_transition_matrix(
    df: pd.DataFrame,
    regime_column: str,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate regime transition probabilities.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime column
    regime_column : str
        Name of regime column
    normalize : bool
        Whether to normalize to probabilities
        
    Returns
    -------
    pd.DataFrame
        Transition matrix
    """
    # Get current and next regime
    current_regime = df[regime_column]
    next_regime = df[regime_column].shift(-1)
    
    # Remove last row (no next regime)
    current_regime = current_regime[:-1]
    next_regime = next_regime[:-1]
    
    # Create transition matrix
    transition_counts = pd.crosstab(current_regime, next_regime)
    
    if normalize:
        # Convert to probabilities
        transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    else:
        transition_matrix = transition_counts
    
    return transition_matrix