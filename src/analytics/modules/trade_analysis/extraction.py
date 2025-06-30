"""Trade extraction from signal data."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime


def extract_trades(
    signals_df: pd.DataFrame,
    signal_col: str = 'signal',
    price_col: str = 'close',
    timestamp_col: str = 'timestamp',
    min_hold_periods: int = 1
) -> pd.DataFrame:
    """
    Extract trades from signal data.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with signals and prices
    signal_col : str
        Name of signal column (-1, 0, 1)
    price_col : str
        Name of price column for entry/exit
    timestamp_col : str
        Name of timestamp column
    min_hold_periods : int
        Minimum periods to hold position
        
    Returns
    -------
    pd.DataFrame
        DataFrame with trade details
        
    Examples
    --------
    >>> signals = pd.DataFrame({
    ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
    ...     'signal': [0, 1, 1, 0, -1, -1, 0, 0, 1, 0],
    ...     'close': [100, 101, 102, 103, 102, 101, 100, 100, 101, 102]
    ... })
    >>> trades = extract_trades(signals)
    """
    if signals_df.empty:
        return pd.DataFrame()
    
    df = signals_df.copy()
    df = df.sort_values(timestamp_col)
    
    trades = []
    position = 0
    entry_price = None
    entry_time = None
    entry_idx = None
    
    for idx, row in df.iterrows():
        signal = row[signal_col]
        
        # Check for entry
        if position == 0 and signal != 0:
            position = signal
            entry_price = row[price_col]
            entry_time = row[timestamp_col]
            entry_idx = idx
            
        # Check for exit
        elif position != 0:
            # Exit if signal flips or goes to 0
            should_exit = (signal == -position or signal == 0)
            
            # Check minimum hold period
            if should_exit and entry_idx is not None:
                periods_held = idx - entry_idx
                if periods_held >= min_hold_periods:
                    # Record trade
                    exit_price = row[price_col]
                    exit_time = row[timestamp_col]
                    
                    # Calculate return
                    if position > 0:  # Long
                        pct_return = (exit_price - entry_price) / entry_price
                    else:  # Short
                        pct_return = (entry_price - exit_price) / entry_price
                    
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'long' if position > 0 else 'short',
                        'pct_return': pct_return,
                        'return': pct_return * 100,  # Percentage
                        'duration_periods': periods_held,
                    }
                    
                    # Add duration in time if timestamp is datetime
                    if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                        duration = exit_time - entry_time
                        trade['duration_minutes'] = duration.total_seconds() / 60
                        trade['duration_hours'] = duration.total_seconds() / 3600
                    
                    # Add metadata if present
                    for col in ['symbol', 'strategy', 'strategy_hash']:
                        if col in row:
                            trade[col] = row[col]
                    
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    entry_price = None
                    entry_time = None
                    entry_idx = None
    
    return pd.DataFrame(trades)


def extract_trades_with_stops(
    signals_df: pd.DataFrame,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    trailing_stop: Optional[float] = None,
    signal_col: str = 'signal',
    ohlc_cols: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Extract trades with stop loss and take profit levels.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with signals and OHLC data
    stop_loss : float, optional
        Stop loss percentage (e.g., 0.02 for 2%)
    take_profit : float, optional
        Take profit percentage
    trailing_stop : float, optional
        Trailing stop percentage
    signal_col : str
        Name of signal column
    ohlc_cols : dict, optional
        Mapping of OHLC column names
        
    Returns
    -------
    pd.DataFrame
        Trades with exit reasons
    """
    if ohlc_cols is None:
        ohlc_cols = {
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close'
        }
    
    df = signals_df.copy()
    trades = []
    
    position = 0
    entry_price = None
    entry_time = None
    entry_idx = None
    trailing_high = None
    trailing_low = None
    
    for idx, row in df.iterrows():
        signal = row[signal_col]
        
        # Entry logic
        if position == 0 and signal != 0:
            position = signal
            entry_price = row[ohlc_cols['close']]
            entry_time = row['timestamp']
            entry_idx = idx
            trailing_high = row[ohlc_cols['high']] if position > 0 else None
            trailing_low = row[ohlc_cols['low']] if position < 0 else None
            
        # Exit logic
        elif position != 0:
            exit_reason = None
            exit_price = None
            
            # Update trailing stops
            if position > 0 and trailing_stop:
                trailing_high = max(trailing_high, row[ohlc_cols['high']])
                trailing_stop_price = trailing_high * (1 - trailing_stop)
                
            elif position < 0 and trailing_stop:
                trailing_low = min(trailing_low, row[ohlc_cols['low']])
                trailing_stop_price = trailing_low * (1 + trailing_stop)
            
            # Check stop conditions for long position
            if position > 0:
                # Stop loss
                if stop_loss and row[ohlc_cols['low']] <= entry_price * (1 - stop_loss):
                    exit_price = entry_price * (1 - stop_loss)
                    exit_reason = 'stop_loss'
                    
                # Take profit
                elif take_profit and row[ohlc_cols['high']] >= entry_price * (1 + take_profit):
                    exit_price = entry_price * (1 + take_profit)
                    exit_reason = 'take_profit'
                    
                # Trailing stop
                elif trailing_stop and row[ohlc_cols['low']] <= trailing_stop_price:
                    exit_price = trailing_stop_price
                    exit_reason = 'trailing_stop'
                    
                # Signal exit
                elif signal <= 0:
                    exit_price = row[ohlc_cols['close']]
                    exit_reason = 'signal'
                    
            # Check stop conditions for short position
            elif position < 0:
                # Stop loss
                if stop_loss and row[ohlc_cols['high']] >= entry_price * (1 + stop_loss):
                    exit_price = entry_price * (1 + stop_loss)
                    exit_reason = 'stop_loss'
                    
                # Take profit
                elif take_profit and row[ohlc_cols['low']] <= entry_price * (1 - take_profit):
                    exit_price = entry_price * (1 - take_profit)
                    exit_reason = 'take_profit'
                    
                # Trailing stop
                elif trailing_stop and row[ohlc_cols['high']] >= trailing_stop_price:
                    exit_price = trailing_stop_price
                    exit_reason = 'trailing_stop'
                    
                # Signal exit
                elif signal >= 0:
                    exit_price = row[ohlc_cols['close']]
                    exit_reason = 'signal'
            
            # Execute exit
            if exit_reason:
                # Calculate return
                if position > 0:
                    pct_return = (exit_price - entry_price) / entry_price
                else:
                    pct_return = (entry_price - exit_price) / entry_price
                
                trade = {
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': 'long' if position > 0 else 'short',
                    'pct_return': pct_return,
                    'return': pct_return * 100,
                    'exit_reason': exit_reason,
                    'duration_periods': idx - entry_idx
                }
                
                trades.append(trade)
                
                # Reset
                position = 0
                entry_price = None
                entry_time = None
                entry_idx = None
                trailing_high = None
                trailing_low = None
    
    return pd.DataFrame(trades)


def merge_trades_with_metadata(
    trades_df: pd.DataFrame,
    metadata: Dict,
    prefix: str = 'strategy_'
) -> pd.DataFrame:
    """
    Merge trade data with strategy metadata.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        Trades DataFrame
    metadata : dict
        Strategy metadata
    prefix : str
        Prefix for metadata columns
        
    Returns
    -------
    pd.DataFrame
        Trades with metadata columns
    """
    df = trades_df.copy()
    
    for key, value in metadata.items():
        col_name = f"{prefix}{key}" if prefix else key
        df[col_name] = value
    
    return df