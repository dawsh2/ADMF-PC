"""
Generate expected trades from signal data for validation.

This module creates "ideal" trades from signal data to compare with actual
system execution results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def generate_expected_trades(
    signal_df: pd.DataFrame,
    strategy_id: str,
    signal_hash: str,
    symbol: str = None,
    commission_per_trade: float = 0.0
) -> pd.DataFrame:
    """
    Generate expected trades from signal data.
    
    This creates the "ideal" trades assuming:
    - Perfect execution at signal bar close
    - No slippage
    - Exits happen exactly when signals indicate
    
    Args:
        signal_df: DataFrame with columns [timestamp, signal, close, ...]
        strategy_id: Strategy identifier
        signal_hash: Hash of the strategy configuration
        symbol: Trading symbol (if not in signal_df)
        commission_per_trade: Fixed commission per trade (default 0)
        
    Returns:
        DataFrame with expected trade records
    """
    if signal_df.empty:
        return pd.DataFrame()
    
    # Ensure we have required columns
    if 'signal' not in signal_df.columns:
        raise ValueError("signal_df must have 'signal' column")
    if 'close' not in signal_df.columns:
        raise ValueError("signal_df must have 'close' column")
    
    # Get symbol from dataframe if not provided
    if symbol is None:
        if 'symbol' in signal_df.columns:
            symbol = signal_df['symbol'].iloc[0]
        else:
            symbol = 'UNKNOWN'
    
    trades = []
    current_trade = None
    trade_counter = 0
    
    # Reset index to ensure we have numeric index
    signal_df = signal_df.reset_index(drop=True)
    
    for idx, row in signal_df.iterrows():
        signal = row['signal']
        
        # Entry signal (non-zero)
        if signal != 0 and current_trade is None:
            trade_counter += 1
            current_trade = {
                'trade_id': f"E{trade_counter:06d}",  # E for Expected
                'symbol': symbol,
                'strategy_id': strategy_id,
                'signal_hash': signal_hash,
                'entry_bar_idx': idx,
                'entry_time': row.get('timestamp', idx),
                'entry_signal_strength': abs(signal),
                'entry_price': row['close'],
                'direction': 'long' if signal > 0 else 'short',
                # Expected trades have perfect execution
                'entry_order_price': row['close'],
                'entry_fill_price': row['close'],
                'slippage_entry': 0.0,
            }
        
        # Exit signal (zero) or direction change
        elif current_trade is not None and (
            signal == 0 or 
            (signal > 0 and current_trade['direction'] == 'short') or
            (signal < 0 and current_trade['direction'] == 'long')
        ):
            # Complete the current trade
            exit_price = row['close']
            
            # Calculate PnL
            if current_trade['direction'] == 'long':
                pnl = (exit_price - current_trade['entry_price']) * 100  # Assume 100 shares
            else:
                pnl = (current_trade['entry_price'] - exit_price) * 100
            
            # Subtract commission
            pnl -= commission_per_trade * 2  # Entry and exit
            
            current_trade.update({
                'exit_bar_idx': idx,
                'exit_time': row.get('timestamp', idx),
                'exit_price': exit_price,
                'exit_signal_strength': abs(signal) if signal != 0 else 0,
                'exit_reason': 'signal',
                # Expected trades have perfect execution
                'exit_order_price': exit_price,
                'exit_fill_price': exit_price,
                'slippage_exit': 0.0,
                # Metrics
                'pnl': pnl,
                'duration_bars': idx - current_trade['entry_bar_idx'],
                'commission': commission_per_trade * 2,
            })
            
            # Calculate duration_time if timestamps are available
            if pd.notna(current_trade['entry_time']) and pd.notna(current_trade['exit_time']):
                try:
                    if isinstance(current_trade['entry_time'], str):
                        entry_dt = pd.to_datetime(current_trade['entry_time'])
                        exit_dt = pd.to_datetime(current_trade['exit_time'])
                    else:
                        entry_dt = current_trade['entry_time']
                        exit_dt = current_trade['exit_time']
                    current_trade['duration_time'] = str(exit_dt - entry_dt)
                except:
                    current_trade['duration_time'] = None
            else:
                current_trade['duration_time'] = None
            
            trades.append(current_trade)
            
            # If signal changed direction, immediately open new trade
            if signal != 0:
                trade_counter += 1
                current_trade = {
                    'trade_id': f"E{trade_counter:06d}",
                    'symbol': symbol,
                    'strategy_id': strategy_id,
                    'signal_hash': signal_hash,
                    'entry_bar_idx': idx,
                    'entry_time': row.get('timestamp', idx),
                    'entry_signal_strength': abs(signal),
                    'entry_price': row['close'],
                    'direction': 'long' if signal > 0 else 'short',
                    'entry_order_price': row['close'],
                    'entry_fill_price': row['close'],
                    'slippage_entry': 0.0,
                }
            else:
                current_trade = None
    
    # Handle unclosed trade at end
    if current_trade is not None:
        logger.warning(f"Unclosed trade at end of data for {strategy_id}")
        # Could optionally close at last bar or leave open
    
    return pd.DataFrame(trades)


def aggregate_expected_trades(
    signal_files: Dict[str, str],
    strategy_index: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate expected trades for multiple strategies.
    
    Args:
        signal_files: Dict mapping strategy_id to signal file path
        strategy_index: Optional strategy index for metadata lookup
        
    Returns:
        Combined DataFrame of all expected trades
    """
    all_trades = []
    
    for strategy_id, signal_path in signal_files.items():
        try:
            # Load signal data
            signal_df = pd.read_parquet(signal_path)
            
            # Get strategy hash from index if available
            signal_hash = None
            if strategy_index is not None and strategy_id in strategy_index.index:
                signal_hash = strategy_index.loc[strategy_id, 'strategy_hash']
            
            # Generate expected trades
            trades = generate_expected_trades(
                signal_df,
                strategy_id,
                signal_hash or 'unknown'
            )
            
            all_trades.append(trades)
            logger.info(f"Generated {len(trades)} expected trades for {strategy_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate expected trades for {strategy_id}: {e}")
    
    if all_trades:
        return pd.concat(all_trades, ignore_index=True)
    else:
        return pd.DataFrame()