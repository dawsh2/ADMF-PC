"""
Compare trades between universal analysis and execution engine to understand profit target differences.
"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
import pyarrow.parquet as pq

def extract_trades(signals_df, prevent_reentry=True):
    """Extract trades from signal data with optional re-entry prevention."""
    trades = []
    position = 0
    entry_time = None
    entry_price = None
    entry_signal = None
    
    for idx, row in signals_df.iterrows():
        signal = row['signal_type']
        
        if position == 0 and signal in ['LONG', 'SHORT']:
            # Enter position
            position = 1 if signal == 'LONG' else -1
            entry_time = row['timestamp']
            entry_price = row['close']
            entry_signal = signal
            
        elif position != 0:
            # Check for exit
            should_exit = False
            
            if prevent_reentry:
                # Exit on any opposite or neutral signal
                if (position > 0 and signal in ['SHORT', 'NEUTRAL']) or \
                   (position < 0 and signal in ['LONG', 'NEUTRAL']):
                    should_exit = True
            else:
                # Only exit on opposite signal
                if (position > 0 and signal == 'SHORT') or \
                   (position < 0 and signal == 'LONG'):
                    should_exit = True
            
            if should_exit:
                # Record trade
                exit_time = row['timestamp']
                exit_price = row['close']
                
                if position > 0:
                    ret = (exit_price - entry_price) / entry_price
                else:
                    ret = (entry_price - exit_price) / entry_price
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'entry_signal': entry_signal,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'exit_signal': signal,
                    'return': ret,
                    'position': position
                })
                
                # Clear position
                position = 0
                entry_time = None
                entry_price = None
                
                # If exit signal is opposite direction and not prevent_reentry, enter new position
                if not prevent_reentry and signal in ['LONG', 'SHORT']:
                    position = 1 if signal == 'LONG' else -1
                    entry_time = row['timestamp']
                    entry_price = row['close']
                    entry_signal = signal
    
    return pd.DataFrame(trades)

def apply_stop_target(trades_df, stop_loss=0.01, profit_target=0.02, price_data=None):
    """Apply stop loss and profit target exits to trades."""
    if len(trades_df) == 0:
        return trades_df
    
    trades_with_exits = []
    
    for _, trade in trades_df.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        original_exit_time = pd.to_datetime(trade['exit_time'])
        entry_price = trade['entry_price']
        position = trade['position']
        
        # Calculate stop and target prices
        if position > 0:  # Long
            stop_price = entry_price * (1 - stop_loss)
            target_price = entry_price * (1 + profit_target)
        else:  # Short
            stop_price = entry_price * (1 + stop_loss)
            target_price = entry_price * (1 - profit_target)
        
        # Get price data during trade
        trade_prices = price_data[
            (price_data['timestamp'] > entry_time) & 
            (price_data['timestamp'] <= original_exit_time)
        ]
        
        # Find first bar that hits stop or target
        exit_type = 'signal'
        exit_time = original_exit_time
        exit_price = trade['exit_price']
        
        for _, bar in trade_prices.iterrows():
            if position > 0:  # Long position
                if bar['low'] <= stop_price:
                    exit_type = 'stop'
                    exit_time = bar['timestamp']
                    exit_price = stop_price
                    break
                elif bar['high'] >= target_price:
                    exit_type = 'target'
                    exit_time = bar['timestamp']
                    exit_price = target_price
                    break
            else:  # Short position
                if bar['high'] >= stop_price:
                    exit_type = 'stop'
                    exit_time = bar['timestamp']
                    exit_price = stop_price
                    break
                elif bar['low'] <= target_price:
                    exit_type = 'target'
                    exit_time = bar['timestamp']
                    exit_price = target_price
                    break
        
        # Calculate return with new exit
        if position > 0:
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price
        
        trade_copy = trade.copy()
        trade_copy['exit_time'] = exit_time
        trade_copy['exit_price'] = exit_price
        trade_copy['return'] = ret
        trade_copy['exit_type'] = exit_type
        trades_with_exits.append(trade_copy)
    
    return pd.DataFrame(trades_with_exits)

def load_execution_trades(workspace_dir: str, strategy_id: str):
    """Load trades from execution engine position traces."""
    conn = duckdb.connect()
    
    # First, let's see what we have in position traces
    query = f"""
    SELECT * 
    FROM read_parquet('{workspace_dir}/position_traces/*/*.parquet')
    WHERE strategy_id = '{strategy_id}'
    ORDER BY timestamp
    LIMIT 20
    """
    
    sample = conn.execute(query).df()
    print("\nSample position trace data:")
    print(sample.columns.tolist())
    print(sample.head())
    
    # Load actual trades from position traces
    query = f"""
    WITH position_data AS (
        SELECT 
            *,
            LAG(position_size, 1, 0) OVER (ORDER BY timestamp) as prev_position,
            LAG(timestamp, 1) OVER (ORDER BY timestamp) as prev_timestamp
        FROM read_parquet('{workspace_dir}/position_traces/*/*.parquet')
        WHERE strategy_id = '{strategy_id}'
    ),
    entries AS (
        SELECT 
            timestamp as entry_time,
            entry_price,
            position_size,
            realized_pnl,
            exit_time,
            exit_price,
            exit_reason
        FROM position_data
        WHERE position_size != 0 AND prev_position = 0  -- New position opened
    )
    SELECT 
        e.*,
        CASE 
            WHEN e.position_size > 0 THEN (e.exit_price - e.entry_price) / e.entry_price
            ELSE (e.entry_price - e.exit_price) / e.entry_price
        END as return
    FROM entries e
    WHERE e.exit_time IS NOT NULL
    ORDER BY e.entry_time
    """
    
    trades = conn.execute(query).df()
    return trades

def main():
    workspace_dir = "workspaces/bollinger_rsi_workspace_20250118_165732"
    strategy_id = "bollinger_10"
    
    # Load price data first
    print("Loading price data...")
    conn = duckdb.connect()
    price_data = conn.execute(f"""
        SELECT timestamp, open, high, low, close
        FROM read_parquet('{workspace_dir}/position_data/*/*.parquet')
        ORDER BY timestamp
    """).df()
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    
    # Load signals for universal analysis
    print("\nLoading signals for universal analysis...")
    signals_df = conn.execute(f"""
        SELECT 
            s.timestamp,
            s.signal_type,
            s.signal_strength,
            p.close,
            p.high,
            p.low
        FROM read_parquet('{workspace_dir}/sparse_signals/*/*.parquet') s
        LEFT JOIN read_parquet('{workspace_dir}/position_data/*/*.parquet') p 
            ON s.source_file = p.source_file 
            AND s.timestamp = p.timestamp
        WHERE s.strategy_id = '{strategy_id}'
        ORDER BY s.timestamp
    """).df()
    
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    
    # Extract trades using universal analysis approach
    print("\nExtracting trades with universal analysis...")
    universal_trades = extract_trades(signals_df, prevent_reentry=True)
    print(f"Raw trades: {len(universal_trades)}")
    
    # Apply stop/target exits
    universal_trades_with_exits = apply_stop_target(
        universal_trades, 
        stop_loss=0.01,
        profit_target=0.02,
        price_data=price_data
    )
    
    print(f"\nUniversal analysis results:")
    print(f"Total trades: {len(universal_trades_with_exits)}")
    print(f"Profit targets hit: {(universal_trades_with_exits['exit_type'] == 'target').sum()}")
    print(f"Stops hit: {(universal_trades_with_exits['exit_type'] == 'stop').sum()}")
    print(f"Signal exits: {(universal_trades_with_exits['exit_type'] == 'signal').sum()}")
    
    # Load execution engine trades
    print("\n" + "="*50)
    print("Loading execution engine trades...")
    execution_trades = load_execution_trades(workspace_dir, strategy_id)
    
    print(f"\nExecution engine results:")
    print(f"Total trades: {len(execution_trades)}")
    if 'exit_reason' in execution_trades.columns:
        print(f"Exit reasons distribution:")
        print(execution_trades['exit_reason'].value_counts())
    
    # Compare first 10 trades in detail
    print("\n" + "="*50)
    print("DETAILED COMPARISON OF FIRST 10 TRADES")
    print("="*50)
    
    for i in range(min(10, len(universal_trades_with_exits), len(execution_trades))):
        u_trade = universal_trades_with_exits.iloc[i]
        
        # Find matching execution trade
        u_entry_time = pd.to_datetime(u_trade['entry_time'])
        time_diffs = abs(pd.to_datetime(execution_trades['entry_time']) - u_entry_time)
        closest_idx = time_diffs.argmin()
        
        if time_diffs.iloc[closest_idx] <= pd.Timedelta(seconds=60):
            e_trade = execution_trades.iloc[closest_idx]
            
            print(f"\nTrade {i+1}:")
            print(f"  Entry: {u_entry_time}")
            print(f"  Entry Price: Universal ${u_trade['entry_price']:.2f}, Execution ${e_trade['entry_price']:.2f}")
            
            print(f"\n  Universal Exit:")
            print(f"    Time: {u_trade['exit_time']}")
            print(f"    Price: ${u_trade['exit_price']:.2f}")
            print(f"    Type: {u_trade['exit_type']}")
            print(f"    Return: {u_trade['return']*100:.2f}%")
            
            print(f"\n  Execution Exit:")
            print(f"    Time: {e_trade['exit_time']}")
            print(f"    Price: ${e_trade['exit_price']:.2f}")
            print(f"    Type: {e_trade.get('exit_reason', 'unknown')}")
            print(f"    Return: {e_trade['return']*100:.2f}%")
            
            # Get price data during trade to verify
            trade_start = u_entry_time
            trade_end = max(pd.to_datetime(u_trade['exit_time']), 
                          pd.to_datetime(e_trade['exit_time']))
            
            trade_prices = price_data[
                (price_data['timestamp'] >= trade_start) & 
                (price_data['timestamp'] <= trade_end)
            ]
            
            if len(trade_prices) > 0:
                print(f"\n  Price Range During Trade:")
                print(f"    Low: ${trade_prices['low'].min():.2f}")
                print(f"    High: ${trade_prices['high'].max():.2f}")
                
                # Check if targets/stops should have been hit
                if u_trade['position'] > 0:  # Long
                    target_price = u_trade['entry_price'] * 1.02
                    stop_price = u_trade['entry_price'] * 0.99
                    print(f"    Target: ${target_price:.2f} (Hit: {trade_prices['high'].max() >= target_price})")
                    print(f"    Stop: ${stop_price:.2f} (Hit: {trade_prices['low'].min() <= stop_price})")
                
                if u_trade['exit_type'] != e_trade.get('exit_reason', 'unknown'):
                    print(f"  ⚠️  EXIT TYPE MISMATCH!")
        else:
            print(f"\nTrade {i+1}: No matching execution trade found")

if __name__ == "__main__":
    main()