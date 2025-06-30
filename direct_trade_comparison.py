"""
Direct comparison of trade extraction with stop/target logic.
Shows exactly where universal analysis and execution engine differ.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_price_data():
    """Load SPY 1-minute price data."""
    # Try parquet first, then CSV
    try:
        df = pd.read_parquet('data/SPY_1m.parquet')
    except:
        df = pd.read_csv('data/SPY_1m.csv')
    
    # Ensure timestamp column
    if 'timestamp' not in df.columns and 'Datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Datetime'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure OHLC columns
    col_mapping = {
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    return df.sort_values('timestamp')

def generate_bollinger_signals(df, period=20, num_std=2.0, rsi_period=14):
    """Generate Bollinger Band + RSI signals similar to the strategies."""
    
    # Calculate Bollinger Bands
    df['sma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['sma'] + (num_std * df['std'])
    df['lower_band'] = df['sma'] - (num_std * df['std'])
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    df['signal'] = 0
    
    # Long signal: price touches lower band and RSI < 30
    long_condition = (df['low'] <= df['lower_band']) & (df['rsi'] < 30)
    df.loc[long_condition, 'signal'] = 1
    
    # Short signal: price touches upper band and RSI > 70
    short_condition = (df['high'] >= df['upper_band']) & (df['rsi'] > 70)
    df.loc[short_condition, 'signal'] = -1
    
    return df

def extract_trades_universal(df, prevent_reentry=True, stop_loss=0.01, profit_target=0.02):
    """
    Extract trades using the universal analysis approach.
    This mimics the logic from analytics functions.
    """
    trades = []
    position = 0
    entry_idx = None
    entry_price = None
    entry_time = None
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        signal = row['signal']
        
        if position == 0 and signal != 0:
            # Enter position
            position = signal
            entry_idx = idx
            entry_price = row['close']
            entry_time = row['timestamp']
            
        elif position != 0:
            # Check for exit conditions
            should_exit = False
            exit_reason = 'signal'
            
            # Signal-based exit
            if prevent_reentry and signal != 0 and signal != position:
                should_exit = True
            elif not prevent_reentry and signal == -position:
                should_exit = True
            
            if should_exit:
                # Now apply stop/target logic by looking at bars AFTER entry
                actual_exit_idx = idx
                actual_exit_price = row['close']
                actual_exit_time = row['timestamp']
                exit_type = 'signal'
                
                # Check all bars between entry and signal exit for stop/target
                for check_idx in range(entry_idx + 1, idx + 1):
                    check_row = df.iloc[check_idx]
                    
                    if position > 0:  # Long position
                        stop_price = entry_price * (1 - stop_loss)
                        target_price = entry_price * (1 + profit_target)
                        
                        if check_row['low'] <= stop_price:
                            exit_type = 'stop'
                            actual_exit_price = stop_price
                            actual_exit_time = check_row['timestamp']
                            actual_exit_idx = check_idx
                            break
                        elif check_row['high'] >= target_price:
                            exit_type = 'target'
                            actual_exit_price = target_price
                            actual_exit_time = check_row['timestamp']
                            actual_exit_idx = check_idx
                            break
                    
                    else:  # Short position
                        stop_price = entry_price * (1 + stop_loss)
                        target_price = entry_price * (1 - profit_target)
                        
                        if check_row['high'] >= stop_price:
                            exit_type = 'stop'
                            actual_exit_price = stop_price
                            actual_exit_time = check_row['timestamp']
                            actual_exit_idx = check_idx
                            break
                        elif check_row['low'] <= target_price:
                            exit_type = 'target'
                            actual_exit_price = target_price
                            actual_exit_time = check_row['timestamp']
                            actual_exit_idx = check_idx
                            break
                
                # Calculate return
                if position > 0:
                    ret = (actual_exit_price - entry_price) / entry_price
                else:
                    ret = (entry_price - actual_exit_price) / entry_price
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_idx': entry_idx,
                    'entry_price': entry_price,
                    'exit_time': actual_exit_time,
                    'exit_idx': actual_exit_idx,
                    'exit_price': actual_exit_price,
                    'exit_type': exit_type,
                    'return': ret,
                    'position': position
                })
                
                # Clear position
                position = 0
                
                # Re-enter if needed
                if not prevent_reentry and signal != 0:
                    position = signal
                    entry_idx = idx
                    entry_price = row['close']
                    entry_time = row['timestamp']
    
    return pd.DataFrame(trades)

def extract_trades_execution(df, prevent_reentry=True, stop_loss=0.01, profit_target=0.02):
    """
    Extract trades simulating execution engine logic.
    This might check prices in a different order or have different logic.
    """
    trades = []
    position = 0
    entry_idx = None
    entry_price = None
    entry_time = None
    stop_price = None
    target_price = None
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        signal = row['signal']
        
        # If we have a position, check stop/target FIRST
        if position != 0:
            hit_stop = False
            hit_target = False
            
            if position > 0:  # Long position
                # Check if current bar hits stop or target
                if row['low'] <= stop_price:
                    hit_stop = True
                elif row['high'] >= target_price:
                    hit_target = True
            else:  # Short position
                if row['high'] >= stop_price:
                    hit_stop = True
                elif row['low'] <= target_price:
                    hit_target = True
            
            # Exit if stop or target hit
            if hit_stop or hit_target:
                exit_price = stop_price if hit_stop else target_price
                exit_type = 'stop' if hit_stop else 'target'
                
                # Calculate return
                if position > 0:
                    ret = (exit_price - entry_price) / entry_price
                else:
                    ret = (entry_price - exit_price) / entry_price
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_idx': entry_idx,
                    'entry_price': entry_price,
                    'exit_time': row['timestamp'],
                    'exit_idx': idx,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'return': ret,
                    'position': position
                })
                
                # Clear position
                position = 0
                continue
        
        # Now check signals
        if position == 0 and signal != 0:
            # Enter position
            position = signal
            entry_idx = idx
            entry_price = row['close']
            entry_time = row['timestamp']
            
            # Set stop and target prices
            if position > 0:
                stop_price = entry_price * (1 - stop_loss)
                target_price = entry_price * (1 + profit_target)
            else:
                stop_price = entry_price * (1 + stop_loss)
                target_price = entry_price * (1 - profit_target)
                
        elif position != 0:
            # Check for signal-based exit
            should_exit = False
            
            if prevent_reentry and signal != 0 and signal != position:
                should_exit = True
            elif not prevent_reentry and signal == -position:
                should_exit = True
            
            if should_exit:
                # Exit on signal
                exit_price = row['close']
                
                # Calculate return
                if position > 0:
                    ret = (exit_price - entry_price) / entry_price
                else:
                    ret = (entry_price - exit_price) / entry_price
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_idx': entry_idx,
                    'entry_price': entry_price,
                    'exit_time': row['timestamp'],
                    'exit_idx': idx,
                    'exit_price': exit_price,
                    'exit_type': 'signal',
                    'return': ret,
                    'position': position
                })
                
                # Clear position
                position = 0
                
                # Re-enter if needed
                if not prevent_reentry and signal != 0:
                    position = signal
                    entry_idx = idx
                    entry_price = row['close']
                    entry_time = row['timestamp']
                    
                    # Set stop and target prices
                    if position > 0:
                        stop_price = entry_price * (1 - stop_loss)
                        target_price = entry_price * (1 + profit_target)
                    else:
                        stop_price = entry_price * (1 + stop_loss)
                        target_price = entry_price * (1 - profit_target)
    
    return pd.DataFrame(trades)

def compare_trades(universal_trades, execution_trades):
    """Compare trades between two extraction methods."""
    print("\n" + "="*80)
    print("TRADE COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nUniversal Analysis:")
    print(f"  Total trades: {len(universal_trades)}")
    print(f"  Profit targets hit: {(universal_trades['exit_type'] == 'target').sum()}")
    print(f"  Stops hit: {(universal_trades['exit_type'] == 'stop').sum()}")
    print(f"  Signal exits: {(universal_trades['exit_type'] == 'signal').sum()}")
    
    print(f"\nExecution Engine:")
    print(f"  Total trades: {len(execution_trades)}")
    print(f"  Profit targets hit: {(execution_trades['exit_type'] == 'target').sum()}")
    print(f"  Stops hit: {(execution_trades['exit_type'] == 'stop').sum()}")
    print(f"  Signal exits: {(execution_trades['exit_type'] == 'signal').sum()}")
    
    # Find matching trades
    print("\n" + "="*80)
    print("DETAILED TRADE-BY-TRADE COMPARISON (First 10 trades)")
    print("="*80)
    
    for i in range(min(10, len(universal_trades), len(execution_trades))):
        u_trade = universal_trades.iloc[i]
        
        # Find matching execution trade by entry time
        e_matches = execution_trades[
            execution_trades['entry_time'] == u_trade['entry_time']
        ]
        
        if len(e_matches) > 0:
            e_trade = e_matches.iloc[0]
            
            print(f"\nTrade {i+1}:")
            print(f"  Entry: {u_trade['entry_time']} @ ${u_trade['entry_price']:.2f}")
            
            if u_trade['exit_type'] != e_trade['exit_type']:
                print(f"  ⚠️  EXIT TYPE MISMATCH!")
                
            print(f"  Universal: Exit {u_trade['exit_type']:6} @ {u_trade['exit_time']} "
                  f"Price ${u_trade['exit_price']:.2f} Return {u_trade['return']*100:6.2f}%")
            print(f"  Execution: Exit {e_trade['exit_type']:6} @ {e_trade['exit_time']} "
                  f"Price ${e_trade['exit_price']:.2f} Return {e_trade['return']*100:6.2f}%")
            
            # If there's a mismatch, investigate
            if u_trade['exit_type'] != e_trade['exit_type']:
                print(f"  Investigation:")
                if u_trade['exit_type'] == 'target' and e_trade['exit_type'] != 'target':
                    print(f"    Universal hit target but execution didn't")
                    print(f"    This suggests execution might be checking signals before targets")
                elif e_trade['exit_type'] == 'target' and u_trade['exit_type'] != 'target':
                    print(f"    Execution hit target but universal didn't")
                    print(f"    This is unexpected - universal should catch all targets")

def main():
    print("DIRECT TRADE COMPARISON: Universal vs Execution Logic")
    print("=" * 80)
    
    # Load price data
    print("Loading SPY 1-minute data...")
    df = load_price_data()
    print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Generate signals
    print("\nGenerating Bollinger Band + RSI signals...")
    df = generate_bollinger_signals(df)
    
    # Filter to recent data with signals
    df_signals = df[df['signal'] != 0].copy()
    print(f"Found {len(df_signals)} signal bars")
    
    # Get a subset with signals for testing
    test_start = '2025-01-01'
    test_end = '2025-02-28'
    df_test = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)].copy()
    
    print(f"\nTesting on period: {test_start} to {test_end}")
    print(f"Test bars: {len(df_test)}")
    print(f"Test signals: {(df_test['signal'] != 0).sum()}")
    
    # Extract trades using both methods
    print("\nExtracting trades using Universal Analysis method...")
    universal_trades = extract_trades_universal(df_test, prevent_reentry=True)
    
    print("\nExtracting trades using Execution Engine method...")
    execution_trades = extract_trades_execution(df_test, prevent_reentry=True)
    
    # Compare results
    compare_trades(universal_trades, execution_trades)
    
    # Show specific examples where they differ
    print("\n" + "="*80)
    print("INVESTIGATING KEY DIFFERENCE")
    print("="*80)
    print("\nThe key difference appears to be in the ORDER of checking:")
    print("1. Universal: Checks for signal exit first, THEN applies stop/target to the path")
    print("2. Execution: Checks stop/target on EVERY bar, exits immediately if hit")
    print("\nThis means execution can exit on stop/target BEFORE seeing an opposite signal,")
    print("while universal only applies stop/target to trades that would exit on signal.")

if __name__ == "__main__":
    main()