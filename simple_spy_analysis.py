#!/usr/bin/env python3
"""
Simple performance analysis for signal generation workspaces
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

def analyze_workspace_performance(workspace_path):
    """Analyze performance of signals in a workspace"""
    
    workspace = Path(workspace_path)
    if not workspace.exists():
        print(f"Error: Workspace not found at {workspace_path}")
        return
    
    # Load SPY price data
    spy_data = pd.read_csv("./data/SPY.csv")
    spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'])
    spy_data = spy_data.set_index('timestamp')
    
    # Find signal files in new directory structure
    signal_files = list(workspace.glob("traces/*/signals/*/*.parquet"))
    
    if not signal_files:
        # Try sparse format
        signal_files = list(workspace.glob("*_signals_*.parquet"))
    
    if not signal_files:
        # Try old format
        signal_files = list(workspace.glob("*.parquet"))
        signal_files = [f for f in signal_files if 'signal' in f.name.lower()]
    
    if not signal_files:
        print("No signal files found")
        return
    
    # Load all signals
    all_signals = []
    for f in signal_files:
        try:
            df = pd.read_parquet(f)
            if not df.empty:
                all_signals.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not all_signals:
        print("No signals loaded")
        return
    
    # Combine all signals
    signals_df = pd.concat(all_signals, ignore_index=True)
    
    # Handle different column naming conventions
    if 'ts' in signals_df.columns:
        signals_df['timestamp'] = pd.to_datetime(signals_df['ts'])
        signals_df['signal_value'] = signals_df['val']
    else:
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        
    signals_df = signals_df.sort_values('timestamp')
    
    # Simple backtest logic
    positions = []
    current_position = None
    
    for _, signal in signals_df.iterrows():
        ts = signal['timestamp']
        
        # Skip if no price data
        if ts not in spy_data.index:
            continue
            
        price = spy_data.loc[ts, 'close']
        
        if signal['signal_value'] != 0 and current_position is None:
            # Enter position
            current_position = {
                'entry_time': ts,
                'entry_price': price,
                'direction': 1 if signal['signal_value'] > 0 else -1,
                'signal_value': signal['signal_value']
            }
        elif signal['signal_value'] == 0 and current_position is not None:
            # Exit position
            exit_price = price
            
            # Calculate return
            if current_position['direction'] > 0:
                ret = (exit_price - current_position['entry_price']) / current_position['entry_price']
            else:
                ret = (current_position['entry_price'] - exit_price) / current_position['entry_price']
            
            current_position['exit_time'] = ts
            current_position['exit_price'] = exit_price
            current_position['return'] = ret
            current_position['bps'] = ret * 10000
            current_position['duration'] = (ts - current_position['entry_time']).total_seconds() / 60  # minutes
            
            positions.append(current_position)
            current_position = None
        elif signal['signal_value'] != 0 and current_position is not None:
            # Signal change while in position - close and reopen
            exit_price = price
            
            # Close current
            if current_position['direction'] > 0:
                ret = (exit_price - current_position['entry_price']) / current_position['entry_price']
            else:
                ret = (current_position['entry_price'] - exit_price) / current_position['entry_price']
            
            current_position['exit_time'] = ts
            current_position['exit_price'] = exit_price
            current_position['return'] = ret
            current_position['bps'] = ret * 10000
            current_position['duration'] = (ts - current_position['entry_time']).total_seconds() / 60
            
            positions.append(current_position)
            
            # Open new position
            current_position = {
                'entry_time': ts,
                'entry_price': price,
                'direction': 1 if signal['signal_value'] > 0 else -1,
                'signal_value': signal['signal_value']
            }
    
    if not positions:
        print("No completed trades")
        return
    
    # Convert to DataFrame for analysis
    trades_df = pd.DataFrame(positions)
    
    # Calculate metrics
    total_trades = len(trades_df)
    avg_bps = trades_df['bps'].mean()
    win_rate = (trades_df['bps'] > 0).mean() * 100
    
    # Calculate trading frequency
    first_date = signals_df['timestamp'].min()
    last_date = signals_df['timestamp'].max()
    trading_days = np.busday_count(first_date.date(), last_date.date())
    trades_per_day = total_trades / trading_days if trading_days > 0 else 0
    
    # Output results
    print(f"Total trades: {total_trades}")
    print(f"Average bps per trade: {avg_bps:.2f}")
    print(f"Trades per day: {trades_per_day:.2f}")
    print(f"Win rate: {win_rate:.1f}%")
    
    # Net performance
    cost_bps = 1.0  # Assume 1 bps cost
    net_bps = avg_bps - cost_bps
    annual_return = net_bps * trades_per_day * 252 / 100
    
    print(f"Net bps (after {cost_bps}bp cost): {net_bps:.2f}")
    print(f"Annualized return: {annual_return:.1f}%")
    
    # Additional stats
    if total_trades > 10:
        print(f"\nAdditional Statistics:")
        print(f"Median bps: {trades_df['bps'].median():.2f}")
        print(f"Std dev bps: {trades_df['bps'].std():.2f}")
        print(f"Best trade: {trades_df['bps'].max():.2f} bps")
        print(f"Worst trade: {trades_df['bps'].min():.2f} bps")
        print(f"Avg duration: {trades_df['duration'].mean():.1f} minutes")


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_spy_analysis.py <workspace_path>")
        sys.exit(1)
    
    workspace_path = sys.argv[1]
    analyze_workspace_performance(workspace_path)


if __name__ == "__main__":
    main()