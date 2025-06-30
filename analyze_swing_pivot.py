#!/usr/bin/env python3
"""Analyze swing pivot bounce zones strategy results."""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

def analyze_swing_pivot_workspace(workspace_path: str):
    """Analyze swing pivot bounce zones optimization."""
    
    workspace = Path(workspace_path)
    signal_pattern = str(workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
    signal_files = sorted(glob(signal_pattern))
    
    print(f"Analyzing swing pivot bounce zones: {workspace}")
    print(f"Found {len(signal_files)} strategies\n")
    
    results = []
    
    for signal_file in signal_files:
        try:
            signals_df = pd.read_parquet(signal_file)
            if signals_df.empty:
                continue
            
            # Extract strategy ID
            strategy_name = Path(signal_file).stem
            strategy_id = int(strategy_name.split('_')[-1])
            
            # Count trades
            total_signals = len(signals_df[signals_df['val'] != 0])
            
            # For very low frequency strategies, show actual signals
            if len(signals_df) <= 10:
                print(f"\nStrategy {strategy_id} - Only {len(signals_df)} signal changes:")
                for _, row in signals_df.iterrows():
                    print(f"  {row['ts']}: Signal {row['val']} at price {row['px']}")
            
            if total_signals == 0:
                continue
            
            # Calculate returns
            trade_returns = []
            entry_price = None
            entry_signal = None
            
            for _, row in signals_df.iterrows():
                signal = row['val']
                price = row['px']
                
                if signal != 0 and entry_price is None:
                    entry_price = price
                    entry_signal = signal
                elif entry_price is not None and (signal == 0 or signal == -entry_signal):
                    log_return = np.log(price / entry_price) * entry_signal * 0.9998
                    trade_returns.append(log_return)
                    
                    if signal != 0:
                        entry_price = price
                        entry_signal = signal
                    else:
                        entry_price = None
                        entry_signal = None
            
            if not trade_returns:
                continue
            
            # Calculate metrics
            trade_returns_bps = [r * 10000 for r in trade_returns]
            edge_bps = np.mean(trade_returns_bps)
            
            # Time span
            first_ts = pd.to_datetime(signals_df['ts'].iloc[0])
            last_ts = pd.to_datetime(signals_df['ts'].iloc[-1])
            trading_days = (last_ts - first_ts).days or 1
            
            results.append({
                'strategy_id': int(strategy_id),
                'signal_changes': len(signals_df),
                'total_trades': len(trade_returns),
                'edge_bps': edge_bps,
                'total_return_bps': sum(trade_returns_bps),
                'trading_days': trading_days
            })
            
        except Exception as e:
            print(f"Error processing {strategy_name}: {e}")
            continue
    
    if not results:
        print("No tradeable strategies found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by edge
    df = df.sort_values('edge_bps', ascending=False)
    
    print("\n=== SWING PIVOT BOUNCE RESULTS ===\n")
    print("ID  | Signals | Trades | Edge(bps) | Total Return | Days")
    print("----|---------|--------|-----------|--------------|-----")
    
    for _, row in df.head(20).iterrows():
        print(f"{row['strategy_id']:3d} | {row['signal_changes']:7d} | {row['total_trades']:6d} | "
              f"{row['edge_bps']:9.2f} | {row['total_return_bps']:12.0f} | {row['trading_days']:4d}")
    
    # Summary stats
    print(f"\n\nSUMMARY:")
    print(f"Total strategies: {len(df)}")
    print(f"Strategies with trades: {len(df[df['total_trades'] > 0])}")
    print(f"Average trades per strategy: {df['total_trades'].mean():.1f}")
    print(f"Best edge: {df['edge_bps'].max():.2f} bps")
    
    # Check for very low frequency
    low_freq = df[df['total_trades'] < 10]
    if len(low_freq) > 0:
        print(f"\nWARNING: {len(low_freq)} strategies have <10 trades!")
        print("This strategy appears to trade very infrequently.")

if __name__ == "__main__":
    workspace = "workspaces/signal_generation_ae5ce1b4"
    analyze_swing_pivot_workspace(workspace)