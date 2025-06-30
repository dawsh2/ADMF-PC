#!/usr/bin/env python3
"""
Analyze true RSI divergence strategy results from sparse trace format
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def analyze_divergence_strategy(workspace_path):
    """Analyze the true divergence strategy results"""
    
    workspace = Path(workspace_path)
    print(f"\nAnalyzing True RSI Divergence Strategy")
    print("="*60)
    
    # Find signal files
    signal_files = list(workspace.rglob("*/signals/*/*.parquet"))
    
    if not signal_files:
        print("No signal files found!")
        return
    
    # Load signals
    print(f"Found {len(signal_files)} signal file(s)")
    
    for file in signal_files:
        print(f"\nAnalyzing: {file.name}")
        df = pd.read_parquet(file)
        
        print(f"Total signals: {len(df)}")
        
        # Signal breakdown
        signal_counts = df['val'].value_counts()
        print("\nSignal Breakdown:")
        for sig_val, count in sorted(signal_counts.items()):
            sig_type = 'LONG' if sig_val > 0 else ('SHORT' if sig_val < 0 else 'EXIT/FLAT')
            print(f"  {sig_type} ({sig_val}): {count}")
        
        # Time analysis
        df['timestamp'] = pd.to_datetime(df['ts'])
        print(f"\nTime Period:")
        print(f"  First signal: {df['timestamp'].min()}")
        print(f"  Last signal: {df['timestamp'].max()}")
        duration = (df['timestamp'].max() - df['timestamp'].min())
        print(f"  Duration: {duration.days} days ({duration.total_seconds() / 3600:.1f} hours)")
        
        # Entry/Exit analysis
        entry_signals = df[df['val'] != 0]
        exit_signals = df[df['val'] == 0]
        
        print(f"\nEntry Signals: {len(entry_signals)}")
        print(f"  Longs: {len(df[df['val'] > 0])}")
        print(f"  Shorts: {len(df[df['val'] < 0])}")
        print(f"Exit Signals: {len(exit_signals)}")
        
        # Sort by time
        df_sorted = df.sort_values('timestamp').copy()
        
        # Track trades
        trades = []
        current_position = 0
        entry_idx = None
        entry_price = None
        entry_time = None
        entry_type = None
        
        for idx, row in df_sorted.iterrows():
            sig_val = row['val']
            
            if current_position == 0 and sig_val != 0:
                # Entry
                current_position = sig_val
                entry_idx = row['idx']
                entry_price = row['px']
                entry_time = row['timestamp']
                entry_type = 'LONG' if sig_val > 0 else 'SHORT'
            
            elif current_position != 0 and (sig_val == 0 or sig_val * current_position < 0):
                # Exit or reversal
                exit_price = row['px']
                exit_time = row['timestamp']
                
                if entry_price and exit_price:
                    if current_position > 0:  # Long
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:  # Short
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                    
                    trades.append({
                        'type': entry_type,
                        'entry_idx': entry_idx,
                        'exit_idx': row['idx'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'duration_bars': row['idx'] - entry_idx,
                        'duration_mins': (exit_time - entry_time).total_seconds() / 60
                    })
                
                # If reversal, set up new position
                if sig_val != 0:
                    current_position = sig_val
                    entry_idx = row['idx']
                    entry_price = row['px']
                    entry_time = row['timestamp']
                    entry_type = 'LONG' if sig_val > 0 else 'SHORT'
                else:
                    current_position = 0
                    entry_price = None
                    entry_time = None
        
        # Analyze trades
        if trades:
            print(f"\n{'='*60}")
            print("Trade Analysis:")
            print("="*60)
            
            print(f"\nCompleted Trades: {len(trades)}")
            
            # Win rate
            wins = [t for t in trades if t['pnl_pct'] > 0]
            print(f"Win Rate: {len(wins)/len(trades)*100:.1f}%")
            
            # PnL stats
            pnls = [t['pnl_pct'] for t in trades]
            print(f"\nPnL Statistics (%):")
            print(f"  Mean: {np.mean(pnls):.3f}%")
            print(f"  Median: {np.median(pnls):.3f}%")
            print(f"  Total: {np.sum(pnls):.2f}%")
            print(f"  Best: {np.max(pnls):.3f}%")
            print(f"  Worst: {np.min(pnls):.3f}%")
            print(f"  Std Dev: {np.std(pnls):.3f}%")
            
            # Holding period
            hold_bars = [t['duration_bars'] for t in trades]
            hold_mins = [t['duration_mins'] for t in trades]
            
            print(f"\nHolding Period:")
            print(f"  Mean: {np.mean(hold_bars):.1f} bars ({np.mean(hold_mins):.1f} mins)")
            print(f"  Median: {np.median(hold_bars):.0f} bars ({np.median(hold_mins):.0f} mins)")
            print(f"  Min: {np.min(hold_bars)} bars ({np.min(hold_mins):.0f} mins)")
            print(f"  Max: {np.max(hold_bars)} bars ({np.max(hold_mins):.0f} mins)")
            
            # By type
            long_trades = [t for t in trades if t['type'] == 'LONG']
            short_trades = [t for t in trades if t['type'] == 'SHORT']
            
            if long_trades:
                long_pnls = [t['pnl_pct'] for t in long_trades]
                print(f"\nLong Trades: {len(long_trades)}")
                print(f"  Win Rate: {len([p for p in long_pnls if p > 0])/len(long_pnls)*100:.1f}%")
                print(f"  Mean PnL: {np.mean(long_pnls):.3f}%")
                print(f"  Total PnL: {np.sum(long_pnls):.2f}%")
            
            if short_trades:
                short_pnls = [t['pnl_pct'] for t in short_trades]
                print(f"\nShort Trades: {len(short_trades)}")
                print(f"  Win Rate: {len([p for p in short_pnls if p > 0])/len(short_pnls)*100:.1f}%")
                print(f"  Mean PnL: {np.mean(short_pnls):.3f}%")
                print(f"  Total PnL: {np.sum(short_pnls):.2f}%")
            
            # Signal frequency
            total_bars = df_sorted.iloc[-1]['idx'] - df_sorted.iloc[0]['idx']
            print(f"\nSignal Frequency:")
            print(f"  Total bars in period: {total_bars}")
            print(f"  Signals per 1000 bars: {len(df) / total_bars * 1000:.1f}")
            print(f"  Trades per 1000 bars: {len(trades) / total_bars * 1000:.1f}")
            
            # Estimate annual performance
            # Assuming 1-minute bars, ~390 bars per day, ~98,280 bars per year
            bars_per_year = 390 * 252
            trades_per_year = len(trades) / total_bars * bars_per_year
            annual_return = np.mean(pnls) * trades_per_year
            
            print(f"\nAnnualized Estimates (rough):")
            print(f"  Trades per year: {trades_per_year:.0f}")
            print(f"  Expected annual return: {annual_return:.1f}%")
            print(f"  Note: Before transaction costs")

def main():
    workspace = "workspaces/signal_generation_84f6c6a0"
    analyze_divergence_strategy(workspace)

if __name__ == "__main__":
    main()