#!/usr/bin/env python3
"""
Analyze true RSI divergence strategy results
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

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
    
    # Load all signals
    all_signals = []
    for file in signal_files:
        try:
            df = pd.read_parquet(file)
            all_signals.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_signals:
        print("No signals could be loaded!")
        return
    
    # Combine all signals
    signals_df = pd.concat(all_signals, ignore_index=True)
    
    print(f"\nTotal Signals: {len(signals_df)}")
    
    # Signal breakdown
    if 'signal_value' in signals_df.columns:
        signal_counts = signals_df['signal_value'].value_counts()
        print("\nSignal Breakdown:")
        for sig_val, count in sorted(signal_counts.items()):
            sig_type = 'LONG' if sig_val > 0 else ('SHORT' if sig_val < 0 else 'FLAT/EXIT')
            print(f"  {sig_type} ({sig_val}): {count}")
    
    # Analyze metadata
    if 'metadata' in signals_df.columns:
        print("\nSignal Types from Metadata:")
        signal_types = {}
        for idx, row in signals_df.iterrows():
            if pd.notna(row['metadata']) and isinstance(row['metadata'], dict):
                sig_type = row['metadata'].get('signal_type', 'unknown')
                signal_types[sig_type] = signal_types.get(sig_type, 0) + 1
        
        for sig_type, count in sorted(signal_types.items()):
            print(f"  {sig_type}: {count}")
    
    # Time analysis
    if 'timestamp' in signals_df.columns:
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        print(f"\nTime Period:")
        print(f"  First signal: {signals_df['timestamp'].min()}")
        print(f"  Last signal: {signals_df['timestamp'].max()}")
        print(f"  Duration: {(signals_df['timestamp'].max() - signals_df['timestamp'].min()).days} days")
    
    # Entry analysis
    entry_signals = signals_df[signals_df['signal_value'] != 0].copy()
    if len(entry_signals) > 0:
        print(f"\nEntry Signals: {len(entry_signals)}")
        
        # Long vs Short
        longs = entry_signals[entry_signals['signal_value'] > 0]
        shorts = entry_signals[entry_signals['signal_value'] < 0]
        
        print(f"  Longs: {len(longs)}")
        print(f"  Shorts: {len(shorts)}")
        
        # Analyze divergence strength
        if 'metadata' in entry_signals.columns:
            divergence_strengths = []
            for idx, row in entry_signals.iterrows():
                if pd.notna(row['metadata']) and isinstance(row['metadata'], dict):
                    strength = row['metadata'].get('divergence_strength', 0)
                    if strength > 0:
                        divergence_strengths.append(strength)
            
            if divergence_strengths:
                print(f"\nDivergence Strength Statistics:")
                print(f"  Mean: {np.mean(divergence_strengths):.2f}")
                print(f"  Median: {np.median(divergence_strengths):.2f}")
                print(f"  Min: {np.min(divergence_strengths):.2f}")
                print(f"  Max: {np.max(divergence_strengths):.2f}")
    
    # Exit analysis
    exit_signals = signals_df[signals_df['signal_value'] == 0]
    if len(exit_signals) > 0:
        print(f"\nExit Signals: {len(exit_signals)}")
        
        # Analyze bars since entry
        if 'metadata' in exit_signals.columns:
            bars_since_entry = []
            for idx, row in exit_signals.iterrows():
                if pd.notna(row['metadata']) and isinstance(row['metadata'], dict):
                    bars = row['metadata'].get('bars_since_entry', 0)
                    if bars > 0:
                        bars_since_entry.append(bars)
            
            if bars_since_entry:
                print(f"\nBars Since Entry (Holding Period):")
                print(f"  Mean: {np.mean(bars_since_entry):.1f}")
                print(f"  Median: {np.median(bars_since_entry):.1f}")
                print(f"  Min: {np.min(bars_since_entry)}")
                print(f"  Max: {np.max(bars_since_entry)}")
    
    # Performance estimation (rough)
    if len(entry_signals) > 0:
        print("\n" + "="*60)
        print("Performance Estimation (Rough):")
        print("="*60)
        
        # Sort by timestamp
        signals_sorted = signals_df.sort_values('timestamp').copy()
        
        # Track trades
        trades = []
        current_position = 0
        entry_price = None
        entry_time = None
        entry_type = None
        
        for idx, row in signals_sorted.iterrows():
            sig_val = row['signal_value']
            
            if current_position == 0 and sig_val != 0:
                # Entry
                current_position = sig_val
                if pd.notna(row['metadata']) and isinstance(row['metadata'], dict):
                    entry_price = row['metadata'].get('entry_price', 0)
                    entry_time = row['timestamp']
                    entry_type = 'LONG' if sig_val > 0 else 'SHORT'
            
            elif current_position != 0 and sig_val == 0:
                # Exit
                if pd.notna(row['metadata']) and isinstance(row['metadata'], dict):
                    exit_price = row['metadata'].get('exit_price', 0)
                    if entry_price and exit_price:
                        if current_position > 0:  # Long
                            pnl_pct = (exit_price - entry_price) / entry_price * 100
                        else:  # Short
                            pnl_pct = (entry_price - exit_price) / entry_price * 100
                        
                        trades.append({
                            'type': entry_type,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pct': pnl_pct,
                            'duration': (row['timestamp'] - entry_time).total_seconds() / 60 if entry_time else 0
                        })
                
                current_position = 0
                entry_price = None
                entry_time = None
        
        if trades:
            print(f"\nCompleted Trades: {len(trades)}")
            
            # Win rate
            wins = [t for t in trades if t['pnl_pct'] > 0]
            print(f"Win Rate: {len(wins)/len(trades)*100:.1f}%")
            
            # PnL stats
            pnls = [t['pnl_pct'] for t in trades]
            print(f"\nPnL Statistics (%):")
            print(f"  Mean: {np.mean(pnls):.3f}%")
            print(f"  Total: {np.sum(pnls):.2f}%")
            print(f"  Best: {np.max(pnls):.3f}%")
            print(f"  Worst: {np.min(pnls):.3f}%")
            
            # By type
            long_trades = [t for t in trades if t['type'] == 'LONG']
            short_trades = [t for t in trades if t['type'] == 'SHORT']
            
            if long_trades:
                long_pnls = [t['pnl_pct'] for t in long_trades]
                print(f"\nLong Trades: {len(long_trades)}")
                print(f"  Win Rate: {len([p for p in long_pnls if p > 0])/len(long_pnls)*100:.1f}%")
                print(f"  Mean PnL: {np.mean(long_pnls):.3f}%")
            
            if short_trades:
                short_pnls = [t['pnl_pct'] for t in short_trades]
                print(f"\nShort Trades: {len(short_trades)}")
                print(f"  Win Rate: {len([p for p in short_pnls if p > 0])/len(short_pnls)*100:.1f}%")
                print(f"  Mean PnL: {np.mean(short_pnls):.3f}%")

def main():
    workspace = "workspaces/signal_generation_f6455215"
    analyze_divergence_strategy(workspace)

if __name__ == "__main__":
    main()