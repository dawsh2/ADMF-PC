#!/usr/bin/env python3
"""
Quick analysis script to find top performers from sparse trace data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_strategy_signals(parquet_file):
    """Analyze a single strategy's performance from sparse signals."""
    
    # Load the parquet file
    df = pd.read_parquet(parquet_file)
    print(f"\nAnalyzing: {parquet_file.name}")
    print(f"Total signals: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 signals:")
    print(df.head())
    
    # Calculate basic performance metrics from sparse signals
    # Sparse format: only signal changes are stored
    if 'val' in df.columns and 'px' in df.columns:
        trades = []
        entry_price = None
        entry_signal = None
        
        for idx, row in df.iterrows():
            signal = row['val']
            price = row['px']
            
            if entry_price is None and signal != 0:
                # Opening position
                entry_price = price
                entry_signal = signal
            elif entry_price is not None and (signal == 0 or signal != entry_signal):
                # Closing position
                if entry_signal > 0:  # Was long
                    pnl = (price - entry_price) / entry_price
                else:  # Was short
                    pnl = (entry_price - price) / entry_price
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl_pct': pnl * 100,
                    'signal_type': 'long' if entry_signal > 0 else 'short'
                })
                
                # Check if opening new position
                if signal != 0:
                    entry_price = price
                    entry_signal = signal
                else:
                    entry_price = None
                    entry_signal = None
        
        if trades:
            trades_df = pd.DataFrame(trades)
            print(f"\nTotal trades: {len(trades_df)}")
            print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.2%}")
            print(f"Average PnL per trade: {trades_df['pnl_pct'].mean():.2f}%")
            print(f"Total return: {trades_df['pnl_pct'].sum():.2f}%")
            print(f"Max win: {trades_df['pnl_pct'].max():.2f}%")
            print(f"Max loss: {trades_df['pnl_pct'].min():.2f}%")
            
            # Show trade breakdown
            print(f"\nTrade breakdown:")
            print(f"Long trades: {(trades_df['signal_type'] == 'long').sum()}")
            print(f"Short trades: {(trades_df['signal_type'] == 'short').sum()}")
            
            return trades_df
    
    return None

def main():
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_00cb24c3")
    traces_path = workspace_path / "traces"
    
    # Find all parquet files
    parquet_files = list(traces_path.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    # Analyze each file
    all_results = []
    for pf in parquet_files:
        result = analyze_strategy_signals(pf)
        if result is not None:
            all_results.append({
                'file': pf.name,
                'trades': len(result),
                'total_return': result['pnl_pct'].sum(),
                'avg_return': result['pnl_pct'].mean(),
                'win_rate': (result['pnl_pct'] > 0).mean()
            })
    
    # Summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n" + "="*60)
        print("SUMMARY - Top Performers by Total Return:")
        print("="*60)
        sorted_results = results_df.sort_values('total_return', ascending=False)
        print(sorted_results.to_string(index=False))

if __name__ == "__main__":
    main()