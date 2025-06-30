#!/usr/bin/env python3
"""
Comprehensive strategy analysis for workspace signal_generation_a15a871e
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_strategy_signals(parquet_file):
    """Analyze a single strategy's performance from sparse signals."""
    
    try:
        df = pd.read_parquet(parquet_file)
        
        # Calculate trades from sparse signals
        trades = []
        entry_price = None
        entry_signal = None
        entry_idx = None
        
        for idx, row in df.iterrows():
            signal = row['val']
            price = row['px']
            bar_idx = row['idx']
            
            if entry_price is None and signal != 0:
                # Opening position
                entry_price = price
                entry_signal = signal
                entry_idx = bar_idx
            elif entry_price is not None and (signal == 0 or signal != entry_signal):
                # Closing position
                if entry_signal > 0:  # Was long
                    pnl = (price - entry_price) / entry_price
                else:  # Was short
                    pnl = (entry_price - price) / entry_price
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': bar_idx,
                    'duration': bar_idx - entry_idx,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl_pct': pnl * 100,
                    'signal_type': 'long' if entry_signal > 0 else 'short'
                })
                
                # Check if opening new position
                if signal != 0:
                    entry_price = price
                    entry_signal = signal
                    entry_idx = bar_idx
                else:
                    entry_price = None
                    entry_signal = None
                    entry_idx = None
        
        return pd.DataFrame(trades) if trades else pd.DataFrame()
        
    except Exception as e:
        print(f"Error processing {parquet_file}: {e}")
        return pd.DataFrame()

def calculate_metrics(trades_df):
    """Calculate comprehensive performance metrics."""
    if len(trades_df) == 0:
        return {}
    
    metrics = {
        'num_trades': len(trades_df),
        'win_rate': (trades_df['pnl_pct'] > 0).mean(),
        'avg_pnl': trades_df['pnl_pct'].mean(),
        'total_return': trades_df['pnl_pct'].sum(),
        'max_win': trades_df['pnl_pct'].max(),
        'max_loss': trades_df['pnl_pct'].min(),
        'avg_duration': trades_df['duration'].mean(),
        'long_trades': (trades_df['signal_type'] == 'long').sum(),
        'short_trades': (trades_df['signal_type'] == 'short').sum(),
    }
    
    # Calculate Sharpe-like metric (simplified)
    if trades_df['pnl_pct'].std() > 0:
        metrics['sharpe_proxy'] = metrics['avg_pnl'] / trades_df['pnl_pct'].std()
    else:
        metrics['sharpe_proxy'] = 0
    
    # Calculate profit factor
    wins = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    losses = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    metrics['profit_factor'] = wins / losses if losses > 0 else float('inf')
    
    return metrics

def main():
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_45c186a6")
    traces_path = workspace_path / "traces"
    
    # Find all parquet files
    parquet_files = list(traces_path.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} strategy files")
    
    # Group by strategy type
    strategy_groups = defaultdict(list)
    for pf in parquet_files:
        strategy_type = pf.parent.name
        strategy_groups[strategy_type].append(pf)
    
    # Print summary by strategy type
    print("\nStrategy distribution:")
    for stype, files in strategy_groups.items():
        print(f"  {stype}: {len(files)} variants")
    
    # Analyze each strategy
    all_results = []
    
    for pf in parquet_files:
        trades_df = analyze_strategy_signals(pf)
        if len(trades_df) > 0:
            metrics = calculate_metrics(trades_df)
            metrics['strategy_type'] = pf.parent.name
            metrics['file'] = pf.stem
            all_results.append(metrics)
    
    if not all_results:
        print("\nNo trades found in any strategy!")
        return
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Top performers by total return
    print("\n" + "="*60)
    print("TOP 10 STRATEGIES BY TOTAL RETURN:")
    print("="*60)
    top_return = results_df.nlargest(10, 'total_return')[
        ['strategy_type', 'file', 'total_return', 'num_trades', 'win_rate', 'avg_pnl']
    ]
    print(top_return.to_string(index=False, float_format='%.2f'))
    
    # Top by Sharpe proxy
    print("\n" + "="*60)
    print("TOP 10 STRATEGIES BY RISK-ADJUSTED RETURN (SHARPE PROXY):")
    print("="*60)
    top_sharpe = results_df.nlargest(10, 'sharpe_proxy')[
        ['strategy_type', 'file', 'sharpe_proxy', 'avg_pnl', 'win_rate', 'num_trades']
    ]
    print(top_sharpe.to_string(index=False, float_format='%.2f'))
    
    # Summary by strategy type
    print("\n" + "="*60)
    print("PERFORMANCE BY STRATEGY TYPE:")
    print("="*60)
    type_summary = results_df.groupby('strategy_type').agg({
        'total_return': ['mean', 'max', 'std'],
        'win_rate': 'mean',
        'num_trades': 'mean',
        'sharpe_proxy': 'mean'
    }).round(2)
    print(type_summary)
    
    # Best of each type
    print("\n" + "="*60)
    print("BEST PERFORMER FROM EACH STRATEGY TYPE:")
    print("="*60)
    for stype in results_df['strategy_type'].unique():
        type_df = results_df[results_df['strategy_type'] == stype]
        best = type_df.nlargest(1, 'total_return').iloc[0]
        print(f"\n{stype}:")
        print(f"  File: {best['file']}")
        print(f"  Total Return: {best['total_return']:.2f}%")
        print(f"  Win Rate: {best['win_rate']:.2%}")
        print(f"  Trades: {best['num_trades']}")
        print(f"  Sharpe Proxy: {best['sharpe_proxy']:.2f}")

if __name__ == "__main__":
    main()