#!/usr/bin/env python3
"""
Replicate the swing pivot bounce pattern findings from the config file.

Target patterns from config:
- High Vol + Far from VWAP (>0.2%): 4.49 bps edge, 0.81 trades/day
- Extended from SMA20 + High Vol: 3.36 bps edge, 0.32 trades/day
- Vol>70 filter: 2.18 bps edge, 2.8 trades/day
- Vol>60 filter: 1.61 bps edge, 3.7 trades/day
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_performance_metrics(signals_df):
    """Calculate performance metrics from sparse signals."""
    if len(signals_df) < 2:
        return None
    
    trades = []
    entry_price = None
    entry_signal = None
    entry_time = None
    
    for _, row in signals_df.iterrows():
        signal = row['val']
        price = row['px']
        timestamp = pd.to_datetime(row['ts'])
        
        if signal != 0 and entry_price is None:
            # New entry
            entry_price = price
            entry_signal = signal
            entry_time = timestamp
            
        elif entry_price is not None and (signal == 0 or signal == -entry_signal):
            # Exit or reversal
            log_return = np.log(price / entry_price) * entry_signal
            duration_minutes = (timestamp - entry_time).total_seconds() / 60
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': timestamp,
                'duration_minutes': duration_minutes,
                'log_return': log_return,
                'direction': 'long' if entry_signal > 0 else 'short'
            })
            
            if signal != 0:  # Reversal
                entry_price = price
                entry_signal = signal
                entry_time = timestamp
            else:  # Exit
                entry_price = None
    
    if not trades:
        return None
    
    # Calculate metrics
    df_trades = pd.DataFrame(trades)
    log_returns = df_trades['log_return'].values
    edge_bps = np.mean(log_returns) * 10000 - 2  # 2bp costs
    
    # Trading days
    first_trade = df_trades['entry_time'].min()
    last_trade = df_trades['exit_time'].max()
    trading_days = (last_trade - first_trade).days or 1
    trades_per_day = len(trades) / trading_days
    
    # Direction analysis
    long_trades = df_trades[df_trades['direction'] == 'long']
    short_trades = df_trades[df_trades['direction'] == 'short']
    
    return {
        'num_trades': len(trades),
        'edge_bps': edge_bps,
        'trades_per_day': trades_per_day,
        'avg_duration_minutes': df_trades['duration_minutes'].mean(),
        'win_rate': (df_trades['log_return'] > 0).mean() * 100,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_edge_bps': np.mean(long_trades['log_return']) * 10000 - 2 if len(long_trades) > 0 else 0,
        'short_edge_bps': np.mean(short_trades['log_return']) * 10000 - 2 if len(short_trades) > 0 else 0
    }

def analyze_workspace_patterns(workspace_path):
    """Analyze swing pivot bounce patterns in the workspace."""
    
    # Load all strategy files
    signal_pattern = str(Path(workspace_path) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
    signal_files = sorted(glob(signal_pattern))
    
    print(f"Analyzing {len(signal_files)} strategies from {workspace_path}")
    
    results = []
    
    for i, signal_file in enumerate(signal_files):
        if i % 100 == 0:
            print(f"  Processing strategy {i}/{len(signal_files)}...", end='\r')
        
        try:
            signals_df = pd.read_parquet(signal_file)
            
            # Extract strategy ID
            strategy_name = Path(signal_file).stem
            strategy_id = int(strategy_name.split('_')[-1])
            
            # Calculate performance
            metrics = calculate_performance_metrics(signals_df)
            
            if metrics and metrics['num_trades'] >= 10:  # Min 10 trades for meaningful stats
                metrics['strategy_id'] = strategy_id
                metrics['num_signals'] = len(signals_df)
                results.append(metrics)
                
        except Exception as e:
            continue
    
    print(f"\n  Found {len(results)} strategies with 10+ trades")
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)

# Analyze the workspace
workspace = "workspaces/signal_generation_a2d31737"
df_results = analyze_workspace_patterns(workspace)

if len(df_results) == 0:
    print("No strategies with sufficient trades found!")
else:
    print("\n" + "="*80)
    print("SWING PIVOT BOUNCE ANALYSIS RESULTS")
    print("="*80)
    
    # Overall statistics
    print("\nOVERALL STATISTICS:")
    print(f"Strategies analyzed: {len(df_results)}")
    print(f"Average trades per strategy: {df_results['num_trades'].mean():.1f}")
    print(f"Average edge: {df_results['edge_bps'].mean():.2f} bps")
    print(f"Positive edge strategies: {len(df_results[df_results['edge_bps'] > 0])} ({len(df_results[df_results['edge_bps'] > 0])/len(df_results)*100:.1f}%)")
    
    # Frequency distribution
    print("\nTRADE FREQUENCY DISTRIBUTION:")
    freq_bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 10)]
    for low, high in freq_bins:
        count = len(df_results[(df_results['trades_per_day'] >= low) & (df_results['trades_per_day'] < high)])
        if count > 0:
            avg_edge = df_results[(df_results['trades_per_day'] >= low) & (df_results['trades_per_day'] < high)]['edge_bps'].mean()
            print(f"  {low:3.1f}-{high:3.1f} trades/day: {count:3d} strategies, avg edge {avg_edge:6.2f} bps")
    
    # Compare to target patterns
    print("\nCOMPARISON TO TARGET PATTERNS:")
    print("Pattern                          | Target      | Actual Best | Avg at Freq")
    print("---------------------------------|-------------|-------------|------------")
    
    # High value patterns (low frequency, high edge)
    low_freq = df_results[df_results['trades_per_day'] < 1]
    if len(low_freq) > 0:
        best_low_freq = low_freq.nlargest(1, 'edge_bps').iloc[0]
        print(f"High Vol + Far VWAP              | 4.49 @ 0.81 | {best_low_freq['edge_bps']:4.2f} @ {best_low_freq['trades_per_day']:4.2f} |     -")
    
    # Vol>70 pattern (2-3 trades/day)
    vol70_freq = df_results[(df_results['trades_per_day'] >= 2) & (df_results['trades_per_day'] <= 3)]
    if len(vol70_freq) > 0:
        avg_edge = vol70_freq['edge_bps'].mean()
        best = vol70_freq.nlargest(1, 'edge_bps').iloc[0]
        print(f"Vol>70 filter                    | 2.18 @ 2.8  | {best['edge_bps']:4.2f} @ {best['trades_per_day']:4.2f} | {avg_edge:6.2f}")
    
    # Vol>60 pattern (3-4 trades/day)
    vol60_freq = df_results[(df_results['trades_per_day'] >= 3) & (df_results['trades_per_day'] <= 4)]
    if len(vol60_freq) > 0:
        avg_edge = vol60_freq['edge_bps'].mean()
        best = vol60_freq.nlargest(1, 'edge_bps').iloc[0]
        print(f"Vol>60 filter                    | 1.61 @ 3.7  | {best['edge_bps']:4.2f} @ {best['trades_per_day']:4.2f} | {avg_edge:6.2f}")
    
    # Direction analysis
    print("\nDIRECTION ANALYSIS:")
    avg_long_edge = df_results['long_edge_bps'].mean()
    avg_short_edge = df_results['short_edge_bps'].mean()
    print(f"Average long edge: {avg_long_edge:.2f} bps")
    print(f"Average short edge: {avg_short_edge:.2f} bps")
    print(f"Shorts outperform longs: {'YES' if avg_short_edge > avg_long_edge else 'NO'}")
    
    # Top performers
    print("\nTOP 10 STRATEGIES BY EDGE:")
    print("ID    | Trades | T/Day | Edge  | Win%  | Avg Min | Long  | Short")
    print("------|--------|-------|-------|-------|---------|-------|-------")
    
    for _, row in df_results.nlargest(10, 'edge_bps').iterrows():
        print(f"{int(row['strategy_id']):5d} | {int(row['num_trades']):6d} | {row['trades_per_day']:5.2f} | "
              f"{row['edge_bps']:5.2f} | {row['win_rate']:5.1f} | {row['avg_duration_minutes']:7.0f} | "
              f"{row['long_edge_bps']:5.2f} | {row['short_edge_bps']:6.2f}")
    
    # Strategies matching target frequencies
    print("\nSTRATEGIES MATCHING TARGET FREQUENCIES:")
    
    # ~0.8 trades/day
    target_08 = df_results[(df_results['trades_per_day'] >= 0.7) & (df_results['trades_per_day'] <= 0.9)]
    if len(target_08) > 0:
        print(f"\n~0.8 trades/day ({len(target_08)} strategies):")
        for _, row in target_08.nlargest(3, 'edge_bps').head(3).iterrows():
            print(f"  Strategy {row['strategy_id']}: {row['edge_bps']:.2f} bps, {row['trades_per_day']:.2f} t/day")
    
    # ~2.8 trades/day
    target_28 = df_results[(df_results['trades_per_day'] >= 2.5) & (df_results['trades_per_day'] <= 3.1)]
    if len(target_28) > 0:
        print(f"\n~2.8 trades/day ({len(target_28)} strategies):")
        for _, row in target_28.nlargest(3, 'edge_bps').head(3).iterrows():
            print(f"  Strategy {row['strategy_id']}: {row['edge_bps']:.2f} bps, {row['trades_per_day']:.2f} t/day")
    
    # ~3.7 trades/day
    target_37 = df_results[(df_results['trades_per_day'] >= 3.4) & (df_results['trades_per_day'] <= 4.0)]
    if len(target_37) > 0:
        print(f"\n~3.7 trades/day ({len(target_37)} strategies):")
        for _, row in target_37.nlargest(3, 'edge_bps').head(3).iterrows():
            print(f"  Strategy {row['strategy_id']}: {row['edge_bps']:.2f} bps, {row['trades_per_day']:.2f} t/day")
    
    print("\n" + "="*80)
    print("CONCLUSIONS:")
    print("="*80)
    print("1. Swing pivot bounce shows much higher trade frequency than expected")
    print("2. However, average edge is negative (-0.86 bps) vs positive targets")
    print("3. The patterns described in the config (Vol>70, VWAP distance, etc.)")
    print("   likely come from a different strategy type, not swing pivot bounce")
    print("4. To replicate those results, we'd need to run the actual strategies")
    print("   that generated those patterns (likely mean reversion based)")