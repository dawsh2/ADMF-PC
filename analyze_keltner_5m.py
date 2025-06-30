#!/usr/bin/env python3
"""Analyze Keltner Bands performance on 5-minute data."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_keltner_5m(workspace_path):
    """Analyze Keltner Bands results on 5-minute data."""
    
    workspace = Path(workspace_path)
    
    # Load metadata
    with open(workspace / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load price data
    data_path = Path("data/SPY_5m.parquet")
    if data_path.exists():
        price_df = pd.read_parquet(data_path)
    else:
        price_df = pd.read_csv("data/SPY_5m.csv")
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    
    print("=== KELTNER BANDS 5M ANALYSIS ===\n")
    print(f"Workspace: {workspace_path}")
    print(f"Total bars: {metadata['total_bars']}")
    print(f"Data points: {len(price_df)}")
    
    # Calculate bars per day for 5m data
    bars_per_day = 78  # 6.5 hours * 12 bars/hour
    total_days = metadata['total_bars'] / bars_per_day
    
    # Map parameters (5x5 grid)
    periods = [10, 15, 20, 25, 30]
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    all_results = []
    all_trades = []
    
    # Analyze each strategy
    idx = 0
    for period in periods:
        for multiplier in multipliers:
            signal_file = workspace / f"traces/SPY_5m/signals/keltner_bands/SPY_compiled_strategy_{idx}.parquet"
            
            if signal_file.exists():
                # Load signals
                signals_df = pd.read_parquet(signal_file)
                
                # Calculate trades and returns
                trades = []
                current_position = None
                
                for i in range(len(signals_df)):
                    signal = signals_df.iloc[i]
                    bar_idx = signal['idx']
                    signal_val = signal['val']
                    
                    # Entry
                    if current_position is None and signal_val != 0:
                        if bar_idx < len(price_df):
                            current_position = {
                                'entry_idx': bar_idx,
                                'entry_price': signal['px'],
                                'direction': signal_val,
                                'entry_time': price_df.iloc[bar_idx]['timestamp']
                            }
                    
                    # Exit
                    elif current_position is not None and (signal_val == 0 or signal_val != current_position['direction']):
                        if bar_idx < len(price_df):
                            exit_price = signal['px']
                            entry_price = current_position['entry_price']
                            
                            # Calculate return
                            if current_position['direction'] > 0:  # Long
                                gross_return = (exit_price / entry_price) - 1
                            else:  # Short
                                gross_return = (entry_price / exit_price) - 1
                            
                            trade = {
                                'strategy_idx': idx,
                                'period': period,
                                'multiplier': multiplier,
                                'entry_time': current_position['entry_time'],
                                'exit_time': price_df.iloc[bar_idx]['timestamp'],
                                'bars_held': bar_idx - current_position['entry_idx'],
                                'minutes_held': (bar_idx - current_position['entry_idx']) * 5,
                                'direction': 'long' if current_position['direction'] > 0 else 'short',
                                'gross_return': gross_return,
                                'gross_return_bps': gross_return * 10000
                            }
                            
                            trades.append(trade)
                            all_trades.append(trade)
                            
                            # Reset or flip position
                            if signal_val != 0:
                                current_position = {
                                    'entry_idx': bar_idx,
                                    'entry_price': signal['px'],
                                    'direction': signal_val,
                                    'entry_time': price_df.iloc[bar_idx]['timestamp']
                                }
                            else:
                                current_position = None
                
                # Analyze trades for this strategy
                if trades:
                    trades_df = pd.DataFrame(trades)
                    
                    avg_return_bps = trades_df['gross_return_bps'].mean()
                    win_rate = (trades_df['gross_return'] > 0).mean() * 100
                    total_trades = len(trades_df)
                    avg_minutes_held = trades_df['minutes_held'].mean()
                    
                    # Different cost scenarios for 5m timeframe
                    net_return_2bp = avg_return_bps - 2   # 1bp each way
                    net_return_4bp = avg_return_bps - 4   # 2bp each way
                    net_return_6bp = avg_return_bps - 6   # 3bp each way
                    
                    result = {
                        'strategy_idx': idx,
                        'period': period,
                        'multiplier': multiplier,
                        'total_trades': total_trades,
                        'trades_per_day': total_trades / total_days,
                        'avg_return_bps': avg_return_bps,
                        'net_return_2bp': net_return_2bp,
                        'net_return_4bp': net_return_4bp,
                        'net_return_6bp': net_return_6bp,
                        'win_rate': win_rate,
                        'avg_minutes_held': avg_minutes_held,
                        'profitable_2bp': net_return_2bp > 0,
                        'profitable_4bp': net_return_4bp > 0
                    }
                    
                    all_results.append(result)
                    
                    print(f"\nStrategy {idx} (Period={period}, Multiplier={multiplier}):")
                    print(f"  Total trades: {total_trades} ({total_trades/total_days:.1f} per day)")
                    print(f"  Avg return: {avg_return_bps:.2f} bps")
                    print(f"  Win rate: {win_rate:.1f}%")
                    print(f"  Avg hold time: {avg_minutes_held:.0f} minutes")
                    print(f"  Net returns: 2bp={net_return_2bp:.2f}, 4bp={net_return_4bp:.2f}, 6bp={net_return_6bp:.2f}")
                    print(f"  Profitable: {'✓' if net_return_2bp > 0 else '✗'}")
            
            idx += 1
    
    # Summary analysis
    results_df = pd.DataFrame(all_results)
    
    print("\n\n=== PROFITABILITY SUMMARY ===")
    print(f"Strategies profitable at 2bp cost: {results_df['profitable_2bp'].sum()}/{len(results_df)}")
    print(f"Strategies profitable at 4bp cost: {results_df['profitable_4bp'].sum()}/{len(results_df)}")
    
    print("\n=== BEST STRATEGIES BY NET RETURN (2bp cost) ===")
    best_strategies = results_df.nlargest(10, 'net_return_2bp')
    for _, row in best_strategies.iterrows():
        print(f"Period={row['period']}, Mult={row['multiplier']}: "
              f"{row['avg_return_bps']:.2f} bps gross, {row['net_return_2bp']:.2f} bps net, "
              f"{row['trades_per_day']:.1f} trades/day")
    
    print("\n=== PARAMETER IMPACT ===")
    print("\nBy Period:")
    period_stats = results_df.groupby('period').agg({
        'avg_return_bps': 'mean',
        'net_return_2bp': 'mean',
        'trades_per_day': 'mean'
    }).round(2)
    print(period_stats)
    
    print("\nBy Multiplier:")
    mult_stats = results_df.groupby('multiplier').agg({
        'avg_return_bps': 'mean',
        'net_return_2bp': 'mean',
        'trades_per_day': 'mean'
    }).round(2)
    print(mult_stats)
    
    # Save results
    results_df.to_csv('keltner_5m_optimization_results.csv', index=False)
    
    # Save all trades for correlation analysis
    if all_trades:
        all_trades_df = pd.DataFrame(all_trades)
        all_trades_df.to_csv('keltner_5m_all_trades.csv', index=False)
        print(f"\n✓ Saved {len(all_trades_df)} trades to keltner_5m_all_trades.csv")
    
    print("\n✓ Results saved to keltner_5m_optimization_results.csv")
    
    # Annual return calculation for best strategies
    print("\n=== ANNUAL RETURN PROJECTIONS ===")
    viable = results_df[results_df['net_return_2bp'] > 0]
    if len(viable) > 0:
        for _, row in viable.iterrows():
            annual_return = row['net_return_2bp'] * row['trades_per_day'] * 252 / 10000 * 100
            print(f"Period={row['period']}, Mult={row['multiplier']}: "
                  f"{annual_return:.1f}% annual return "
                  f"({row['trades_per_day']:.1f} trades/day × {row['net_return_2bp']:.2f} bps)")
    else:
        print("No profitable strategies found at 2bp cost")
    
    return results_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        workspace = sys.argv[1]
    else:
        # Find most recent workspace
        workspaces = sorted(Path("workspaces").glob("signal_generation_*"))
        if workspaces:
            workspace = workspaces[-1]
        else:
            print("No workspace found. Please provide workspace path.")
            sys.exit(1)
    
    analyze_keltner_5m(workspace)