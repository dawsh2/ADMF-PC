#!/usr/bin/env python3
"""Analyze Keltner Bands returns and profitability."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

workspace_path = Path("workspaces/signal_generation_11d63547")

# Load metadata
with open(workspace_path / "metadata.json", 'r') as f:
    metadata = json.load(f)

# Load SPY price data to calculate returns
data_path = Path("data/SPY_1m.parquet")
if data_path.exists():
    price_df = pd.read_parquet(data_path)
else:
    data_path = Path("data/SPY_1m.csv")
    price_df = pd.read_csv(data_path)

print("=== KELTNER BANDS RETURN ANALYSIS ===\n")

# From config: period: [10, 20, 30], multiplier: [1.5, 2.0, 2.5]
periods = [10, 20, 30]
multipliers = [1.5, 2.0, 2.5]

all_results = []

# Analyze each strategy
idx = 0
for period in periods:
    for multiplier in multipliers:
        signal_file = workspace_path / f"traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_{idx}.parquet"
        
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
                    current_position = {
                        'entry_idx': bar_idx,
                        'entry_price': signal['px'],
                        'direction': signal_val
                    }
                
                # Exit
                elif current_position is not None and (signal_val == 0 or signal_val != current_position['direction']):
                    exit_price = signal['px']
                    entry_price = current_position['entry_price']
                    
                    # Calculate return
                    if current_position['direction'] > 0:  # Long
                        gross_return = (exit_price / entry_price) - 1
                    else:  # Short
                        gross_return = (entry_price / exit_price) - 1
                    
                    trades.append({
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': bar_idx,
                        'bars_held': bar_idx - current_position['entry_idx'],
                        'direction': 'long' if current_position['direction'] > 0 else 'short',
                        'gross_return': gross_return,
                        'gross_return_bps': gross_return * 10000
                    })
                    
                    # Reset or flip position
                    if signal_val != 0:
                        current_position = {
                            'entry_idx': bar_idx,
                            'entry_price': signal['px'],
                            'direction': signal_val
                        }
                    else:
                        current_position = None
            
            # Analyze trades
            if trades:
                trades_df = pd.DataFrame(trades)
                
                # Calculate metrics
                avg_return_bps = trades_df['gross_return_bps'].mean()
                win_rate = (trades_df['gross_return'] > 0).mean() * 100
                total_trades = len(trades_df)
                avg_bars_held = trades_df['bars_held'].mean()
                
                # Calculate net returns with different cost assumptions
                net_return_1bp = avg_return_bps - 1  # 0.5bp each way
                net_return_2bp = avg_return_bps - 2  # 1bp each way
                net_return_4bp = avg_return_bps - 4  # 2bp each way
                
                all_results.append({
                    'strategy_idx': idx,
                    'period': period,
                    'multiplier': multiplier,
                    'total_trades': total_trades,
                    'trades_per_day': total_trades / (metadata['total_bars'] / 390),
                    'avg_return_bps': avg_return_bps,
                    'net_return_1bp': net_return_1bp,
                    'net_return_2bp': net_return_2bp,
                    'net_return_4bp': net_return_4bp,
                    'win_rate': win_rate,
                    'avg_bars_held': avg_bars_held,
                    'profitable_1bp': net_return_1bp > 0,
                    'profitable_2bp': net_return_2bp > 0,
                    'profitable_4bp': net_return_4bp > 0
                })
                
                print(f"Strategy {idx} (Period={period}, Multiplier={multiplier}):")
                print(f"  Total trades: {total_trades}")
                print(f"  Avg return: {avg_return_bps:.2f} bps")
                print(f"  Win rate: {win_rate:.1f}%")
                print(f"  Net after costs: 1bp={net_return_1bp:.2f}, 2bp={net_return_2bp:.2f}, 4bp={net_return_4bp:.2f}")
                print(f"  Profitable at: 1bp={'✓' if net_return_1bp > 0 else '✗'}, 2bp={'✓' if net_return_2bp > 0 else '✗'}, 4bp={'✓' if net_return_4bp > 0 else '✗'}")
                print()
            else:
                print(f"Strategy {idx}: No trades found\n")
        
        idx += 1

# Convert to DataFrame for analysis
results_df = pd.DataFrame(all_results)

print("\n=== PROFITABILITY SUMMARY ===")
print(f"Strategies profitable at 1bp cost: {results_df['profitable_1bp'].sum()}/9")
print(f"Strategies profitable at 2bp cost: {results_df['profitable_2bp'].sum()}/9")
print(f"Strategies profitable at 4bp cost: {results_df['profitable_4bp'].sum()}/9")

print("\n=== BEST STRATEGIES BY NET RETURN (2bp cost) ===")
best_strategies = results_df.nlargest(5, 'net_return_2bp')[['strategy_idx', 'period', 'multiplier', 'avg_return_bps', 'net_return_2bp', 'trades_per_day', 'win_rate']]
print(best_strategies)

print("\n=== PARAMETER IMPACT ON RETURNS ===")
print("\nBy Period:")
period_impact = results_df.groupby('period').agg({
    'avg_return_bps': 'mean',
    'net_return_2bp': 'mean',
    'win_rate': 'mean',
    'trades_per_day': 'mean'
}).round(2)
print(period_impact)

print("\nBy Multiplier:")
multiplier_impact = results_df.groupby('multiplier').agg({
    'avg_return_bps': 'mean',
    'net_return_2bp': 'mean',
    'win_rate': 'mean',
    'trades_per_day': 'mean'
}).round(2)
print(multiplier_impact)

print("\n=== OPTIMAL CONFIGURATIONS ===")
# Find configurations that balance returns and frequency
viable_strategies = results_df[results_df['net_return_2bp'] > 0].copy()
viable_strategies['annual_return'] = viable_strategies['net_return_2bp'] * viable_strategies['trades_per_day'] * 252 / 10000

print("\nViable strategies (positive after 2bp cost):")
for _, row in viable_strategies.iterrows():
    print(f"Period={row['period']}, Multiplier={row['multiplier']}:")
    print(f"  {row['trades_per_day']:.1f} trades/day × {row['net_return_2bp']:.2f} bps = {row['annual_return']:.1f}% annual")

# Save results
results_df.to_csv('keltner_optimization_results.csv', index=False)
print("\n✓ Results saved to keltner_optimization_results.csv")