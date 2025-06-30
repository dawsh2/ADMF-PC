#!/usr/bin/env python3
"""
Analyze multiple Keltner strategies to find the high-performing ones.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

def calculate_basic_performance(signals_df: pd.DataFrame, execution_cost_bps: float = 0.5) -> dict:
    """Calculate basic performance metrics from signals."""
    if signals_df.empty or len(signals_df) < 2:
        return {'trades': 0, 'rpt_bps': 0, 'win_rate': 0}
    
    # Sort by index
    signals_df = signals_df.sort_values('idx').reset_index(drop=True)
    
    trades = []
    current_position = None
    execution_cost_multiplier = 1 - (execution_cost_bps / 10000)
    
    for i in range(len(signals_df)):
        row = signals_df.iloc[i]
        signal = row['val']
        price = row['px']
        
        if signal != 0:
            # Close existing position
            if current_position is not None:
                if current_position['direction'] == 'long':
                    trade_return = np.log(price / current_position['entry_price'])
                else:
                    trade_return = -np.log(price / current_position['entry_price'])
                trade_return *= execution_cost_multiplier
                trades.append(trade_return * 10000)  # Convert to bps
            
            # Open new position
            current_position = {
                'entry_price': price,
                'direction': 'long' if signal > 0 else 'short'
            }
        elif signal == 0 and current_position is not None:
            # Exit signal
            if current_position['direction'] == 'long':
                trade_return = np.log(price / current_position['entry_price'])
            else:
                trade_return = -np.log(price / current_position['entry_price'])
            trade_return *= execution_cost_multiplier
            trades.append(trade_return * 10000)
            current_position = None
    
    if trades:
        wins = [t for t in trades if t > 0]
        return {
            'trades': len(trades),
            'rpt_bps': np.mean(trades),
            'win_rate': len(wins) / len(trades)
        }
    return {'trades': 0, 'rpt_bps': 0, 'win_rate': 0}

# Analyze all strategies
workspace = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"
signals_path = Path(workspace) / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands"

print("=== Analyzing All Keltner Strategies ===\n")
print(f"{'Strategy':<20} {'Trades':<10} {'RPT (bps)':<12} {'Win Rate':<10}")
print("-" * 55)

results = []

# Get all strategy files
strategy_files = sorted(signals_path.glob("SPY_5m_compiled_strategy_*.parquet"))

for strategy_file in strategy_files:
    try:
        signals_df = pd.read_parquet(strategy_file)
        perf = calculate_basic_performance(signals_df)
        
        strategy_name = strategy_file.stem
        strategy_num = int(strategy_name.split('_')[-1])
        
        results.append({
            'strategy': strategy_name,
            'num': strategy_num,
            **perf
        })
        
        print(f"{strategy_name:<20} {perf['trades']:<10} {perf['rpt_bps']:>10.2f} {perf['win_rate']*100:>9.1f}%")
    except Exception as e:
        print(f"Error processing {strategy_file.name}: {e}")

# Find best performers
print("\n=== Top 5 Strategies by Return per Trade ===")
df = pd.DataFrame(results)
df = df.sort_values('rpt_bps', ascending=False)
top5 = df.head(5)

for idx, row in top5.iterrows():
    print(f"Strategy {row['num']:>2}: {row['rpt_bps']:>6.2f} bps/trade, "
          f"{row['trades']:>4} trades, {row['win_rate']*100:>5.1f}% win rate")

# Check if any match the previous analysis
print("\n=== Looking for High-Performance Strategies ===")
high_perf = df[df['rpt_bps'] > 2.0]
if len(high_perf) > 0:
    print(f"Found {len(high_perf)} strategies with >2 bps/trade:")
    for idx, row in high_perf.iterrows():
        print(f"  {row['strategy']}")
else:
    print("No strategies found with >2 bps/trade performance")
    print("\nThis suggests we may be looking at:")
    print("1. A different workspace than the high-performing one")
    print("2. Different execution cost assumptions")
    print("3. Different signal interpretation")