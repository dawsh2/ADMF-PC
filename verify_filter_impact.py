#!/usr/bin/env python3
"""
Verify the filter impact by comparing workspaces.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_strategy(workspace_path: str, strategy_num: int = 4):
    """Analyze a specific strategy in a workspace."""
    signals_path = Path(workspace_path) / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands"
    strategy_file = signals_path / f"SPY_5m_compiled_strategy_{strategy_num}.parquet"
    
    if not strategy_file.exists():
        return None
    
    signals_df = pd.read_parquet(strategy_file)
    signals_df = signals_df.sort_values('idx').reset_index(drop=True)
    
    # Calculate returns
    trades = []
    current_position = None
    
    for i in range(len(signals_df)):
        row = signals_df.iloc[i]
        signal = row['val']
        price = row['px']
        
        if signal != 0:
            if current_position is not None:
                if current_position['direction'] == 'long':
                    ret = np.log(price / current_position['entry_price'])
                else:
                    ret = -np.log(price / current_position['entry_price'])
                trades.append(ret * 10000)
            
            current_position = {
                'entry_price': price,
                'direction': 'long' if signal > 0 else 'short'
            }
        elif signal == 0 and current_position is not None:
            if current_position['direction'] == 'long':
                ret = np.log(price / current_position['entry_price'])
            else:
                ret = -np.log(price / current_position['entry_price'])
            trades.append(ret * 10000)
            current_position = None
    
    if trades:
        exec_mult = 1 - (0.5 / 10000)
        adj_trades = [t * exec_mult for t in trades]
        wins = [t for t in adj_trades if t > 0]
        
        return {
            'signals': len(signals_df),
            'trades': len(trades),
            'rpt_bps': np.mean(adj_trades),
            'win_rate': len(wins) / len(trades),
            'total_return': np.exp(sum(t/10000 for t in adj_trades)) - 1
        }
    return None

print("=== Verifying Filter Impact ===\n")

# Analyze both workspaces
workspace1 = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"
workspace2 = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_5m_20250622_103740"

print("Workspace 1 (with filters):")
result1 = analyze_strategy(workspace1, 4)
if result1:
    print(f"  Trades: {result1['trades']}")
    print(f"  RPT: {result1['rpt_bps']:.2f} bps")
    print(f"  Win rate: {result1['win_rate']*100:.1f}%")
    print(f"  Total return: {result1['total_return']*100:.1f}%")

print("\nWorkspace 2 (without filters?):")
result2 = analyze_strategy(workspace2, 4)
if result2:
    print(f"  Trades: {result2['trades']}")
    print(f"  RPT: {result2['rpt_bps']:.2f} bps")
    print(f"  Win rate: {result2['win_rate']*100:.1f}%")
    print(f"  Total return: {result2['total_return']*100:.1f}%")

if result1 and result2:
    print(f"\nFilter impact multiplier: {result1['rpt_bps'] / result2['rpt_bps']:.1f}x")

# The key insight: Let's check if the 2.70 bps came from a DIFFERENT calculation method
print("\n=== Checking Alternative Calculation Methods ===")

# Maybe the previous analysis was using a different execution cost or fee structure
exec_costs = [0, 0.1, 0.5, 1.0, 2.0]
print("\nStrategy 4 performance with different execution costs:")
for cost in exec_costs:
    result = analyze_strategy(workspace1, 4)
    if result:
        # Recalculate with different cost
        signals_df = pd.read_parquet(Path(workspace1) / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands" / "SPY_5m_compiled_strategy_4.parquet")
        signals_df = signals_df.sort_values('idx').reset_index(drop=True)
        
        trades = []
        current_position = None
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            signal = row['val']
            price = row['px']
            
            if signal != 0:
                if current_position is not None:
                    if current_position['direction'] == 'long':
                        ret = np.log(price / current_position['entry_price'])
                    else:
                        ret = -np.log(price / current_position['entry_price'])
                    trades.append(ret * 10000)
                
                current_position = {
                    'entry_price': price,
                    'direction': 'long' if signal > 0 else 'short'
                }
            elif signal == 0 and current_position is not None:
                if current_position['direction'] == 'long':
                    ret = np.log(price / current_position['entry_price'])
                else:
                    ret = -np.log(price / current_position['entry_price'])
                trades.append(ret * 10000)
                current_position = None
        
        # Apply execution cost
        exec_mult = (1 - (cost / 10000)) ** 2  # Applied on entry AND exit
        adj_trades = [t * exec_mult for t in trades]
        
        print(f"  {cost} bps: {np.mean(adj_trades):.2f} bps/trade")

# Check if the 2.70 came from gross returns (no execution costs)
print(f"\nGross returns (no costs): {np.mean(trades):.2f} bps")
print(f"Number from earlier analysis: 2.70 bps")
print(f"Ratio: {2.70 / np.mean(trades):.2f}x")

# The mystery deepens - let's check if we're looking at the right file
print("\n=== Checking File Sizes ===")
import os
strategy_file = Path(workspace1) / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands" / "SPY_5m_compiled_strategy_4.parquet"
file_size = os.path.getsize(strategy_file)
print(f"Strategy 4 file size: {file_size:,} bytes")
print(f"Signal count: {len(signals_df)}")
print(f"Bytes per signal: {file_size / len(signals_df):.1f}")