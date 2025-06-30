#!/usr/bin/env python3
"""Analyze stop loss and trade management for Keltner strategies"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Let's analyze both training and test results
datasets = [
    ("Training", "config/keltner/results/20250622_215020", "SPY_5m_compiled_strategy_209"),  # P=22, M=0.5
    ("Test", "config/keltner/test_top10/results/20250622_220133", "SPY_5m_kb_p22_m05")
]

print("STOP LOSS AND TRADE MANAGEMENT ANALYSIS")
print("="*80)

for dataset_name, results_dir, strategy_file in datasets:
    print(f"\n{dataset_name.upper()} DATASET:")
    print("-"*80)
    
    # Load trace
    trace_file = Path(results_dir) / "traces" / "mean_reversion" / f"{strategy_file}.parquet"
    if not trace_file.exists():
        trace_file = Path(results_dir) / "traces" / "keltner_bands" / f"{strategy_file}.parquet"
    
    df_trace = pd.read_parquet(trace_file)
    
    # Load SPY data
    spy_data = pd.read_csv("data/SPY_5m.csv")
    
    # Analyze each trade in detail
    trades = []
    in_trade = False
    
    for i in range(len(df_trace)):
        current_signal = df_trace.iloc[i]['val']
        current_idx = df_trace.iloc[i]['idx']
        current_price = df_trace.iloc[i]['px']
        
        if not in_trade and current_signal != 0:
            # Entry
            in_trade = True
            entry_idx = current_idx
            entry_price = current_price
            entry_signal = current_signal
            trade_bars = 0
            high_water = current_price if entry_signal > 0 else current_price
            low_water = current_price if entry_signal < 0 else current_price
            
        elif in_trade and current_signal == 0:
            # Exit
            in_trade = False
            exit_idx = current_idx
            exit_price = current_price
            
            # Calculate returns
            if entry_signal > 0:  # Long
                trade_return = (exit_price - entry_price) / entry_price
                max_profit = (high_water - entry_price) / entry_price
                max_loss = (low_water - entry_price) / entry_price
            else:  # Short
                trade_return = (entry_price - exit_price) / entry_price
                max_profit = (entry_price - low_water) / entry_price
                max_loss = (entry_price - high_water) / entry_price
            
            # Store trade info
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'duration_bars': exit_idx - entry_idx,
                'direction': 'Long' if entry_signal > 0 else 'Short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': trade_return,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'profit_given_back': max_profit - trade_return if trade_return > 0 else 0,
                'loss_avoided': trade_return - max_loss if trade_return < 0 else 0
            })
            
        elif in_trade:
            # Update high/low water marks during trade
            if i + 1 < len(df_trace):
                next_idx = df_trace.iloc[i + 1]['idx']
                # Update prices between sparse points
                for j in range(current_idx, min(next_idx, len(spy_data))):
                    if j < len(spy_data):
                        price = spy_data.iloc[j]['close']
                        if entry_signal > 0:  # Long
                            high_water = max(high_water, price)
                            low_water = min(low_water, price)
                        else:  # Short
                            high_water = max(high_water, price)
                            low_water = min(low_water, price)
    
    # Convert to DataFrame
    df_trades = pd.DataFrame(trades)
    
    if len(df_trades) == 0:
        print("No trades found!")
        continue
    
    # Basic statistics
    print(f"\nTotal trades: {len(df_trades)}")
    print(f"Win rate: {(df_trades['return'] > 0).mean() * 100:.1f}%")
    print(f"Average return: {df_trades['return'].mean() * 100:.4f}%")
    print(f"Average duration: {df_trades['duration_bars'].mean():.1f} bars")
    
    # Separate winners and losers
    winners = df_trades[df_trades['return'] > 0]
    losers = df_trades[df_trades['return'] <= 0]
    
    print(f"\nWINNERS ({len(winners)} trades):")
    if len(winners) > 0:
        print(f"  Average return: {winners['return'].mean() * 100:.4f}%")
        print(f"  Average max profit: {winners['max_profit'].mean() * 100:.4f}%")
        print(f"  Average profit given back: {winners['profit_given_back'].mean() * 100:.4f}%")
        print(f"  % of max profit captured: {(winners['return'] / winners['max_profit']).mean() * 100:.1f}%")
        print(f"  Average duration: {winners['duration_bars'].mean():.1f} bars")
    
    print(f"\nLOSERS ({len(losers)} trades):")
    if len(losers) > 0:
        print(f"  Average return: {losers['return'].mean() * 100:.4f}%")
        print(f"  Average max loss: {losers['max_loss'].mean() * 100:.4f}%")
        print(f"  Average loss avoided: {losers['loss_avoided'].mean() * 100:.4f}%")
        print(f"  % of max loss realized: {(losers['return'] / losers['max_loss']).mean() * 100:.1f}%")
        print(f"  Average duration: {losers['duration_bars'].mean():.1f} bars")
    
    # Distribution analysis
    print(f"\nRETURN DISTRIBUTION:")
    print(f"  Best trade: {df_trades['return'].max() * 100:.2f}%")
    print(f"  Worst trade: {df_trades['return'].min() * 100:.2f}%")
    print(f"  Std dev: {df_trades['return'].std() * 100:.2f}%")
    print(f"  Skew: {df_trades['return'].skew():.2f}")
    
    # Check for stop loss patterns
    print(f"\nPOTENTIAL STOP LOSS ISSUES:")
    
    # Large losses that could have been stopped
    large_losses = losers[losers['return'] < -0.01]  # Losses worse than -1%
    if len(large_losses) > 0:
        print(f"  Trades with >1% loss: {len(large_losses)}")
        print(f"  Average max loss before exit: {large_losses['max_loss'].mean() * 100:.2f}%")
        print(f"  These could save: {(large_losses['return'] - large_losses['max_loss']).sum() * 100:.2f}% total")
    
    # Winners that gave back too much
    greedy_winners = winners[winners['profit_given_back'] > winners['return'] * 0.5]
    if len(greedy_winners) > 0:
        print(f"\n  Winners giving back >50% of profit: {len(greedy_winners)}")
        print(f"  Average profit given back: {greedy_winners['profit_given_back'].mean() * 100:.2f}%")
        print(f"  Potential gain from better exits: {greedy_winners['profit_given_back'].sum() * 100:.2f}% total")
    
    # Long vs Short analysis
    print(f"\nDIRECTIONAL ANALYSIS:")
    for direction in ['Long', 'Short']:
        dir_trades = df_trades[df_trades['direction'] == direction]
        if len(dir_trades) > 0:
            print(f"  {direction}: {len(dir_trades)} trades, "
                  f"Win rate: {(dir_trades['return'] > 0).mean() * 100:.1f}%, "
                  f"Avg return: {dir_trades['return'].mean() * 100:.4f}%")

# Save results for comparison
all_results = {}

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)