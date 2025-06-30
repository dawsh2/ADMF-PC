#!/usr/bin/env python3
"""
Compare the two Bollinger Bands implementations
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load both signal files
old_signals = pd.read_parquet("/Users/daws/ADMF-PC/workspaces/signal_generation_f88793ad/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")
new_signals = pd.read_parquet("/Users/daws/ADMF-PC/workspaces/signal_generation_cc984d99/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")

print("="*60)
print("IMPLEMENTATION COMPARISON")
print("="*60)

print(f"\nOld implementation: {len(old_signals)} signals")
print(f"New implementation: {len(new_signals)} signals")

# Load price data for context
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
prices['bb_position'] = (prices['Close'] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])

# Analyze entry patterns for both
def analyze_implementation(signals, name):
    print(f"\n{name} IMPLEMENTATION:")
    print("-" * 40)
    
    # Extract trades
    trades = []
    entry_idx = None
    entry_signal = None
    
    for _, row in signals.iterrows():
        signal = row['val']
        bar_idx = row['idx']
        
        if entry_idx is None and signal != 0:
            if bar_idx < len(prices):
                entry_row = prices.iloc[bar_idx]
                entry_idx = bar_idx
                entry_signal = signal
                entry_price = row['px']
                entry_bb_pos = entry_row['bb_position']
                
        elif entry_idx is not None and (signal == 0 or signal != entry_signal):
            # Exit
            if entry_signal > 0:
                pnl_pct = (row['px'] - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - row['px']) / entry_price * 100
                
            trades.append({
                'duration': bar_idx - entry_idx,
                'pnl_pct': pnl_pct,
                'signal_type': 'long' if entry_signal > 0 else 'short',
                'entry_bb_position': entry_bb_pos
            })
            
            # Check for re-entry
            if signal != 0:
                if bar_idx < len(prices):
                    entry_row = prices.iloc[bar_idx]
                    entry_idx = bar_idx
                    entry_signal = signal
                    entry_price = row['px']
                    entry_bb_pos = entry_row['bb_position']
            else:
                entry_idx = None
    
    trades_df = pd.DataFrame(trades)
    
    # Entry position analysis
    print("\nEntry BB positions:")
    if 'entry_bb_position' in trades_df.columns:
        bb_buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        trades_df['bb_bucket'] = pd.cut(trades_df['entry_bb_position'], bins=bb_buckets)
        print(trades_df['bb_bucket'].value_counts().sort_index())
    
    # Duration analysis
    print("\nDuration analysis (first 10 durations):")
    for d in range(1, 11):
        d_trades = trades_df[trades_df['duration'] == d]
        if len(d_trades) > 0:
            net = d_trades['pnl_pct'].sum() - len(d_trades) * 0.01
            win_rate = (d_trades['pnl_pct'] > 0).mean()
            avg_pnl = d_trades['pnl_pct'].mean()
            print(f"  Duration {d:2d}: {len(d_trades):4d} trades, {win_rate:5.1%} win, "
                  f"{avg_pnl:6.3f}% avg, {net:7.2f}% net")
    
    # What makes 2-5 bar trades work/fail?
    trades_2_5 = trades_df[trades_df['duration'].between(2, 5)]
    if len(trades_2_5) > 100:
        print(f"\n2-5 bar trade analysis ({len(trades_2_5)} trades):")
        
        # By BB position
        for interval in [(0, 0.2), (0.2, 0.8), (0.8, 1.0)]:
            bb_trades = trades_2_5[trades_2_5['entry_bb_position'].between(interval[0], interval[1])]
            if len(bb_trades) > 10:
                net = bb_trades['pnl_pct'].sum() - len(bb_trades) * 0.01
                win_rate = (bb_trades['pnl_pct'] > 0).mean()
                print(f"  BB position {interval}: {len(bb_trades):4d} trades, "
                      f"{win_rate:5.1%} win, {net:7.2f}% net")
    
    return trades_df

# Analyze both
old_trades = analyze_implementation(old_signals, "OLD")
new_trades = analyze_implementation(new_signals, "NEW")

# Direct comparison
print("\n" + "="*60)
print("KEY DIFFERENCES")
print("="*60)

print("\n1. Entry distribution:")
old_middle = len(old_trades[old_trades['entry_bb_position'].between(0.2, 0.8)]) / len(old_trades) * 100
new_middle = len(new_trades[new_trades['entry_bb_position'].between(0.2, 0.8)]) / len(new_trades) * 100
print(f"  Old: {old_middle:.1f}% entries in middle zone (0.2-0.8)")
print(f"  New: {new_middle:.1f}% entries in middle zone (0.2-0.8)")

print("\n2. Average trade duration:")
print(f"  Old: {old_trades['duration'].mean():.1f} bars")
print(f"  New: {new_trades['duration'].mean():.1f} bars")

print("\n3. Why old 2-5 bar trades worked:")
# Sample some winning 2-5 bar trades from old
old_2_5_winners = old_trades[(old_trades['duration'].between(2, 5)) & (old_trades['pnl_pct'] > 0)]
if len(old_2_5_winners) > 0:
    print(f"  {len(old_2_5_winners)} winning 2-5 bar trades in old version")
    print(f"  Average entry BB position: {old_2_5_winners['entry_bb_position'].mean():.3f}")
    print(f"  Entry position distribution:")
    for interval in [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]:
        count = len(old_2_5_winners[old_2_5_winners['entry_bb_position'].between(interval[0], interval[1])])
        print(f"    {interval}: {count} trades ({count/len(old_2_5_winners)*100:.1f}%)")