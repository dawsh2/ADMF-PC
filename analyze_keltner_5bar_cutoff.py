"""Analyze Keltner Bands with mandatory 5-bar exit for ALL trades"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_d5807cc2")
signal_file = workspace / "traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_0.parquet"

print("=== KELTNER BANDS WITH 5-BAR MANDATORY EXIT ===\n")

# Load signals
signals = pd.read_parquet(signal_file)

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})
spy_subset = spy_1m.iloc[:81787].copy()

# Calculate indicators
spy_subset['returns'] = spy_subset['close'].pct_change()
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_subset['vol_percentile'] = spy_subset['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# Collect trades with MANDATORY 5-bar exit
trades = []
entry_data = None

for i in range(len(signals)):
    curr = signals.iloc[i]
    
    if entry_data is None and curr['val'] != 0:
        entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
    
    elif entry_data is not None:
        # Check if we should exit
        should_exit = False
        bars_held = curr['idx'] - entry_data['idx']
        
        # Exit conditions:
        # 1. Natural exit (signal changes to 0 or opposite)
        # 2. MANDATORY exit after 5 bars
        if curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal']):
            should_exit = True
        elif bars_held >= 5:  # FORCE EXIT after 5 bars
            should_exit = True
            
        if should_exit and entry_data['idx'] < len(spy_subset) and curr['idx'] < len(spy_subset):
            # Use actual exit price at bar 5 if forced exit
            if bars_held >= 5 and curr['val'] == entry_data['signal']:
                # Find the exact 5-bar exit point
                exit_idx = entry_data['idx'] + 5
                if exit_idx < len(spy_subset):
                    exit_price = spy_subset.iloc[exit_idx]['close']
                else:
                    continue
            else:
                exit_price = curr['px']
                exit_idx = curr['idx']
            
            entry_conditions = spy_subset.iloc[entry_data['idx']]
            pct_return = (exit_price / entry_data['price'] - 1) * entry_data['signal'] * 100
            actual_duration = min(exit_idx - entry_data['idx'], 5)  # Cap at 5
            
            trade = {
                'pct_return': pct_return,
                'bps_return': pct_return,  # Already in percentage, same as bps
                'direction': 'short' if entry_data['signal'] < 0 else 'long',
                'duration': actual_duration,
                'forced_exit': bars_held >= 5 and curr['val'] == entry_data['signal'],
                'vol_percentile': entry_conditions.get('vol_percentile', 50),
                'entry_idx': entry_data['idx'],
                'exit_idx': exit_idx
            }
            trades.append(trade)
            
            # Clear entry after forced exit
            if bars_held >= 5:
                entry_data = None
            else:
                # Natural exit to opposite signal
                if curr['val'] != 0:
                    entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
                else:
                    entry_data = None

trades_df = pd.DataFrame(trades)
total_days = 81787 / 390

print(f"Total trades with 5-bar rule: {len(trades_df)}")
print(f"Average trades per day: {len(trades_df)/total_days:.1f}")

# Overall performance
avg_return_pct = trades_df['pct_return'].mean()
avg_return_bps = avg_return_pct  # Since we stored as percentage
print(f"\n=== OVERALL PERFORMANCE ===")
print(f"Average return per trade: {avg_return_pct:.4f}% = {avg_return_bps:.2f} basis points")
print(f"Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
print(f"Average duration: {trades_df['duration'].mean():.1f} bars")

# Separate natural vs forced exits
natural_exits = trades_df[~trades_df['forced_exit']]
forced_exits = trades_df[trades_df['forced_exit']]

print(f"\n=== EXIT TYPE ANALYSIS ===")
print(f"Natural exits: {len(natural_exits)} ({len(natural_exits)/len(trades_df)*100:.1f}%)")
print(f"  Return: {natural_exits['pct_return'].mean():.4f}% = {natural_exits['pct_return'].mean():.2f} bps")
print(f"  Win rate: {(natural_exits['pct_return'] > 0).mean():.1%}")

print(f"\nForced 5-bar exits: {len(forced_exits)} ({len(forced_exits)/len(trades_df)*100:.1f}%)")
print(f"  Return: {forced_exits['pct_return'].mean():.4f}% = {forced_exits['pct_return'].mean():.2f} bps")
print(f"  Win rate: {(forced_exits['pct_return'] > 0).mean():.1%}")

# By direction
print(f"\n=== PERFORMANCE BY DIRECTION (5-bar cutoff) ===")
for direction in ['long', 'short']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    if len(dir_trades) > 0:
        avg_pct = dir_trades['pct_return'].mean()
        print(f"\n{direction.upper()} trades: {len(dir_trades)}")
        print(f"  Average return: {avg_pct:.4f}% = {avg_pct:.2f} basis points")
        print(f"  Trades/day: {len(dir_trades)/total_days:.1f}")
        print(f"  Win rate: {(dir_trades['pct_return'] > 0).mean():.1%}")
        
        # Natural vs forced
        nat = dir_trades[~dir_trades['forced_exit']]
        forced = dir_trades[dir_trades['forced_exit']]
        if len(nat) > 0:
            print(f"  Natural exits: {nat['pct_return'].mean():.4f}% on {len(nat)} trades")
        if len(forced) > 0:
            print(f"  Forced exits: {forced['pct_return'].mean():.4f}% on {len(forced)} trades")

# Distribution analysis
print(f"\n=== RETURN DISTRIBUTION ===")
print(f"25th percentile: {trades_df['pct_return'].quantile(0.25):.4f}%")
print(f"Median: {trades_df['pct_return'].median():.4f}%")
print(f"75th percentile: {trades_df['pct_return'].quantile(0.75):.4f}%")
print(f"Std dev: {trades_df['pct_return'].std():.4f}%")

# Risk metrics
winners = trades_df[trades_df['pct_return'] > 0]
losers = trades_df[trades_df['pct_return'] < 0]
if len(winners) > 0 and len(losers) > 0:
    avg_win = winners['pct_return'].mean()
    avg_loss = abs(losers['pct_return'].mean())
    print(f"\n=== RISK METRICS ===")
    print(f"Average win: {avg_win:.4f}%")
    print(f"Average loss: {avg_loss:.4f}%")
    print(f"Win/Loss ratio: {avg_win/avg_loss:.2f}")

print(f"\n=== CONCLUSION ===")
print(f"With mandatory 5-bar exit:")
print(f"- Edge: {avg_return_bps:.2f} basis points per trade")
print(f"- Frequency: {len(trades_df)/total_days:.1f} trades per day")
print(f"- Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")

if avg_return_bps >= 1.0 and len(trades_df)/total_days >= 2.0:
    print(f"\n✓ MEETS your requirements of >=1 bps with 2-3+ trades/day")
else:
    print(f"\n✗ Does NOT meet requirements after proper 5-bar cutoff")
    if avg_return_bps < 1.0:
        print(f"  - Edge too low: {avg_return_bps:.2f} bps < 1 bps")
    if len(trades_df)/total_days < 2.0:
        print(f"  - Frequency too low: {len(trades_df)/total_days:.1f} tpd < 2 tpd")