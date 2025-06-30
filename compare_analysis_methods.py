#!/usr/bin/env python3
"""
Compare different analysis methods to understand the discrepancy.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load Strategy 4 signals
workspace = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"
signals_path = Path(workspace) / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands"
strategy_file = signals_path / "SPY_5m_compiled_strategy_4.parquet"

signals_df = pd.read_parquet(strategy_file)
signals_df = signals_df.sort_values('idx').reset_index(drop=True)

print("=== Comparing Analysis Methods ===\n")

# Method 1: Using only position changes (ignoring flat signals in the middle)
print("Method 1: Count only position changes")
trades = []
current_position = None

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    signal = row['val']
    price = row['px']
    
    # Only process if signal changes
    if signal != 0 and (current_position is None or signal != current_position['signal']):
        if current_position is not None:
            # Close existing position
            if current_position['direction'] == 'long':
                ret = np.log(price / current_position['entry_price'])
            else:
                ret = -np.log(price / current_position['entry_price'])
            trades.append(ret * 10000)  # Convert to bps
        
        # Open new position
        current_position = {
            'entry_price': price,
            'direction': 'long' if signal > 0 else 'short',
            'signal': signal
        }
    elif signal == 0 and current_position is not None:
        # Exit signal
        if current_position['direction'] == 'long':
            ret = np.log(price / current_position['entry_price'])
        else:
            ret = -np.log(price / current_position['entry_price'])
        trades.append(ret * 10000)
        current_position = None

exec_mult = 1 - (0.5 / 10000)  # 0.5 bps execution cost
adj_trades = [t * exec_mult for t in trades]

print(f"Trades: {len(trades)}")
print(f"Raw avg return: {np.mean(trades):.2f} bps")
print(f"With 0.5 bps cost: {np.mean(adj_trades):.2f} bps")
print(f"Win rate: {len([t for t in adj_trades if t > 0]) / len(adj_trades) * 100:.1f}%")

# Method 2: Look for the 2.70 bps by checking if there's filtering
print("\nMethod 2: Check if there's additional filtering happening")

# Count actual signal transitions
transitions = 0
last_val = None
for _, row in signals_df.iterrows():
    if last_val is None or row['val'] != last_val:
        transitions += 1
        last_val = row['val']

print(f"Total signal transitions: {transitions}")
print(f"Total signals: {len(signals_df)}")
print(f"Unique signal values: {signals_df['val'].unique()}")

# Method 3: Check if the high performance comes from specific periods
print("\nMethod 3: Performance by time periods")

# Group by rough time periods (every 1000 bars ~ 13 days)
signals_df['period'] = signals_df['idx'] // 1000

period_returns = []
for period, group in signals_df.groupby('period'):
    if len(group) < 2:
        continue
    
    # Calculate returns for this period
    period_trades = []
    pos = None
    
    for _, row in group.iterrows():
        signal = row['val']
        price = row['px']
        
        if signal != 0 and (pos is None or signal != pos['signal']):
            if pos is not None:
                if pos['direction'] == 'long':
                    ret = np.log(price / pos['entry_price'])
                else:
                    ret = -np.log(price / pos['entry_price'])
                period_trades.append(ret * 10000)
            
            pos = {'entry_price': price, 'direction': 'long' if signal > 0 else 'short', 'signal': signal}
        elif signal == 0 and pos is not None:
            if pos['direction'] == 'long':
                ret = np.log(price / pos['entry_price'])
            else:
                ret = -np.log(price / pos['entry_price'])
            period_trades.append(ret * 10000)
            pos = None
    
    if period_trades:
        avg_ret = np.mean(period_trades)
        period_returns.append((period, len(period_trades), avg_ret))

# Print best and worst periods
period_returns.sort(key=lambda x: x[2], reverse=True)
print("\nBest 5 periods:")
for period, trades, ret in period_returns[:5]:
    print(f"  Period {period}: {trades} trades, {ret:.2f} bps/trade")

print("\nWorst 5 periods:")
for period, trades, ret in period_returns[-5:]:
    print(f"  Period {period}: {trades} trades, {ret:.2f} bps/trade")

# Method 4: Check if we're missing a multiplier or scaling factor
print("\nMethod 4: Check for possible scaling differences")
print(f"Average price level: ${signals_df['px'].mean():.2f}")
print(f"Price range: ${signals_df['px'].min():.2f} - ${signals_df['px'].max():.2f}")

# If returns were calculated as simple percentage instead of log returns
simple_pct_trades = []
current_position = None

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    signal = row['val']
    price = row['px']
    
    if signal != 0 and (current_position is None or signal != current_position['signal']):
        if current_position is not None:
            # Using simple percentage returns
            if current_position['direction'] == 'long':
                ret = (price / current_position['entry_price'] - 1)
            else:
                ret = -(price / current_position['entry_price'] - 1)
            simple_pct_trades.append(ret * 10000)  # Convert to bps
        
        current_position = {
            'entry_price': price,
            'direction': 'long' if signal > 0 else 'short',
            'signal': signal
        }

print(f"\nIf using simple % returns instead of log returns:")
print(f"Avg return: {np.mean(simple_pct_trades):.2f} bps")

# The 2.70 might come from a subset or different calculation
print(f"\n=== Checking for 2.70 bps match ===")
print(f"0.42 * 6.4 = {0.42 * 6.4:.2f} (if 6.4x multiplier)")
print(f"0.42 * 2.7 / 0.45 = {0.42 * 2.7 / 0.45:.2f} (ratio adjustment)")
print(f"Filtered trades (IQR): {np.mean(filtered_returns):.2f} bps (from earlier)")

# Check if certain trade characteristics match 2.70
long_trades = [t for i, t in enumerate(adj_trades) if i < len(trades) and signals_df.iloc[i]['val'] > 0]
short_trades = [t for i, t in enumerate(adj_trades) if i < len(trades) and signals_df.iloc[i]['val'] < 0]

if long_trades:
    print(f"\nLong trades only: {np.mean(long_trades):.2f} bps")
if short_trades:
    print(f"Short trades only: {np.mean(short_trades):.2f} bps")