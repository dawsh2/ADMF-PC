#!/usr/bin/env python3
"""Compare current results with notebook expectations."""

import pandas as pd
import json
from pathlib import Path

# Load latest results
results_dir = Path("config/bollinger/results/latest")

# Load position data
positions_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"
positions_df = pd.read_parquet(positions_file)

print("=== Current System Performance ===")
print(f"Total trades: {len(positions_df)}")

# Calculate returns and metrics
returns = []
exit_types = {'signal': 0, 'stop_loss': 0, 'take_profit': 0}

for _, row in positions_df.iterrows():
    metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
    
    entry_price = metadata.get('entry_price', 0)
    exit_price = metadata.get('exit_price', 0)
    exit_type = metadata.get('exit_type', 'signal')
    
    if entry_price > 0:
        ret = (exit_price - entry_price) / entry_price
        # Subtract 1bp execution cost
        net_ret = ret - 0.0001
        returns.append(net_ret)
        exit_types[exit_type] = exit_types.get(exit_type, 0) + 1

# Calculate metrics
total_return = (1 + pd.Series(returns)).prod() - 1
avg_return = pd.Series(returns).mean()
win_rate = (pd.Series(returns) > 0).mean()
sharpe = pd.Series(returns).mean() / pd.Series(returns).std() * (252 * len(returns) / 252) ** 0.5 if pd.Series(returns).std() > 0 else 0

print(f"\nPerformance Metrics:")
print(f"Total Return: {total_return*100:.2f}%")
print(f"Average Return per Trade: {avg_return*100:.3f}%")
print(f"Win Rate: {win_rate*100:.1f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")

print(f"\nExit Type Distribution:")
total = sum(exit_types.values())
for exit_type, count in exit_types.items():
    print(f"  {exit_type}: {count} ({count/total*100:.1f}%)")

print("\n=== Notebook Expected Performance ===")
print("With 0.075% stop loss and 0.1% profit target:")
print("Total Return: 10.27%")
print("Sharpe Ratio: 18.12")
print("Win Rate: 91.1%")
print("Avg Return per Trade: 0.006%")

print("\n=== Key Differences ===")
print(f"1. Trade count: {len(positions_df)} vs 416 expected")
print(f"2. Your avg return/trade: {avg_return*100:.3f}% vs 0.006% expected")
print(f"3. Your win rate: {win_rate*100:.1f}% vs 91.1% expected")

# Check if we're getting too many signals
print("\n=== Signal Generation Analysis ===")
# Group trades by day to see frequency
positions_df['timestamp'] = pd.to_datetime(positions_df['timestamp'])
positions_df['date'] = positions_df['timestamp'].dt.date
trades_per_day = positions_df.groupby('date').size()
print(f"Average trades per day: {trades_per_day.mean():.1f}")
print(f"Max trades per day: {trades_per_day.max()}")
print(f"Trading days: {len(trades_per_day)}")

# Look at trade durations
durations = []
for _, row in positions_df.iterrows():
    metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
    if 'bars_held' in metadata:
        durations.append(metadata['bars_held'])

if durations:
    print(f"\nTrade Duration Analysis:")
    print(f"Average bars held: {pd.Series(durations).mean():.1f}")
    print(f"Median bars held: {pd.Series(durations).median():.0f}")
    print(f"Min bars held: {min(durations)}")
    print(f"Max bars held: {max(durations)}")