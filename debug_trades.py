#!/usr/bin/env python3
"""Debug trade calculations to understand the metrics."""

from src.analytics.signal_reconstruction import SignalReconstructor
import json

# Load and analyze
signal_file = "workspaces/tmp/20250611_171158/signals_strategy_SPY_ma_crossover_5_20_20250611_171158.json"
market_data = "data/SPY_1m.csv"

reconstructor = SignalReconstructor(signal_file, market_data)

# Extract trades with details
trades = reconstructor.extract_trades()

print("Individual Trades:")
print("-" * 80)
for i, trade in enumerate(trades):
    print(f"\nTrade {i+1}:")
    print(f"  Direction: {trade.direction}")
    print(f"  Entry: Bar {trade.entry_bar} @ ${trade.entry_price:.2f}")
    print(f"  Exit: Bar {trade.exit_bar} @ ${trade.exit_price:.2f}") 
    print(f"  Bars held: {trade.bars_held}")
    print(f"  P&L: ${trade.pnl:.4f} ({trade.pnl_pct:.4f}%)")
    print(f"  Winner: {trade.is_winner}")

# Calculate metrics manually
winners = [t for t in trades if t.is_winner]
losers = [t for t in trades if not t.is_winner]

gross_profit = sum(t.pnl for t in winners)
gross_loss = abs(sum(t.pnl for t in losers))
total_pnl = sum(t.pnl for t in trades)

print(f"\nMetrics Breakdown:")
print(f"Total trades: {len(trades)}")
print(f"Winners: {len(winners)}")
print(f"Losers: {len(losers)}")
print(f"Gross profit: ${gross_profit:.4f}")
print(f"Gross loss: ${gross_loss:.4f}")
print(f"Total P&L: ${total_pnl:.4f}")
print(f"Profit factor: {gross_profit/gross_loss if gross_loss > 0 else 'inf':.2f}")

# Check signal frequency vs position time
with open(signal_file, 'r') as f:
    data = json.load(f)

total_bars = data['metadata']['total_bars']
signal_changes = data['metadata']['total_changes']
bars_in_position = sum(t.bars_held for t in trades)

print(f"\nSignal Analysis:")
print(f"Total bars: {total_bars}")
print(f"Signal changes: {signal_changes}")
print(f"Compression ratio: {signal_changes/total_bars:.1%}")
print(f"Bars in position: {bars_in_position}")
print(f"True position frequency: {bars_in_position/total_bars:.1%}")