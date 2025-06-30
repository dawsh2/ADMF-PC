#!/usr/bin/env python3
"""Debug why stop losses didn't trigger for specific trades."""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

workspace = Path("config/bollinger/results/latest")

# Load all data
signals = pd.read_parquet(workspace / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
positions_open = pd.read_parquet(workspace / "traces/portfolio/positions_open/positions_open.parquet")
positions_close = pd.read_parquet(workspace / "traces/portfolio/positions_close/positions_close.parquet")

# Convert timestamps
signals['ts'] = pd.to_datetime(signals['ts'])
positions_open['ts'] = pd.to_datetime(positions_open['ts'])
positions_close['ts'] = pd.to_datetime(positions_close['ts'])

# Focus on Position 1 that should have stopped at -0.119%
print("=== Analyzing Position 1 (Should have stopped) ===")
pos1_open = json.loads(positions_open.iloc[0]['metadata'])
pos1_close = json.loads(positions_close.iloc[0]['metadata'])

print(f"Entry: ${pos1_open['entry_price']} at {positions_open.iloc[0]['ts']}")
print(f"Exit: ${pos1_close['exit_price']} at {positions_close.iloc[0]['ts']}")
print(f"Return: {((pos1_close['exit_price'] - pos1_open['entry_price']) / pos1_open['entry_price'] * 100):.3f}%")
print(f"Stop should trigger at: ${pos1_open['entry_price'] * 0.999:.2f}")

# Find signals around this position
pos1_open_time = positions_open.iloc[0]['ts']
pos1_close_time = positions_close.iloc[0]['ts']

print(f"\n=== Signals during position ===")
during_signals = signals[(signals['ts'] >= pos1_open_time) & (signals['ts'] <= pos1_close_time)]
for _, sig in during_signals.iterrows():
    print(f"  {sig['ts']}: Signal={sig['val']}, Price=${sig['px']:.2f}")

# Load bar data to see all price movements
data_path = Path("./data/SPY_5m.csv")
if data_path.exists():
    bars = pd.read_csv(data_path)
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    
    # Convert position times to match bar data timezone
    # The signals show UTC times like "2024-03-26T19:30:00+00:00"
    # Find bars during the position
    print(f"\n=== Price bars during position ===")
    
    # Get the signal timestamps that opened and closed the position
    open_signal = signals[signals['ts'] == '2024-03-26T19:30:00+00:00'].iloc[0]
    close_signal = signals[signals['ts'] == '2024-03-26T19:50:00+00:00'].iloc[0]
    
    # Find bars between these times
    position_bars = bars[(bars['timestamp'] >= open_signal['ts']) & 
                        (bars['timestamp'] <= close_signal['ts'])]
    
    if len(position_bars) > 0:
        print(f"Found {len(position_bars)} bars")
        for _, bar in position_bars.iterrows():
            pnl_pct = ((bar['close'] - pos1_open['entry_price']) / pos1_open['entry_price'] * 100)
            print(f"  {bar['timestamp']}: O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f} (P&L: {pnl_pct:.3f}%)")
            
            # Check if low went below stop
            low_pnl = ((bar['low'] - pos1_open['entry_price']) / pos1_open['entry_price'] * 100)
            if low_pnl <= -0.1:
                print(f"    ⚠️ LOW BREACHED STOP: {low_pnl:.3f}%")
    else:
        print("No bars found in position timeframe")
        
# Check the actual problem
print("\n=== Key Issue ===")
print("The position lasted from 19:30 to 19:50 (20 minutes)")
print("With 5-minute bars, we should have seen 4 bars")
print("But the position shows bars_held=0, suggesting it closed before the first bar completed")
print("\nThis means:")
print("1. The strategy generated an entry signal at 19:30")
print("2. The strategy generated an exit signal at 19:50") 
print("3. The portfolio likely didn't receive any BAR events between these signals")
print("4. So the exit condition checking on BAR events never ran!")