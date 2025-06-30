# Debug version to understand trailing stop behavior
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250625_173629')
SIGNAL_DIR = RESULTS_DIR / 'traces/signals/bollinger_bands'
DATA_DIR = Path('/Users/daws/ADMF-PC/data')

# Load market data
print("Loading market data...")
market_data = pd.read_csv(DATA_DIR / 'SPY_5m.csv')
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], utc=True).dt.tz_localize(None)

# Load first signal file
signal_files = list(SIGNAL_DIR.glob('*.parquet'))
if not signal_files:
    print("No signal files found!")
    exit()

df = pd.read_parquet(signal_files[0])
df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize(None)
df = df.sort_values('ts')

print(f"\nAnalyzing {signal_files[0].name}")
print(f"Total signals: {len(df)}")
print(f"Non-zero signals: {(df['val'] != 0).sum()}")

# Extract first few trades with detailed logging
trades_analyzed = 0
current_pos = 0
entry = None

# Test config: 0.1% initial stop, 0.05% trail
initial_stop_pct = 0.1
trail_stop_pct = 0.05
initial_target_pct = 0.15

for idx, row in df.iterrows():
    signal = row['val']
    
    if current_pos == 0 and signal != 0:
        # Entry
        current_pos = signal
        entry = {
            'time': row['ts'], 
            'price': row['px'], 
            'direction': signal
        }
        
    elif current_pos != 0 and signal != current_pos:
        # Exit
        if entry and trades_analyzed < 5:  # Analyze first 5 trades
            trades_analyzed += 1
            print(f"\n{'='*60}")
            print(f"TRADE {trades_analyzed}")
            print(f"Entry: {entry['time']} @ ${entry['price']:.2f}")
            print(f"Direction: {'LONG' if entry['direction'] > 0 else 'SHORT'}")
            
            # Find market data
            mask = (market_data['timestamp'] >= entry['time']) & (market_data['timestamp'] <= row['ts'])
            trade_bars = market_data[mask]
            
            if len(trade_bars) > 0 and entry['direction'] > 0:  # Focus on longs
                entry_price = entry['price']
                initial_stop_level = entry_price * (1 - initial_stop_pct/100)
                initial_target_level = entry_price * (1 + initial_target_pct/100)
                
                print(f"Initial stop: ${initial_stop_level:.2f} ({initial_stop_pct}% below)")
                print(f"Initial target: ${initial_target_level:.2f} ({initial_target_pct}% above)")
                print(f"\nPrice movement:")
                
                stop_price = initial_stop_level
                highest_price = entry_price
                stop_has_trailed = False
                
                for bar_idx, (_, bar) in enumerate(trade_bars.iterrows()):
                    if bar_idx < 10 or bar_idx % 10 == 0:  # Show first 10 bars then every 10th
                        print(f"  Bar {bar_idx}: H=${bar['high']:.2f}, L=${bar['low']:.2f}", end="")
                        
                        # Check if price moved up
                        if bar['high'] > highest_price:
                            old_highest = highest_price
                            highest_price = bar['high']
                            
                            # Calculate new trailing stop
                            new_stop = highest_price * (1 - trail_stop_pct/100)
                            if new_stop > stop_price:
                                print(f" → New high! Stop: ${stop_price:.2f} → ${new_stop:.2f}")
                                stop_price = new_stop
                                stop_has_trailed = True
                            else:
                                print(f" → New high ${highest_price:.2f} but stop unchanged")
                        else:
                            print()
                        
                        # Check if stopped out
                        if bar['low'] <= stop_price:
                            print(f"\n  STOPPED OUT at bar {bar_idx}!")
                            print(f"  Exit price: ${stop_price:.2f}")
                            print(f"  Exit type: {'TRAILING STOP' if stop_has_trailed else 'REGULAR STOP'}")
                            break
                        
                        # Check if target hit
                        if bar['high'] >= initial_target_level:
                            print(f"\n  TARGET HIT at bar {bar_idx}!")
                            print(f"  Exit price: ${initial_target_level:.2f}")
                            break
                
                # Calculate the minimum price move needed to start trailing
                min_move_to_trail = (initial_stop_level / (1 - trail_stop_pct/100)) - entry_price
                min_move_pct = min_move_to_trail / entry_price * 100
                print(f"\nMinimum price move to start trailing: ${min_move_to_trail:.4f} ({min_move_pct:.4f}%)")
                
        # Update position
        current_pos = signal
        if signal != 0:
            entry = {'time': row['ts'], 'price': row['px'], 'direction': signal}
        else:
            entry = None
            
        if trades_analyzed >= 5:
            break

# Analysis of the trailing mechanism
print(f"\n{'='*60}")
print("TRAILING STOP ANALYSIS")
print(f"{'='*60}")

print(f"\nWith initial stop {initial_stop_pct}% and trail {trail_stop_pct}%:")
print(f"- The stop trails {trail_stop_pct}% below the highest price")
print(f"- But it starts at {initial_stop_pct}% below entry")

# Calculate crossover point
# At what price increase does trailing stop = initial stop?
# Entry * (1 + X) * (1 - trail%) = Entry * (1 - initial%)
# 1 + X = (1 - initial%) / (1 - trail%)
crossover_increase = (1 - initial_stop_pct/100) / (1 - trail_stop_pct/100) - 1
print(f"\nPrice must increase {crossover_increase*100:.4f}% before trailing stop moves above initial stop")

# Example
entry_example = 100
print(f"\nExample with entry at ${entry_example}:")
print(f"- Initial stop: ${entry_example * (1 - initial_stop_pct/100):.2f}")
print(f"- Price needs to reach ${entry_example * (1 + crossover_increase):.2f} for trailing to begin")
print(f"- At that point, trailing stop = ${entry_example * (1 + crossover_increase) * (1 - trail_stop_pct/100):.2f}")