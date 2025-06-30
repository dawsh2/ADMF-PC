# Debug the signal flipping at 15:50
import pandas as pd
from pathlib import Path

print("🔍 DEBUGGING SIGNAL FLIPPING AT 15:50")
print("="*80)

# Load signals
run_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250627_185448')
strategy_index = pd.read_parquet(run_dir / 'strategy_index.parquet')
strategy_info = strategy_index[strategy_index['strategy_hash'] == '5edc43651004'].iloc[0]
signals = pd.read_parquet(run_dir / strategy_info['trace_path'])
signals['ts'] = pd.to_datetime(signals['ts'])

# Load market data
market_data = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_5m.csv')
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])

# Focus on the period around 15:50
start_time = pd.to_datetime('2024-03-26 15:30').tz_localize('UTC')
end_time = pd.to_datetime('2024-03-26 16:10').tz_localize('UTC')

# Merge signals with market data
window_data = market_data[(market_data['timestamp'] >= start_time) & (market_data['timestamp'] <= end_time)].copy()
window_signals = signals[(signals['ts'] >= start_time) & (signals['ts'] <= end_time)]

merged = window_data.merge(window_signals[['ts', 'val', 'px']], left_on='timestamp', right_on='ts', how='left')
merged['signal'] = merged['val'].ffill().fillna(0)
merged['position'] = merged['signal'].replace({0: 0, 1: 1, -1: -1})
merged['position_change'] = merged['position'].diff().fillna(0)

print("Market data with signals and position changes:")
print(merged[['timestamp', 'close', 'signal', 'position', 'position_change']].to_string())

print("\n💡 ANALYSIS LOGIC CHECK:")

# Simulate the analysis logic
current_trade = None
trades_created = []

for idx, row in merged.iterrows():
    # New position opened
    if row['position_change'] != 0 and row['position'] != 0:
        if current_trade is None:  # KEY: Only if no current trade
            current_trade = {
                'time': row['timestamp'],
                'direction': 'LONG' if row['position'] == 1 else 'SHORT',
                'price': row['close']
            }
            trades_created.append(f"{row['timestamp'].strftime('%H:%M')} - ENTER {current_trade['direction']} at {row['close']:.2f}")
        else:
            trades_created.append(f"{row['timestamp'].strftime('%H:%M')} - IGNORED {row['position']} signal (already in trade)")
    
    # Position closed
    elif current_trade is not None and (row['position'] == 0 or row['position_change'] != 0):
        if row['position'] == 0:
            trades_created.append(f"{row['timestamp'].strftime('%H:%M')} - EXIT (signal=0) at {row['close']:.2f}")
        else:
            trades_created.append(f"{row['timestamp'].strftime('%H:%M')} - EXIT (reversal) at {row['close']:.2f}")
        current_trade = None
        
        # Check if we're also entering a new position
        if row['position'] != 0 and row['position_change'] != 0:
            current_trade = {
                'time': row['timestamp'],
                'direction': 'LONG' if row['position'] == 1 else 'SHORT',
                'price': row['close']
            }
            trades_created.append(f"                  - ENTER {current_trade['direction']} at {row['close']:.2f}")

print("\nAnalysis would create these trades:")
for trade in trades_created:
    print(f"  {trade}")

print("\n✅ So the analysis DOES process signal changes, but:")
print("1. It only enters if current_trade is None")
print("2. At 15:50, it would exit the LONG and NOT enter SHORT (because it's in the same bar)")
print("3. At 15:55, signal goes to 0, so no position")
print("\nThe execution engine must be immediately entering the SHORT at 15:50.")