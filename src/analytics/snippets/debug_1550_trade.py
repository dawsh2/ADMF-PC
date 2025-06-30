# Debug why execution has an extra trade at 15:50 on 2024-03-26
import pandas as pd
from pathlib import Path

print("🔍 DEBUGGING EXTRA TRADE AT 15:50 on 2024-03-26")
print("="*80)

# Load the signal trace for this strategy
run_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250627_185448')
strategy_index = pd.read_parquet(run_dir / 'strategy_index.parquet')

# Find the strategy
strategy_hash = '5edc43651004'
strategy_info = strategy_index[strategy_index['strategy_hash'] == strategy_hash].iloc[0]
trace_path = strategy_info['trace_path']

# Load signals
signals = pd.read_parquet(run_dir / trace_path)
signals['ts'] = pd.to_datetime(signals['ts'])

# Load market data
market_data = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_5m.csv')
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])

# Focus on the time period around the issue
start_time = pd.to_datetime('2024-03-26 15:30').tz_localize('UTC')
end_time = pd.to_datetime('2024-03-26 16:00').tz_localize('UTC')

# Get signals in this window
window_signals = signals[(signals['ts'] >= start_time) & (signals['ts'] <= end_time)].copy()
print(f"Signals between {start_time} and {end_time}:")
print(window_signals[['ts', 'val', 'px']].to_string())

# Get market data in this window  
window_market = market_data[(market_data['timestamp'] >= start_time) & (market_data['timestamp'] <= end_time)].copy()

# Merge to see signal changes
merged = window_market.merge(window_signals[['ts', 'val', 'px']], left_on='timestamp', right_on='ts', how='left')
merged['signal'] = merged['val'].ffill()
merged['signal_change'] = merged['signal'].diff()

print("\n📊 Market data with signals:")
print(merged[['timestamp', 'close', 'signal', 'signal_change', 'px']].to_string())

print("\n💡 ANALYSIS:")
print("The issue appears to be at these timestamps:")

# Find signal changes
signal_changes = merged[merged['signal_change'] != 0].dropna()
for _, row in signal_changes.iterrows():
    print(f"  {row['timestamp']}: Signal changed from {row['signal'] - row['signal_change']:.0f} to {row['signal']:.0f}")

print("\n🐛 HYPOTHESIS:")
print("If the execution engine is reacting to every signal value (even 0),")
print("it might be creating trades that the analysis ignores.")
print("The analysis might be filtering out rapid signal changes or 0 signals.")