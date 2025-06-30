# Debug why Trade #3 doesn't hit its target
import pandas as pd
from pathlib import Path

print("🔍 DEBUGGING TRADE #3 TARGET MISS")
print("="*80)

# Trade #3 details from analysis:
# Entry: 2024-03-26 19:40, LONG at 519.19
# Should exit at TARGET: 519.71 (0.1% above entry)
# But execution exits at SIGNAL: 519.11

# Load market data
market_data = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_5m.csv')
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])

# Get bars after entry
entry_time = pd.to_datetime('2024-03-26 19:40').tz_localize('UTC')
exit_time = pd.to_datetime('2024-03-26 19:45').tz_localize('UTC')  # Next bar

entry_bar = market_data[market_data['timestamp'] == entry_time].iloc[0]
exit_bar = market_data[market_data['timestamp'] == exit_time].iloc[0]

print(f"Entry bar (19:40):")
print(f"  Open:  {entry_bar['open']:.2f}")
print(f"  High:  {entry_bar['high']:.2f}")
print(f"  Low:   {entry_bar['low']:.2f}")
print(f"  Close: {entry_bar['close']:.2f}")

print(f"\nNext bar (19:45):")
print(f"  Open:  {exit_bar['open']:.2f}")
print(f"  High:  {exit_bar['high']:.2f}")
print(f"  Low:   {exit_bar['low']:.2f}")
print(f"  Close: {exit_bar['close']:.2f}")

# Calculate target
entry_price = 519.19
target_price = entry_price * 1.001  # 0.1% above
stop_price = entry_price * 0.99925   # 0.075% below

print(f"\nTrade setup:")
print(f"  Entry:  {entry_price:.2f}")
print(f"  Target: {target_price:.2f}")
print(f"  Stop:   {stop_price:.2f}")

print(f"\nTarget check on exit bar:")
print(f"  High {exit_bar['high']:.2f} >= Target {target_price:.2f}? {exit_bar['high'] >= target_price}")
print(f"  Low {exit_bar['low']:.2f} <= Stop {stop_price:.2f}? {exit_bar['low'] <= stop_price}")

# Check signals
run_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250627_185448')
strategy_index = pd.read_parquet(run_dir / 'strategy_index.parquet')
strategy_info = strategy_index[strategy_index['strategy_hash'] == '5edc43651004'].iloc[0]
signals = pd.read_parquet(run_dir / strategy_info['trace_path'])
signals['ts'] = pd.to_datetime(signals['ts'])

# Get signals around this time
window_signals = signals[(signals['ts'] >= entry_time) & (signals['ts'] <= exit_time)]
print(f"\nSignals:")
for _, sig in window_signals.iterrows():
    print(f"  {sig['ts']}: {sig['val']}")

print("\n💡 HYPOTHESIS:")
print("The execution engine might be:")
print("1. Processing the signal exit BEFORE checking intrabar targets")
print("2. Not checking high/low prices on the current bar")
print("3. Using a different order of operations than the analysis")