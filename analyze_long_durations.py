"""Analyze why filtered strategies have such long trade durations"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_57cd04b8")

# Load SPY data for prices
spy_data = pd.read_csv("./data/SPY.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
spy_data.columns = spy_data.columns.str.lower()
spy_data = spy_data.set_index('timestamp')

strategies = [
    ('keltner_baseline', 'No filter'),
    ('keltner_rsi70', 'RSI < 70'),
]

for strategy_name, filter_desc in strategies:
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Filter: {filter_desc}")
    print("="*60)
    
    # Load signals
    signal_file = workspace / f"traces/SPY_1m/signals/mean_reversion/SPY_{strategy_name}.parquet"
    if not signal_file.exists():
        print(f"File not found: {signal_file}")
        continue
        
    signals_df = pd.read_parquet(signal_file)
    signals_df['timestamp'] = pd.to_datetime(signals_df['ts']).dt.tz_convert('UTC')
    signals_df = signals_df.sort_values('timestamp')
    
    # Count signal changes
    print(f"\nSignal changes: {len(signals_df):,}")
    
    # Analyze signal patterns
    signal_values = signals_df['val'].values
    signal_changes = np.diff(signal_values, prepend=signal_values[0])
    
    # Count different signal transitions
    entries = ((signal_changes != 0) & (signals_df['val'] != 0)).sum()
    exits = ((signal_changes != 0) & (signals_df['val'] == 0)).sum()
    flips = ((signal_changes != 0) & (signals_df['val'].shift(1) != 0) & (signals_df['val'] != 0) & (np.sign(signals_df['val']) != np.sign(signals_df['val'].shift(1)))).sum()
    
    print(f"\nSignal transitions:")
    print(f"  Entries (0 → ±1): {entries}")
    print(f"  Exits (±1 → 0): {exits}")
    print(f"  Flips (1 → -1 or -1 → 1): {flips}")
    
    # Sample some long trades
    positions = []
    current_position = None
    
    for _, signal in signals_df.iterrows():
        ts = signal['timestamp']
        signal_value = signal['val']
        
        if signal_value != 0 and current_position is None:
            # Enter position
            current_position = {
                'entry_time': ts,
                'entry_signal': signal_value,
                'direction': 1 if signal_value > 0 else -1
            }
        elif signal_value == 0 and current_position is not None:
            # Exit position
            duration_min = (ts - current_position['entry_time']).total_seconds() / 60
            current_position['exit_time'] = ts
            current_position['duration_min'] = duration_min
            positions.append(current_position)
            current_position = None
        elif signal_value != 0 and current_position is not None and np.sign(signal_value) != current_position['direction']:
            # Close and reverse
            duration_min = (ts - current_position['entry_time']).total_seconds() / 60
            current_position['exit_time'] = ts
            current_position['duration_min'] = duration_min
            positions.append(current_position)
            
            # Open new position
            current_position = {
                'entry_time': ts,
                'entry_signal': signal_value,
                'direction': 1 if signal_value > 0 else -1
            }
    
    if len(positions) > 0:
        trades_df = pd.DataFrame(positions)
        
        # Duration statistics
        print(f"\nDuration statistics:")
        print(f"  Mean: {trades_df['duration_min'].mean():.1f} minutes")
        print(f"  Median: {trades_df['duration_min'].median():.1f} minutes")
        print(f"  Max: {trades_df['duration_min'].max():.1f} minutes")
        print(f"  Min: {trades_df['duration_min'].min():.1f} minutes")
        
        # Show longest trades
        longest_trades = trades_df.nlargest(5, 'duration_min')
        print(f"\nLongest 5 trades:")
        for _, trade in longest_trades.iterrows():
            print(f"  {trade['entry_time']} → {trade['exit_time']} ({trade['duration_min']:.0f} min)")
            
        # Check if we have current position still open
        if current_position is not None:
            print(f"\nWARNING: Position still open from {current_position['entry_time']}")

# Now let's check the actual filter logic
print("\n" + "="*80)
print("CHECKING FILTER BEHAVIOR")
print("="*80)

# Load baseline signals with all signal changes
baseline_file = workspace / "traces/SPY_1m/signals/mean_reversion/SPY_keltner_baseline.parquet"
baseline_signals = pd.read_parquet(baseline_file)
baseline_signals['timestamp'] = pd.to_datetime(baseline_signals['ts']).dt.tz_convert('UTC')

# Load RSI70 filtered signals
rsi70_file = workspace / "traces/SPY_1m/signals/mean_reversion/SPY_keltner_rsi70.parquet"
rsi70_signals = pd.read_parquet(rsi70_file)
rsi70_signals['timestamp'] = pd.to_datetime(rsi70_signals['ts']).dt.tz_convert('UTC')

# Compare a specific time period
start_time = pd.Timestamp('2024-04-01 09:30:00', tz='UTC')
end_time = pd.Timestamp('2024-04-01 10:30:00', tz='UTC')

baseline_subset = baseline_signals[(baseline_signals['timestamp'] >= start_time) & 
                                  (baseline_signals['timestamp'] <= end_time)]
rsi70_subset = rsi70_signals[(rsi70_signals['timestamp'] >= start_time) & 
                            (rsi70_signals['timestamp'] <= end_time)]

print(f"\nComparing signals from {start_time} to {end_time}:")
print(f"Baseline signal changes: {len(baseline_subset)}")
print(f"RSI70 signal changes: {len(rsi70_subset)}")

# Show the actual signals
print("\nBaseline signals:")
print(baseline_subset[['timestamp', 'val']].to_string())

print("\nRSI70 filtered signals:")
print(rsi70_subset[['timestamp', 'val']].to_string())