"""Check if filters are only affecting one direction"""
import pandas as pd
import glob

# Get sample files from each strategy type
files = {
    'baseline': 'workspaces/signal_generation_310b2aeb/traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_0.parquet',
    'rsi_filter': 'workspaces/signal_generation_310b2aeb/traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_30.parquet',
    'volume_filter': 'workspaces/signal_generation_310b2aeb/traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_40.parquet',
}

for strategy_type, file_path in files.items():
    print(f"\n=== {strategy_type.upper()} ===")
    try:
        df = pd.read_parquet(file_path)
        
        # Count signal values
        signal_counts = df['val'].value_counts().sort_index()
        print(f"Signal value distribution:")
        for val, count in signal_counts.items():
            print(f"  {val:2d}: {count:6d} ({count/len(df)*100:5.1f}%)")
        
        # Count transitions
        print(f"\nSignal transitions:")
        # Add a shifted column to detect transitions
        df['prev_val'] = df['val'].shift(1).fillna(0)
        
        # Count different types of transitions
        long_entries = len(df[(df['prev_val'] == 0) & (df['val'] == 1)])
        long_exits = len(df[(df['prev_val'] == 1) & (df['val'] == 0)])
        short_entries = len(df[(df['prev_val'] == 0) & (df['val'] == -1)])
        short_exits = len(df[(df['prev_val'] == -1) & (df['val'] == 0)])
        
        print(f"  Long entries (0 → 1): {long_entries}")
        print(f"  Long exits (1 → 0): {long_exits}")
        print(f"  Short entries (0 → -1): {short_entries}")
        print(f"  Short exits (-1 → 0): {short_exits}")
        
        # Check if exits match entries
        print(f"\nBalance check:")
        print(f"  Long: {long_entries} entries vs {long_exits} exits (diff: {long_entries - long_exits})")
        print(f"  Short: {short_entries} entries vs {short_exits} exits (diff: {short_entries - short_exits})")
        
    except Exception as e:
        print(f"Error: {e}")

# Let's also check the actual filter expressions being used
print("\n=== FILTER EXPRESSIONS ===")
print("According to the config, filters should be:")
print("- RSI filter: 'signal == 0 or rsi(14) < ${rsi_threshold}'")
print("- Volume filter: 'signal == 0 or volume > volume_sma_20 * ${volume_threshold}'")
print("\nThe 'signal == 0' part should allow ALL exits (transitions to 0)")
print("Filters should only reduce ENTRIES (transitions from 0 to ±1)")