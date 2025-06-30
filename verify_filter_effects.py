"""Verify that filters are actually reducing signal counts"""
import pandas as pd
import glob

# Get all parquet files
parquet_files = glob.glob("workspaces/signal_generation_310b2aeb/traces/SPY_1m/signals/keltner_bands/*.parquet")

# Analyze signal counts by strategy group
results = []

for file in parquet_files:
    strategy_name = file.split('/')[-1].replace('.parquet', '')
    strategy_num = int(strategy_name.split('_')[-1])
    
    # Determine strategy type based on number
    if strategy_num < 25:
        strategy_type = "Baseline (no filter)"
    elif 25 <= strategy_num < 36:
        strategy_type = "RSI filter"
    elif 36 <= strategy_num < 45:
        strategy_type = "Volume filter"
    elif 45 <= strategy_num < 54:
        strategy_type = "Combined RSI+Volume"
    elif 54 <= strategy_num < 70:
        strategy_type = "Directional RSI"
    else:
        strategy_type = "Unknown"
    
    # Read file and count signals
    df = pd.read_parquet(file)
    total_rows = len(df)
    non_zero = len(df[df['val'] != 0])
    signal_rate = non_zero / total_rows * 100 if total_rows > 0 else 0
    
    results.append({
        'strategy': strategy_name,
        'number': strategy_num,
        'type': strategy_type,
        'total_rows': total_rows,
        'non_zero_signals': non_zero,
        'signal_rate': signal_rate
    })

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

# Group by strategy type and show statistics
print("=== Signal Rates by Strategy Type ===")
grouped = results_df.groupby('type').agg({
    'signal_rate': ['mean', 'min', 'max', 'count'],
    'non_zero_signals': ['mean', 'min', 'max']
})
print(grouped)

# Show specific examples
print("\n=== Example Strategies ===")
examples = [
    ("Baseline", 0),
    ("RSI filter", 30),
    ("Volume filter", 40), 
    ("Combined", 50),
    ("Directional", 60)
]

for label, num in examples:
    strategy = results_df[results_df['number'] == num]
    if not strategy.empty:
        row = strategy.iloc[0]
        print(f"\n{label} (strategy_{num}):")
        print(f"  Total rows: {row['total_rows']:,}")
        print(f"  Non-zero signals: {row['non_zero_signals']:,}")
        print(f"  Signal rate: {row['signal_rate']:.1f}%")

# Check if volume filter is working
print("\n=== Volume Filter Analysis ===")
volume_strategies = results_df[results_df['type'] == 'Volume filter']
if not volume_strategies.empty:
    print(f"Average signal rate for volume filtered strategies: {volume_strategies['signal_rate'].mean():.1f}%")
    print(f"Min/Max: {volume_strategies['signal_rate'].min():.1f}% - {volume_strategies['signal_rate'].max():.1f}%")