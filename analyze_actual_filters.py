"""Analyze the actual 45 strategies that were generated"""
import pandas as pd
import glob

# Based on the config and the fact we have 45 strategies, the actual mapping is:
# The system generated 45 parameter combinations as shown in the log output
# Need to figure out which ones were actually generated

parquet_files = sorted(glob.glob("workspaces/signal_generation_310b2aeb/traces/SPY_1m/signals/keltner_bands/*.parquet"))
print(f"Found {len(parquet_files)} strategy files")

# Analyze each file
results = []

for file in parquet_files:
    strategy_name = file.split('/')[-1].replace('.parquet', '')
    strategy_num = int(strategy_name.split('_')[-1])
    
    # Read file
    df = pd.read_parquet(file)
    
    # Count transitions  
    df['prev_val'] = df['val'].shift(1).fillna(0)
    
    long_entries = len(df[(df['prev_val'] == 0) & (df['val'] == 1)])
    short_entries = len(df[(df['prev_val'] == 0) & (df['val'] == -1)])
    total_entries = long_entries + short_entries
    
    results.append({
        'strategy_num': strategy_num,
        'strategy_name': strategy_name,
        'long_entries': long_entries,
        'short_entries': short_entries,
        'total_entries': total_entries,
        'total_bars': len(df),
        'entry_rate': total_entries / len(df) * 100 if len(df) > 0 else 0
    })

# Convert to DataFrame and sort
results_df = pd.DataFrame(results).sort_values('strategy_num')

# Try to identify patterns based on entry counts
# Group strategies by similar entry counts
results_df['entry_group'] = pd.qcut(results_df['total_entries'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

print("\n=== ENTRY COUNT DISTRIBUTION ===")
print(results_df.groupby('entry_group')['total_entries'].agg(['count', 'min', 'max', 'mean']))

# Look for clear groups
print("\n=== STRATEGY GROUPS BY ENTRY COUNT ===")

# Find natural breaks in entry counts
sorted_entries = results_df['total_entries'].sort_values()
diffs = sorted_entries.diff()
large_gaps = diffs[diffs > 500].index

print(f"\nLarge gaps found at indices: {list(large_gaps)}")

# Show strategies with very low entries (likely heavily filtered)
print("\n=== MOST FILTERED STRATEGIES (< 2000 entries) ===")
heavily_filtered = results_df[results_df['total_entries'] < 2000].sort_values('total_entries')
for _, row in heavily_filtered.iterrows():
    print(f"Strategy {row['strategy_num']:2d}: {row['total_entries']:4d} entries ({row['entry_rate']:.1f}% rate)")

# Show strategies with high entries (likely baseline)
print("\n=== LEAST FILTERED STRATEGIES (> 6000 entries) ===")
lightly_filtered = results_df[results_df['total_entries'] > 6000].sort_values('total_entries', ascending=False)
for _, row in lightly_filtered.iterrows():
    print(f"Strategy {row['strategy_num']:2d}: {row['total_entries']:4d} entries ({row['entry_rate']:.1f}% rate)")

# Look for patterns in strategy numbers
print("\n=== ENTRY COUNTS BY STRATEGY NUMBER ===")
fig_data = results_df[['strategy_num', 'total_entries']].set_index('strategy_num').sort_index()

# Print in groups of 10
for i in range(0, 45, 10):
    print(f"\nStrategies {i}-{min(i+9, 44)}:")
    subset = fig_data.loc[i:min(i+9, 44)]
    for idx, row in subset.iterrows():
        print(f"  {idx:2d}: {row['total_entries']:5d} entries")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Min entries: {results_df['total_entries'].min()}")
print(f"Max entries: {results_df['total_entries'].max()}")
print(f"Mean entries: {results_df['total_entries'].mean():.0f}")
print(f"Median entries: {results_df['total_entries'].median():.0f}")
print(f"Std dev: {results_df['total_entries'].std():.0f}")

# Check if filtering is working by comparing min/max
baseline_estimate = results_df['total_entries'].max()
most_filtered = results_df['total_entries'].min()
print(f"\nFiltering effectiveness:")
print(f"Baseline (estimated): {baseline_estimate} entries")
print(f"Most filtered: {most_filtered} entries")
print(f"Maximum reduction: {(1 - most_filtered/baseline_estimate)*100:.1f}%")