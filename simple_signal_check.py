import pandas as pd

# Read the parquet file
df = pd.read_parquet('config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')

print("=== Signal Analysis ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

if 'val' in df.columns:
    unique_vals = sorted(df['val'].unique())
    print(f"\nUnique values in 'val' column: {unique_vals}")
    
    # Check range
    min_val = df['val'].min()
    max_val = df['val'].max()
    print(f"\nMin value: {min_val}")
    print(f"Max value: {max_val}")
    
    # Distribution
    print("\nValue distribution:")
    value_counts = df['val'].value_counts().sort_index()
    for val, count in value_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {val}: {count} ({pct:.2f}%)")
    
    # Check for 2 or -2
    extreme_vals = df[(df['val'] == 2) | (df['val'] == -2)]
    print(f"\nSignals with value 2 or -2: {len(extreme_vals)}")
    
    # Sample non-zero signals
    print("\nFirst 5 non-zero signals:")
    non_zero = df[df['val'] != 0].head(5)
    if len(non_zero) > 0:
        for idx, row in non_zero.iterrows():
            print(f"  Time: {row['time']}, Val: {row['val']}, Strategy: {row.get('strategy_index', 'N/A')}")
else:
    print("\nERROR: 'val' column not found!")