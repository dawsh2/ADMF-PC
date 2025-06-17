import pandas as pd

file_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_acf6c935/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_12_0002_08.parquet'

df = pd.read_parquet(file_path)
print(f'File loaded: {len(df)} rows')

# Examine the val column
print(f'Val column unique values: {df["val"].unique()}')
val_counts = df['val'].value_counts()
print(f'Val distribution: {val_counts.to_dict()}')

# Check for non-zero values
non_zero = df[df['val'] != 0.0]
print(f'Non-zero val entries: {len(non_zero)}')
if len(non_zero) > 0:
    print(f'Sample non-zero vals: {non_zero["val"].head().tolist()}')

# Look at the full dataframe structure
print(f'\nFull dataframe info:')
print(df.info())
print(f'\nSample rows:')
print(df.head(10))

# Check if val column contains regime information
print(f'\nAnalyzing val column as regime states:')
if len(df['val'].unique()) > 1:
    for val in sorted(df['val'].unique()):
        count = (df['val'] == val).sum()
        pct = count / len(df) * 100
        print(f'  State {val}: {count} occurrences ({pct:.1f}%)')