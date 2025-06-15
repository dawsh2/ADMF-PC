import pandas as pd

df = pd.read_parquet('workspaces/expansive_grid_search_ab6d77f1/traces/SPY_1m/signals/chaikin_money_flow_grid/SPY_chaikin_money_flow_grid_10_0.03.parquet')
print(f'Signals: {len(df)} rows')
print(f'Columns: {list(df.columns)}')
print(f'Signal values (val): {sorted(df.val.unique())}')
print(f'Signal distribution: {dict(df.val.value_counts())}')
print('\nFirst few rows:')
print(df[['idx', 'val', 'strat']].head(10))