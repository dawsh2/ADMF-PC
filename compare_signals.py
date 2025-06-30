#!/usr/bin/env python3
"""Compare notebook and test signals."""

import pandas as pd

# Load signals from both sources
notebook_signals = pd.read_parquet('config/bollinger/results/20250625_170201/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')
your_signals = pd.read_parquet('config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')

notebook_signals['ts'] = pd.to_datetime(notebook_signals['ts'])
your_signals['ts'] = pd.to_datetime(your_signals['ts'])

# Count entry signals
def count_entries(df):
    df = df.sort_values('ts')
    df['prev_val'] = df['val'].shift(1).fillna(0)
    entries = df[(df['prev_val'] == 0) & (df['val'] != 0)]
    return len(entries)

nb_entries = count_entries(notebook_signals)
your_entries = count_entries(your_signals)

print(f'Notebook: {len(notebook_signals)} signals, {nb_entries} entries')
print(f'Your run: {len(your_signals)} signals, {your_entries} entries')

# Check time overlap
print(f'\nNotebook period: {notebook_signals.ts.min()} to {notebook_signals.ts.max()}')
print(f'Your period: {your_signals.ts.min()} to {your_signals.ts.max()}')

# Filter your signals to match notebook period
notebook_start = notebook_signals.ts.min()
notebook_end = notebook_signals.ts.max()
your_test_signals = your_signals[(your_signals.ts >= notebook_start) & (your_signals.ts <= notebook_end)]
print(f'\nYour signals in notebook period: {len(your_test_signals)}')

if len(your_test_signals) > 0:
    your_test_entries = count_entries(your_test_signals)
    print(f'Your entries in notebook period: {your_test_entries}')
    
    # Compare values
    print('\nComparing signal values...')
    # Reset indices for comparison
    nb_sorted = notebook_signals.sort_values('ts').reset_index(drop=True)
    yr_sorted = your_test_signals.sort_values('ts').reset_index(drop=True)
    
    # Check if same length
    if len(nb_sorted) == len(yr_sorted):
        print(f'✓ Same number of signals in test period')
        
        # Compare values
        matches = (nb_sorted['val'] == yr_sorted['val']).sum()
        print(f'Matching values: {matches}/{len(nb_sorted)} ({matches/len(nb_sorted)*100:.1f}%)')
        
        if matches < len(nb_sorted):
            # Find first mismatch
            for i in range(len(nb_sorted)):
                if nb_sorted.iloc[i]['val'] != yr_sorted.iloc[i]['val']:
                    print(f'\nFirst mismatch at index {i}:')
                    print(f'  NB: {nb_sorted.iloc[i].ts} = {nb_sorted.iloc[i].val}')
                    print(f'  YOU: {yr_sorted.iloc[i].ts} = {yr_sorted.iloc[i].val}')
                    break
    else:
        print(f'❌ Different signal counts: {len(nb_sorted)} vs {len(yr_sorted)}')
        
        # Show what's different
        nb_times = set(nb_sorted.ts)
        yr_times = set(yr_sorted.ts)
        
        only_nb = nb_times - yr_times
        only_yr = yr_times - nb_times
        
        if only_nb:
            print(f'\nSignals only in notebook: {len(only_nb)} (first: {min(only_nb)})')
        if only_yr:
            print(f'Signals only in your run: {len(only_yr)} (first: {min(only_yr)})')