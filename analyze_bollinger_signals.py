import pandas as pd

# Read signal data  
df = pd.read_parquet('config/bollinger/results/latest/traces/strategies/signals/strategy_signals.parquet')

# Get bollinger strategy signals
bollinger_signals = df[df['source'] == 'SPY_5m_strategy_0'].copy()

print(f'Total bollinger signals: {len(bollinger_signals)}')

# Look at direction distribution
print('\nSignal direction distribution:')
print(bollinger_signals['direction'].value_counts())

# Check for rapid flips by looking at consecutive signals
bollinger_signals = bollinger_signals.sort_values('idx')
bollinger_signals['prev_direction'] = bollinger_signals['direction'].shift(1)

# Count flips
flips = (bollinger_signals['direction'] \!= bollinger_signals['prev_direction']).sum() - 1  # -1 for first row
print(f'\nTotal direction flips: {flips}')
print(f'Flip rate: {flips / len(bollinger_signals) * 100:.1f}%')

# Calculate consecutive bars in same direction
bollinger_signals['direction_numeric'] = bollinger_signals['direction'].map({'LONG': 1, 'SHORT': -1, 'FLAT': 0})
bollinger_signals['direction_changed'] = bollinger_signals['direction'] \!= bollinger_signals['prev_direction']
bollinger_signals['direction_group'] = bollinger_signals['direction_changed'].cumsum()

# Group by direction runs
direction_runs = bollinger_signals.groupby('direction_group').agg({
    'direction': 'first',
    'idx': 'count'
}).rename(columns={'idx': 'consecutive_bars'})

print(f'\nAverage consecutive bars in same direction: {direction_runs["consecutive_bars"].mean():.1f}')

# Show distribution
print('\nConsecutive bars distribution:')
cons_dist = direction_runs['consecutive_bars'].value_counts().sort_index()
for bars, count in cons_dist.head(10).items():
    pct = count / len(direction_runs) * 100
    print(f'  {bars} bars: {count:4d} runs ({pct:4.1f}%)')

# Show signal patterns
print('\nFirst 50 signals pattern:')
for i, (_, row) in enumerate(bollinger_signals.head(50).iterrows()):
    if i % 10 == 0:
        print(f'\n{i:3d}: ', end='')
    print(f'{row["direction"][0]}', end='')  # Just first letter
print()
EOF < /dev/null