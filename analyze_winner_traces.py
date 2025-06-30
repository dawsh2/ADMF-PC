#!/usr/bin/env python3
"""Analyze the keltner winner config trace files"""

import pandas as pd
import json
from pathlib import Path

# Load the trace files
trace_dir = Path('config/keltner/config_winrar/results/20250622_210552/traces/mean_reversion')
p50_file = trace_dir / 'SPY_5m_kb_winner_p50_m1.parquet'
p30_file = trace_dir / 'SPY_5m_kb_winner_p30_m1.parquet'

# Load metadata
metadata_file = Path('config/keltner/config_winrar/results/20250622_210552/metadata.json')
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

print('KELTNER WINNER CONFIG - RESULTS ANALYSIS')
print('='*60)
print(f"Run timestamp: {metadata['timestamp']}")
print(f"Total bars: {metadata['total_bars']:,}")

# Analyze each strategy
for name, file_path in [('Period 50', p50_file), ('Period 30', p30_file)]:
    print(f'\n{name} Strategy:')
    print('-'*40)
    
    # Load trace
    df = pd.read_parquet(file_path)
    
    # Count trades
    trades = 0
    in_trade = False
    for i in range(len(df)):
        signal = df.iloc[i]['val']
        if not in_trade and signal != 0:
            trades += 1
            in_trade = True
        elif in_trade and signal == 0:
            in_trade = False
    
    # Get component info
    comp_name = file_path.stem
    if comp_name in metadata['components']:
        comp_data = metadata['components'][comp_name]
        print(f"  Signal changes: {comp_data['signal_changes']:,}")
        print(f"  Signal frequency: {comp_data['signal_frequency']*100:.2f}%")
        print(f"  Estimated trades: {trades}")
        
        # Trading days
        bars_per_day = 78  # 6.5 hours * 12 (5-min bars per hour)
        trading_days = metadata['total_bars'] / bars_per_day
        trades_per_day = trades / trading_days if trading_days > 0 else 0
        print(f"  Trades per day: {trades_per_day:.2f}")
    
    # Check signal distribution
    if len(df) > 0:
        long_signals = (df['val'] > 0).sum()
        short_signals = (df['val'] < 0).sum()
        neutral = (df['val'] == 0).sum()
        print(f"  Long signals: {long_signals}, Short signals: {short_signals}, Neutral: {neutral}")

print('\n' + '='*60)
print('COMPARISON TO ORIGINAL ANALYSIS:')
print('Original Period 50, Mult 1.0: 1.89 trades/day, 16.92% annual')
print('Original Period 30, Mult 1.0: ~4-8 trades/day (avg across filters)')
print('\nNote: These are baseline strategies without filters.')
print('The original top performers included specific filter combinations.')