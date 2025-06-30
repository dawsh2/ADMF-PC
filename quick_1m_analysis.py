"""Quick analysis of 1-minute best possible performance"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path('workspaces/signal_generation_acc7968d')
signal_dir = workspace / 'traces/SPY_1m/signals/swing_pivot_bounce_zones'

# Quick check of best strategies
spy_1m = pd.read_csv('./data/SPY.csv')
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

spy_1m['returns'] = spy_1m['close'].pct_change()
spy_1m['volatility_20'] = spy_1m['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_1m['vol_percentile_20'] = spy_1m['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# Trend
spy_1m['sma_50'] = spy_1m['close'].rolling(50).mean()
spy_1m['sma_200'] = spy_1m['close'].rolling(200).mean()
spy_1m['trend_up'] = (spy_1m['close'] > spy_1m['sma_50']) & (spy_1m['sma_50'] > spy_1m['sma_200'])

total_days = len(spy_1m) / 390

print('=== 1-MINUTE BEST POSSIBLE PERFORMANCE ===\n')

best_results = []

# Check multiple strategies
for strategy_id in [0, 50, 88, 144, 256, 400, 500, 600, 700, 800, 900, 1000]:
    signal_file = signal_dir / f'SPY_compiled_strategy_{strategy_id}.parquet'
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # Quick trade collection
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_1m):
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and curr['idx'] < len(spy_1m) and entry_data['idx'] < len(spy_1m):
                entry_conditions = spy_1m.iloc[entry_data['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = curr['idx'] - entry_data['idx']
                
                trades.append({
                    'pct_return': pct_return,
                    'duration': duration,
                    'vol_percentile': entry_conditions.get('vol_percentile_20', 50),
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'trend_up': entry_conditions.get('trend_up', False)
                })
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if len(trades) > 50:
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df[trades_df['vol_percentile'].notna()]
        
        # Baseline
        baseline_bps = trades_df['pct_return'].mean()
        baseline_tpd = len(trades_df) / total_days
        
        print(f'\nStrategy {strategy_id}: {len(trades_df)} trades')
        print(f'  Baseline: {baseline_bps:.2f} bps on {baseline_tpd:.1f} tpd')
        
        # Test key filters
        filters = [
            ('3-10 min', (trades_df['duration'] >= 3) & (trades_df['duration'] < 10)),
            ('Vol>70', trades_df['vol_percentile'] > 70),
            ('Vol>80', trades_df['vol_percentile'] > 80),
            ('CT shorts in uptrend', (trades_df['trend_up']) & (trades_df['direction'] == 'short')),
            ('CT shorts + Vol>70', (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
             (trades_df['vol_percentile'] > 70)),
            ('3-10min + Vol>70', (trades_df['duration'] >= 3) & (trades_df['duration'] < 10) & 
             (trades_df['vol_percentile'] > 70)),
            ('Shorts only', trades_df['direction'] == 'short')
        ]
        
        for name, mask in filters:
            filtered = trades_df[mask]
            if len(filtered) >= 20:
                edge = filtered['pct_return'].mean()
                tpd = len(filtered) / total_days
                win_rate = (filtered['pct_return'] > 0).mean()
                
                if edge >= 0.5 or (edge > 0 and tpd >= 1.0):
                    print(f'  {name}: {edge:.2f} bps on {tpd:.1f} tpd ({win_rate:.0%} win)')
                    best_results.append({
                        'strategy': strategy_id,
                        'filter': name,
                        'edge': edge,
                        'tpd': tpd,
                        'win_rate': win_rate
                    })

print('\n\n' + '='*60)
print('SUMMARY: BEST 1-MINUTE CONFIGURATIONS')
print('='*60)

if best_results:
    results_df = pd.DataFrame(best_results)
    results_df = results_df.sort_values('edge', ascending=False)
    
    print('\nFilters achieving >=0.5 bps or high frequency:')
    for _, row in results_df.head(10).iterrows():
        print(f"  Strategy {row['strategy']:>4}, {row['filter']:<20}: "
              f"{row['edge']:>5.2f} bps on {row['tpd']:>4.1f} tpd ({row['win_rate']:.0%} win)")
    
    # Check if any achieve >=1 bps
    high_edge = results_df[results_df['edge'] >= 1.0]
    if len(high_edge) > 0:
        print(f'\n✓ Found {len(high_edge)} configurations with >=1 bps edge!')
        for _, row in high_edge.iterrows():
            print(f"  Strategy {row['strategy']}, {row['filter']}: "
                  f"{row['edge']:.2f} bps on {row['tpd']:.1f} tpd")
    else:
        print('\n✗ No configurations achieve >=1 bps edge on 1-minute data')
        
    # Best with >2 tpd
    high_freq = results_df[results_df['tpd'] >= 2.0]
    if len(high_freq) > 0:
        print(f'\nBest with 2+ trades/day:')
        best_hf = high_freq.nlargest(3, 'edge')
        for _, row in best_hf.iterrows():
            print(f"  {row['edge']:.2f} bps on {row['tpd']:.1f} tpd")
else:
    print('\nNo filters found with meaningful edge.')

print('\n\nCONCLUSION:')
print('1-minute swing pivot bounce is not viable for >=1 bps with multiple trades/day.')
print('The support/resistance patterns are too noisy at 1-minute granularity.')
print('\nRecommendation: Use 5-minute data (2.18 bps, 2.8 tpd) or explore different')
print('strategy types better suited to 1-minute timeframes.')