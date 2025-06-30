"""Analyze 1-minute swing pivot bounce performance"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_acc7968d")
signal_dir = workspace / "traces/SPY/signals/swing_pivot_bounce_zones"

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'])
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

print("=== 1-MINUTE SWING PIVOT BOUNCE ANALYSIS ===\n")

# Calculate indicators
spy_1m['returns'] = spy_1m['close'].pct_change()
spy_1m['volatility_20'] = spy_1m['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_1m['vol_percentile'] = spy_1m['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# Volume
spy_1m['volume_sma_20'] = spy_1m['volume'].rolling(20).mean()
spy_1m['volume_ratio'] = spy_1m['volume'] / spy_1m['volume_sma_20']

# Sample a few strategies to test
strategies_to_test = [0, 100, 200, 500, 1000]  # Sample strategies

results = []

for strategy_id in strategies_to_test:
    signal_file = signal_dir / f"SPY_compiled_strategy_{strategy_id}.parquet"
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # Collect trades
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_1m):
                entry_data = {
                    'idx': curr['idx'],
                    'price': curr['px'],
                    'signal': curr['val']
                }
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and entry_data['idx'] < len(spy_1m) and curr['idx'] < len(spy_1m):
                entry_conditions = spy_1m.iloc[entry_data['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                
                if not pd.isna(entry_conditions['vol_percentile']):
                    trade = {
                        'pct_return': pct_return,
                        'direction': 'short' if entry_data['signal'] < 0 else 'long',
                        'vol_percentile': entry_conditions['vol_percentile'],
                        'volume_ratio': entry_conditions['volume_ratio']
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if not trades:
        continue
    
    trades_df = pd.DataFrame(trades)
    total_days = len(spy_1m) / 390  # 1-min bars per day
    
    # Baseline
    baseline_bps = trades_df['pct_return'].mean() * 100
    baseline_tpd = len(trades_df) / total_days
    
    # Test key filters from 5m analysis
    filters_to_test = [
        ("Vol>70", trades_df['vol_percentile'] > 70),
        ("Vol>85", trades_df['vol_percentile'] > 85),
        ("Shorts only", trades_df['direction'] == 'short'),
        ("Vol>70 + Shorts", (trades_df['vol_percentile'] > 70) & (trades_df['direction'] == 'short')),
        ("Volume>1.2x", trades_df['volume_ratio'] > 1.2)
    ]
    
    print(f"\nStrategy {strategy_id}:")
    print(f"Baseline: {baseline_bps:.2f} bps, {baseline_tpd:.1f} trades/day")
    
    for filter_name, filter_mask in filters_to_test:
        filtered = trades_df[filter_mask]
        if len(filtered) > 10:
            edge_bps = filtered['pct_return'].mean() * 100
            tpd = len(filtered) / total_days
            results.append({
                'strategy_id': strategy_id,
                'filter': filter_name,
                'edge_bps': edge_bps,
                'trades_per_day': tpd,
                'total_trades': len(filtered)
            })
            print(f"  {filter_name}: {edge_bps:.2f} bps, {tpd:.1f} trades/day")

# Find best performers
results_df = pd.DataFrame(results)
if len(results_df) > 0:
    print("\n\nBEST 1-MINUTE CONFIGURATIONS:")
    print("="*60)
    
    # Best for 2-3+ trades/day
    high_freq = results_df[results_df['trades_per_day'] >= 2.0].sort_values('edge_bps', ascending=False)
    if len(high_freq) > 0:
        print("\nHigh frequency (2+ trades/day):")
        for _, row in high_freq.head(5).iterrows():
            print(f"  Strategy {row['strategy_id']}, {row['filter']}: {row['edge_bps']:.2f} bps, {row['trades_per_day']:.1f} tpd")
    
    # Best edge regardless of frequency
    print("\nBest edge overall:")
    for _, row in results_df.sort_values('edge_bps', ascending=False).head(5).iterrows():
        print(f"  Strategy {row['strategy_id']}, {row['filter']}: {row['edge_bps']:.2f} bps, {row['trades_per_day']:.1f} tpd")

print("\n\nCOMPARISON: 1-minute vs 5-minute")
print("="*60)
print("5-minute best: Vol>70 yields 2.18 bps, 2.8 trades/day")
print("Check if 1-minute can match or exceed this performance...")