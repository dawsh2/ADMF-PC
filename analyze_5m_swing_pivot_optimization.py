"""Analyze 5-minute swing pivot bounce zones optimization results"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

workspace = Path("workspaces/signal_generation_bfe07b48")

# Load metadata if available
metadata_file = workspace / "metadata.json"
if metadata_file.exists():
    with open(metadata_file) as f:
        metadata = json.load(f)
    print("Metadata found:")
    print(json.dumps(metadata, indent=2)[:500] + "...\n")

# Load SPY 5m data for reference
spy_5m_file = "./data/SPY_5m.csv"
if Path(spy_5m_file).exists():
    spy_data = pd.read_csv(spy_5m_file)
    spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'])
    print(f"SPY 5m data: {len(spy_data)} bars from {spy_data['timestamp'].min()} to {spy_data['timestamp'].max()}")
else:
    print("SPY 5m data not found, will calculate metrics without price data")
    spy_data = None

print("\n=== ANALYZING 5-MINUTE SWING PIVOT BOUNCE ZONES OPTIMIZATION ===\n")

# Analyze all strategy variations
results = []
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

for i in range(100):  # Check first 100 files
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{i}.parquet"
    if not signal_file.exists():
        continue
        
    try:
        signals = pd.read_parquet(signal_file)
        
        # Count trades (signal changes)
        trades = 0
        for j in range(1, len(signals)):
            prev_signal = signals.iloc[j-1]['val']
            curr_signal = signals.iloc[j]['val']
            
            # Count position changes
            if prev_signal != 0 and (curr_signal == 0 or np.sign(curr_signal) != np.sign(prev_signal)):
                trades += 1
        
        # Calculate basic metrics
        total_bars = signals.iloc[-1]['idx'] - signals.iloc[0]['idx'] + 1
        bars_in_position = (signals['val'] != 0).sum()
        
        results.append({
            'strategy_id': i,
            'total_signals': len(signals),
            'trades': trades,
            'bars_in_position': bars_in_position,
            'total_bars': total_bars,
            'position_pct': bars_in_position / total_bars * 100 if total_bars > 0 else 0
        })
        
        if i % 20 == 0:
            print(f"Processed {i} strategies...")
            
    except Exception as e:
        print(f"Error processing strategy {i}: {e}")
        continue

results_df = pd.DataFrame(results)
print(f"\nAnalyzed {len(results_df)} strategy variations")

# Show distribution of results
print("\n=== TRADE FREQUENCY DISTRIBUTION ===")
print(results_df['trades'].describe())

# Find strategies with reasonable trade counts (not too few, not too many)
reasonable_trades = results_df[(results_df['trades'] > 50) & (results_df['trades'] < 500)]
print(f"\nStrategies with 50-500 trades: {len(reasonable_trades)}")

# Show top strategies by trade count
print("\n=== TOP 10 STRATEGIES BY TRADE COUNT ===")
top_by_trades = results_df.nlargest(10, 'trades')[['strategy_id', 'trades', 'position_pct']]
print(top_by_trades)

# For detailed analysis, let's look at a few specific strategies
print("\n=== DETAILED ANALYSIS OF SELECT STRATEGIES ===")

# Analyze strategies with different trade frequencies
sample_strategies = []
if len(results_df) > 0:
    # Low frequency
    low_freq = results_df[results_df['trades'] < 100].head(1)
    if len(low_freq) > 0:
        sample_strategies.append(('Low Frequency', low_freq.iloc[0]['strategy_id']))
    
    # Medium frequency  
    med_freq = results_df[(results_df['trades'] >= 100) & (results_df['trades'] < 300)].head(1)
    if len(med_freq) > 0:
        sample_strategies.append(('Medium Frequency', med_freq.iloc[0]['strategy_id']))
    
    # High frequency
    high_freq = results_df[results_df['trades'] >= 300].head(1)
    if len(high_freq) > 0:
        sample_strategies.append(('High Frequency', high_freq.iloc[0]['strategy_id']))

for label, strategy_id in sample_strategies:
    print(f"\n{label} - Strategy {strategy_id}:")
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    signals = pd.read_parquet(signal_file)
    
    # Analyze trades in detail
    trades = []
    for j in range(1, len(signals)):
        prev = signals.iloc[j-1]
        curr = signals.iloc[j]
        
        if prev['val'] != 0 and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(prev['val'])):
            entry_idx = prev['idx']
            exit_idx = curr['idx']
            bars_held = exit_idx - entry_idx
            
            # Simple return calculation using signal prices
            pct_return = (curr['px'] / prev['px'] - 1) * prev['val'] * 100
            
            trades.append({
                'bars_held': bars_held,
                'pct_return': pct_return,
                'direction': 'long' if prev['val'] > 0 else 'short'
            })
    
    if trades:
        trades_df = pd.DataFrame(trades)
        avg_return = trades_df['pct_return'].mean()
        win_rate = (trades_df['pct_return'] > 0).mean()
        avg_bars = trades_df['bars_held'].mean()
        
        print(f"  Trades: {len(trades_df)}")
        print(f"  Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Avg bars held: {avg_bars:.1f} (5-min bars)")
        print(f"  Long trades: {(trades_df['direction'] == 'long').sum()}")
        print(f"  Short trades: {(trades_df['direction'] == 'short').sum()}")

print("\n=== RECOMMENDATIONS ===")
print("1. Look for strategies with 100-300 trades (good balance)")
print("2. Focus on those with positive average returns")
print("3. Check parameter patterns in best performers")
print("4. Consider running sparse trace analysis on top candidates")