"""Analyze 5-minute swing pivot bounce zones optimization results - fixed version"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

workspace = Path("workspaces/signal_generation_bfe07b48")

# Load metadata
metadata_file = workspace / "metadata.json"
with open(metadata_file) as f:
    metadata = json.load(f)

print("=== 5-MINUTE SWING PIVOT BOUNCE ZONES OPTIMIZATION ===\n")
print(f"Total bars analyzed: {metadata['total_bars']:,}")
print(f"Total signals: {metadata['total_signals']:,}")
print(f"Stored changes: {metadata['stored_changes']:,}")
print(f"Compression ratio: {metadata['compression_ratio']:.2%}\n")

# Check for data files
spy_5m_file = "./data/SPY_5m.csv"
if Path(spy_5m_file).exists():
    spy_data = pd.read_csv(spy_5m_file)
    print(f"SPY 5m data available: {len(spy_data):,} bars")

# Analyze all strategy variations
results = []
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

print("\nAnalyzing strategy variations...")

for i in range(100):  # Check first 100 files
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{i}.parquet"
    if not signal_file.exists():
        continue
        
    try:
        signals = pd.read_parquet(signal_file)
        
        # Count trades
        trades = []
        for j in range(1, len(signals)):
            prev = signals.iloc[j-1]
            curr = signals.iloc[j]
            
            # Entry
            if prev['val'] == 0 and curr['val'] != 0:
                entry_idx = curr['idx']
                entry_price = curr['px']
                direction = curr['val']
            # Exit
            elif prev['val'] != 0 and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(prev['val'])):
                exit_idx = curr['idx']
                exit_price = curr['px']
                
                pct_return = (exit_price / entry_price - 1) * direction * 100
                bars_held = exit_idx - entry_idx
                
                trades.append({
                    'pct_return': pct_return,
                    'bars_held': bars_held,
                    'direction': 'long' if direction > 0 else 'short'
                })
                
                # Reset for new position if flipping
                if curr['val'] != 0:
                    entry_idx = curr['idx']
                    entry_price = curr['px']
                    direction = curr['val']
        
        # Calculate metrics
        num_trades = len(trades)
        if num_trades > 0:
            trades_df = pd.DataFrame(trades)
            avg_return = trades_df['pct_return'].mean()
            win_rate = (trades_df['pct_return'] > 0).mean()
            avg_bars = trades_df['bars_held'].mean()
            total_return = trades_df['pct_return'].sum()
            
            # Estimate annual metrics (5-min bars, ~78 per day)
            bars_per_year = 78 * 252
            trades_per_year = num_trades * (bars_per_year / metadata['total_bars'])
            
            results.append({
                'strategy_id': i,
                'trades': num_trades,
                'avg_return_pct': avg_return,
                'avg_return_bps': avg_return * 100,
                'win_rate': win_rate,
                'avg_bars_held': avg_bars,
                'total_return': total_return,
                'trades_per_year': trades_per_year,
                'long_trades': (trades_df['direction'] == 'long').sum(),
                'short_trades': (trades_df['direction'] == 'short').sum()
            })
        else:
            results.append({
                'strategy_id': i,
                'trades': 0,
                'avg_return_pct': 0,
                'avg_return_bps': 0,
                'win_rate': 0,
                'avg_bars_held': 0,
                'total_return': 0,
                'trades_per_year': 0,
                'long_trades': 0,
                'short_trades': 0
            })
            
    except Exception as e:
        print(f"Error processing strategy {i}: {e}")
        continue

results_df = pd.DataFrame(results)
print(f"\nAnalyzed {len(results_df)} strategy variations")

# Filter for strategies with reasonable trade counts
active_strategies = results_df[results_df['trades'] > 20]
print(f"Active strategies (>20 trades): {len(active_strategies)}")

if len(active_strategies) > 0:
    # Show top performers by average return
    print("\n=== TOP 10 STRATEGIES BY AVERAGE RETURN (BPS) ===")
    top_by_return = active_strategies.nlargest(10, 'avg_return_bps')[
        ['strategy_id', 'trades', 'avg_return_bps', 'win_rate', 'trades_per_year']
    ]
    print(top_by_return.to_string(index=False))
    
    # Show most active strategies
    print("\n=== TOP 10 MOST ACTIVE STRATEGIES ===")
    top_by_trades = active_strategies.nlargest(10, 'trades')[
        ['strategy_id', 'trades', 'avg_return_bps', 'win_rate', 'trades_per_year']
    ]
    print(top_by_trades.to_string(index=False))
    
    # Find balanced strategies (good return + reasonable activity)
    print("\n=== BALANCED STRATEGIES (>1 bps, 50-200 trades) ===")
    balanced = active_strategies[
        (active_strategies['avg_return_bps'] > 1) & 
        (active_strategies['trades'] >= 50) & 
        (active_strategies['trades'] <= 200)
    ].sort_values('avg_return_bps', ascending=False)
    
    if len(balanced) > 0:
        print(balanced[['strategy_id', 'trades', 'avg_return_bps', 'win_rate', 'trades_per_year']].to_string(index=False))
    else:
        print("No strategies found in this range")
    
    # Best overall (considering edge and frequency)
    print("\n=== BEST OVERALL (Edge × Frequency) ===")
    active_strategies['expected_annual_bps'] = active_strategies['avg_return_bps'] * active_strategies['trades_per_year']
    top_overall = active_strategies.nlargest(10, 'expected_annual_bps')[
        ['strategy_id', 'trades', 'avg_return_bps', 'trades_per_year', 'expected_annual_bps']
    ]
    print(top_overall.to_string(index=False))

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Average trades per strategy: {results_df['trades'].mean():.1f}")
print(f"Average return per trade: {results_df['avg_return_bps'].mean():.2f} bps")
print(f"Strategies with positive returns: {(results_df['avg_return_bps'] > 0).sum()}")
print(f"Strategies with >1 bps returns: {(results_df['avg_return_bps'] > 1).sum()}")
print(f"Strategies with >5 bps returns: {(results_df['avg_return_bps'] > 5).sum()}")

# Compare to 1-minute results
print("\n=== COMPARISON TO 1-MINUTE RESULTS ===")
print("1-minute baseline: 0.04 bps")
print("1-minute with high vol filter: 1.26 bps")
print(f"5-minute best: {active_strategies['avg_return_bps'].max():.2f} bps" if len(active_strategies) > 0 else "No active strategies")
print(f"5-minute average (active): {active_strategies['avg_return_bps'].mean():.2f} bps" if len(active_strategies) > 0 else "No active strategies")