"""Analyze workspace 320d109d swing pivot optimization"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

workspace = Path("workspaces/signal_generation_320d109d")

# Load metadata if available
metadata_file = workspace / "metadata.json"
if metadata_file.exists():
    with open(metadata_file) as f:
        metadata = json.load(f)
    print("=== WORKSPACE 320d109d ANALYSIS ===\n")
    print(f"Total bars: {metadata.get('total_bars', 'N/A'):,}")
    print(f"Total signals: {metadata.get('total_signals', 'N/A'):,}")
    print(f"Compression ratio: {metadata.get('compression_ratio', 0):.2%}\n")

# Analyze all strategy variations
results = []
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

print("Analyzing strategy variations...")

for i in range(120):  # Check up to 120 files
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{i}.parquet"
    if not signal_file.exists():
        continue
        
    try:
        signals = pd.read_parquet(signal_file)
        
        # Count trades more carefully
        trades = []
        entry_price = None
        entry_idx = None
        entry_signal = None
        
        for j in range(len(signals)):
            curr = signals.iloc[j]
            
            # New position
            if entry_price is None and curr['val'] != 0:
                entry_price = curr['px']
                entry_idx = curr['idx']
                entry_signal = curr['val']
            
            # Exit position
            elif entry_price is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_signal)):
                exit_price = curr['px']
                exit_idx = curr['idx']
                
                pct_return = (exit_price / entry_price - 1) * entry_signal * 100
                bars_held = exit_idx - entry_idx
                
                trades.append({
                    'pct_return': pct_return,
                    'bars_held': bars_held,
                    'direction': 'long' if entry_signal > 0 else 'short'
                })
                
                # Check if flipping position
                if curr['val'] != 0:
                    entry_price = curr['px']
                    entry_idx = curr['idx']
                    entry_signal = curr['val']
                else:
                    entry_price = None
                    entry_idx = None
                    entry_signal = None
        
        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            avg_return = trades_df['pct_return'].mean()
            avg_return_bps = avg_return * 100
            win_rate = (trades_df['pct_return'] > 0).mean()
            
            # Annual projection (assuming same data period as previous workspace)
            total_bars = metadata.get('total_bars', 16614)
            bars_per_year = 78 * 252
            trades_per_year = len(trades) * (bars_per_year / total_bars) if total_bars > 0 else 0
            
            results.append({
                'strategy_id': i,
                'trades': len(trades),
                'avg_return_pct': avg_return,
                'avg_return_bps': avg_return_bps,
                'win_rate': win_rate,
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
                'trades_per_year': 0,
                'long_trades': 0,
                'short_trades': 0
            })
            
    except Exception as e:
        print(f"Error processing strategy {i}: {e}")
        continue

results_df = pd.DataFrame(results)
print(f"\nAnalyzed {len(results_df)} strategy variations")

# Filter for active strategies
active_strategies = results_df[results_df['trades'] > 10]
print(f"Active strategies (>10 trades): {len(active_strategies)}")

if len(active_strategies) > 0:
    # Top by return
    print("\n=== TOP 15 STRATEGIES BY AVERAGE RETURN (BPS) ===")
    top_by_return = active_strategies.nlargest(15, 'avg_return_bps')[
        ['strategy_id', 'trades', 'avg_return_bps', 'win_rate', 'trades_per_year']
    ].round(2)
    print(top_by_return.to_string(index=False))
    
    # Most active
    print("\n=== TOP 10 MOST ACTIVE STRATEGIES ===")
    top_by_trades = active_strategies.nlargest(10, 'trades')[
        ['strategy_id', 'trades', 'avg_return_bps', 'win_rate', 'trades_per_year']
    ].round(2)
    print(top_by_trades.to_string(index=False))
    
    # Best overall (edge × frequency)
    active_strategies['annual_bps'] = active_strategies['avg_return_bps'] * active_strategies['trades_per_year']
    
    print("\n=== TOP 15 BY EXPECTED ANNUAL BPS ===")
    top_overall = active_strategies.nlargest(15, 'annual_bps')[
        ['strategy_id', 'trades', 'avg_return_bps', 'trades_per_year', 'annual_bps']
    ].round(2)
    print(top_overall.to_string(index=False))
    
    # Strategies with good balance
    print("\n=== BALANCED STRATEGIES (>100 trades, >1 bps) ===")
    balanced = active_strategies[
        (active_strategies['trades'] > 100) & 
        (active_strategies['avg_return_bps'] > 1.0)
    ].sort_values('annual_bps', ascending=False)
    
    if len(balanced) > 0:
        print(balanced[['strategy_id', 'trades', 'avg_return_bps', 'trades_per_year', 'annual_bps']].round(2).to_string(index=False))
    else:
        print("No strategies meet these criteria")
    
    # Distribution analysis
    print("\n=== PERFORMANCE DISTRIBUTION ===")
    print(f"Strategies with positive edge: {(active_strategies['avg_return_bps'] > 0).sum()}")
    print(f"Strategies with >1 bps edge: {(active_strategies['avg_return_bps'] > 1).sum()}")
    print(f"Strategies with >5 bps edge: {(active_strategies['avg_return_bps'] > 5).sum()}")
    print(f"Strategies with >10 bps edge: {(active_strategies['avg_return_bps'] > 10).sum()}")
    
    # Compare workspaces
    print("\n=== WORKSPACE COMPARISON ===")
    print("\nWorkspace bfe07b48 (previous):")
    print("- Best edge: 1.89 bps")
    print("- Best annual: 167.6 bps")
    print("- Most trades: 307")
    
    print(f"\nWorkspace 320d109d (current):")
    if len(active_strategies) > 0:
        print(f"- Best edge: {active_strategies['avg_return_bps'].max():.2f} bps")
        print(f"- Best annual: {active_strategies['annual_bps'].max():.2f} bps")
        print(f"- Most trades: {active_strategies['trades'].max()}")
    
    # Detailed look at top performer
    if len(active_strategies) > 0:
        best_id = active_strategies.nlargest(1, 'annual_bps')['strategy_id'].iloc[0]
        print(f"\n=== DETAILED ANALYSIS: STRATEGY {best_id} ===")
        
        signal_file = signal_dir / f"SPY_5m_compiled_strategy_{best_id}.parquet"
        signals = pd.read_parquet(signal_file)
        
        # Re-analyze for more detail
        trades = []
        entry_price = None
        
        for j in range(len(signals)):
            curr = signals.iloc[j]
            
            if entry_price is None and curr['val'] != 0:
                entry_price = curr['px']
                entry_signal = curr['val']
            elif entry_price is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_signal)):
                pct_return = (curr['px'] / entry_price - 1) * entry_signal * 100
                trades.append(pct_return)
                
                if curr['val'] != 0:
                    entry_price = curr['px']
                    entry_signal = curr['val']
                else:
                    entry_price = None
        
        if trades:
            returns = pd.Series(trades)
            print(f"Total trades: {len(returns)}")
            print(f"Average return: {returns.mean():.4f}% ({returns.mean()*100:.2f} bps)")
            print(f"Win rate: {(returns > 0).mean():.1%}")
            print(f"Best trade: {returns.max():.2f}%")
            print(f"Worst trade: {returns.min():.2f}%")
            print(f"Sharpe ratio: {returns.mean() / returns.std():.2f}")