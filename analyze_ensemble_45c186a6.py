#!/usr/bin/env python3
"""
Ensemble and filtering analysis for workspace signal_generation_45c186a6
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns

def load_all_strategies(workspace_path):
    """Load all strategy signals and metadata."""
    traces_path = workspace_path / "traces"
    parquet_files = list(traces_path.rglob("*.parquet"))
    
    strategies = {}
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            strategy_type = pf.parent.name
            strategy_id = pf.stem
            
            # Convert to time series of positions
            positions = pd.Series(0, index=range(df['idx'].max() + 1))
            
            current_pos = 0
            for _, row in df.iterrows():
                current_pos = row['val']
                positions[row['idx']:] = current_pos
            
            strategies[strategy_id] = {
                'type': strategy_type,
                'positions': positions,
                'signals_df': df,
                'file': pf
            }
        except Exception as e:
            print(f"Error loading {pf}: {e}")
    
    return strategies

def calculate_strategy_metrics(strategies):
    """Calculate performance metrics for each strategy."""
    metrics = []
    
    for strat_id, strat_data in strategies.items():
        df = strat_data['signals_df']
        
        # Calculate trades
        trades = []
        entry_price = None
        entry_signal = None
        entry_idx = None
        
        for _, row in df.iterrows():
            signal = row['val']
            price = row['px']
            bar_idx = row['idx']
            
            if entry_price is None and signal != 0:
                entry_price = price
                entry_signal = signal
                entry_idx = bar_idx
            elif entry_price is not None and (signal == 0 or signal != entry_signal):
                if entry_signal > 0:
                    pnl_pct = (price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - price) / entry_price * 100
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': bar_idx,
                    'duration': bar_idx - entry_idx,
                    'pnl_pct': pnl_pct
                })
                
                if signal != 0:
                    entry_price = price
                    entry_signal = signal
                    entry_idx = bar_idx
                else:
                    entry_price = None
        
        if trades:
            trades_df = pd.DataFrame(trades)
            metrics.append({
                'strategy_id': strat_id,
                'strategy_type': strat_data['type'],
                'num_trades': len(trades_df),
                'total_return': trades_df['pnl_pct'].sum(),
                'avg_return': trades_df['pnl_pct'].mean(),
                'win_rate': (trades_df['pnl_pct'] > 0).mean(),
                'avg_duration': trades_df['duration'].mean(),
                'max_duration': trades_df['duration'].max(),
                'sharpe': trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() if trades_df['pnl_pct'].std() > 0 else 0
            })
    
    return pd.DataFrame(metrics)

def analyze_correlations(strategies, metrics_df):
    """Analyze correlations between strategies."""
    # Get top performers by type
    top_by_type = {}
    for stype in metrics_df['strategy_type'].unique():
        type_df = metrics_df[metrics_df['strategy_type'] == stype]
        # Filter for reasonable trade counts and positive returns
        filtered = type_df[(type_df['num_trades'] > 10) & (type_df['total_return'] > 0)]
        if len(filtered) > 0:
            top = filtered.nlargest(3, 'total_return')
            top_by_type[stype] = top['strategy_id'].tolist()
    
    # Calculate position correlations
    print("\nPosition Correlations Between Top Strategies:")
    print("=" * 60)
    
    position_data = {}
    for stype, strat_ids in top_by_type.items():
        for strat_id in strat_ids:
            if strat_id in strategies:
                position_data[f"{stype}_{strat_id[-2:]}"] = strategies[strat_id]['positions']
    
    if position_data:
        positions_df = pd.DataFrame(position_data)
        corr_matrix = positions_df.corr()
        
        # Find low correlation pairs
        low_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) < 0.3:  # Low correlation threshold
                    low_corr_pairs.append({
                        'strat1': corr_matrix.columns[i],
                        'strat2': corr_matrix.columns[j],
                        'correlation': corr
                    })
        
        print("\nLow Correlation Strategy Pairs (|corr| < 0.3):")
        for pair in sorted(low_corr_pairs, key=lambda x: abs(x['correlation']))[:10]:
            print(f"{pair['strat1']} <-> {pair['strat2']}: {pair['correlation']:.3f}")
    
    return top_by_type, position_data

def create_filtered_ensemble(strategies, metrics_df):
    """Create ensemble strategies with various filters."""
    print("\n" + "="*60)
    print("ENSEMBLE ANALYSIS")
    print("="*60)
    
    # Filter 1: Quality strategies (high win rate, reasonable trades)
    quality_filter = metrics_df[
        (metrics_df['win_rate'] > 0.7) & 
        (metrics_df['num_trades'].between(50, 500)) &
        (metrics_df['total_return'] > 5)
    ].copy()
    
    print(f"\nQuality Filter (win_rate > 70%, 50-500 trades, return > 5%):")
    print(f"Found {len(quality_filter)} strategies")
    
    if len(quality_filter) > 0:
        # Calculate ensemble metrics
        avg_trades = quality_filter['num_trades'].mean()
        avg_return = quality_filter['total_return'].mean()
        
        # Estimate ensemble with execution costs
        cost_per_trade = 0.01  # 1bp round trip
        avg_cost = avg_trades * cost_per_trade
        net_return = avg_return - avg_cost
        
        print(f"Average gross return: {avg_return:.2f}%")
        print(f"Average trades: {avg_trades:.0f}")
        print(f"Estimated cost: {avg_cost:.2f}%")
        print(f"Estimated net return: {net_return:.2f}%")
        
        print("\nTop quality strategies:")
        print(quality_filter.nlargest(5, 'total_return')[
            ['strategy_type', 'strategy_id', 'total_return', 'win_rate', 'num_trades']
        ].to_string(index=False))
    
    # Filter 2: Low frequency, high return per trade
    efficient_filter = metrics_df[
        (metrics_df['avg_return'] > 0.05) & 
        (metrics_df['num_trades'] < 200) &
        (metrics_df['win_rate'] > 0.8)
    ].copy()
    
    print(f"\n\nEfficient Filter (avg return > 0.05%, <200 trades, win_rate > 80%):")
    print(f"Found {len(efficient_filter)} strategies")
    
    if len(efficient_filter) > 0:
        avg_trades = efficient_filter['num_trades'].mean()
        avg_return = efficient_filter['total_return'].mean()
        avg_cost = avg_trades * cost_per_trade
        net_return = avg_return - avg_cost
        
        print(f"Average gross return: {avg_return:.2f}%")
        print(f"Average trades: {avg_trades:.0f}")
        print(f"Estimated cost: {avg_cost:.2f}%")
        print(f"Estimated net return: {net_return:.2f}%")
    
    # Ensemble by strategy type
    print("\n\nOptimal Mix by Strategy Type:")
    print("-" * 40)
    
    ensemble_components = []
    for stype in metrics_df['strategy_type'].unique():
        type_df = metrics_df[metrics_df['strategy_type'] == stype]
        
        # Find best net return strategy after costs
        type_df['net_return'] = type_df['total_return'] - type_df['num_trades'] * cost_per_trade
        best = type_df.nlargest(1, 'net_return')
        
        if len(best) > 0 and best.iloc[0]['net_return'] > 0:
            ensemble_components.append({
                'type': stype,
                'strategy': best.iloc[0]['strategy_id'],
                'gross_return': best.iloc[0]['total_return'],
                'trades': best.iloc[0]['num_trades'],
                'net_return': best.iloc[0]['net_return'],
                'win_rate': best.iloc[0]['win_rate']
            })
    
    ensemble_df = pd.DataFrame(ensemble_components)
    if len(ensemble_df) > 0:
        print(ensemble_df.to_string(index=False))
        
        # Equal weight ensemble
        print(f"\nEqual-weight ensemble estimated net return: {ensemble_df['net_return'].mean():.2f}%")
        
        # Sharpe-weighted ensemble
        if 'sharpe' in metrics_df.columns:
            sharpe_weights = []
            for comp in ensemble_components:
                sharpe = metrics_df[metrics_df['strategy_id'] == comp['strategy']]['sharpe'].iloc[0]
                sharpe_weights.append(max(0, sharpe))
            
            if sum(sharpe_weights) > 0:
                sharpe_weights = np.array(sharpe_weights) / sum(sharpe_weights)
                sharpe_weighted_return = sum(sharpe_weights * ensemble_df['net_return'])
                print(f"Sharpe-weighted ensemble estimated net return: {sharpe_weighted_return:.2f}%")

def analyze_regime_patterns(strategies, metrics_df):
    """Analyze when different strategies perform well."""
    print("\n" + "="*60)
    print("TRADE TIMING ANALYSIS")
    print("="*60)
    
    # Get one good strategy from each type
    examples = {}
    for stype in metrics_df['strategy_type'].unique():
        type_df = metrics_df[metrics_df['strategy_type'] == stype]
        type_df['net_return'] = type_df['total_return'] - type_df['num_trades'] * 0.01
        best = type_df[type_df['net_return'] > 0].nlargest(1, 'net_return')
        if len(best) > 0:
            examples[stype] = best.iloc[0]['strategy_id']
    
    # Analyze trading patterns
    trade_patterns = defaultdict(list)
    
    for stype, strat_id in examples.items():
        if strat_id in strategies:
            signals = strategies[strat_id]['signals_df']
            for _, row in signals.iterrows():
                if row['val'] != 0:  # Trade entry
                    trade_patterns[stype].append(row['idx'])
    
    # Find overlapping and unique trading periods
    print("\nTrading Activity by Bar Index:")
    for stype, indices in trade_patterns.items():
        if indices:
            print(f"{stype}: {len(indices)} entries, bars {min(indices)}-{max(indices)}")
    
    # Check for complementary patterns
    if len(trade_patterns) >= 2:
        types = list(trade_patterns.keys())
        for i in range(len(types)):
            for j in range(i+1, len(types)):
                set1 = set(trade_patterns[types[i]])
                set2 = set(trade_patterns[types[j]])
                overlap = len(set1 & set2)
                total = len(set1 | set2)
                overlap_pct = overlap / total * 100 if total > 0 else 0
                print(f"\n{types[i]} vs {types[j]}: {overlap_pct:.1f}% overlap")
                
                if overlap_pct < 30:
                    print(f"  -> Good ensemble candidates (low overlap)")

def main():
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_45c186a6")
    
    print("Loading strategies...")
    strategies = load_all_strategies(workspace_path)
    print(f"Loaded {len(strategies)} strategies")
    
    print("\nCalculating metrics...")
    metrics_df = calculate_strategy_metrics(strategies)
    
    # Correlation analysis
    top_by_type, position_data = analyze_correlations(strategies, metrics_df)
    
    # Ensemble creation
    create_filtered_ensemble(strategies, metrics_df)
    
    # Regime/timing analysis
    analyze_regime_patterns(strategies, metrics_df)
    
    # Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. Focus on VWAP deviation strategies with <200 trades")
    print("2. Consider pivot bounces ONLY with strict trade filters")
    print("3. Ensemble low-correlation strategies from different types")
    print("4. Apply time-of-day or volatility filters to reduce trade count")

if __name__ == "__main__":
    main()