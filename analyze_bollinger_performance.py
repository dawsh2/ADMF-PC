#!/usr/bin/env python3
"""
Analyze Bollinger Bands parameter sweep with actual performance metrics.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def decode_parameters(strategy_num, periods, std_devs):
    """Decode strategy number to parameters."""
    std_idx = strategy_num % len(std_devs)
    period_idx = strategy_num // len(std_devs)
    
    if period_idx < len(periods):
        return periods[period_idx], std_devs[std_idx]
    return None, None

def calculate_performance(signals_path, market_data):
    """Calculate performance metrics for a strategy."""
    try:
        signals = pd.read_parquet(signals_path)
        
        # Convert sparse signals to full array
        max_idx = min(signals['idx'].max(), len(market_data) - 1)
        full_signals = np.zeros(len(market_data))
        
        for _, row in signals.iterrows():
            idx = int(row['idx'])
            if idx < len(full_signals):
                full_signals[idx] = row['val']
        
        # Forward fill signals
        for i in range(1, len(full_signals)):
            if full_signals[i] == 0:
                full_signals[i] = full_signals[i-1]
        
        # Calculate returns
        returns = market_data['close'].pct_change()
        positions = pd.Series(full_signals).shift(1).fillna(0)
        strategy_returns = positions * returns
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        if strategy_returns.std() > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 78)
        else:
            sharpe = 0
        
        # Win rate
        winning_days = (strategy_returns > 0).sum()
        losing_days = (strategy_returns < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        
        # Number of trades
        position_changes = positions.diff().fillna(0)
        num_trades = (position_changes != 0).sum() // 2
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'signal_count': len(signals)
        }
    except Exception as e:
        return None

def main():
    print("Bollinger Bands Performance Analysis")
    print("=" * 50)
    
    # Load market data
    market_data = pd.read_parquet('data/SPY_5m.parquet')
    print(f"Market data: {len(market_data)} bars")
    
    # Parameter grid from config
    periods = list(range(10, 50, 1))  # 40 values
    std_devs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # 7 values
    total_combinations = len(periods) * len(std_devs)
    
    print(f"\nParameter grid:")
    print(f"  Periods: {periods[0]} to {periods[-1]} (step 1)")
    print(f"  Std devs: {std_devs}")
    print(f"  Total combinations: {total_combinations}")
    
    # Path to results
    results_path = Path("config/bollinger/results/latest")
    metadata_path = results_path / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Analyze each strategy
    results = []
    
    print(f"\nAnalyzing {len(metadata['components'])} strategies...")
    
    for i, (comp_id, comp_data) in enumerate(metadata['components'].items()):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(metadata['components'])}")
            
        if comp_data['component_type'] != 'strategy':
            continue
        
        # Extract strategy number
        if 'compiled_strategy_' in comp_id:
            strategy_num = int(comp_id.split('compiled_strategy_')[1])
        else:
            continue
        
        # Decode parameters
        period, std_dev = decode_parameters(strategy_num, periods, std_devs)
        if period is None:
            continue
        
        # Calculate performance
        signal_path = results_path / comp_data['signal_file_path']
        perf = calculate_performance(signal_path, market_data)
        
        if perf:
            result = {
                'strategy_num': strategy_num,
                'period': period,
                'std_dev': std_dev,
                **perf
            }
            results.append(result)
    
    # Create DataFrame and analyze
    df = pd.DataFrame(results)
    
    # Sort by total return
    df_sorted = df.sort_values('total_return', ascending=False)
    
    print(f"\n\nTop 20 strategies by total return:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Period':<7} {'StdDev':<7} {'Return':<10} {'Sharpe':<8} {'WinRate':<8} {'Trades':<8}")
    print("-" * 80)
    
    for i, row in df_sorted.head(20).iterrows():
        rank = df_sorted.index.get_loc(i) + 1
        print(f"{rank:<5} {row['period']:<7} {row['std_dev']:<7.1f} "
              f"{row['total_return']:>9.2%} {row['sharpe']:>7.2f} "
              f"{row['win_rate']:>7.1%} {row['num_trades']:>7}")
    
    # Find specific combination
    target = df[(df['period'] == 11) & (df['std_dev'] == 2.0)]
    if not target.empty:
        row = target.iloc[0]
        rank = df_sorted.index.get_loc(target.index[0]) + 1
        print(f"\n\nTarget strategy (period=11, std_dev=2.0):")
        print(f"  Rank: {rank} out of {len(df)}")
        print(f"  Total return: {row['total_return']:.2%}")
        print(f"  Sharpe ratio: {row['sharpe']:.2f}")
        print(f"  Win rate: {row['win_rate']:.1%}")
        print(f"  Number of trades: {row['num_trades']}")
    
    # Heatmap data
    print("\n\nGenerating performance heatmap...")
    pivot_return = df.pivot_table(values='total_return', index='period', columns='std_dev')
    pivot_sharpe = df.pivot_table(values='sharpe', index='period', columns='std_dev')
    
    # Find best parameters
    best_idx = df['total_return'].idxmax()
    best = df.loc[best_idx]
    print(f"\nBest performing strategy:")
    print(f"  Period: {best['period']}, Std Dev: {best['std_dev']}")
    print(f"  Total return: {best['total_return']:.2%}")
    print(f"  Sharpe ratio: {best['sharpe']:.2f}")
    print(f"  Win rate: {best['win_rate']:.1%}")
    
    # Save results
    df.to_csv('bollinger_performance_analysis.csv', index=False)
    pivot_return.to_csv('bollinger_returns_heatmap.csv')
    pivot_sharpe.to_csv('bollinger_sharpe_heatmap.csv')
    
    print("\n\nResults saved to:")
    print("  - bollinger_performance_analysis.csv (all results)")
    print("  - bollinger_returns_heatmap.csv (returns by period/stddev)")
    print("  - bollinger_sharpe_heatmap.csv (sharpe by period/stddev)")

if __name__ == "__main__":
    main()