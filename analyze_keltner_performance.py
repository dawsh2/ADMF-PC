"""Analyze performance of Keltner Bands strategies using sparse trace analysis"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Import the sparse trace analysis modules
from src.analytics.sparse_trace_analysis.performance_calculation import (
    calculate_log_returns_with_costs, 
    ExecutionCostConfig
)
from src.analytics.sparse_trace_analysis.strategy_analysis import load_strategy_signals

# Set up workspace path
workspace_path = Path("workspaces/signal_generation_310b2aeb")

# Get all strategy files
strategy_files = sorted(glob.glob(str(workspace_path / "traces/SPY_1m/signals/keltner_bands/*.parquet")))
print(f"Found {len(strategy_files)} strategy files")

# Set up execution costs
# 1bp (0.01%) per side = 0.02% round trip = 0.9998 multiplier
cost_config = ExecutionCostConfig(cost_multiplier=0.9998)

# Analyze each strategy
results = []

for file_path in strategy_files:
    try:
        # Extract strategy name
        strategy_path = Path(file_path)
        strategy_name = strategy_path.stem
        strategy_num = int(strategy_name.split('_')[-1])
        
        # Load sparse signal data
        signals_df = load_strategy_signals(strategy_path)
        
        if signals_df is None:
            continue
            
        # Calculate performance
        performance = calculate_log_returns_with_costs(
            signals_df=signals_df,
            cost_config=cost_config,
            initial_capital=10000.0
        )
        
        # Extract key metrics
        if performance and performance['num_trades'] > 0:
            # Calculate average bars per trade
            avg_bars = sum(t['bars_held'] for t in performance['trades']) / len(performance['trades'])
            
            # Calculate average percentage return per trade
            avg_pct_per_trade = (np.exp(performance['avg_trade_log_return']) - 1) * 100
            
            results.append({
                'strategy_num': strategy_num,
                'strategy_name': strategy_name,
                'num_trades': performance['num_trades'],
                'total_log_return': performance['total_log_return'],
                'total_pct_return': performance['percentage_return'] * 100,  # Convert to percentage
                'avg_log_return_per_trade': performance['avg_trade_log_return'],
                'avg_pct_return_per_trade': avg_pct_per_trade,
                'win_rate': performance['win_rate'],
                'max_drawdown_pct': performance['max_drawdown_pct'] * 100,  # Convert to percentage
                'avg_bars_per_trade': avg_bars
            })
        
    except Exception as e:
        print(f"Error processing {strategy_name}: {e}")

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

if len(results_df) > 0:
    # Sort by average return per trade
    results_df = results_df.sort_values('avg_pct_return_per_trade', ascending=False)
    
    # Calculate basis points (already in percentage, so multiply by 100)
    results_df['avg_bps_per_trade'] = results_df['avg_pct_return_per_trade'] * 100
    results_df['total_return_bps'] = results_df['total_pct_return'] * 100
    
    # Calculate trades per day (assuming ~390 minutes per day for 1m data)
    # Total bars in dataset: 81,787
    total_days = 81787 / 390  
    results_df['trades_per_day'] = results_df['num_trades'] / total_days
    
    print("\n=== TOP PERFORMERS BY AVERAGE BPS PER TRADE ===")
    top_10 = results_df.head(10)
    for _, row in top_10.iterrows():
        print(f"\nStrategy {row['strategy_num']:2d}:")
        print(f"  Avg per trade: {row['avg_bps_per_trade']:.2f} bps")
        print(f"  Total return: {row['total_return_bps']:.2f} bps")
        print(f"  Trades/day: {row['trades_per_day']:.1f}")
        print(f"  Win rate: {row['win_rate']:.1%}")
        print(f"  Num trades: {row['num_trades']}")
        print(f"  Avg bars/trade: {row['avg_bars_per_trade']:.0f}")
    
    # Find strategies meeting criteria (>1 bps edge, 2-3 trades/day)
    print("\n=== STRATEGIES MEETING CRITERIA ===")
    print("Criteria: >1 bps per trade after costs, 2-3+ trades per day")
    
    criteria_met = results_df[
        (results_df['avg_bps_per_trade'] > 1.0) & 
        (results_df['trades_per_day'] >= 2.0)
    ]
    
    if len(criteria_met) > 0:
        print(f"\nFound {len(criteria_met)} strategies meeting criteria:")
        criteria_met_sorted = criteria_met.sort_values('avg_bps_per_trade', ascending=False)
        for _, row in criteria_met_sorted.iterrows():
            print(f"\nStrategy {row['strategy_num']:2d}:")
            print(f"  Avg per trade: {row['avg_bps_per_trade']:.2f} bps")
            print(f"  Trades/day: {row['trades_per_day']:.1f}")
            print(f"  Total return: {row['total_return_bps']:.2f} bps")
            print(f"  Win rate: {row['win_rate']:.1%}")
            print(f"  Avg bars/trade: {row['avg_bars_per_trade']:.0f}")
    else:
        print("\nNo strategies met the criteria")
    
    # Summary statistics by trade frequency
    print("\n=== PERFORMANCE BY TRADE FREQUENCY ===")
    results_df['frequency_group'] = pd.cut(
        results_df['trades_per_day'], 
        bins=[0, 1, 3, 5, 10, 100],
        labels=['<1/day', '1-3/day', '3-5/day', '5-10/day', '>10/day']
    )
    
    freq_summary = results_df.groupby('frequency_group', observed=True).agg({
        'avg_bps_per_trade': ['mean', 'std', 'count'],
        'win_rate': 'mean',
        'avg_bars_per_trade': 'mean'
    })
    print(freq_summary)
    
    # Show worst performers to understand filter impact
    print("\n=== WORST PERFORMERS (NEGATIVE EDGE) ===")
    worst = results_df.tail(5)
    for _, row in worst.iterrows():
        print(f"\nStrategy {row['strategy_num']:2d}:")
        print(f"  Avg per trade: {row['avg_bps_per_trade']:.2f} bps")
        print(f"  Trades/day: {row['trades_per_day']:.1f}")
        print(f"  Win rate: {row['win_rate']:.1%}")
    
    # Save detailed results
    results_df.to_csv('keltner_performance_results.csv', index=False)
    print("\n\nDetailed results saved to keltner_performance_results.csv")
    
    # Summary statistics
    print("\n=== OVERALL SUMMARY ===")
    print(f"Total strategies analyzed: {len(results_df)}")
    print(f"Strategies with positive edge: {len(results_df[results_df['avg_bps_per_trade'] > 0])}")
    print(f"Strategies meeting criteria: {len(criteria_met)}")
    print(f"Best avg bps/trade: {results_df['avg_bps_per_trade'].max():.2f}")
    print(f"Worst avg bps/trade: {results_df['avg_bps_per_trade'].min():.2f}")
    print(f"Mean avg bps/trade: {results_df['avg_bps_per_trade'].mean():.2f}")
    
else:
    print("\nNo results to analyze")