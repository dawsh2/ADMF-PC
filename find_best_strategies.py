#!/usr/bin/env python3
"""
Find best swing pivot strategies across workspaces with realistic criteria.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Import our analyzer
sys.path.append('.')
from analyze_swing_pivot_advanced import AdvancedSwingPivotAnalyzer

def analyze_workspace_with_criteria(workspace_path: str, 
                                  min_return_per_trade_bps: float = 1.0,
                                  min_trades_per_day: float = 0.3,
                                  stop_losses_to_test: list = None):
    """Analyze workspace and find best strategies with various stop losses."""
    
    if stop_losses_to_test is None:
        stop_losses_to_test = [None, 0.002, 0.003, 0.005, 0.0075, 0.01]
    
    analyzer = AdvancedSwingPivotAnalyzer(workspace_path)
    workspace_name = Path(workspace_path).name
    
    all_results = []
    
    for stop_loss in stop_losses_to_test:
        print(f"\nAnalyzing {workspace_name} with stop loss: {stop_loss*100 if stop_loss else 'None'}%")
        
        results = analyzer.analyze_all_strategies_with_filters(
            min_return_per_trade_bps=min_return_per_trade_bps,
            min_trades_per_day=min_trades_per_day,
            stop_loss_pct=stop_loss,
            limit=None  # Analyze all
        )
        
        results['stop_loss_pct'] = stop_loss
        results['stop_loss_bps'] = stop_loss * 10000 if stop_loss else 0
        results['workspace'] = workspace_name
        
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)

def main():
    # Workspaces to analyze
    workspaces = [
        "/Users/daws/ADMF-PC/workspaces/signal_generation_a2d31737",
        "/Users/daws/ADMF-PC/workspaces/swing_pivot_bounce_zones_5m_high_freq_20250622_100151"
    ]
    
    # Criteria
    min_return_per_trade_bps = 1.0  # 1 bps minimum
    min_trades_per_day = 0.3  # More realistic given the data
    
    all_workspace_results = []
    
    for workspace in workspaces:
        print(f"\n{'='*60}")
        print(f"Analyzing workspace: {workspace}")
        print(f"{'='*60}")
        
        try:
            results = analyze_workspace_with_criteria(
                workspace,
                min_return_per_trade_bps=min_return_per_trade_bps,
                min_trades_per_day=min_trades_per_day
            )
            all_workspace_results.append(results)
        except Exception as e:
            print(f"Error analyzing {workspace}: {e}")
            continue
    
    # Combine all results
    if all_workspace_results:
        combined_df = pd.concat(all_workspace_results, ignore_index=True)
        
        # Save all results
        combined_df.to_csv("all_strategies_stop_loss_analysis.csv", index=False)
        
        # Filter to best performers
        best_strategies = combined_df[
            (combined_df['avg_return_per_trade_bps'] >= min_return_per_trade_bps) &
            (combined_df['trades_per_day'] >= min_trades_per_day)
        ].sort_values('avg_return_per_trade_bps', ascending=False)
        
        print(f"\n{'='*60}")
        print("SUMMARY: Best Strategies Found")
        print(f"{'='*60}")
        print(f"Total strategies analyzed: {len(combined_df)}")
        print(f"Strategies meeting criteria: {len(best_strategies)}")
        
        if len(best_strategies) > 0:
            print(f"\nTop 20 strategies by return per trade:")
            print(f"{'Strategy':<40} {'Workspace':<30} {'Stop Loss':<10} {'RPT (bps)':<10} {'TPD':<8} {'Win%':<8} {'Trades':<8}")
            print("-" * 120)
            
            for idx, row in best_strategies.head(20).iterrows():
                stop_str = f"{row['stop_loss_bps']:.0f}bps" if row['stop_loss_pct'] else "None"
                print(f"{row['strategy_id']:<40} {row['workspace'][:28]:<30} {stop_str:<10} "
                      f"{row['avg_return_per_trade_bps']:>8.2f} {row['trades_per_day']:>6.2f} "
                      f"{row['win_rate']*100:>6.1f}% {row['num_trades']:>7}")
            
            # Analyze stop loss impact
            print(f"\n{'='*60}")
            print("Stop Loss Impact Analysis")
            print(f"{'='*60}")
            
            stop_loss_summary = best_strategies.groupby('stop_loss_bps').agg({
                'strategy_id': 'count',
                'avg_return_per_trade_bps': 'mean',
                'stop_rate': 'mean',
                'win_rate': 'mean'
            }).round(2)
            
            stop_loss_summary.columns = ['Count', 'Avg RPT (bps)', 'Avg Stop Rate', 'Avg Win Rate']
            print(stop_loss_summary)
            
            # Save best strategies
            best_strategies.to_csv("best_swing_pivot_strategies.csv", index=False)
            print(f"\nBest strategies saved to: best_swing_pivot_strategies.csv")
            
            # Additional insights
            print(f"\n{'='*60}")
            print("Key Insights")
            print(f"{'='*60}")
            
            # Best without stop loss
            no_stop = best_strategies[best_strategies['stop_loss_pct'].isna()]
            if len(no_stop) > 0:
                print(f"\nBest strategy without stop loss:")
                best_no_stop = no_stop.iloc[0]
                print(f"  {best_no_stop['strategy_id']} ({best_no_stop['workspace']})")
                print(f"  Return per trade: {best_no_stop['avg_return_per_trade_bps']:.2f} bps")
                print(f"  Trades per day: {best_no_stop['trades_per_day']:.2f}")
                print(f"  Win rate: {best_no_stop['win_rate']*100:.1f}%")
            
            # Best with stop loss
            with_stop = best_strategies[best_strategies['stop_loss_pct'].notna()]
            if len(with_stop) > 0:
                print(f"\nBest strategy with stop loss:")
                best_with_stop = with_stop.iloc[0]
                print(f"  {best_with_stop['strategy_id']} ({best_with_stop['workspace']})")
                print(f"  Stop loss: {best_with_stop['stop_loss_bps']:.0f} bps")
                print(f"  Return per trade: {best_with_stop['avg_return_per_trade_bps']:.2f} bps")
                print(f"  Trades per day: {best_with_stop['trades_per_day']:.2f}")
                print(f"  Win rate: {best_with_stop['win_rate']*100:.1f}%")
                print(f"  Stop rate: {best_with_stop['stop_rate']*100:.1f}%")
            
            # Optimal stop loss
            stop_loss_perf = best_strategies.groupby('stop_loss_bps')['avg_return_per_trade_bps'].agg(['mean', 'count'])
            optimal_stop = stop_loss_perf['mean'].idxmax()
            print(f"\nOptimal stop loss appears to be: {optimal_stop:.0f} bps")
            print(f"  Strategies with this stop: {stop_loss_perf.loc[optimal_stop, 'count']}")
            print(f"  Average return per trade: {stop_loss_perf.loc[optimal_stop, 'mean']:.2f} bps")


if __name__ == "__main__":
    main()